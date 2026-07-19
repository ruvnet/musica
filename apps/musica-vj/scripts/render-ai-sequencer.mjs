import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const SAMPLE_RATE = 48_000;
const CHANNELS = 2;
const BPM = 72;
const STEPS_PER_BAR = 16;
const STEP_SECONDS = 60 / BPM / 4;
const BARS = 16;
const TAIL_SECONDS = 3;
const SOURCE = resolve("samples/lyria/moonlight-sonata-ai-timbre.mp3");
const OUTPUT = resolve("exports/moonlight-lyria-sequenced-composition.mp3");
const TEMP_WAV = resolve("exports/.moonlight-lyria-sequenced-composition.wav");

const aiTones = {
  bass: { baseNote: 37, grainSeconds: 1.08, level: 0.23, brightness: 0.28, windowStartSeconds: 16, windowDurationSeconds: 28 },
  chords: { baseNote: 61, grainSeconds: 0.74, level: 0.18, brightness: 0.46, windowStartSeconds: 0, windowDurationSeconds: 18 },
  lead: { baseNote: 73, grainSeconds: 0.44, level: 0.2, brightness: 0.82, windowStartSeconds: 10, windowDurationSeconds: 20 },
  voice: { baseNote: 73, grainSeconds: 0.58, level: 0.12, brightness: 0.68, windowStartSeconds: 10, windowDurationSeconds: 20 },
  texture: { baseNote: 49, grainSeconds: 1.36, level: 0.16, brightness: 0.34, windowStartSeconds: 22, windowDurationSeconds: 28 },
};

const tracks = {
  bass: { pattern: [0, 8], notes: [25, 32, 25, 32, 24, 31, 24, 31, 23, 30, 23, 30, 21, 28, 21, 28], pan: -0.08, volume: 0.48 },
  chords: { pattern: [0, 4, 8, 12], notes: [49, 49, 48, 48, 47, 47, 45, 45, 44, 44, 42, 42, 41, 41, 44, 44], pan: -0.18, volume: 0.52 },
  lead: {
    pattern: [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
    notes: [68, 61, 64, 68, 61, 64, 68, 61, 66, 68, 61, 66, 68, 60, 64, 68, 60, 64, 68, 59, 63, 68, 59, 63, 68, 57, 61, 68, 57, 61, 68, 56],
    pan: 0.12,
    volume: 0.5,
  },
  voice: { pattern: [3, 7, 11, 15], notes: [64, 63, 61, 59, 57, 56, 57, 59, 61, 63, 64, 66, 64, 61, 59, 56], pan: 0.26, volume: 0.24 },
  texture: { pattern: [0, 8], notes: [37, 44, 49, 56, 61, 56, 49, 44, 36, 43, 48, 55, 60, 55, 48, 43], pan: 0.32, volume: 0.42 },
};

function run(command, args, options = {}) {
  const result = spawnSync(command, args, { stdio: ["ignore", "pipe", "pipe"], maxBuffer: 256 * 1024 * 1024, ...options });
  if (result.status !== 0) {
    throw new Error(`${command} failed: ${result.stderr.toString("utf8").trim()}`);
  }
  return result.stdout;
}

function decodeSource() {
  if (!existsSync(SOURCE)) throw new Error(`Missing Lyria tone source: ${SOURCE}`);
  const pcm = run("ffmpeg", ["-v", "error", "-i", SOURCE, "-f", "f32le", "-ac", String(CHANNELS), "-ar", String(SAMPLE_RATE), "pipe:1"]);
  return new Float32Array(pcm.buffer, pcm.byteOffset, pcm.byteLength / Float32Array.BYTES_PER_ELEMENT);
}

function sampleSource(source, frame, channel) {
  const maxFrame = source.length / CHANNELS - 2;
  if (frame < 0 || frame >= maxFrame) return 0;
  const base = Math.floor(frame);
  const fraction = frame - base;
  const left = source[base * CHANNELS + channel] ?? 0;
  const right = source[(base + 1) * CHANNELS + channel] ?? 0;
  return left + (right - left) * fraction;
}

function midiToRatio(note, baseNote) {
  return 2 ** ((note - baseNote) / 12);
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function addGrain(output, source, note, timeSeconds, track, tone, durationScale = 1, chordGain = 1) {
  const startsAt = Math.max(0, Math.floor(timeSeconds * SAMPLE_RATE));
  const grainSeconds = clamp(tone.grainSeconds * durationScale, 0.08, 2.4);
  const frames = Math.floor(grainSeconds * SAMPLE_RATE);
  const attackFrames = Math.max(16, Math.floor(Math.min(0.05, grainSeconds * 0.18) * SAMPLE_RATE));
  const releaseStart = Math.floor(frames * 0.72);
  const sourceFrames = source.length / CHANNELS;
  const windowStart = clamp(tone.windowStartSeconds * SAMPLE_RATE, 0, Math.max(0, sourceFrames - 1));
  const windowFrames = clamp(tone.windowDurationSeconds * SAMPLE_RATE, SAMPLE_RATE * 0.05, Math.max(SAMPLE_RATE * 0.05, sourceFrames - windowStart - frames));
  const seed = Math.abs(Math.sin(note * 12.9898 + timeSeconds * 78.233)) % 1;
  const sourceStart = clamp(windowStart + seed * Math.max(1, windowFrames - frames), 0, Math.max(0, sourceFrames - frames - 2));
  const ratio = midiToRatio(note, tone.baseNote);
  const pan = clamp(track.pan, -1, 1);
  const leftGain = Math.cos((pan + 1) * Math.PI * 0.25);
  const rightGain = Math.sin((pan + 1) * Math.PI * 0.25);
  const level = tone.level * track.volume * chordGain;

  for (let index = 0; index < frames; index += 1) {
    const outFrame = startsAt + index;
    if (outFrame >= output.length / CHANNELS) break;
    const sourceFrame = sourceStart + index * ratio;
    const attack = index < attackFrames ? index / attackFrames : 1;
    const release = index > releaseStart ? 1 - (index - releaseStart) / Math.max(1, frames - releaseStart) : 1;
    const envelope = Math.sin(Math.PI * clamp(Math.min(attack, release), 0, 1) * 0.5);
    const lowpass = 0.62 + tone.brightness * 0.38;
    const sample = (sampleSource(source, sourceFrame, 0) + sampleSource(source, sourceFrame, 1)) * 0.5 * envelope * level * lowpass;
    output[outFrame * CHANNELS] += sample * leftGain;
    output[outFrame * CHANNELS + 1] += sample * rightGain;
  }
}

function renderComposition(source) {
  const durationSeconds = BARS * STEPS_PER_BAR * STEP_SECONDS + TAIL_SECONDS;
  const output = new Float32Array(Math.ceil(durationSeconds * SAMPLE_RATE) * CHANNELS);
  const totalSteps = BARS * STEPS_PER_BAR;
  for (let absoluteStep = 0; absoluteStep < totalSteps; absoluteStep += 1) {
    const step = absoluteStep % STEPS_PER_BAR;
    const time = absoluteStep * STEP_SECONDS;
    for (const [id, track] of Object.entries(tracks)) {
      if (!track.pattern.includes(step)) continue;
      const tone = aiTones[id];
      const note = track.notes[Math.floor(absoluteStep / 2) % track.notes.length];
      if (id === "chords") {
        for (const interval of [0, 3, 7]) addGrain(output, source, note + interval, time, track, tone, 1.08, 0.48);
      } else if (id === "texture") {
        addGrain(output, source, note, time, track, tone, 1.24, 0.8);
        addGrain(output, source, note + 12, time + 0.03, track, tone, 1.1, 0.42);
      } else {
        addGrain(output, source, note, time, track, tone);
      }
    }
  }

  let peak = 0;
  for (const sample of output) peak = Math.max(peak, Math.abs(sample));
  const gain = peak > 0 ? 0.89 / peak : 1;
  for (let index = 0; index < output.length; index += 1) output[index] = clamp(output[index] * gain, -0.98, 0.98);
  return output;
}

function writeWav(file, samples) {
  const bytesPerSample = 2;
  const dataBytes = samples.length * bytesPerSample;
  const buffer = Buffer.alloc(44 + dataBytes);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataBytes, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(CHANNELS, 22);
  buffer.writeUInt32LE(SAMPLE_RATE, 24);
  buffer.writeUInt32LE(SAMPLE_RATE * CHANNELS * bytesPerSample, 28);
  buffer.writeUInt16LE(CHANNELS * bytesPerSample, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataBytes, 40);
  for (let index = 0; index < samples.length; index += 1) {
    buffer.writeInt16LE(Math.round(clamp(samples[index], -1, 1) * 32767), 44 + index * bytesPerSample);
  }
  writeFileSync(file, buffer);
}

mkdirSync(dirname(OUTPUT), { recursive: true });
const source = decodeSource();
const rendered = renderComposition(source);
writeWav(TEMP_WAV, rendered);
run("ffmpeg", ["-y", "-v", "error", "-i", TEMP_WAV, "-codec:a", "libmp3lame", "-b:a", "192k", OUTPUT]);
console.log(OUTPUT);
