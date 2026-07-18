import { describe, expect, it } from "vitest";
import { analyzeDecodedPcm, inspectEncodedAudio } from "../src/audio/audioAnalysis";

function writeAscii(bytes: Uint8Array, offset: number, value: string): void {
  for (let index = 0; index < value.length; index += 1) bytes[offset + index] = value.charCodeAt(index);
}

function createWavHeader(sampleRateHz = 48_000, channels = 2, seconds = 1): Uint8Array {
  const bitDepth = 16;
  const blockAlign = channels * (bitDepth / 8);
  const byteRate = sampleRateHz * blockAlign;
  const dataBytes = byteRate * seconds;
  const bytes = new Uint8Array(44 + dataBytes);
  const view = new DataView(bytes.buffer);
  writeAscii(bytes, 0, "RIFF");
  view.setUint32(4, bytes.length - 8, true);
  writeAscii(bytes, 8, "WAVE");
  writeAscii(bytes, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRateHz, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeAscii(bytes, 36, "data");
  view.setUint32(40, dataBytes, true);
  return bytes;
}

function createMp3Frames(frameCount = 2): Uint8Array {
  // MPEG-1 Layer III, 128 kbps, 44.1 kHz, stereo. Each unpadded frame is 417 bytes.
  const frameLength = 417;
  const bytes = new Uint8Array(frameLength * frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    const offset = frame * frameLength;
    bytes.set([0xff, 0xfb, 0x90, 0x00], offset);
  }
  return bytes;
}

function createMeasuredTrack(sampleRateHz = 48_000, durationSeconds = 16): Float32Array {
  const samples = new Float32Array(sampleRateHz * durationSeconds);
  const beatSamples = sampleRateHz / 2;
  for (let index = 0; index < samples.length; index += 1) {
    const time = index / sampleRateHz;
    const sectionAmplitude = time < 4 ? 0.035 : time < 8 ? 0.18 : time < 12 ? 0.55 : 0.08;
    const tonal = sectionAmplitude * Math.sin(2 * Math.PI * (time < 8 ? 110 : 440) * time);
    const beatOffset = index % beatSamples;
    const click = beatOffset < 700 ? 0.8 * Math.exp(-beatOffset / 100) * (beatOffset % 2 === 0 ? 1 : -1) : 0;
    samples[index] = Math.max(-1, Math.min(1, tonal + click));
  }
  return samples;
}

function createAMinorFixture(sampleRateHz = 48_000, durationSeconds = 6): Float32Array {
  const samples = new Float32Array(sampleRateHz * durationSeconds);
  const partials = [
    { frequency: 110, amplitude: 0.3 },
    { frequency: 220, amplitude: 0.24 },
    { frequency: 261.6256, amplitude: 0.18 },
    { frequency: 329.6276, amplitude: 0.16 },
  ];
  for (let index = 0; index < samples.length; index += 1) {
    const time = index / sampleRateHz;
    samples[index] = partials.reduce(
      (value, partial) => value + partial.amplitude * Math.sin(2 * Math.PI * partial.frequency * time),
      0,
    );
  }
  return samples;
}

function createChromaticFixture(sampleRateHz = 48_000, durationSeconds = 3): Float32Array {
  const samples = new Float32Array(sampleRateHz * durationSeconds);
  const frequencies = Array.from({ length: 12 }, (_, pitchClass) => 261.6256 * 2 ** (pitchClass / 12));
  for (let index = 0; index < samples.length; index += 1) {
    const time = index / sampleRateHz;
    samples[index] = frequencies.reduce(
      (value, frequency) => value + 0.045 * Math.sin(2 * Math.PI * frequency * time),
      0,
    );
  }
  return samples;
}

function createSineFixture(amplitude = 0.1, frequency = 1_000, sampleRateHz = 48_000, durationSeconds = 3): Float32Array {
  const samples = new Float32Array(sampleRateHz * durationSeconds);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = amplitude * Math.sin((2 * Math.PI * frequency * index) / sampleRateHz);
  }
  return samples;
}

describe("encoded audio inspection", () => {
  it("reads actual WAV metadata without assuming the provider sample rate", () => {
    const metadata = inspectEncodedAudio(createWavHeader(48_000, 2, 1), "audio/wav");
    expect(metadata).toMatchObject({
      codec: "wav",
      sampleRateHz: 48_000,
      channels: 2,
      bitDepth: 16,
      frameCount: 48_000,
    });
    expect(metadata.durationSeconds).toBeCloseTo(1, 8);
  });

  it("reads MPEG frame metadata and rejects a mismatched declared MIME type", () => {
    const metadata = inspectEncodedAudio(createMp3Frames(), "audio/mpeg");
    expect(metadata).toMatchObject({
      codec: "mp3",
      sampleRateHz: 44_100,
      channels: 2,
      bitRateKbps: 128,
      frameCount: 2,
    });
    expect(metadata.durationSeconds).toBeCloseTo((2 * 1152) / 44_100, 8);
    expect(() => inspectEncodedAudio(createMp3Frames(), "audio/wav")).toThrow(/MIME type/i);
  });
});

describe("decoded PCM analysis", () => {
  it("detects a synthetic 120 BPM pulse within two BPM", () => {
    const analysis = analyzeDecodedPcm({ sampleRateHz: 48_000, channels: [createMeasuredTrack()] });
    expect(analysis.bpm).not.toBeNull();
    expect(Math.abs((analysis.bpm ?? 0) - 120)).toBeLessThanOrEqual(2);
    expect(analysis.beatGridSeconds.length).toBeGreaterThan(25);
    expect(analysis.onsetMap.length).toBeGreaterThan(20);
  });

  it("does not cancel audible anti-phase stereo during analysis", () => {
    const left = createMeasuredTrack();
    const right = Float32Array.from(left, (sample) => -sample);
    const analysis = analyzeDecodedPcm({ sampleRateHz: 48_000, channels: [left, right] });

    expect(analysis.bpm).not.toBeNull();
    expect(Math.abs((analysis.bpm ?? 0) - 120)).toBeLessThanOrEqual(2);
    expect(analysis.loudnessLufs).toBeGreaterThan(-30);
    expect(analysis.onsetMap.length).toBeGreaterThan(20);
  });

  it("uses K-weighted gated channel energy for integrated loudness", () => {
    const tone = createSineFixture();
    const mono = analyzeDecodedPcm({ sampleRateHz: 48_000, channels: [tone] });
    const stereo = analyzeDecodedPcm({ sampleRateHz: 48_000, channels: [tone, tone] });

    expect(mono.loudnessLufs).toBeCloseTo(-23, 0);
    expect(stereo.loudnessLufs - mono.loudnessLufs).toBeCloseTo(3.01, 1);
  });

  it("produces measured, contiguous sections and stable visual mapping invariants", () => {
    const input = createMeasuredTrack();
    const analysis = analyzeDecodedPcm({ sampleRateHz: 48_000, channels: [input, input] }, { waveformBuckets: 64 });
    expect(analysis.durationSeconds).toBe(16);
    expect(analysis.channels).toBe(2);
    expect(analysis.waveform).toHaveLength(64);
    expect(analysis.sections.length).toBeGreaterThanOrEqual(2);
    expect(analysis.sections[0]?.start).toBe(0);
    expect(analysis.sections.at(-1)?.end).toBe(16);
    for (let index = 0; index < analysis.sections.length; index += 1) {
      const section = analysis.sections[index]!;
      expect(section.end).toBeGreaterThan(section.start);
      if (index > 0) expect(section.start).toBe(analysis.sections[index - 1]?.end);
    }
    const energyTotal =
      analysis.spectralProfile.lowEnergy + analysis.spectralProfile.midEnergy + analysis.spectralProfile.highEnergy;
    expect(energyTotal).toBeCloseTo(1, 5);
    expect(analysis.visualMapping).toEqual({
      bass: "camera_displacement",
      kick: "radial_pulse",
      highFrequencyEnergy: "particle_density",
      sectionChange: "scene_transition",
    });
    expect(["tunnel", "bloom", "terrain"]).toContain(analysis.recommendedScene);
    expect(analysis.visualIntensity).toBeGreaterThanOrEqual(0.25);
    expect(analysis.visualIntensity).toBeLessThanOrEqual(1);
  });

  it("estimates musical key from local PCM and returns null for silence or ambiguous chroma", () => {
    const tonal = analyzeDecodedPcm({ sampleRateHz: 48_000, channels: [createAMinorFixture()] });
    const silence = analyzeDecodedPcm({ sampleRateHz: 48_000, channels: [new Float32Array(48_000 * 2)] });
    const ambiguous = analyzeDecodedPcm({ sampleRateHz: 48_000, channels: [createChromaticFixture()] });

    expect(tonal.key).toBe("A minor");
    expect(silence.key).toBeNull();
    expect(ambiguous.key).toBeNull();
  });

  it("analyzes a full three minute stereo asset within the five second target", () => {
    const sampleRateHz = 48_000;
    const durationSeconds = 180;
    const samples = new Float32Array(sampleRateHz * durationSeconds);
    const beatSamples = sampleRateHz / 2;
    for (let index = 0; index < samples.length; index += 1) {
      const beatOffset = index % beatSamples;
      samples[index] =
        0.08 * Math.sin((2 * Math.PI * 110 * index) / sampleRateHz) +
        (beatOffset < 240 ? 0.65 * Math.exp(-beatOffset / 45) : 0);
    }

    const startedAt = performance.now();
    const analysis = analyzeDecodedPcm({ sampleRateHz, channels: [samples, samples] });
    const elapsedMs = performance.now() - startedAt;

    expect(analysis.durationSeconds).toBe(durationSeconds);
    expect(analysis.waveform).toHaveLength(256);
    expect(elapsedMs).toBeLessThan(5_000);
  }, 15_000);
});
