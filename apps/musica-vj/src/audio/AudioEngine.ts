import {
  clamp,
  createTrackDefinitions,
  defaultMix,
  effectiveTrackGain,
  hashSeed,
  midiToFrequency,
  mutatePattern,
  patternFromSteps,
  secondsPerStep,
  STEPS_PER_BAR,
} from "../core/music";
import { defaultPerformanceTemplate } from "../core/presets";
import { TRACK_IDS, type AudioMetrics, type PerformanceTemplate, type TrackDefinition, type TrackId, type TrackMix, type TrackSnapshot, type TrackTemplate } from "../core/types";
import {
  analyzeDecodedPcm,
  inspectEncodedAudio,
  type AudioAnalysisResult,
  type EncodedAudioMetadata,
} from "./audioAnalysis";

interface TrackRuntime {
  definition: TrackDefinition;
  mix: TrackMix;
  input: GainNode;
  gain: GainNode;
  pan: StereoPannerNode;
  analyser: AnalyserNode;
  fxSend: GainNode;
  meterBuffer: Uint8Array<ArrayBuffer>;
  loadedBuffer?: AudioBuffer;
  loadedFile?: string;
  loadedLoop: boolean;
  loadedStartedAt?: number;
  loopSource?: AudioBufferSourceNode;
  aiTone?: AiToneRuntime;
}

interface ScheduledSourceMetadata {
  startsAt: number;
  imported: boolean;
}

export interface AiToneOptions {
  baseNote?: number;
  grainSeconds?: number;
  level?: number;
  brightness?: number;
  windowStartSeconds?: number;
  windowDurationSeconds?: number;
}

interface AiToneRuntime extends Required<AiToneOptions> {
  buffer: AudioBuffer;
  fileName: string;
}

export interface LoadAudioOptions {
  declaredMimeType?: string;
  loop?: boolean;
  requireEncodedValidation?: boolean;
}

export interface LoadedAudioDetails {
  encoded?: EncodedAudioMetadata;
  analysis: AudioAnalysisResult;
}

export interface LoadedAiToneDetails extends LoadedAudioDetails {
  fileName: string;
}

export interface EngineSnapshot {
  tracks: TrackSnapshot[];
  bpm: number;
  playing: boolean;
  currentStep: number;
  masterVolume: number;
  droppedLateSteps: number;
}

type EngineListener = (snapshot: EngineSnapshot) => void;

const LOOK_AHEAD_SECONDS = 0.12;
const SCHEDULER_INTERVAL_MS = 25;
const MAX_IMPORT_BYTES = 250 * 1024 * 1024;
const MAX_CLIP_DURATION_SECONDS = 10 * 60;
const MIN_SCHEDULE_LEAD_SECONDS = 0.01;
const LATE_STEP_TOLERANCE_SECONDS = 0.02;
const REALTIME_PRIMARY_TRACK_DUCK = 0.08;

export function countLateSteps(nextStepTime: number, currentTime: number, stepSeconds: number): number {
  if (nextStepTime >= currentTime - LATE_STEP_TOLERANCE_SECONDS) return 0;
  return Math.max(1, Math.ceil((currentTime + MIN_SCHEDULE_LEAD_SECONDS - nextStepTime) / stepSeconds));
}

export function importedAudioStartTime(loop: boolean, currentTime: number, nextBarTime: number): number {
  return loop ? nextBarTime : currentTime + MIN_SCHEDULE_LEAD_SECONDS;
}

export interface TempoClockRebase {
  transportStartedAt: number;
  nextStepTime: number;
  currentStep: number;
}

export function performanceStepTime(step: number, absoluteStep: number, scheduledTime: number, stepSeconds: number): number {
  const swing = step % 2 === 1 ? stepSeconds * 0.16 : 0;
  const humanize = ((((absoluteStep * 1103515245 + 12345) >>> 8) & 0xff) / 255 - 0.5) * 0.006;
  return scheduledTime + swing + humanize;
}

export function rebaseTempoClock(
  currentTime: number,
  transportStartedAt: number,
  previousBpm: number,
  nextBpm: number,
): TempoClockRebase {
  if (currentTime < transportStartedAt) {
    const nextStepTime = currentTime + MIN_SCHEDULE_LEAD_SECONDS;
    return { transportStartedAt: nextStepTime, nextStepTime, currentStep: 0 };
  }

  const previousStepSeconds = secondsPerStep(previousBpm);
  const nextStepSeconds = secondsPerStep(nextBpm);
  const elapsedSteps = Math.max(0, (currentTime - transportStartedAt) / previousStepSeconds);
  const phaseInBar = elapsedSteps % STEPS_PER_BAR;
  const rebasedStart = currentTime - phaseInBar * nextStepSeconds;
  const schedulingHorizon = currentTime + MIN_SCHEDULE_LEAD_SECONDS;
  const nextStepIndex = Math.max(
    0,
    Math.floor((schedulingHorizon - rebasedStart) / nextStepSeconds + 1e-9) + 1,
  );
  return {
    transportStartedAt: rebasedStart,
    nextStepTime: rebasedStart + nextStepIndex * nextStepSeconds,
    currentStep: nextStepIndex % STEPS_PER_BAR,
  };
}

export function createEngineSnapshotFromTemplate(template: PerformanceTemplate): EngineSnapshot {
  const definitions = createTrackDefinitions(template.id);
  const tracks = definitions.map((definition) => {
    const templateTrack = template.tracks[definition.id];
    const mix = defaultMix();
    return {
      ...definition,
      pattern: patternFromSteps(templateTrack.pattern),
      notes: templateTrack.notes.slice(0, 64),
      ...mix,
      volume: clamp(templateTrack.volume ?? mix.volume, 0, 1),
      pan: clamp(templateTrack.pan ?? mix.pan, -1, 1),
      muted: false,
      solo: false,
    };
  });
  return {
    tracks,
    bpm: template.bpm,
    playing: false,
    currentStep: 0,
    masterVolume: 0.82,
    droppedLateSteps: 0,
  };
}

export class AudioEngine {
  private context?: AudioContext;
  private definitions = createTrackDefinitions();
  private pendingMix = new Map<TrackId, TrackMix>(this.definitions.map((track) => [track.id, defaultMix()]));
  private tracks = new Map<TrackId, TrackRuntime>();
  private masterGain?: GainNode;
  private compressor?: DynamicsCompressorNode;
  private fxDelay?: DelayNode;
  private fxFeedback?: GainNode;
  private fxWet?: GainNode;
  private reverb?: ConvolverNode;
  private reverbWet?: GainNode;
  private realtimeOutputGain?: GainNode;
  private masterAnalyser?: AnalyserNode;
  private captureDestination?: MediaStreamAudioDestinationNode;
  private noiseBuffer?: AudioBuffer;
  private frequencyBuffer = new Uint8Array(1024);
  private waveformBuffer = new Uint8Array(2048);
  private scheduledSources = new Map<AudioScheduledSourceNode, ScheduledSourceMetadata>();
  private listeners = new Set<EngineListener>();
  private schedulerTimer?: number;
  private bpm = 112;
  private masterVolume = 0.82;
  private currentStep = 0;
  private transportStepCounter = 0;
  private nextStepTime = 0;
  private transportStartedAt = 0;
  private playing = false;
  private droppedLateSteps = 0;
  private realtimeStreamTime = 0;
  private realtimeStreamPrimary = false;

  constructor(initialTemplate: PerformanceTemplate = defaultPerformanceTemplate()) {
    this.definitions = createTrackDefinitions(initialTemplate.id);
    this.pendingMix = new Map<TrackId, TrackMix>(this.definitions.map((track) => [track.id, defaultMix()]));
    this.bpm = initialTemplate.bpm;
    this.applyTrackTemplates(initialTemplate.tracks, false);
  }

  async initialize(): Promise<void> {
    if (this.context) {
      if (this.context.state === "suspended") await this.context.resume();
      return;
    }

    const context = new AudioContext({ latencyHint: "interactive", sampleRate: 48_000 });
    const masterGain = context.createGain();
    const compressor = context.createDynamicsCompressor();
    const fxDelay = context.createDelay(1.5);
    const fxFeedback = context.createGain();
    const fxWet = context.createGain();
    const reverb = context.createConvolver();
    const reverbWet = context.createGain();
    const realtimeOutputGain = context.createGain();
    const masterAnalyser = context.createAnalyser();
    const captureDestination = context.createMediaStreamDestination();

    masterGain.gain.value = this.masterVolume * this.masterVolume;
    compressor.threshold.value = -16;
    compressor.knee.value = 10;
    compressor.ratio.value = 5;
    compressor.attack.value = 0.004;
    compressor.release.value = 0.24;
    fxDelay.delayTime.value = secondsPerStep(this.bpm) * 3;
    fxFeedback.gain.value = 0.34;
    fxWet.gain.value = 0.24;
    reverb.buffer = this.createImpulseResponse(context, 1.7, 2.4);
    reverbWet.gain.value = 0.18;
    realtimeOutputGain.gain.value = 0.94;
    masterAnalyser.fftSize = 2048;
    masterAnalyser.smoothingTimeConstant = 0.76;

    masterGain.connect(compressor);
    fxDelay.connect(fxFeedback).connect(fxDelay);
    fxDelay.connect(fxWet).connect(compressor);
    reverb.connect(reverbWet).connect(compressor);
    realtimeOutputGain.connect(masterGain);
    compressor.connect(masterAnalyser);
    masterAnalyser.connect(context.destination);
    masterAnalyser.connect(captureDestination);

    this.context = context;
    this.masterGain = masterGain;
    this.compressor = compressor;
    this.fxDelay = fxDelay;
    this.fxFeedback = fxFeedback;
    this.fxWet = fxWet;
    this.reverb = reverb;
    this.reverbWet = reverbWet;
    this.realtimeOutputGain = realtimeOutputGain;
    this.masterAnalyser = masterAnalyser;
    this.captureDestination = captureDestination;
    this.frequencyBuffer = new Uint8Array(masterAnalyser.frequencyBinCount);
    this.waveformBuffer = new Uint8Array(masterAnalyser.fftSize);
    this.noiseBuffer = this.createNoiseBuffer(context);

    for (const definition of this.definitions) this.createTrackRuntime(definition);
    this.applyMix(false, true);
    await context.resume();
    this.emit();
  }

  private createTrackRuntime(definition: TrackDefinition): void {
    const context = this.requireContext();
    const master = this.masterGain;
    if (!master) throw new Error("Audio output is not initialized");

    const input = context.createGain();
    const gain = context.createGain();
    const pan = context.createStereoPanner();
    const analyser = context.createAnalyser();
    const fxSend = context.createGain();
    const mix = { ...(this.pendingMix.get(definition.id) ?? defaultMix()) };
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.68;
    gain.gain.value = effectiveTrackGain(mix, false);
    pan.pan.value = mix.pan;
    fxSend.gain.value = this.defaultFxSend(definition.id);

    input.connect(gain);
    gain.connect(pan);
    pan.connect(analyser);
    analyser.connect(master);
    analyser.connect(fxSend);
    if (this.fxDelay) fxSend.connect(this.fxDelay);
    if (this.reverb) fxSend.connect(this.reverb);

    this.tracks.set(definition.id, {
      definition,
      mix,
      input,
      gain,
      pan,
      analyser,
      fxSend,
      meterBuffer: new Uint8Array(analyser.fftSize),
      loadedLoop: true,
    });
  }

  async toggle(): Promise<void> {
    if (this.playing) this.stop();
    else await this.start();
  }

  async start(): Promise<void> {
    await this.initialize();
    const context = this.requireContext();
    await context.resume();
    if (this.playing) return;

    this.playing = true;
    this.droppedLateSteps = 0;
    this.currentStep = 0;
    this.transportStepCounter = 0;
    this.nextStepTime = context.currentTime + 0.06;
    this.transportStartedAt = this.nextStepTime;
    this.startImportedLoops(this.nextStepTime);
    this.schedulerTimer = window.setInterval(() => this.schedulerTick(), SCHEDULER_INTERVAL_MS);
    this.schedulerTick();
    this.emit();
  }

  stop(): void {
    if (this.schedulerTimer !== undefined) window.clearInterval(this.schedulerTimer);
    this.schedulerTimer = undefined;
    this.playing = false;
    this.currentStep = 0;
    this.transportStepCounter = 0;
    for (const source of this.scheduledSources.keys()) {
      try {
        source.stop();
      } catch {
        // A source may already have ended between iteration and stop.
      }
    }
    this.scheduledSources.clear();
    for (const track of this.tracks.values()) {
      track.loopSource = undefined;
      track.loadedStartedAt = undefined;
    }
    this.emit();
  }

  private schedulerTick(): void {
    const context = this.context;
    if (!context || !this.playing) return;
    const stepSeconds = secondsPerStep(this.bpm);
    const lateSteps = countLateSteps(this.nextStepTime, context.currentTime, stepSeconds);
    if (lateSteps > 0) {
      this.nextStepTime += lateSteps * stepSeconds;
      this.currentStep = (this.currentStep + lateSteps) % STEPS_PER_BAR;
      this.transportStepCounter += lateSteps;
      this.droppedLateSteps += lateSteps;
    }
    while (this.nextStepTime < context.currentTime + LOOK_AHEAD_SECONDS) {
      this.scheduleStep(
        this.currentStep,
        this.transportStepCounter,
        performanceStepTime(this.currentStep, this.transportStepCounter, this.nextStepTime, stepSeconds),
      );
      this.nextStepTime += stepSeconds;
      this.currentStep = (this.currentStep + 1) % STEPS_PER_BAR;
      this.transportStepCounter += 1;
      this.emit();
    }
  }

  private scheduleStep(step: number, absoluteStep: number, time: number): void {
    for (const track of this.tracks.values()) {
      if (track.loadedBuffer || !track.definition.pattern[step]) continue;
      const noteIndex = Math.floor(absoluteStep / 2) % track.definition.notes.length;
      const note = track.definition.notes[noteIndex];
      switch (track.definition.instrument) {
        case "drums":
          this.triggerDrum(track, note, absoluteStep, time);
          break;
        case "bass":
          this.triggerBass(track, note, time);
          break;
        case "poly":
          this.triggerChord(track, note, time);
          break;
        case "lead":
          this.triggerLead(track, note, time);
          break;
        case "voice":
          this.triggerVoice(track, note, time);
          break;
        case "texture":
          this.triggerTexture(track, note, time);
          break;
      }
    }
  }

  triggerTrack(id: TrackId): void {
    const track = this.tracks.get(id);
    const context = this.context;
    if (!track || !context) return;
    const time = context.currentTime + 0.01;
    const note = track.definition.notes[this.transportStepCounter % track.definition.notes.length];
    this.triggerTrackNote(track, note, time);
  }

  triggerNote(id: TrackId, note: number): void {
    const track = this.tracks.get(id);
    const context = this.context;
    if (!track || !context || !Number.isFinite(note)) return;
    this.triggerTrackNote(track, Math.round(clamp(note, 0, 127)), context.currentTime + 0.01);
  }

  private triggerTrackNote(track: TrackRuntime, note: number, time: number): void {
    if (track.definition.instrument === "drums") this.triggerDrum(track, note, this.transportStepCounter, time);
    else if (track.definition.instrument === "bass") this.triggerBass(track, note, time);
    else if (track.definition.instrument === "poly") this.triggerChord(track, note, time);
    else if (track.definition.instrument === "lead") this.triggerLead(track, note, time);
    else if (track.definition.instrument === "voice") this.triggerVoice(track, note, time);
    else this.triggerTexture(track, note, time);
  }

  private triggerDrum(track: TrackRuntime, note: number, absoluteStep: number, time: number): void {
    const context = this.requireContext();
    const step = absoluteStep % STEPS_PER_BAR;
    if (note <= 36) {
      const oscillator = context.createOscillator();
      const click = context.createBufferSource();
      const clickFilter = context.createBiquadFilter();
      const clickEnvelope = context.createGain();
      const envelope = context.createGain();
      click.buffer = this.noiseBuffer ?? null;
      clickFilter.type = "highpass";
      clickFilter.frequency.value = 2200;
      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(step % 8 === 0 ? 152 : 112, time);
      oscillator.frequency.exponentialRampToValueAtTime(42, time + 0.22);
      envelope.gain.setValueAtTime(step % 8 === 0 ? 0.92 : 0.62, time);
      envelope.gain.exponentialRampToValueAtTime(0.001, time + 0.32);
      clickEnvelope.gain.setValueAtTime(step % 8 === 0 ? 0.16 : 0.09, time);
      clickEnvelope.gain.exponentialRampToValueAtTime(0.001, time + 0.026);
      oscillator.connect(envelope).connect(track.input);
      click.connect(clickFilter).connect(clickEnvelope).connect(track.input);
      this.startSource(oscillator, time, time + 0.34);
      this.startSource(click, time, time + 0.04);
      return;
    }

    const source = context.createBufferSource();
    const filter = context.createBiquadFilter();
    const envelope = context.createGain();
    source.buffer = this.noiseBuffer ?? null;
    filter.type = note === 38 ? "bandpass" : "highpass";
    filter.Q.value = note === 38 ? 2.4 : note >= 46 ? 0.9 : 0.65;
    filter.frequency.value = note === 38 ? 840 : note >= 46 ? 4800 : 8200;
    envelope.gain.setValueAtTime(note === 38 ? 0.42 : note >= 46 ? 0.2 : 0.09, time);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + (note === 38 ? 0.26 : note >= 46 ? 0.24 : 0.052));
    source.connect(filter).connect(envelope).connect(track.input);
    this.startSource(source, time, time + (note === 38 ? 0.3 : 0.25));
  }

  private triggerBass(track: TrackRuntime, note: number, time: number): void {
    const context = this.requireContext();
    const sub = context.createOscillator();
    const mid = context.createOscillator();
    const filter = context.createBiquadFilter();
    const envelope = context.createGain();
    const subGain = context.createGain();
    const midGain = context.createGain();
    sub.type = "sine";
    sub.frequency.value = midiToFrequency(note - 12);
    mid.type = "sawtooth";
    mid.frequency.value = midiToFrequency(note);
    mid.detune.value = -5;
    filter.type = "lowpass";
    filter.Q.value = 7;
    filter.frequency.setValueAtTime(980, time);
    filter.frequency.exponentialRampToValueAtTime(145, time + 0.28);
    envelope.gain.setValueAtTime(0.001, time);
    envelope.gain.exponentialRampToValueAtTime(0.34, time + 0.018);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + 0.42);
    subGain.gain.value = 0.9;
    midGain.gain.value = 0.42;
    sub.connect(subGain).connect(filter);
    mid.connect(midGain).connect(filter);
    filter.connect(envelope).connect(track.input);
    this.startSource(sub, time, time + 0.44);
    this.startSource(mid, time, time + 0.44);
    this.triggerAiToneGrain(track, note, time, 0.48, 0.72);
  }

  private triggerChord(track: TrackRuntime, root: number, time: number): void {
    for (const [index, interval] of [0, 3, 7, 10].entries()) {
      this.triggerTone(track, root + interval, time + index * 0.006, 0.9, "triangle", 0.07, 1800, index % 2 === 0 ? -6 : 7, 0.08);
    }
  }

  private triggerLead(track: TrackRuntime, note: number, time: number): void {
    this.triggerTone(track, note, time, 0.28, "sawtooth", 0.09, 4200, -9, 0.012);
    this.triggerTone(track, note + 12, time + 0.002, 0.22, "square", 0.045, 5200, 11, 0.006);
  }

  private triggerVoice(track: TrackRuntime, note: number, time: number): void {
    const context = this.requireContext();
    const source = context.createBufferSource();
    const formant = context.createBiquadFilter();
    const secondFormant = context.createBiquadFilter();
    const envelope = context.createGain();
    source.buffer = this.noiseBuffer ?? null;
    formant.type = "bandpass";
    formant.Q.value = 10;
    secondFormant.type = "bandpass";
    secondFormant.Q.value = 8;
    formant.frequency.setValueAtTime(midiToFrequency(note) * 3.2, time);
    formant.frequency.linearRampToValueAtTime(midiToFrequency(note) * 4.8, time + 0.42);
    secondFormant.frequency.setValueAtTime(midiToFrequency(note) * 6.2, time);
    secondFormant.frequency.linearRampToValueAtTime(midiToFrequency(note) * 5.6, time + 0.42);
    envelope.gain.setValueAtTime(0.001, time);
    envelope.gain.linearRampToValueAtTime(0.18, time + 0.08);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + 0.62);
    source.connect(formant).connect(secondFormant).connect(envelope).connect(track.input);
    this.startSource(source, time, time + 0.66);
    this.triggerAiToneGrain(track, note, time, 0.62, 0.62);
  }

  private triggerTexture(track: TrackRuntime, note: number, time: number): void {
    const context = this.requireContext();
    const carrier = context.createOscillator();
    const modulator = context.createOscillator();
    const modGain = context.createGain();
    const air = context.createBufferSource();
    const airFilter = context.createBiquadFilter();
    const envelope = context.createGain();
    const airEnvelope = context.createGain();
    carrier.type = "sine";
    carrier.frequency.value = midiToFrequency(note);
    modulator.type = "sine";
    modulator.frequency.value = midiToFrequency(note - 17);
    modGain.gain.value = 26;
    air.buffer = this.noiseBuffer ?? null;
    airFilter.type = "bandpass";
    airFilter.frequency.value = midiToFrequency(note + 24);
    airFilter.Q.value = 3.6;
    envelope.gain.setValueAtTime(0.001, time);
    envelope.gain.linearRampToValueAtTime(0.1, time + 0.28);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + 1.45);
    airEnvelope.gain.setValueAtTime(0.001, time);
    airEnvelope.gain.linearRampToValueAtTime(0.052, time + 0.45);
    airEnvelope.gain.exponentialRampToValueAtTime(0.001, time + 1.6);
    modulator.connect(modGain).connect(carrier.frequency);
    carrier.connect(envelope).connect(track.input);
    air.connect(airFilter).connect(airEnvelope).connect(track.input);
    this.startSource(modulator, time, time + 1.48);
    this.startSource(carrier, time, time + 1.48);
    this.startSource(air, time, time + 1.62);
    this.triggerAiToneGrain(track, note, time, 1.48, 0.78);
  }

  private triggerTone(
    track: TrackRuntime,
    note: number,
    time: number,
    duration: number,
    type: OscillatorType,
    level: number,
    cutoff: number,
    detune = 0,
    attack = 0.015,
  ): void {
    const context = this.requireContext();
    const oscillator = context.createOscillator();
    const filter = context.createBiquadFilter();
    const envelope = context.createGain();
    oscillator.type = type;
    oscillator.frequency.value = midiToFrequency(note);
    oscillator.detune.value = detune;
    filter.type = "lowpass";
    filter.frequency.value = cutoff;
    filter.Q.value = 2;
    envelope.gain.setValueAtTime(0.001, time);
    envelope.gain.exponentialRampToValueAtTime(level, time + attack);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + duration);
    oscillator.connect(filter).connect(envelope).connect(track.input);
    this.startSource(oscillator, time, time + duration + 0.02);
    this.triggerAiToneGrain(track, note, time, duration, 1);
  }

  private triggerAiToneGrain(track: TrackRuntime, note: number, time: number, duration: number, intensity: number): void {
    const aiTone = track.aiTone;
    if (!aiTone) return;
    const context = this.requireContext();
    const source = context.createBufferSource();
    const filter = context.createBiquadFilter();
    const envelope = context.createGain();
    const grainSeconds = clamp(Math.min(aiTone.grainSeconds, duration + 0.32), 0.08, 2.4);
    const attack = Math.min(0.045, grainSeconds * 0.18);
    const releaseStart = Math.max(attack + 0.02, grainSeconds * 0.72);
    const windowStart = clamp(aiTone.windowStartSeconds, 0, Math.max(0, aiTone.buffer.duration - 0.05));
    const windowEnd = clamp(windowStart + aiTone.windowDurationSeconds, windowStart + 0.05, aiTone.buffer.duration);
    const windowWidth = Math.max(0.05, windowEnd - windowStart - grainSeconds);
    const fractionalSeed = Math.abs(Math.sin(note * 12.9898 + time * 78.233)) % 1;
    const offset = clamp(windowStart + fractionalSeed * windowWidth, 0, Math.max(0, aiTone.buffer.duration - grainSeconds));
    const semitones = note - aiTone.baseNote;

    source.buffer = aiTone.buffer;
    source.playbackRate.value = 2 ** (semitones / 12);
    filter.type = "lowpass";
    filter.frequency.value = 700 + aiTone.brightness * 7_600;
    filter.Q.value = 0.9 + aiTone.brightness * 2.4;
    envelope.gain.setValueAtTime(0.0001, time);
    envelope.gain.linearRampToValueAtTime(aiTone.level * intensity, time + attack);
    envelope.gain.setValueAtTime(aiTone.level * intensity, time + releaseStart);
    envelope.gain.exponentialRampToValueAtTime(0.0001, time + grainSeconds);
    source.connect(filter).connect(envelope).connect(track.input);
    this.scheduledSources.set(source, { startsAt: time, imported: false });
    source.addEventListener("ended", () => this.scheduledSources.delete(source), { once: true });
    source.start(time, offset, grainSeconds);
    source.stop(time + grainSeconds + 0.02);
  }

  private startSource(source: AudioScheduledSourceNode, startsAt: number, stopsAt: number): void {
    this.scheduledSources.set(source, { startsAt, imported: false });
    source.addEventListener("ended", () => this.scheduledSources.delete(source), { once: true });
    source.start(startsAt);
    source.stop(stopsAt);
  }

  private startImportedLoops(startsAt: number): void {
    for (const track of this.tracks.values()) {
      if (!track.loadedBuffer) continue;
      this.startImportedLoop(track, startsAt);
    }
  }

  private startImportedLoop(track: TrackRuntime, startsAt: number): void {
    const source = this.requireContext().createBufferSource();
    source.buffer = track.loadedBuffer ?? null;
    source.loop = track.loadedLoop;
    source.connect(track.input);
    source.addEventListener("ended", () => {
      this.scheduledSources.delete(source);
      if (track.loopSource === source) {
        track.loopSource = undefined;
        track.loadedStartedAt = undefined;
      }
    }, { once: true });
    track.loopSource = source;
    track.loadedStartedAt = startsAt;
    this.scheduledSources.set(source, { startsAt, imported: true });
    source.start(startsAt);
  }

  private nextBarTime(): number {
    const context = this.requireContext();
    const barSeconds = secondsPerStep(this.bpm) * STEPS_PER_BAR;
    const elapsed = Math.max(0, context.currentTime - this.transportStartedAt);
    return this.transportStartedAt + (Math.floor(elapsed / barSeconds) + 1) * barSeconds;
  }

  async loadAudioFile(
    id: TrackId,
    bytes: ArrayBuffer,
    fileName: string,
    options: LoadAudioOptions = {},
  ): Promise<LoadedAudioDetails> {
    if (bytes.byteLength > MAX_IMPORT_BYTES) throw new Error("Audio files are limited to 250 MB");
    let encoded: EncodedAudioMetadata | undefined;
    try {
      encoded = inspectEncodedAudio(bytes, options.declaredMimeType);
    } catch (error) {
      if (options.requireEncodedValidation) throw error;
    }
    await this.initialize();
    const track = this.tracks.get(id);
    if (!track) throw new Error(`Unknown track: ${id}`);
    const buffer = await this.requireContext().decodeAudioData(bytes.slice(0));
    if (buffer.duration > MAX_CLIP_DURATION_SECONDS) throw new Error("Audio clips are limited to 10 minutes");
    if (buffer.numberOfChannels > 8) throw new Error("Audio clips are limited to 8 channels");
    if (buffer.sampleRate > 192_000) throw new Error("Audio clips are limited to 192 kHz");
    const analysis = analyzeDecodedPcm({
      sampleRateHz: buffer.sampleRate,
      channels: Array.from({ length: buffer.numberOfChannels }, (_, channel) => buffer.getChannelData(channel)),
    });
    track.loadedBuffer = buffer;
    track.loadedFile = fileName;
    track.loadedLoop = options.loop ?? true;
    if (this.playing) {
      track.loopSource?.stop();
      const context = this.requireContext();
      this.startImportedLoop(
        track,
        importedAudioStartTime(track.loadedLoop, context.currentTime, this.nextBarTime()),
      );
    }
    this.emit();
    return { encoded, analysis };
  }

  async loadTrackToneFile(
    id: TrackId,
    bytes: ArrayBuffer,
    fileName: string,
    options: LoadAudioOptions & AiToneOptions = {},
  ): Promise<LoadedAiToneDetails> {
    if (bytes.byteLength > MAX_IMPORT_BYTES) throw new Error("AI tone files are limited to 250 MB");
    let encoded: EncodedAudioMetadata | undefined;
    try {
      encoded = inspectEncodedAudio(bytes, options.declaredMimeType);
    } catch (error) {
      if (options.requireEncodedValidation) throw error;
    }
    await this.initialize();
    const track = this.tracks.get(id);
    if (!track) throw new Error(`Unknown track: ${id}`);
    const buffer = await this.requireContext().decodeAudioData(bytes.slice(0));
    if (buffer.duration > MAX_CLIP_DURATION_SECONDS) throw new Error("AI tone files are limited to 10 minutes");
    if (buffer.numberOfChannels > 8) throw new Error("AI tone files are limited to 8 channels");
    if (buffer.sampleRate > 192_000) throw new Error("AI tone files are limited to 192 kHz");
    const analysis = analyzeDecodedPcm({
      sampleRateHz: buffer.sampleRate,
      channels: Array.from({ length: buffer.numberOfChannels }, (_, channel) => buffer.getChannelData(channel)),
    });
    track.aiTone = {
      buffer,
      fileName,
      baseNote: options.baseNote ?? 60,
      grainSeconds: clamp(options.grainSeconds ?? 0.7, 0.08, 2.4),
      level: clamp(options.level ?? 0.055, 0, 0.35),
      brightness: clamp(options.brightness ?? 0.5, 0, 1),
      windowStartSeconds: Math.max(0, options.windowStartSeconds ?? 0),
      windowDurationSeconds: Math.max(0.05, options.windowDurationSeconds ?? buffer.duration),
    };
    this.emit();
    return { encoded, analysis, fileName };
  }

  clearTrackToneFile(id: TrackId): void {
    const track = this.tracks.get(id);
    if (!track) return;
    track.aiTone = undefined;
    this.emit();
  }

  clearAudioFile(id: TrackId): void {
    const track = this.tracks.get(id);
    if (!track) return;
    track.loopSource?.stop();
    track.loopSource = undefined;
    track.loadedStartedAt = undefined;
    track.loadedBuffer = undefined;
    track.loadedFile = undefined;
    track.loadedLoop = true;
    this.emit();
  }

  toggleStep(id: TrackId, step: number): void {
    const track = this.tracks.get(id);
    const definition = track?.definition ?? this.definitions.find((candidate) => candidate.id === id);
    if (!definition || step < 0 || step >= STEPS_PER_BAR) return;
    definition.pattern[step] = !definition.pattern[step];
    this.emit();
  }

  setStep(id: TrackId, step: number, active: boolean): void {
    const track = this.tracks.get(id);
    const definition = track?.definition ?? this.definitions.find((candidate) => candidate.id === id);
    if (!definition || step < 0 || step >= STEPS_PER_BAR || definition.pattern[step] === active) return;
    definition.pattern[step] = active;
    this.emit();
  }

  setTrackVolume(id: TrackId, volume: number): void {
    const track = this.tracks.get(id);
    const mix = track?.mix ?? this.pendingMix.get(id);
    if (!mix) return;
    mix.volume = clamp(volume, 0, 1);
    this.applyMix();
  }

  setTrackPan(id: TrackId, pan: number): void {
    const track = this.tracks.get(id);
    const mix = track?.mix ?? this.pendingMix.get(id);
    if (!mix) return;
    mix.pan = clamp(pan, -1, 1);
    if (!track) {
      this.emit();
      return;
    }
    const now = this.context?.currentTime ?? 0;
    track.pan.pan.setTargetAtTime(track.mix.pan, now, 0.012);
    this.emit();
  }

  toggleMute(id: TrackId): void {
    const track = this.tracks.get(id);
    const mix = track?.mix ?? this.pendingMix.get(id);
    if (!mix) return;
    mix.muted = !mix.muted;
    this.applyMix();
  }

  toggleSolo(id: TrackId): void {
    const track = this.tracks.get(id);
    const mix = track?.mix ?? this.pendingMix.get(id);
    if (!mix) return;
    mix.solo = !mix.solo;
    this.applyMix();
  }

  setRealtimeStreamPrimary(enabled: boolean): void {
    if (this.realtimeStreamPrimary === enabled) return;
    this.realtimeStreamPrimary = enabled;
    this.applyMix();
  }

  private applyMix(shouldEmit = true, immediate = false): void {
    if (this.tracks.size === 0) {
      if (shouldEmit) this.emit();
      return;
    }
    const anySolo = [...this.tracks.values()].some((track) => track.mix.solo);
    const now = this.context?.currentTime ?? 0;
    const realtimeDuck = this.realtimeStreamPrimary ? REALTIME_PRIMARY_TRACK_DUCK : 1;
    for (const track of this.tracks.values()) {
      const gain = effectiveTrackGain(track.mix, anySolo) * realtimeDuck;
      if (immediate) track.gain.gain.value = gain;
      else track.gain.gain.setTargetAtTime(gain, now, 0.012);
    }
    if (shouldEmit) this.emit();
  }

  setBpm(value: number): void {
    const nextBpm = Math.round(clamp(value, 60, 200));
    if (nextBpm === this.bpm) return;
    const context = this.context;
    if (this.playing && context) {
      const cancellationBoundary = context.currentTime + MIN_SCHEDULE_LEAD_SECONDS;
      for (const [source, metadata] of this.scheduledSources) {
        if (metadata.imported || metadata.startsAt <= cancellationBoundary) continue;
        try {
          source.stop();
        } catch {
          // A scheduled source may have ended while the tempo change was being applied.
        }
        this.scheduledSources.delete(source);
      }
      const rebased = rebaseTempoClock(context.currentTime, this.transportStartedAt, this.bpm, nextBpm);
      this.transportStartedAt = rebased.transportStartedAt;
      this.nextStepTime = rebased.nextStepTime;
      this.currentStep = rebased.currentStep;
      this.transportStepCounter = Math.max(this.transportStepCounter, rebased.currentStep);
    }
    this.bpm = nextBpm;
    this.fxDelay?.delayTime.setTargetAtTime(secondsPerStep(nextBpm) * 3, this.context?.currentTime ?? 0, 0.08);
    if (this.playing) this.schedulerTick();
    this.emit();
  }

  setMasterVolume(value: number): void {
    this.masterVolume = clamp(value, 0, 1);
    this.masterGain?.gain.setTargetAtTime(this.masterVolume * this.masterVolume, this.context?.currentTime ?? 0, 0.015);
    this.emit();
  }

  applyPerformanceTemplate(template: PerformanceTemplate): void {
    this.setBpm(template.bpm);
    this.applyTrackTemplates(template.tracks);
  }

  applyImportedMidi(tracks: Partial<Record<TrackId, TrackTemplate>>, bpm?: number): void {
    if (bpm !== undefined) this.setBpm(bpm);
    this.applyTrackTemplates(tracks);
  }

  private applyTrackTemplates(tracks: Partial<Record<TrackId, TrackTemplate>>, emit = true): void {
    const definitions = this.tracks.size > 0 ? [...this.tracks.values()].map((track) => track.definition) : this.definitions;
    for (const definition of definitions) {
      const trackTemplate = tracks[definition.id];
      if (!trackTemplate) continue;
      definition.pattern = patternFromSteps(trackTemplate.pattern);
      definition.notes = trackTemplate.notes.slice(0, 64);
      const mix = this.tracks.get(definition.id)?.mix ?? this.pendingMix.get(definition.id);
      if (mix) {
        mix.volume = clamp(trackTemplate.volume ?? mix.volume, 0, 1);
        mix.pan = clamp(trackTemplate.pan ?? mix.pan, -1, 1);
        mix.muted = false;
        mix.solo = false;
      }
      const runtime = this.tracks.get(definition.id);
      if (runtime) {
        runtime.pan.pan.setTargetAtTime(runtime.mix.pan, this.context?.currentTime ?? 0, 0.012);
      }
    }
    if (emit) this.applyMix();
  }

  mutate(seedPhrase: string): void {
    const seed = hashSeed(seedPhrase);
    let offset = 0;
    const definitions = this.tracks.size > 0 ? [...this.tracks.values()].map((track) => track.definition) : this.definitions;
    for (const definition of definitions) {
      definition.pattern = mutatePattern(definition.pattern, seed + offset, definition.id === "drums" ? 0.1 : 0.2);
      offset += 997;
    }
    this.emit();
  }

  getMetrics(): AudioMetrics {
    const analyser = this.masterAnalyser;
    if (analyser) {
      analyser.getByteFrequencyData(this.frequencyBuffer);
      analyser.getByteTimeDomainData(this.waveformBuffer);
    } else {
      this.frequencyBuffer.fill(0);
      this.waveformBuffer.fill(128);
    }

    const trackLevels = Object.fromEntries(TRACK_IDS.map((id) => [id, 0])) as Record<TrackId, number>;
    for (const [id, track] of this.tracks) {
      track.analyser.getByteTimeDomainData(track.meterBuffer);
      let energy = 0;
      for (const value of track.meterBuffer) {
        const sample = (value - 128) / 128;
        energy += sample * sample;
      }
      trackLevels[id] = Math.min(1, Math.sqrt(energy / track.meterBuffer.length) * 2.4);
    }

    let masterEnergy = 0;
    for (const value of this.waveformBuffer) {
      const sample = (value - 128) / 128;
      masterEnergy += sample * sample;
    }
    const context = this.context;
    const stepSeconds = secondsPerStep(this.bpm);
    const beatPhase = this.playing && context ? ((context.currentTime - this.transportStartedAt) / (stepSeconds * 4)) % 1 : 0;

    return {
      frequency: this.frequencyBuffer,
      waveform: this.waveformBuffer,
      trackLevels,
      masterLevel: Math.min(1, Math.sqrt(masterEnergy / this.waveformBuffer.length) * 2),
      beatPhase: Math.max(0, beatPhase),
      currentStep: this.audibleStep(),
      bpm: this.bpm,
      playing: this.playing,
    };
  }

  getCaptureStream(): MediaStream | undefined {
    return this.captureDestination?.stream;
  }

  getLoadedTrackPosition(id: TrackId): number | undefined {
    const context = this.context;
    const track = this.tracks.get(id);
    if (!this.playing || !context || !track) return undefined;
    const startsAt = track.loadedStartedAt;
    const duration = track.loadedBuffer?.duration;
    if (startsAt === undefined || duration === undefined || duration <= 0) return undefined;
    const elapsed = context.currentTime - startsAt;
    if (elapsed < 0) return undefined;
    return track.loadedLoop ? elapsed % duration : Math.min(elapsed, duration);
  }

  async playRealtimePcm16(bytes: Uint8Array, sampleRateHz: number, channels: number): Promise<void> {
    if (bytes.byteLength < 4 || channels < 1 || channels > 2 || sampleRateHz < 8_000 || sampleRateHz > 384_000) return;
    await this.initialize();
    const context = this.requireContext();
    const frameCount = Math.floor(bytes.byteLength / 2 / channels);
    if (frameCount <= 0) return;
    const buffer = context.createBuffer(2, frameCount, sampleRateHz);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const left = buffer.getChannelData(0);
    const right = buffer.getChannelData(1);
    for (let frame = 0; frame < frameCount; frame += 1) {
      const leftSample = view.getInt16((frame * channels) * 2, true) / 32768;
      const rightSample = channels === 1 ? leftSample : view.getInt16((frame * channels + 1) * 2, true) / 32768;
      left[frame] = leftSample;
      right[frame] = rightSample;
    }
    const output = this.realtimeOutputGain ?? this.masterGain;
    if (!output) return;
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(output);
    const earliest = context.currentTime + 0.08;
    if (this.realtimeStreamTime < earliest || this.realtimeStreamTime > context.currentTime + 1.8) {
      this.realtimeStreamTime = earliest;
    }
    const startsAt = this.realtimeStreamTime;
    this.realtimeStreamTime += buffer.duration;
    this.scheduledSources.set(source, { startsAt, imported: true });
    source.addEventListener("ended", () => this.scheduledSources.delete(source), { once: true });
    source.start(startsAt);
  }

  getSnapshot(): EngineSnapshot {
    const tracks = this.tracks.size > 0
      ? [...this.tracks.values()].map((track) => ({
          ...track.definition,
          pattern: [...track.definition.pattern],
          notes: [...track.definition.notes],
          ...track.mix,
          loadedFile: track.loadedFile,
          aiToneFile: track.aiTone?.fileName,
        }))
      : this.definitions.map((definition) => ({
          ...definition,
          pattern: [...definition.pattern],
          notes: [...definition.notes],
          ...(this.pendingMix.get(definition.id) ?? defaultMix()),
        }));
    return {
      tracks,
      bpm: this.bpm,
      playing: this.playing,
      currentStep: this.audibleStep(),
      masterVolume: this.masterVolume,
      droppedLateSteps: this.droppedLateSteps,
    };
  }

  subscribe(listener: EngineListener): () => void {
    this.listeners.add(listener);
    listener(this.getSnapshot());
    return () => this.listeners.delete(listener);
  }

  private emit(): void {
    const snapshot = this.getSnapshot();
    for (const listener of this.listeners) listener(snapshot);
  }

  private createNoiseBuffer(context: AudioContext): AudioBuffer {
    const buffer = context.createBuffer(1, context.sampleRate * 2, context.sampleRate);
    const channel = buffer.getChannelData(0);
    let seed = 0x5eed1234;
    for (let index = 0; index < channel.length; index += 1) {
      seed = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      seed ^= seed + Math.imul(seed ^ (seed >>> 7), 61 | seed);
      channel[index] = (((seed ^ (seed >>> 14)) >>> 0) / 2147483648 - 1) * 0.72;
    }
    return buffer;
  }

  private createImpulseResponse(context: AudioContext, seconds: number, decay: number): AudioBuffer {
    const length = Math.max(1, Math.floor(context.sampleRate * seconds));
    const buffer = context.createBuffer(2, length, context.sampleRate);
    let seed = 0x72657662;
    for (let channelIndex = 0; channelIndex < buffer.numberOfChannels; channelIndex += 1) {
      const channel = buffer.getChannelData(channelIndex);
      for (let index = 0; index < length; index += 1) {
        seed = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        seed ^= seed + Math.imul(seed ^ (seed >>> 7), 61 | seed);
        const noise = ((seed ^ (seed >>> 14)) >>> 0) / 2147483648 - 1;
        channel[index] = noise * (1 - index / length) ** decay;
      }
    }
    return buffer;
  }

  private defaultFxSend(id: TrackId): number {
    if (id === "drums") return 0.08;
    if (id === "bass") return 0.03;
    if (id === "chords") return 0.28;
    if (id === "lead") return 0.22;
    if (id === "voice") return 0.34;
    return 0.46;
  }

  private audibleStep(): number {
    const context = this.context;
    if (!this.playing || !context || context.currentTime <= this.transportStartedAt) return 0;
    return Math.floor((context.currentTime - this.transportStartedAt) / secondsPerStep(this.bpm)) % STEPS_PER_BAR;
  }

  private requireContext(): AudioContext {
    if (!this.context) throw new Error("Audio engine has not been initialized");
    return this.context;
  }
}
