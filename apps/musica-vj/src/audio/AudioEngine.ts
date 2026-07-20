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

export type RealtimeDeckId = "main" | "sequence" | "vocal";

export type SfxKind = "siren" | "airhorn" | "riser" | "impact" | "laser";

export const SFX_KINDS: ReadonlyArray<{ id: SfxKind; label: string }> = [
  { id: "siren", label: "SIREN" },
  { id: "airhorn", label: "HORN" },
  { id: "riser", label: "RISER" },
  { id: "impact", label: "IMPACT" },
  { id: "laser", label: "LASER" },
];

interface PadLoopSlot {
  buffer?: AudioBuffer;
  name?: string;
  source?: AudioBufferSourceNode;
  gain?: GainNode;
  playing: boolean;
}

interface RealtimeDeckControl {
  volume: number;
  muted: boolean;
  pitchSemitones: number;
  beatNudgeMs: number;
}

interface RealtimeDeckRuntime {
  gain: GainNode;
  streamTime: number;
}

const REALTIME_DECK_DEFAULTS: Record<RealtimeDeckId, RealtimeDeckControl> = {
  main: { volume: 0.72, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
  sequence: { volume: 0.42, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
  vocal: { volume: 0.42, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
};

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

export function buildSoftClipCurve(edge: number): Float32Array<ArrayBuffer> {
  const k = 1.5 + Math.max(0, Math.min(1, edge)) * 4.5;
  const curve = new Float32Array(1_024);
  for (let index = 0; index < curve.length; index += 1) {
    const x = (index / (curve.length - 1)) * 2 - 1;
    curve[index] = Math.tanh(k * x) / Math.tanh(k);
  }
  return curve;
}

export function buildQuantizeCurve(bits: number): Float32Array<ArrayBuffer> {
  const steps = 2 + Math.round(Math.max(0, Math.min(1, bits)) * 22);
  const curve = new Float32Array(1_024);
  for (let index = 0; index < curve.length; index += 1) {
    const x = (index / (curve.length - 1)) * 2 - 1;
    curve[index] = Math.round(x * steps) / steps;
  }
  return curve;
}

export interface MasterEffectsState {
  flanger: number;
  sweep: number;
  reverb: number;
  echo: number;
  drive: number;
  crush: number;
  phaser: number;
}

export const MASTER_EFFECT_IDS = ["flanger", "phaser", "drive", "crush", "sweep", "reverb", "echo"] as const;

export interface MasterEffectParams {
  flangerRate: number;
  flangerDepth: number;
  flangerFeedback: number;
  phaserRate: number;
  phaserDepth: number;
  driveEdge: number;
  crushBits: number;
  sweepRate: number;
  sweepReso: number;
}

export const DEFAULT_MASTER_EFFECT_PARAMS: Readonly<MasterEffectParams> = Object.freeze({
  flangerRate: 0.2,
  flangerDepth: 0.5,
  flangerFeedback: 0.6,
  phaserRate: 0.25,
  phaserDepth: 0.6,
  driveEdge: 0.4,
  crushBits: 0.45,
  sweepRate: 0.3,
  sweepReso: 0.55,
});

export const MASTER_EFFECT_PARAM_IDS: Readonly<Record<(typeof MASTER_EFFECT_IDS)[number], Array<{ id: keyof MasterEffectParams; label: string }>>> = Object.freeze({
  flanger: [
    { id: "flangerRate", label: "RATE" },
    { id: "flangerDepth", label: "DEPTH" },
    { id: "flangerFeedback", label: "FEEDBACK" },
  ],
  phaser: [
    { id: "phaserRate", label: "RATE" },
    { id: "phaserDepth", label: "DEPTH" },
  ],
  drive: [{ id: "driveEdge", label: "EDGE" }],
  crush: [{ id: "crushBits", label: "BITS" }],
  sweep: [
    { id: "sweepRate", label: "RATE" },
    { id: "sweepReso", label: "RESO" },
  ],
  reverb: [],
  echo: [],
});

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
  private masterLimiter?: DynamicsCompressorNode;
  private flangerWet?: GainNode;
  private flangerFeedback?: GainNode;
  private flangerDepth?: GainNode;
  private flangerLfo?: OscillatorNode;
  private sweepFilter?: BiquadFilterNode;
  private sweepDepth?: GainNode;
  private sweepLfo?: OscillatorNode;
  private masterReverbSend?: GainNode;
  private masterEchoSend?: GainNode;
  private phaserWet?: GainNode;
  private phaserDry?: GainNode;
  private phaserDepth?: GainNode;
  private phaserLfo?: OscillatorNode;
  private phaserStages: BiquadFilterNode[] = [];
  private driveDry?: GainNode;
  private driveWet?: GainNode;
  private drivePre?: GainNode;
  private crushDry?: GainNode;
  private crushWet?: GainNode;
  private driveShaper?: WaveShaperNode;
  private crushShaper?: WaveShaperNode;
  private masterEffects: MasterEffectsState = { flanger: 0, sweep: 0, reverb: 0, echo: 0, drive: 0, crush: 0, phaser: 0 };
  private masterEffectParams: MasterEffectParams = { ...DEFAULT_MASTER_EFFECT_PARAMS };
  private readonly padLoops = new Map<number, PadLoopSlot>();
  private sfxLevel = 0.5;
  private fxDelay?: DelayNode;
  private fxFeedback?: GainNode;
  private fxWet?: GainNode;
  private reverb?: ConvolverNode;
  private reverbWet?: GainNode;
  private realtimeDecks = new Map<RealtimeDeckId, RealtimeDeckRuntime>();
  private realtimeDeckControls: Record<RealtimeDeckId, RealtimeDeckControl> = {
    main: { ...REALTIME_DECK_DEFAULTS.main },
    sequence: { ...REALTIME_DECK_DEFAULTS.sequence },
    vocal: { ...REALTIME_DECK_DEFAULTS.vocal },
  };
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
  private realtimeStreamPrimary = false;
  private realtimeAnchor = 0;

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
    const masterAnalyser = context.createAnalyser();
    const captureDestination = context.createMediaStreamDestination();

    const rumbleFilter = context.createBiquadFilter();
    const airShelf = context.createBiquadFilter();
    const limiter = context.createDynamicsCompressor();

    masterGain.gain.value = this.masterVolume * this.masterVolume;
    rumbleFilter.type = "highpass";
    rumbleFilter.frequency.value = 28;
    rumbleFilter.Q.value = 0.71;
    compressor.threshold.value = -18;
    compressor.knee.value = 24;
    compressor.ratio.value = 2.6;
    compressor.attack.value = 0.008;
    compressor.release.value = 0.22;
    airShelf.type = "highshelf";
    airShelf.frequency.value = 11_500;
    airShelf.gain.value = 1.1;
    limiter.threshold.value = -2.5;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.001;
    limiter.release.value = 0.09;
    fxDelay.delayTime.value = secondsPerStep(this.bpm) * 3;
    fxFeedback.gain.value = 0.34;
    fxWet.gain.value = 0.24;
    reverb.buffer = this.createImpulseResponse(context, 1.7, 2.4);
    reverbWet.gain.value = 0.18;
    masterAnalyser.fftSize = 2048;
    masterAnalyser.smoothingTimeConstant = 0.76;

    const flangerDry = context.createGain();
    const flangerDelay = context.createDelay(0.05);
    const flangerWet = context.createGain();
    const flangerFeedback = context.createGain();
    const flangerDepth = context.createGain();
    const flangerLfo = context.createOscillator();
    const flangerSum = context.createGain();
    const sweepFilter = context.createBiquadFilter();
    const sweepDepth = context.createGain();
    const sweepLfo = context.createOscillator();
    const masterReverbSend = context.createGain();
    const masterEchoSend = context.createGain();

    flangerDelay.delayTime.value = 0.0035;
    flangerWet.gain.value = 0;
    flangerFeedback.gain.value = 0;
    flangerDepth.gain.value = 0;
    flangerLfo.type = "sine";
    flangerLfo.frequency.value = 0.23;
    sweepFilter.type = "lowpass";
    sweepFilter.frequency.value = 18_500;
    sweepFilter.Q.value = 0.9;
    sweepDepth.gain.value = 0;
    sweepLfo.type = "sine";
    sweepLfo.frequency.value = 0.16;
    masterReverbSend.gain.value = 0;
    masterEchoSend.gain.value = 0;

    const phaserDry = context.createGain();
    const phaserWet = context.createGain();
    const phaserSum = context.createGain();
    const phaserDepth = context.createGain();
    const phaserLfo = context.createOscillator();
    const phaserStages = Array.from({ length: 4 }, () => {
      const stage = context.createBiquadFilter();
      stage.type = "allpass";
      stage.frequency.value = 620;
      stage.Q.value = 0.6;
      return stage;
    });
    phaserWet.gain.value = 0;
    phaserDepth.gain.value = 0;
    phaserLfo.type = "sine";
    phaserLfo.frequency.value = 0.35;

    const driveDry = context.createGain();
    const drivePre = context.createGain();
    const driveShaper = context.createWaveShaper();
    const driveWet = context.createGain();
    const driveSum = context.createGain();
    driveShaper.curve = buildSoftClipCurve(this.masterEffectParams.driveEdge);
    driveShaper.oversample = "4x";
    drivePre.gain.value = 1;
    driveWet.gain.value = 0;

    const crushDry = context.createGain();
    const crushShaper = context.createWaveShaper();
    const crushWet = context.createGain();
    const crushSum = context.createGain();
    crushShaper.curve = buildQuantizeCurve(this.masterEffectParams.crushBits);
    crushWet.gain.value = 0;

    masterGain.connect(flangerDry).connect(flangerSum);
    masterGain.connect(flangerDelay);
    flangerDelay.connect(flangerWet).connect(flangerSum);
    flangerDelay.connect(flangerFeedback).connect(flangerDelay);
    flangerLfo.connect(flangerDepth).connect(flangerDelay.delayTime);

    flangerSum.connect(phaserDry).connect(phaserSum);
    flangerSum.connect(phaserStages[0]!);
    for (let index = 0; index < phaserStages.length - 1; index += 1) {
      phaserStages[index]!.connect(phaserStages[index + 1]!);
    }
    phaserStages[phaserStages.length - 1]!.connect(phaserWet).connect(phaserSum);
    for (const stage of phaserStages) phaserLfo.connect(phaserDepth).connect(stage.frequency);

    phaserSum.connect(driveDry).connect(driveSum);
    phaserSum.connect(drivePre).connect(driveShaper).connect(driveWet).connect(driveSum);

    driveSum.connect(crushDry).connect(crushSum);
    driveSum.connect(crushShaper).connect(crushWet).connect(crushSum);

    crushSum.connect(sweepFilter);
    sweepLfo.connect(sweepDepth).connect(sweepFilter.frequency);
    sweepFilter.connect(rumbleFilter);
    sweepFilter.connect(masterReverbSend).connect(reverb);
    sweepFilter.connect(masterEchoSend).connect(fxDelay);
    flangerLfo.start();
    sweepLfo.start();
    phaserLfo.start();

    rumbleFilter.connect(compressor);
    fxDelay.connect(fxFeedback).connect(fxDelay);
    fxDelay.connect(fxWet).connect(rumbleFilter);
    reverb.connect(reverbWet).connect(rumbleFilter);
    for (const deck of ["main", "sequence", "vocal"] as const) {
      const gain = context.createGain();
      const control = this.realtimeDeckControls[deck];
      gain.gain.value = control.muted ? 0 : control.volume;
      gain.connect(masterGain);
      this.realtimeDecks.set(deck, { gain, streamTime: 0 });
    }
    compressor.connect(airShelf);
    airShelf.connect(limiter);
    limiter.connect(masterAnalyser);
    masterAnalyser.connect(context.destination);
    masterAnalyser.connect(captureDestination);

    this.context = context;
    this.masterGain = masterGain;
    this.compressor = compressor;
    this.masterLimiter = limiter;
    this.flangerWet = flangerWet;
    this.flangerFeedback = flangerFeedback;
    this.flangerDepth = flangerDepth;
    this.flangerLfo = flangerLfo;
    this.driveShaper = driveShaper;
    this.crushShaper = crushShaper;
    this.sweepFilter = sweepFilter;
    this.sweepDepth = sweepDepth;
    this.sweepLfo = sweepLfo;
    this.masterReverbSend = masterReverbSend;
    this.masterEchoSend = masterEchoSend;
    this.phaserWet = phaserWet;
    this.phaserDry = phaserDry;
    this.phaserDepth = phaserDepth;
    this.phaserLfo = phaserLfo;
    this.phaserStages = phaserStages;
    this.driveDry = driveDry;
    this.driveWet = driveWet;
    this.drivePre = drivePre;
    this.crushDry = crushDry;
    this.crushWet = crushWet;
    this.applyMasterEffects();
    this.fxDelay = fxDelay;
    this.fxFeedback = fxFeedback;
    this.fxWet = fxWet;
    this.reverb = reverb;
    this.reverbWet = reverbWet;
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
      this.nextStepTime += stepSeconds;
      this.currentStep = (this.currentStep + 1) % STEPS_PER_BAR;
      this.transportStepCounter += 1;
      this.emit();
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

  setRealtimeDeckControl(deck: RealtimeDeckId, update: Partial<RealtimeDeckControl>): void {
    const current = this.realtimeDeckControls[deck];
    const next = {
      volume: clamp(update.volume ?? current.volume, 0, 1),
      muted: update.muted ?? current.muted,
      pitchSemitones: clamp(update.pitchSemitones ?? current.pitchSemitones, -12, 12),
      beatNudgeMs: clamp(update.beatNudgeMs ?? current.beatNudgeMs, -500, 500),
    };
    this.realtimeDeckControls[deck] = next;
    const runtime = this.realtimeDecks.get(deck);
    if (runtime && this.context) {
      if (runtime.streamTime > 0 && next.beatNudgeMs !== current.beatNudgeMs) {
        runtime.streamTime += (next.beatNudgeMs - current.beatNudgeMs) / 1_000;
      }
      runtime.gain.gain.setTargetAtTime(next.muted ? 0 : next.volume, this.context.currentTime, 0.012);
    }
  }

  async synchronizeRealtimeDeckClocks(leadSeconds = 0.45): Promise<number> {
    await this.initialize();
    const context = this.requireContext();
    const anchor = context.currentTime + clamp(leadSeconds, 0.1, 2);
    this.realtimeAnchor = anchor;
    for (const [deck, runtime] of this.realtimeDecks) {
      const nudgeSeconds = this.realtimeDeckControls[deck].beatNudgeMs / 1_000;
      runtime.streamTime = Math.max(context.currentTime + 0.02, anchor + nudgeSeconds);
    }
    return anchor;
  }

  async synchronizeRealtimeDeckClockToNextBar(deck: RealtimeDeckId, leadSeconds = 0.75): Promise<number> {
    await this.initialize();
    const context = this.requireContext();
    const runtime = this.realtimeDecks.get(deck);
    if (!runtime) return context.currentTime;
    const minimumStart = context.currentTime + clamp(leadSeconds, 0.1, 2);
    const barSeconds = secondsPerStep(this.bpm) * STEPS_PER_BAR;
    const anchor = this.realtimeAnchor > 0 ? this.realtimeAnchor : minimumStart;
    const elapsedBars = Math.max(0, Math.ceil((minimumStart - anchor) / barSeconds));
    const startsAt = anchor + elapsedBars * barSeconds;
    runtime.streamTime = startsAt + this.realtimeDeckControls[deck].beatNudgeMs / 1_000;
    return startsAt;
  }

  resetRealtimeDeckClock(deck: RealtimeDeckId): void {
    const runtime = this.realtimeDecks.get(deck);
    if (runtime) runtime.streamTime = 0;
  }

  private applyMix(shouldEmit = true, immediate = false): void {
    if (this.tracks.size === 0) {
      if (shouldEmit) this.emit();
      return;
    }
    const anySolo = [...this.tracks.values()].some((track) => track.mix.solo);
    const now = this.context?.currentTime ?? 0;
    const realtimeDuck = this.realtimeStreamPrimary ? 0 : 1;
    for (const track of this.tracks.values()) {
      const gain = track.loadedBuffer ? effectiveTrackGain(track.mix, anySolo) * realtimeDuck : 0;
      if (immediate) track.gain.gain.value = gain;
      else track.gain.gain.setTargetAtTime(gain, now, 0.012);
    }
    if (shouldEmit) this.emit();
  }

  setMasterEffect(effect: keyof MasterEffectsState, amount: number): void {
    this.masterEffects = { ...this.masterEffects, [effect]: clamp(amount, 0, 1) };
    this.applyMasterEffects();
  }

  getMasterEffects(): MasterEffectsState {
    return { ...this.masterEffects };
  }

  setMasterEffectParam(param: keyof MasterEffectParams, value: number): void {
    const bounded = clamp(value, 0, 1);
    this.masterEffectParams = { ...this.masterEffectParams, [param]: bounded };
    if (param === "driveEdge" && this.driveShaper) this.driveShaper.curve = buildSoftClipCurve(bounded);
    if (param === "crushBits" && this.crushShaper) this.crushShaper.curve = buildQuantizeCurve(bounded);
    this.applyMasterEffects();
  }

  getMasterEffectParams(): MasterEffectParams {
    return { ...this.masterEffectParams };
  }

  setSfxLevel(level: number): void {
    this.sfxLevel = clamp(level, 0, 1);
  }

  getSfxLevel(): number {
    return this.sfxLevel;
  }

  async playSfx(kind: SfxKind): Promise<void> {
    await this.initialize();
    const context = this.requireContext();
    const master = this.masterGain;
    if (!master) return;
    const now = context.currentTime + 0.02;
    const out = context.createGain();
    out.connect(master);
    const envelope = (basePeak: number, attack: number, hold: number, release: number) => {
      const peak = Math.max(0.0002, basePeak * this.sfxLevel * this.sfxLevel * 1.6);
      out.gain.setValueAtTime(0.0001, now);
      out.gain.exponentialRampToValueAtTime(peak, now + Math.max(0.005, attack));
      out.gain.setValueAtTime(peak, now + attack + hold);
      out.gain.exponentialRampToValueAtTime(0.0001, now + attack + hold + release);
    };
    const noiseSource = (seconds: number) => {
      const buffer = context.createBuffer(1, Math.ceil(context.sampleRate * seconds), context.sampleRate);
      const data = buffer.getChannelData(0);
      let state = 0x9e3779b9 ^ Math.floor(seconds * 1_000);
      for (let index = 0; index < data.length; index += 1) {
        state = Math.imul(state ^ (state >>> 15), 1 | state);
        data[index] = (((state ^ (state >>> 14)) >>> 0) / 4294967296) * 2 - 1;
      }
      const source = context.createBufferSource();
      source.buffer = buffer;
      return source;
    };
    const stopAll = (nodes: Array<OscillatorNode | AudioBufferSourceNode>, seconds: number) => {
      for (const node of nodes) {
        node.start(now);
        node.stop(now + seconds);
      }
      window.setTimeout(() => out.disconnect(), (seconds + 1.2) * 1_000);
    };

    if (kind === "siren") {
      const oscillators = [context.createOscillator(), context.createOscillator()];
      const lfo = context.createOscillator();
      const lfoDepth = context.createGain();
      lfo.type = "triangle";
      lfo.frequency.value = 3.4;
      lfoDepth.gain.value = 320;
      for (const [index, oscillator] of oscillators.entries()) {
        oscillator.type = "sawtooth";
        oscillator.frequency.value = 880 + index * 6;
        lfo.connect(lfoDepth).connect(oscillator.frequency);
        oscillator.connect(out);
      }
      envelope(0.34, 0.03, 1.7, 0.5);
      stopAll([...oscillators, lfo], 2.4);
    } else if (kind === "airhorn") {
      const oscillators = [0, 1, 2].map(() => context.createOscillator());
      for (const [index, oscillator] of oscillators.entries()) {
        oscillator.type = "sawtooth";
        oscillator.frequency.setValueAtTime(508 + index * 5, now);
        oscillator.frequency.exponentialRampToValueAtTime(415 + index * 4, now + 0.14);
        oscillator.connect(out);
      }
      envelope(0.4, 0.01, 0.75, 0.35);
      stopAll(oscillators, 1.3);
    } else if (kind === "riser") {
      const noise = noiseSource(2.2);
      const filter = context.createBiquadFilter();
      filter.type = "bandpass";
      filter.Q.value = 1.4;
      filter.frequency.setValueAtTime(220, now);
      filter.frequency.exponentialRampToValueAtTime(9_500, now + 2.1);
      noise.connect(filter).connect(out);
      envelope(0.42, 0.4, 1.5, 0.25);
      stopAll([noise], 2.25);
    } else if (kind === "impact") {
      const sub = context.createOscillator();
      sub.type = "sine";
      sub.frequency.setValueAtTime(72, now);
      sub.frequency.exponentialRampToValueAtTime(28, now + 0.9);
      sub.connect(out);
      const burst = noiseSource(0.24);
      const burstFilter = context.createBiquadFilter();
      burstFilter.type = "lowpass";
      burstFilter.frequency.value = 2_400;
      burst.connect(burstFilter).connect(out);
      envelope(0.85, 0.005, 0.12, 1);
      stopAll([sub, burst], 1.25);
    } else {
      const oscillator = context.createOscillator();
      oscillator.type = "square";
      oscillator.frequency.setValueAtTime(2_400, now);
      oscillator.frequency.exponentialRampToValueAtTime(160, now + 0.34);
      oscillator.connect(out);
      envelope(0.3, 0.005, 0.08, 0.3);
      stopAll([oscillator], 0.5);
    }
  }

  async loadPadLoop(slot: number, data: ArrayBuffer, fileName: string): Promise<void> {
    await this.initialize();
    const context = this.requireContext();
    const buffer = await context.decodeAudioData(data.slice(0));
    this.stopPadLoop(slot);
    this.padLoops.set(slot, { buffer, name: fileName, playing: false });
  }

  togglePadLoop(slot: number): boolean {
    const pad = this.padLoops.get(slot);
    const context = this.context;
    const master = this.masterGain;
    if (!pad?.buffer || !context || !master) return false;
    if (pad.playing) {
      this.stopPadLoop(slot);
      return false;
    }
    const source = context.createBufferSource();
    source.buffer = pad.buffer;
    source.loop = true;
    const gain = context.createGain();
    gain.gain.value = 0.72;
    source.connect(gain).connect(master);
    const startsAt = this.playing ? this.nextBarTime() : context.currentTime + MIN_SCHEDULE_LEAD_SECONDS;
    source.start(startsAt);
    pad.source = source;
    pad.gain = gain;
    pad.playing = true;
    return true;
  }

  stopPadLoop(slot: number): void {
    const pad = this.padLoops.get(slot);
    if (!pad) return;
    try {
      pad.source?.stop();
    } catch {
      // The pad source may already have ended.
    }
    pad.gain?.disconnect();
    pad.source = undefined;
    pad.gain = undefined;
    pad.playing = false;
  }

  getPadLoops(): Array<{ slot: number; name?: string; playing: boolean }> {
    return [0, 1, 2, 3].map((slot) => {
      const pad = this.padLoops.get(slot);
      return { slot, name: pad?.name, playing: pad?.playing ?? false };
    });
  }

  private applyMasterEffects(): void {
    const now = this.context?.currentTime ?? 0;
    const smooth = (param: AudioParam | undefined, value: number) => param?.setTargetAtTime(value, now, 0.06);
    const { flanger, sweep, reverb, echo, drive, crush, phaser } = this.masterEffects;
    const params = this.masterEffectParams;
    smooth(this.flangerWet?.gain, flanger * 0.85);
    smooth(this.flangerFeedback?.gain, flanger * (0.15 + params.flangerFeedback * 0.7));
    smooth(this.flangerDepth?.gain, flanger * (0.0008 + params.flangerDepth * 0.0035));
    smooth(this.flangerLfo?.frequency, 0.05 + params.flangerRate * 1.15);
    smooth(this.phaserDry?.gain, 1 - phaser * 0.5);
    smooth(this.phaserWet?.gain, phaser * 0.95);
    smooth(this.phaserDepth?.gain, phaser * (120 + params.phaserDepth * 680));
    smooth(this.phaserLfo?.frequency, 0.05 + params.phaserRate * 1.4);
    smooth(this.driveDry?.gain, 1 - drive);
    smooth(this.drivePre?.gain, 1 + drive * 14);
    smooth(this.driveWet?.gain, drive * (0.9 - drive * 0.35));
    smooth(this.crushDry?.gain, 1 - crush);
    smooth(this.crushWet?.gain, crush * 0.9);
    smooth(this.sweepFilter?.frequency, 18_500 - sweep * 16_800);
    smooth(this.sweepFilter?.Q, 0.9 + sweep * (0.5 + params.sweepReso * 7));
    smooth(this.sweepDepth?.gain, sweep * 1_350);
    smooth(this.sweepLfo?.frequency, 0.05 + params.sweepRate * 0.95);
    // The shared reverb/echo returns are attenuated (0.18 / 0.24), so the master
    // sends run hot to land at a clearly audible wet level at full amount.
    smooth(this.masterReverbSend?.gain, reverb * 2.2);
    smooth(this.masterEchoSend?.gain, echo * 1.8);
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

  async playRealtimePcm16(
    bytes: Uint8Array,
    sampleRateHz: number,
    channels: number,
    deck: RealtimeDeckId = "main",
  ): Promise<void> {
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
    const runtime = this.realtimeDecks.get(deck);
    if (!runtime) return;
    const control = this.realtimeDeckControls[deck];
    const source = context.createBufferSource();
    source.buffer = buffer;
    const playbackRate = 2 ** (control.pitchSemitones / 12);
    source.playbackRate.value = playbackRate;
    source.connect(runtime.gain);
    const earliest = Math.max(context.currentTime + 0.02, context.currentTime + 0.08 + control.beatNudgeMs / 1_000);
    if (runtime.streamTime < context.currentTime + 0.02 || runtime.streamTime > context.currentTime + 10) {
      runtime.streamTime = earliest;
    }
    const startsAt = runtime.streamTime;
    runtime.streamTime += buffer.duration / playbackRate;
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
