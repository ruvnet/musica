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
import { TRACK_IDS, type AudioMetrics, type PerformanceTemplate, type TrackDefinition, type TrackId, type TrackMix, type TrackSnapshot } from "../core/types";
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
  meterBuffer: Uint8Array<ArrayBuffer>;
  loadedBuffer?: AudioBuffer;
  loadedFile?: string;
  loadedLoop: boolean;
  loadedStartedAt?: number;
  loopSource?: AudioBufferSourceNode;
}

interface ScheduledSourceMetadata {
  startsAt: number;
  imported: boolean;
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

export class AudioEngine {
  private context?: AudioContext;
  private definitions = createTrackDefinitions();
  private pendingMix = new Map<TrackId, TrackMix>(this.definitions.map((track) => [track.id, defaultMix()]));
  private tracks = new Map<TrackId, TrackRuntime>();
  private masterGain?: GainNode;
  private compressor?: DynamicsCompressorNode;
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

  async initialize(): Promise<void> {
    if (this.context) {
      if (this.context.state === "suspended") await this.context.resume();
      return;
    }

    const context = new AudioContext({ latencyHint: "interactive", sampleRate: 48_000 });
    const masterGain = context.createGain();
    const compressor = context.createDynamicsCompressor();
    const masterAnalyser = context.createAnalyser();
    const captureDestination = context.createMediaStreamDestination();

    masterGain.gain.value = this.masterVolume * this.masterVolume;
    compressor.threshold.value = -12;
    compressor.knee.value = 8;
    compressor.ratio.value = 8;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.18;
    masterAnalyser.fftSize = 2048;
    masterAnalyser.smoothingTimeConstant = 0.76;

    masterGain.connect(compressor);
    compressor.connect(masterAnalyser);
    masterAnalyser.connect(context.destination);
    masterAnalyser.connect(captureDestination);

    this.context = context;
    this.masterGain = masterGain;
    this.compressor = compressor;
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
    const mix = { ...(this.pendingMix.get(definition.id) ?? defaultMix()) };
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.68;
    gain.gain.value = effectiveTrackGain(mix, false);
    pan.pan.value = mix.pan;

    input.connect(gain);
    gain.connect(pan);
    pan.connect(analyser);
    analyser.connect(master);

    this.tracks.set(definition.id, {
      definition,
      mix,
      input,
      gain,
      pan,
      analyser,
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
      this.scheduleStep(this.currentStep, this.transportStepCounter, this.nextStepTime);
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
      const envelope = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(step % 8 === 0 ? 152 : 112, time);
      oscillator.frequency.exponentialRampToValueAtTime(42, time + 0.22);
      envelope.gain.setValueAtTime(step % 8 === 0 ? 0.92 : 0.62, time);
      envelope.gain.exponentialRampToValueAtTime(0.001, time + 0.32);
      oscillator.connect(envelope).connect(track.input);
      this.startSource(oscillator, time, time + 0.34);
      return;
    }

    const source = context.createBufferSource();
    const filter = context.createBiquadFilter();
    const envelope = context.createGain();
    source.buffer = this.noiseBuffer ?? null;
    filter.type = note === 38 ? "bandpass" : "highpass";
    filter.Q.value = note === 38 ? 1.8 : 0.7;
    filter.frequency.value = note === 38 ? 980 : note >= 46 ? 5200 : 7600;
    envelope.gain.setValueAtTime(note === 38 ? 0.38 : note >= 46 ? 0.18 : 0.1, time);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + (note === 38 ? 0.22 : note >= 46 ? 0.18 : 0.055));
    source.connect(filter).connect(envelope).connect(track.input);
    this.startSource(source, time, time + (note === 38 ? 0.24 : 0.2));
  }

  private triggerBass(track: TrackRuntime, note: number, time: number): void {
    const context = this.requireContext();
    const oscillator = context.createOscillator();
    const filter = context.createBiquadFilter();
    const envelope = context.createGain();
    oscillator.type = "sawtooth";
    oscillator.frequency.value = midiToFrequency(note);
    filter.type = "lowpass";
    filter.Q.value = 5;
    filter.frequency.setValueAtTime(780, time);
    filter.frequency.exponentialRampToValueAtTime(190, time + 0.24);
    envelope.gain.setValueAtTime(0.001, time);
    envelope.gain.exponentialRampToValueAtTime(0.3, time + 0.012);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + 0.3);
    oscillator.connect(filter).connect(envelope).connect(track.input);
    this.startSource(oscillator, time, time + 0.32);
  }

  private triggerChord(track: TrackRuntime, root: number, time: number): void {
    for (const interval of [0, 3, 7]) this.triggerTone(track, root + interval, time, 0.56, "triangle", 0.1, 1500);
  }

  private triggerLead(track: TrackRuntime, note: number, time: number): void {
    this.triggerTone(track, note, time, 0.25, "square", 0.13, 3200, 8);
  }

  private triggerVoice(track: TrackRuntime, note: number, time: number): void {
    const context = this.requireContext();
    const source = context.createBufferSource();
    const formant = context.createBiquadFilter();
    const envelope = context.createGain();
    source.buffer = this.noiseBuffer ?? null;
    formant.type = "bandpass";
    formant.Q.value = 14;
    formant.frequency.setValueAtTime(midiToFrequency(note) * 4, time);
    formant.frequency.linearRampToValueAtTime(midiToFrequency(note) * 5.5, time + 0.42);
    envelope.gain.setValueAtTime(0.001, time);
    envelope.gain.linearRampToValueAtTime(0.23, time + 0.06);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + 0.48);
    source.connect(formant).connect(envelope).connect(track.input);
    this.startSource(source, time, time + 0.5);
  }

  private triggerTexture(track: TrackRuntime, note: number, time: number): void {
    const context = this.requireContext();
    const carrier = context.createOscillator();
    const modulator = context.createOscillator();
    const modGain = context.createGain();
    const envelope = context.createGain();
    carrier.type = "sine";
    carrier.frequency.value = midiToFrequency(note);
    modulator.type = "sine";
    modulator.frequency.value = midiToFrequency(note - 17);
    modGain.gain.value = 38;
    envelope.gain.setValueAtTime(0.001, time);
    envelope.gain.linearRampToValueAtTime(0.12, time + 0.2);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + 0.9);
    modulator.connect(modGain).connect(carrier.frequency);
    carrier.connect(envelope).connect(track.input);
    this.startSource(modulator, time, time + 0.92);
    this.startSource(carrier, time, time + 0.92);
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
    envelope.gain.exponentialRampToValueAtTime(level, time + 0.015);
    envelope.gain.exponentialRampToValueAtTime(0.001, time + duration);
    oscillator.connect(filter).connect(envelope).connect(track.input);
    this.startSource(oscillator, time, time + duration + 0.02);
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

  private applyMix(shouldEmit = true, immediate = false): void {
    if (this.tracks.size === 0) {
      if (shouldEmit) this.emit();
      return;
    }
    const anySolo = [...this.tracks.values()].some((track) => track.mix.solo);
    const now = this.context?.currentTime ?? 0;
    for (const track of this.tracks.values()) {
      const gain = effectiveTrackGain(track.mix, anySolo);
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
    const definitions = this.tracks.size > 0 ? [...this.tracks.values()].map((track) => track.definition) : this.definitions;
    for (const definition of definitions) {
      const trackTemplate = template.tracks[definition.id];
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
    this.applyMix();
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

  getSnapshot(): EngineSnapshot {
    const tracks = this.tracks.size > 0
      ? [...this.tracks.values()].map((track) => ({
          ...track.definition,
          pattern: [...track.definition.pattern],
          notes: [...track.definition.notes],
          ...track.mix,
          loadedFile: track.loadedFile,
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
