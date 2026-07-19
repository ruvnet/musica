export const TRACK_IDS = ["drums", "bass", "chords", "lead", "voice", "texture"] as const;

export type TrackId = (typeof TRACK_IDS)[number];
export type InstrumentKind = "drums" | "bass" | "poly" | "lead" | "voice" | "texture";
export type VisualSceneId =
  | "tunnel"
  | "bloom"
  | "terrain"
  | "lasergrid"
  | "aurora"
  | "monolith"
  | "pulsefield"
  | "chromawave";
export type VisualSceneMode = "tunnel" | "bloom" | "terrain";

export interface VisualSceneMeta {
  id: VisualSceneId;
  mode: VisualSceneMode;
  name: string;
  label: string;
  color: string;
  accent: string;
}

export interface VisualTemporalControls {
  speed: number;
  strobe: number;
  trail: number;
  morph: number;
  camera: number;
  phase: number;
}

export interface VisualPreset {
  id: string;
  name: string;
  scene: VisualSceneId;
  intensity: number;
  artDirection: {
    sculpture: number;
    motion: number;
    atmosphere: number;
    ribbon: number;
  };
  temporal: VisualTemporalControls;
}

export interface TrackDefinition {
  id: TrackId;
  name: string;
  shortName: string;
  color: string;
  instrument: InstrumentKind;
  pattern: boolean[];
  notes: number[];
}

export interface TrackMix {
  volume: number;
  pan: number;
  muted: boolean;
  solo: boolean;
}

export interface TrackTemplate {
  pattern: number[];
  notes: number[];
  volume?: number;
  pan?: number;
}

export interface PerformanceTemplate {
  id: string;
  name: string;
  description: string;
  bpm: number;
  prompt: string;
  scene: VisualSceneId;
  intensity: number;
  artDirection: {
    sculpture: number;
    motion: number;
    atmosphere: number;
    ribbon: number;
  };
  temporal?: VisualTemporalControls;
  tracks: Record<TrackId, TrackTemplate>;
}

export interface TrackSnapshot extends TrackDefinition, TrackMix {
  loadedFile?: string;
}

export interface AudioMetrics {
  frequency: Uint8Array<ArrayBuffer>;
  waveform: Uint8Array<ArrayBuffer>;
  trackLevels: Record<TrackId, number>;
  masterLevel: number;
  beatPhase: number;
  currentStep: number;
  bpm: number;
  playing: boolean;
}

export type ControlAction =
  | "transport.toggle"
  | "transport.stop"
  | "transport.record"
  | "tempo.tap"
  | "tempo.delta"
  | "track.next"
  | "track.previous"
  | "track.mute"
  | "track.solo"
  | "track.trigger"
  | "master.delta"
  | "visual.next"
  | "visual.previous"
  | "visual.scene.select"
  | "visual.intensity.delta"
  | "visual.sculpture.delta"
  | "visual.motion.delta"
  | "visual.atmosphere.delta"
  | "visual.ribbon.delta"
  | "visual.temporal.speed.delta"
  | "visual.temporal.strobe.delta"
  | "visual.temporal.trail.delta"
  | "visual.temporal.morph.delta"
  | "visual.temporal.camera.delta"
  | "visual.temporal.phase.delta"
  | "performance.template.select";

export interface ControlMessage {
  action: ControlAction;
  value?: number;
  source: "keyboard" | "shortcut" | "logitech" | "midi" | "ui";
  timestamp: number;
}

export interface SocialPreset {
  id: "reel-6" | "reel-9" | "reel-15" | "reel-30" | "square-15";
  label: string;
  width: number;
  height: number;
  fps: 30 | 60;
  durationSeconds: number;
  videoBitsPerSecond: number;
}

export interface ProviderStatus {
  available: boolean;
  provider: string;
  endpointHost?: string;
  reason?: string;
  model?: string;
  unitCostUsd?: number;
  maxDurationSeconds?: number;
}

export interface GenerationSection {
  timeSeconds: number;
  section: string;
}

export interface ReferenceAsset {
  mimeType: string;
  storageUri: string;
}

export interface GenerationRequest {
  prompt: string;
  durationSeconds: number;
  instrumental: boolean;
  seed?: number;
  language?: string;
  bpm?: number;
  lyrics?: string;
  structure?: GenerationSection[];
  outputFormat?: "mp3" | "wav";
  referenceAssets?: ReferenceAsset[];
  maxCostUsd?: number;
  candidateCount?: number;
  maxAttempts?: number;
  rightsDeclared?: boolean;
  clientRequestId?: string;
}

export interface GenerationProvenance {
  requestId: string;
  promptHash: string;
  generatedAt: string;
  modelVersion: string;
  pricingVersion: string;
  termsVersion?: string;
  synthidExpected: boolean;
  c2paExpected: boolean;
  c2paStatus: "preserved_unverified" | "present" | "valid" | "invalid" | "unsupported";
  providerBillingVerified: boolean;
}

export interface GenerationTask {
  id: string;
  status: "queued" | "processing" | "complete" | "failed" | "cancelled";
  title?: string;
  audioUrl?: string;
  provider: string;
  model?: string;
  hasAudio: boolean;
  audioMimeType?: string;
  lyrics?: string;
  structure?: string;
  actualDurationSeconds?: number;
  sampleRateHz?: number;
  channels?: number;
  reservedCostUsd?: number;
  generationCostUsd?: number;
  provenance?: GenerationProvenance;
  errorCode?: string;
  cancellationRequested: boolean;
  providerCancelConfirmed: boolean;
  completedAfterCancel: boolean;
}
