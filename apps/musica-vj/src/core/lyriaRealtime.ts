import { invoke, isTauri } from "@tauri-apps/api/core";
import type { PerformanceTemplate } from "./types";

export type LyriaRealtimeScale =
  | "C_MAJOR_A_MINOR"
  | "D_FLAT_MAJOR_B_FLAT_MINOR"
  | "D_MAJOR_B_MINOR"
  | "E_FLAT_MAJOR_C_MINOR"
  | "E_MAJOR_D_FLAT_MINOR"
  | "F_MAJOR_D_MINOR"
  | "G_FLAT_MAJOR_E_FLAT_MINOR"
  | "G_MAJOR_E_MINOR"
  | "A_FLAT_MAJOR_F_MINOR"
  | "A_MAJOR_G_FLAT_MINOR"
  | "B_FLAT_MAJOR_G_MINOR"
  | "B_MAJOR_A_FLAT_MINOR"
  | "SCALE_UNSPECIFIED";

export type LyriaRealtimeMode = "QUALITY" | "DIVERSITY" | "VOCALIZATION";

export interface LyriaWeightedPrompt {
  text: string;
  weight: number;
}

export interface LyriaRealtimeConfig {
  bpm: number;
  guidance: number;
  density: number;
  brightness: number;
  temperature: number;
  topK: number;
  seed?: number;
  scale: LyriaRealtimeScale;
  muteBass: boolean;
  muteDrums: boolean;
  onlyBassAndDrums: boolean;
  musicGenerationMode: LyriaRealtimeMode;
}

export interface LyriaRealtimeRequest {
  weightedPrompts: LyriaWeightedPrompt[];
  config: LyriaRealtimeConfig;
}

export interface LyriaRealtimeStylePreset {
  id: string;
  label: string;
  prompts: LyriaWeightedPrompt[];
  config: Partial<LyriaRealtimeConfig>;
}

export interface LyriaRealtimeStatus {
  available: boolean;
  provider: "lyria_realtime" | string;
  model: string;
  sampleRateHz: number;
  channels: number;
  audioFormat: "pcm16" | string;
  instrumentalOnly: boolean;
  reason?: string;
  activeSessionId?: string;
  bufferedAudioBytes: number;
  streamedAudioBytes: number;
  warning?: string;
}

export interface LyriaRealtimeSession {
  id: string;
  provider: string;
  model: string;
  state: string;
  weightedPrompts: LyriaWeightedPrompt[];
  config: LyriaRealtimeConfig;
  sampleRateHz: number;
  channels: number;
  audioFormat: string;
}

export interface LyriaRealtimeAudioPoll {
  sessionId?: string;
  sampleRateHz: number;
  channels: number;
  audioFormat: string;
  chunks: number[][];
  bufferedAudioBytes: number;
  streamedAudioBytes: number;
  warning?: string;
}

export const DEFAULT_LYRIA_REALTIME_CONFIG: LyriaRealtimeConfig = {
  bpm: 118,
  guidance: 4,
  density: 0.52,
  brightness: 0.42,
  temperature: 1.1,
  topK: 40,
  scale: "E_FLAT_MAJOR_C_MINOR",
  muteBass: false,
  muteDrums: false,
  onlyBassAndDrums: false,
  musicGenerationMode: "QUALITY",
};

export const DEFAULT_LYRIA_REALTIME_PROMPTS: LyriaWeightedPrompt[] = [
  { text: "high quality instrumental performance bed, clear groove, polished mix, musical transitions", weight: 1 },
  { text: "support the local sequencer and piano, leave space for live lead notes", weight: 0.7 },
];

export const LYRIA_REALTIME_STYLE_PRESETS: LyriaRealtimeStylePreset[] = [
  {
    id: "samba",
    label: "Samba",
    prompts: [
      { text: "Brazilian samba groove, nylon guitar, cavaquinho, surdo, tamborim, warm percussion", weight: 1 },
      { text: "sunny dance energy, syncopated acoustic rhythm section, polished instrumental mix", weight: 0.75 },
    ],
    config: { bpm: 104, density: 0.72, brightness: 0.64, guidance: 4.4, scale: "C_MAJOR_A_MINOR" },
  },
  {
    id: "rock",
    label: "Rock",
    prompts: [
      { text: "tight live rock band, electric guitars, punchy bass, real drum kit, anthemic instrumental hook", weight: 1 },
      { text: "clean production, strong backbeat, energetic but not noisy", weight: 0.72 },
    ],
    config: { bpm: 126, density: 0.68, brightness: 0.58, guidance: 4.2, scale: "E_MAJOR_D_FLAT_MINOR" },
  },
  {
    id: "jazz",
    label: "Jazz",
    prompts: [
      { text: "modern jazz trio, piano, upright bass, brushed drums, tasteful harmonic movement", weight: 1 },
      { text: "blue note color, human swing feel, intimate club recording", weight: 0.68 },
    ],
    config: { bpm: 112, density: 0.5, brightness: 0.38, guidance: 4.6, scale: "B_FLAT_MAJOR_G_MINOR", muteDrums: false },
  },
  {
    id: "techno",
    label: "Techno",
    prompts: [
      { text: "deep warehouse techno, solid kick, rolling bass, hypnotic stabs, club ready instrumental", weight: 1 },
      { text: "minimal arrangement, powerful low end, crisp hats, dark atmosphere", weight: 0.76 },
    ],
    config: { bpm: 132, density: 0.62, brightness: 0.46, guidance: 4.1, scale: "F_MAJOR_D_MINOR" },
  },
  {
    id: "classical",
    label: "Classical",
    prompts: [
      { text: "classical piano and chamber strings, public domain romantic harmony, expressive arpeggios", weight: 1 },
      { text: "cinematic concert hall sound, elegant phrasing, restrained dynamics", weight: 0.7 },
    ],
    config: { bpm: 76, density: 0.36, brightness: 0.32, guidance: 5.0, scale: "E_FLAT_MAJOR_C_MINOR", muteDrums: true },
  },
  {
    id: "cinematic",
    label: "Cinema",
    prompts: [
      { text: "cinematic electronic score, low pulses, shimmering piano, orchestral pads, widescreen drama", weight: 1 },
      { text: "clear emotional arc, polished trailer-quality instrumental bed", weight: 0.72 },
    ],
    config: { bpm: 104, density: 0.52, brightness: 0.44, guidance: 4.5, scale: "C_MAJOR_A_MINOR" },
  },
];

const TEMPLATE_STYLE_MAP: Record<string, string> = {
  "moonlight-sequencer": "classical",
  "warehouse-techno": "techno",
  "liquid-breaks": "techno",
  "ambient-dub": "cinematic",
  "synthwave-drive": "cinematic",
  "footwork-cuts": "techno",
  "cinematic-pulse": "cinematic",
  "uk-garage-neon": "techno",
  "afro-cosmic-house": "samba",
  "idm-crystalline": "techno",
  "hyperpop-rush": "rock",
};

export async function getLyriaRealtimeStatus(): Promise<LyriaRealtimeStatus> {
  if (!isTauri()) {
    return {
      available: false,
      provider: "browser_preview",
      model: "models/lyria-realtime-exp",
      sampleRateHz: 48_000,
      channels: 2,
      audioFormat: "pcm16",
      instrumentalOnly: true,
      reason: "Lyria RealTime requires the desktop app so the Gemini key stays out of React",
      bufferedAudioBytes: 0,
      streamedAudioBytes: 0,
    };
  }
  return invoke<LyriaRealtimeStatus>("lyria_realtime_status");
}

export function lyriaRealtimeStyleById(id: string): LyriaRealtimeStylePreset {
  return LYRIA_REALTIME_STYLE_PRESETS.find((preset) => preset.id === id) ?? LYRIA_REALTIME_STYLE_PRESETS[0];
}

export function createLyriaRealtimeRequestFromStyle(style: LyriaRealtimeStylePreset, bpm?: number): LyriaRealtimeRequest {
  return {
    weightedPrompts: style.prompts.map((prompt) => ({ ...prompt })),
    config: {
      ...DEFAULT_LYRIA_REALTIME_CONFIG,
      ...style.config,
      bpm: bpm ?? style.config.bpm ?? DEFAULT_LYRIA_REALTIME_CONFIG.bpm,
      onlyBassAndDrums: style.config.onlyBassAndDrums ?? false,
      muteBass: style.config.muteBass ?? false,
      muteDrums: style.config.muteDrums ?? false,
    },
  };
}

export function createLyriaRealtimeRequestForTemplate(template: PerformanceTemplate): LyriaRealtimeRequest {
  const style = lyriaRealtimeStyleById(TEMPLATE_STYLE_MAP[template.id] ?? "cinematic");
  const request = createLyriaRealtimeRequestFromStyle(style, template.bpm);
  return {
    weightedPrompts: [
      ...request.weightedPrompts,
      {
        text: `follow the Musica preset: ${template.name}; ${template.description}; leave room for local MIDI and sequencer accents`,
        weight: 0.58,
      },
    ].slice(0, 4),
    config: request.config,
  };
}

export async function startLyriaRealtime(request: LyriaRealtimeRequest): Promise<LyriaRealtimeSession> {
  if (!isTauri()) throw new Error("Lyria RealTime requires the desktop app");
  return invoke<LyriaRealtimeSession>("lyria_realtime_start", { request });
}

export async function updateLyriaRealtime(request: LyriaRealtimeRequest): Promise<LyriaRealtimeSession> {
  if (!isTauri()) throw new Error("Lyria RealTime requires the desktop app");
  return invoke<LyriaRealtimeSession>("lyria_realtime_update", { request });
}

export async function stopLyriaRealtime(): Promise<void> {
  if (!isTauri()) return;
  await invoke<void>("lyria_realtime_stop");
}

export async function pollLyriaRealtimeAudio(): Promise<LyriaRealtimeAudioPoll> {
  if (!isTauri()) {
    return {
      sampleRateHz: 48_000,
      channels: 2,
      audioFormat: "pcm16",
      chunks: [],
      bufferedAudioBytes: 0,
      streamedAudioBytes: 0,
    };
  }
  return invoke<LyriaRealtimeAudioPoll>("lyria_realtime_poll_audio");
}
