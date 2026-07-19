import { invoke, isTauri } from "@tauri-apps/api/core";

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
  { text: "prepared piano, dark ambient techno pulse, intimate felt attack", weight: 1 },
  { text: "subtle modular synth pads, restrained hall reverb, cinematic space", weight: 0.55 },
];

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
