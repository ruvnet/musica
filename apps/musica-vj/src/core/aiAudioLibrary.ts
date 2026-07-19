import type { TrackId } from "./types";

export interface AiTonePreset {
  id: string;
  name: string;
  description: string;
  fileName: string;
  url: string;
  mimeType: "audio/mpeg" | "audio/wav";
  targetTracks: TrackId[];
  baseNote: number;
  grainSeconds: number;
  level: number;
  brightness: number;
  windowStartSeconds: number;
  windowDurationSeconds: number;
}

const moonlightSource = new URL("../../samples/lyria/moonlight-sonata-ai-timbre.mp3", import.meta.url).href;

export const AI_TONE_LIBRARY: AiTonePreset[] = [
  {
    id: "moonlight-bowed-bass",
    name: "Moonlight Bowed Bass",
    description: "Low bowed-synth body for pedal tones",
    fileName: "moonlight-sonata-ai-timbre.mp3",
    url: moonlightSource,
    mimeType: "audio/mpeg",
    targetTracks: ["bass", "texture"],
    baseNote: 37,
    grainSeconds: 1.08,
    level: 0.064,
    brightness: 0.28,
    windowStartSeconds: 16,
    windowDurationSeconds: 28,
  },
  {
    id: "moonlight-felt-piano",
    name: "Moonlight Felt",
    description: "Soft piano transients and close harmonic body",
    fileName: "moonlight-sonata-ai-timbre.mp3",
    url: moonlightSource,
    mimeType: "audio/mpeg",
    targetTracks: ["chords", "lead"],
    baseNote: 61,
    grainSeconds: 0.74,
    level: 0.07,
    brightness: 0.46,
    windowStartSeconds: 0,
    windowDurationSeconds: 18,
  },
  {
    id: "moonlight-glass-bells",
    name: "Moonlight Glass",
    description: "Bell harmonics for lead and voice gestures",
    fileName: "moonlight-sonata-ai-timbre.mp3",
    url: moonlightSource,
    mimeType: "audio/mpeg",
    targetTracks: ["lead", "voice"],
    baseNote: 73,
    grainSeconds: 0.44,
    level: 0.06,
    brightness: 0.82,
    windowStartSeconds: 10,
    windowDurationSeconds: 20,
  },
  {
    id: "moonlight-dark-pad",
    name: "Moonlight Pad",
    description: "Dark sustained synthetic resonance",
    fileName: "moonlight-sonata-ai-timbre.mp3",
    url: moonlightSource,
    mimeType: "audio/mpeg",
    targetTracks: ["chords", "texture"],
    baseNote: 49,
    grainSeconds: 1.36,
    level: 0.052,
    brightness: 0.34,
    windowStartSeconds: 22,
    windowDurationSeconds: 28,
  },
];

export const DEFAULT_AI_TONE_BANK: Partial<Record<TrackId, string>> = {
  bass: "moonlight-bowed-bass",
  chords: "moonlight-felt-piano",
  lead: "moonlight-glass-bells",
  voice: "moonlight-glass-bells",
  texture: "moonlight-dark-pad",
};

export function aiTonePresetById(id: string): AiTonePreset {
  return AI_TONE_LIBRARY.find((preset) => preset.id === id) ?? AI_TONE_LIBRARY[0];
}
