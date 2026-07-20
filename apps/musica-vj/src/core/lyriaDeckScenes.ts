import type { LyriaRealtimeDeckId } from "./lyriaRealtime";

export interface LyriaDeckControl {
  volume: number;
  muted: boolean;
  pitchSemitones: number;
  beatNudgeMs: number;
}

export interface LyriaDeckScene {
  id: string;
  name: string;
  styleId: string;
  bpm: number;
  enabled: Record<LyriaRealtimeDeckId, boolean>;
  controls: Record<LyriaRealtimeDeckId, LyriaDeckControl>;
}

export const LYRIA_DECK_SCENE_STORAGE_KEY = "musica.lyria.deck-scenes.v1";

export const DEFAULT_LYRIA_DECK_CONTROLS: Record<LyriaRealtimeDeckId, LyriaDeckControl> = {
  main: { volume: 0.72, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
  sequence: { volume: 0.42, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
  vocal: { volume: 0.42, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
};

export const DEFAULT_LYRIA_DECK_SCENES: LyriaDeckScene[] = [
  {
    id: "club",
    name: "Rock",
    styleId: "rock",
    bpm: 126,
    enabled: { main: true, sequence: false, vocal: false },
    controls: {
      main: { volume: 0.76, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
      sequence: { volume: 0.38, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
      vocal: { volume: 0.42, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
    },
  },
  {
    id: "peak",
    name: "Peak",
    styleId: "techno",
    bpm: 132,
    enabled: { main: true, sequence: true, vocal: false },
    controls: {
      main: { volume: 0.7, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
      sequence: { volume: 0.58, muted: false, pitchSemitones: 0, beatNudgeMs: -20 },
      vocal: { volume: 0.42, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
    },
  },
  {
    id: "vocalize",
    name: "Vocalize",
    styleId: "cinematic",
    bpm: 104,
    enabled: { main: true, sequence: false, vocal: true },
    controls: {
      main: { volume: 0.58, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
      sequence: { volume: 0.22, muted: false, pitchSemitones: 0, beatNudgeMs: 15 },
      vocal: { volume: 0.24, muted: false, pitchSemitones: 0, beatNudgeMs: 35 },
    },
  },
  {
    id: "breaks",
    name: "Breaks",
    styleId: "drum-bass",
    bpm: 174,
    enabled: { main: true, sequence: true, vocal: false },
    controls: {
      main: { volume: 0.58, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
      sequence: { volume: 0.72, muted: false, pitchSemitones: 0, beatNudgeMs: -15 },
      vocal: { volume: 0.42, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
    },
  },
];

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, Number.isFinite(value) ? value : minimum));
}

function normalizeControl(value: Partial<LyriaDeckControl> | undefined, fallback: LyriaDeckControl): LyriaDeckControl {
  return {
    volume: clamp(value?.volume ?? fallback.volume, 0, 1),
    muted: value?.muted ?? fallback.muted,
    pitchSemitones: Math.round(clamp(value?.pitchSemitones ?? fallback.pitchSemitones, -7, 7)),
    beatNudgeMs: Math.round(clamp(value?.beatNudgeMs ?? fallback.beatNudgeMs, -250, 250) / 5) * 5,
  };
}

export function normalizeLyriaDeckScene(value: Partial<LyriaDeckScene>, fallback: LyriaDeckScene): LyriaDeckScene {
  return {
    id: fallback.id,
    name: String(value.name ?? fallback.name).trim().slice(0, 18) || fallback.name,
    styleId: String(value.styleId ?? fallback.styleId).trim().slice(0, 40) || fallback.styleId,
    bpm: Math.round(clamp(value.bpm ?? fallback.bpm, 60, 200)),
    enabled: {
      main: value.enabled?.main ?? fallback.enabled.main,
      sequence: value.enabled?.sequence ?? fallback.enabled.sequence,
      vocal: value.enabled?.vocal ?? fallback.enabled.vocal,
    },
    controls: {
      main: normalizeControl(value.controls?.main, fallback.controls.main),
      sequence: normalizeControl(value.controls?.sequence, fallback.controls.sequence),
      vocal: normalizeControl(value.controls?.vocal, fallback.controls.vocal),
    },
  };
}

export function cloneLyriaDeckScene(scene: LyriaDeckScene): LyriaDeckScene {
  return normalizeLyriaDeckScene(scene, scene);
}

export function loadLyriaDeckScenes(serialized?: string | null): LyriaDeckScene[] {
  if (!serialized) return DEFAULT_LYRIA_DECK_SCENES.map(cloneLyriaDeckScene);
  try {
    const parsed = JSON.parse(serialized) as unknown;
    if (!Array.isArray(parsed)) return DEFAULT_LYRIA_DECK_SCENES.map(cloneLyriaDeckScene);
    return DEFAULT_LYRIA_DECK_SCENES.map((fallback) => {
      const candidate = parsed.find((scene): scene is Partial<LyriaDeckScene> => (
        typeof scene === "object" && scene !== null && "id" in scene && scene.id === fallback.id
      ));
      return normalizeLyriaDeckScene(candidate ?? fallback, fallback);
    });
  } catch {
    return DEFAULT_LYRIA_DECK_SCENES.map(cloneLyriaDeckScene);
  }
}
