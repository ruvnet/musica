import type { TrackDefinition, TrackId, TrackMix } from "./types";

export const STEPS_PER_BAR = 16;
export const MIN_BPM = 60;
export const MAX_BPM = 200;

const COLORS: Record<TrackId, string> = {
  drums: "#ff4f86",
  bass: "#ff9c42",
  chords: "#ffe066",
  lead: "#75f4c5",
  voice: "#70a9ff",
  texture: "#b889ff",
};

const DEFAULT_PATTERNS: Record<TrackId, number[]> = {
  drums: [0, 2, 4, 6, 8, 10, 11, 12, 14, 15],
  bass: [0, 3, 6, 8, 11, 14],
  chords: [0, 7, 12],
  lead: [2, 5, 7, 10, 13, 15],
  voice: [4, 11, 15],
  texture: [0, 4, 8, 12, 14],
};

const SCALE_PALETTES = [
  { root: 36, intervals: [0, 2, 3, 7, 10], progression: [0, 7, 3, 10] },
  { root: 38, intervals: [0, 2, 5, 7, 9], progression: [0, 5, 9, 7] },
  { root: 41, intervals: [0, 3, 5, 7, 10], progression: [0, 10, 7, 3] },
  { root: 43, intervals: [0, 2, 4, 7, 9, 11], progression: [0, 9, 4, 7] },
];

export function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

export function secondsPerStep(bpm: number): number {
  return 60 / clamp(bpm, MIN_BPM, MAX_BPM) / 4;
}

export function midiToFrequency(note: number): number {
  return 440 * 2 ** ((note - 69) / 12);
}

export function patternFromSteps(steps: readonly number[], length = STEPS_PER_BAR): boolean[] {
  const pattern = Array.from({ length }, () => false);
  for (const step of steps) {
    if (Number.isInteger(step) && step >= 0 && step < length) pattern[step] = true;
  }
  return pattern;
}

export function hashSeed(input: string): number {
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

export function createTrackDefinitions(seed = "musica-vj"): TrackDefinition[] {
  const random = mulberry32(hashSeed(seed));
  const palette = SCALE_PALETTES[Math.floor(random() * SCALE_PALETTES.length)];
  const notes = Array.from({ length: 32 }, (_, index) => {
    const chordRoot = palette.progression[Math.floor(index / 8) % palette.progression.length];
    const interval = palette.intervals[(index + Math.floor(random() * palette.intervals.length)) % palette.intervals.length];
    const octave = index % 8 > 4 ? 12 : 0;
    return palette.root + chordRoot + interval + octave;
  });
  const bassNotes = palette.progression.flatMap((interval) => [
    palette.root + interval - 12,
    palette.root + interval - 12,
    palette.root + interval - 5,
    palette.root + interval - 12,
  ]);
  const chordNotes = palette.progression.flatMap((interval) => [
    palette.root + interval + 12,
    palette.root + interval + 15,
    palette.root + interval + 19,
    palette.root + interval + 24,
  ]);
  const drumNotes = [36, 42, 38, 42, 36, 46, 38, 42, 36, 42, 38, 46, 36, 42, 38, 49];

  const definition = (
    id: TrackId,
    name: string,
    shortName: string,
    instrument: TrackDefinition["instrument"],
    trackNotes: number[],
  ): TrackDefinition => ({
    id,
    name,
    shortName,
    instrument,
    color: COLORS[id],
    pattern: patternFromSteps(DEFAULT_PATTERNS[id]),
    notes: trackNotes,
  });

  return [
    definition("drums", "Pulse", "PLS", "drums", drumNotes),
    definition("bass", "Gravity", "GRV", "bass", bassNotes),
    definition("chords", "Prism", "PRS", "poly", chordNotes),
    definition("lead", "Signal", "SIG", "lead", notes.map((note, index) => note + 24 + (index % 11 === 0 ? 12 : 0))),
    definition("voice", "Breath", "BRH", "voice", notes.map((note) => note + 12)),
    definition("texture", "Dust", "DST", "texture", chordNotes.map((note, index) => note - (index % 2 === 0 ? 12 : 0))),
  ];
}

export function defaultMix(): TrackMix {
  return { volume: 0.78, pan: 0, muted: false, solo: false };
}

export function effectiveTrackGain(mix: TrackMix, anySolo: boolean): number {
  if (mix.muted || (anySolo && !mix.solo)) return 0;
  const normalized = clamp(mix.volume, 0, 1);
  return normalized * normalized;
}

export function mutatePattern(pattern: readonly boolean[], seed: number, density = 0.16): boolean[] {
  const random = mulberry32(seed);
  return pattern.map((active, index) => {
    if (index === 0) return active;
    return random() < clamp(density, 0, 0.5) ? !active : active;
  });
}
