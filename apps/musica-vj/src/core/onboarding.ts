import {
  createLyriaRealtimeRequestFromStyle,
  lyriaRealtimeStyleById,
  type LyriaRealtimeRequest,
  type LyriaRealtimeStylePreset,
} from "./lyriaRealtime";
import type { VisualPaletteId, VisualSceneId } from "./types";
import type { VisualAnimationStyle } from "../visual/VisualEngine";

export const ONBOARDING_STORAGE_KEY = "musica.onboarding.v1";

export type MusicFormat = "instrumental" | "hybrid" | "vocal-led";
export type MusicPace = "slow" | "mid" | "fast";
export type VocalStyle = "none" | "male" | "female" | "other";
export type VocalRole = "sparse" | "chorus" | "experimental";

export interface OnboardingPreferences {
  version: 1;
  format: MusicFormat;
  styleId: string;
  pace: MusicPace;
  bpm: number;
  vocalStyle: VocalStyle;
  vocalRole: VocalRole;
  experimental: boolean;
  direction: string;
  visualScene: VisualSceneId;
  visualPalette: VisualPaletteId;
  visualAnimation: VisualAnimationStyle;
  visualIntensity: number;
}

export const DEFAULT_ONBOARDING_PREFERENCES: OnboardingPreferences = {
  version: 1,
  format: "instrumental",
  styleId: "rock",
  pace: "fast",
  bpm: 126,
  vocalStyle: "none",
  vocalRole: "sparse",
  experimental: false,
  direction: "Tight live-band energy, memorable guitar hook, strong dynamics, and a clean modern mix.",
  visualScene: "terrain",
  visualPalette: "ember",
  visualAnimation: "scan",
  visualIntensity: 0.72,
};

const FORMATS: MusicFormat[] = ["instrumental", "hybrid", "vocal-led"];
const PACES: MusicPace[] = ["slow", "mid", "fast"];
const VOCAL_STYLES: VocalStyle[] = ["none", "male", "female", "other"];
const VOCAL_ROLES: VocalRole[] = ["sparse", "chorus", "experimental"];
const VISUAL_SCENES: VisualSceneId[] = ["tunnel", "bloom", "terrain", "lasergrid", "aurora", "monolith", "pulsefield", "chromawave"];
const VISUAL_PALETTES: VisualPaletteId[] = ["scene", "neon", "ember", "ice", "prism", "mono"];
const VISUAL_ANIMATIONS: VisualAnimationStyle[] = ["flow", "orbit", "warp", "shards", "scan", "minimal"];

function boundedNumber(value: unknown, fallback: number, minimum: number, maximum: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? Math.max(minimum, Math.min(maximum, Math.round(parsed))) : fallback;
}

function enumValue<T extends string>(value: unknown, options: readonly T[], fallback: T): T {
  return typeof value === "string" && options.includes(value as T) ? value as T : fallback;
}

export function normalizeOnboardingPreferences(value: unknown): OnboardingPreferences {
  const source = value && typeof value === "object" ? value as Partial<OnboardingPreferences> : {};
  const style = lyriaRealtimeStyleById(typeof source.styleId === "string" ? source.styleId : DEFAULT_ONBOARDING_PREFERENCES.styleId);
  const legacyRockDirection = source.direction === DEFAULT_ONBOARDING_PREFERENCES.direction && style.id !== "rock";
  const vocalStyle = enumValue(source.vocalStyle, VOCAL_STYLES, DEFAULT_ONBOARDING_PREFERENCES.vocalStyle);
  const format = enumValue(source.format, FORMATS, DEFAULT_ONBOARDING_PREFERENCES.format);
  return {
    version: 1,
    format,
    styleId: style.id,
    pace: enumValue(source.pace, PACES, DEFAULT_ONBOARDING_PREFERENCES.pace),
    bpm: boundedNumber(source.bpm, style.config.bpm ?? DEFAULT_ONBOARDING_PREFERENCES.bpm, 60, 200),
    vocalStyle: format === "instrumental" ? "none" : vocalStyle === "none" ? "other" : vocalStyle,
    vocalRole: enumValue(source.vocalRole, VOCAL_ROLES, DEFAULT_ONBOARDING_PREFERENCES.vocalRole),
    experimental: source.experimental === true,
    direction: legacyRockDirection
      ? style.description.slice(0, 180)
      : typeof source.direction === "string"
      ? source.direction.trim().slice(0, 180)
      : DEFAULT_ONBOARDING_PREFERENCES.direction,
    visualScene: enumValue(source.visualScene, VISUAL_SCENES, DEFAULT_ONBOARDING_PREFERENCES.visualScene),
    visualPalette: enumValue(source.visualPalette, VISUAL_PALETTES, DEFAULT_ONBOARDING_PREFERENCES.visualPalette),
    visualAnimation: enumValue(source.visualAnimation, VISUAL_ANIMATIONS, DEFAULT_ONBOARDING_PREFERENCES.visualAnimation),
    visualIntensity: boundedNumber(
      typeof source.visualIntensity === "number" ? source.visualIntensity * 100 : undefined,
      DEFAULT_ONBOARDING_PREFERENCES.visualIntensity * 100,
      5,
      100,
    ) / 100,
  };
}

export function loadOnboardingPreferences(storage: Pick<Storage, "getItem"> = window.localStorage): OnboardingPreferences | undefined {
  try {
    const raw = storage.getItem(ONBOARDING_STORAGE_KEY);
    return raw ? normalizeOnboardingPreferences(JSON.parse(raw)) : undefined;
  } catch {
    return undefined;
  }
}

export function saveOnboardingPreferences(
  preferences: OnboardingPreferences,
  storage: Pick<Storage, "setItem"> = window.localStorage,
): OnboardingPreferences {
  const normalized = normalizeOnboardingPreferences(preferences);
  try {
    storage.setItem(ONBOARDING_STORAGE_KEY, JSON.stringify(normalized));
  } catch {
    // The session can continue when storage is unavailable.
  }
  return normalized;
}

export function bpmForPace(style: LyriaRealtimeStylePreset, pace: MusicPace): number {
  const center = style.config.bpm ?? 118;
  if (pace === "slow") return Math.max(60, Math.min(108, center - 22));
  if (pace === "fast") return Math.max(124, Math.min(200, center + (center < 120 ? 20 : 0)));
  return Math.max(96, Math.min(132, center));
}

export function createOnboardingRealtimeRequest(
  preferences: OnboardingPreferences,
  includeWelcomeCue = false,
): LyriaRealtimeRequest {
  const normalized = normalizeOnboardingPreferences(preferences);
  const style = lyriaRealtimeStyleById(normalized.styleId);
  const base = createLyriaRealtimeRequestFromStyle(style, normalized.bpm);
  const cue = includeWelcomeCue
    ? "Open once with a concise four-note Musica sonic logo: low root, rising fifth, bright octave, resolved major second; clean two-second phrase, then land on bar 1. "
    : "";
  const personal = normalized.direction || "memorable musical identity with coherent phrasing and live-set dynamics";
  const experiment = normalized.experimental
    ? " Add controlled generative variation, unusual timbral motion, and one surprising detail per eight bars without losing pulse or key."
    : " Preserve a clear motif, stable pulse, and disciplined arrangement.";
  const format = normalized.format === "instrumental"
    ? "Instrumental main stream; leave no vocal part."
    : "Main arrangement leaves harmonic and rhythmic space for a synchronized voice-only companion deck.";

  return {
    weightedPrompts: [
      { ...base.weightedPrompts[0], text: `${cue}${base.weightedPrompts[0].text}`.slice(0, 240) },
      { text: `${format} Personal direction: ${personal}.${experiment}`.slice(0, 240), weight: 1.08 },
      base.weightedPrompts[2] ?? base.weightedPrompts[1],
      base.weightedPrompts.find((prompt) => prompt.weight < 0) ?? { text: "tempo drift, key clash, muddy mix, random genre changes", weight: -1 },
    ].map((prompt) => ({ ...prompt, text: prompt.text.slice(0, 240) })),
    config: { ...base.config, bpm: normalized.bpm },
  };
}

export function createOnboardingVocalGuidance(preferences: OnboardingPreferences): string {
  const normalized = normalizeOnboardingPreferences(preferences);
  if (normalized.vocalStyle === "none") return "Voice deck disabled; preserve instrumental space and avoid vocal-like lead phrases";
  const voice = normalized.vocalStyle === "other"
    ? "androgynous or nontraditional lead voice"
    : `${normalized.vocalStyle} lead voice`;
  const role = normalized.vocalRole === "chorus"
    ? "develop a repeatable chorus hook over bars 9-16 and 21-28"
    : normalized.vocalRole === "experimental"
      ? "use controlled extended vocal textures and evolving wordless phrases at section boundaries"
      : "sing sparse call-and-response phrases with long rests between them";
  return `${voice}; ${role}; voice only, same key and BPM as main, no instruments or accompaniment`.slice(0, 240);
}
