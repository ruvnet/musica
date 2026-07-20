/// AI-generated visual plugin specs (ADR-177 tier 1): pure data interpreted
/// by VisualEngine's generic plugin renderer. Never code — validation here
/// mirrors the Rust-side bounds so imported/persisted specs are re-checked.

export type PluginBase = "particles" | "ribbons" | "rings";
export type PluginBassTarget = "scale" | "speed" | "none";
export type PluginHighTarget = "brightness" | "jitter" | "none";

export interface VisualPluginSpec {
  id: string;
  name: string;
  base: PluginBase;
  count: number;
  size: number;
  spread: number;
  motion: { orbit: number; pulse: number; drift: number; twist: number };
  audio: { bassTo: PluginBassTarget; highTo: PluginHighTarget };
  colors: { primary: string; accent: string; background: string };
  fog: number;
  exposure: number;
}

const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;
const BASES: PluginBase[] = ["particles", "ribbons", "rings"];
const BASS_TARGETS: PluginBassTarget[] = ["scale", "speed", "none"];
const HIGH_TARGETS: PluginHighTarget[] = ["brightness", "jitter", "none"];

function unit(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : fallback;
}

function pick<T extends string>(value: unknown, options: readonly T[], fallback: T): T {
  return typeof value === "string" && options.includes(value as T) ? value as T : fallback;
}

function hex(value: unknown, fallback: string): string {
  return typeof value === "string" && HEX_COLOR.test(value) ? value : fallback;
}

export function normalizeVisualPluginSpec(value: unknown): VisualPluginSpec | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const raw = value as Partial<VisualPluginSpec> & { motion?: Partial<VisualPluginSpec["motion"]>; audio?: Partial<VisualPluginSpec["audio"]>; colors?: Partial<VisualPluginSpec["colors"]> };
  const name = typeof raw.name === "string" ? raw.name.trim().slice(0, 24) : "";
  if (!name) return undefined;
  const id = typeof raw.id === "string" && raw.id.startsWith("plugin-") ? raw.id : `plugin-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  const count = typeof raw.count === "number" && Number.isFinite(raw.count) ? Math.round(Math.max(50, Math.min(4_000, raw.count))) : 1_200;
  return {
    id,
    name,
    base: pick(raw.base, BASES, "particles"),
    count,
    size: unit(raw.size, 0.4),
    spread: unit(raw.spread, 0.6),
    motion: {
      orbit: unit(raw.motion?.orbit, 0.4),
      pulse: unit(raw.motion?.pulse, 0.5),
      drift: unit(raw.motion?.drift, 0.3),
      twist: unit(raw.motion?.twist, 0.3),
    },
    audio: {
      bassTo: pick(raw.audio?.bassTo, BASS_TARGETS, "scale"),
      highTo: pick(raw.audio?.highTo, HIGH_TARGETS, "brightness"),
    },
    colors: {
      primary: hex(raw.colors?.primary, "#35dcff"),
      accent: hex(raw.colors?.accent, "#b06bf2"),
      background: hex(raw.colors?.background, "#030107"),
    },
    fog: unit(raw.fog, 0.4),
    exposure: unit(raw.exposure, 0.5),
  };
}

export function normalizeVisualPluginList(value: unknown): VisualPluginSpec[] {
  if (!Array.isArray(value)) return [];
  return value
    .map(normalizeVisualPluginSpec)
    .filter((spec): spec is VisualPluginSpec => spec !== undefined)
    .slice(0, 12);
}

/// Keyword fallback generator for offline plugin creation.
export function localVisualPluginSpec(description: string): VisualPluginSpec {
  const normalized = description.toLowerCase();
  const spec = (overrides: Partial<VisualPluginSpec> & { name: string }): VisualPluginSpec =>
    normalizeVisualPluginSpec({ ...overrides, id: undefined })!;
  if (/star|space|galaxy|cosmos|warp/.test(normalized)) {
    return spec({ name: "Starfield", base: "particles", count: 2_400, size: 0.25, spread: 0.95, motion: { orbit: 0.15, pulse: 0.3, drift: 0.85, twist: 0.1 }, audio: { bassTo: "speed", highTo: "brightness" }, colors: { primary: "#dfe8ff", accent: "#7c96ff", background: "#01020a" }, fog: 0.2, exposure: 0.6 });
  }
  if (/ring|orbit|saturn|halo|portal/.test(normalized)) {
    return spec({ name: "Halo Rings", base: "rings", count: 900, size: 0.5, spread: 0.7, motion: { orbit: 0.7, pulse: 0.6, drift: 0.2, twist: 0.55 }, audio: { bassTo: "scale", highTo: "brightness" }, colors: { primary: "#75f4c5", accent: "#35dcff", background: "#020409" }, fog: 0.35, exposure: 0.55 });
  }
  if (/ribbon|wave|silk|flow|aurora/.test(normalized)) {
    return spec({ name: "Silk Ribbons", base: "ribbons", count: 700, size: 0.6, spread: 0.8, motion: { orbit: 0.3, pulse: 0.45, drift: 0.6, twist: 0.7 }, audio: { bassTo: "scale", highTo: "jitter" }, colors: { primary: "#ffb7eb", accent: "#70a9ff", background: "#050108" }, fog: 0.5, exposure: 0.5 });
  }
  if (/fire|ember|lava|burn/.test(normalized)) {
    return spec({ name: "Emberstorm", base: "particles", count: 1_800, size: 0.45, spread: 0.55, motion: { orbit: 0.5, pulse: 0.85, drift: 0.5, twist: 0.4 }, audio: { bassTo: "scale", highTo: "brightness" }, colors: { primary: "#ff9c42", accent: "#ff315f", background: "#070202" }, fog: 0.45, exposure: 0.65 });
  }
  return spec({ name: "Nebula Drift", base: "particles", count: 1_400, size: 0.5, spread: 0.75, motion: { orbit: 0.4, pulse: 0.5, drift: 0.5, twist: 0.35 }, audio: { bassTo: "scale", highTo: "brightness" }, colors: { primary: "#35dcff", accent: "#b06bf2", background: "#030107" }, fog: 0.4, exposure: 0.55 });
}
