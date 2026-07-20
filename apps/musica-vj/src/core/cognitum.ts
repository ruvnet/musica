import { invoke, isTauri } from "@tauri-apps/api/core";
import type { LyriaRealtimeConfig, LyriaWeightedPrompt } from "./lyriaRealtime";
import { normalizeVisualPluginSpec, type VisualPluginSpec } from "./visualPlugins";

export type CognitumCapability = "advanced-prompting" | "autopilot" | "learning" | "realtime-vocals";

export interface CognitumStatus {
  signedIn: boolean;
  pending: boolean;
  account?: string;
  capabilities: CognitumCapability[];
  authHost: string;
  reason?: string;
}

export interface CognitumStylePack {
  label: string;
  description: string;
  prompts: LyriaWeightedPrompt[];
  config: Partial<LyriaRealtimeConfig>;
}

export const COGNITUM_CAPABILITY_LABELS: Record<CognitumCapability, { label: string; detail: string }> = {
  "advanced-prompting": { label: "ADV PROMPTING", detail: "AI-generated style packs and richer plan briefs" },
  autopilot: { label: "AUTOPILOT+", detail: "Meta-LLM Auto DJ phrase briefs with set memory" },
  learning: { label: "LEARNING", detail: "Adapts direction to your saved sets over time" },
  "realtime-vocals": { label: "RT VOCALS", detail: "Guided vocal deck direction and hooks" },
};

const OFFLINE_STATUS: CognitumStatus = {
  signedIn: false,
  pending: false,
  capabilities: [],
  authHost: "cognitum.one",
  reason: "Cognitum sign-in requires the desktop app",
};

export async function getCognitumStatus(): Promise<CognitumStatus> {
  if (!isTauri()) return OFFLINE_STATUS;
  return invoke<CognitumStatus>("cognitum_status");
}

export async function startCognitumSignIn(): Promise<{ authUrl: string }> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<{ authUrl: string }>("cognitum_auth_start");
}

export async function signOutCognitum(): Promise<void> {
  if (!isTauri()) return;
  await invoke("cognitum_sign_out");
}

export async function startCognitumManualSignIn(): Promise<{ authUrl: string }> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<{ authUrl: string }>("cognitum_auth_manual_start");
}

export async function completeCognitumManualSignIn(code: string): Promise<void> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  await invoke("cognitum_auth_manual_complete", { code });
}

/// Exchanges the current Cognitum sign-in for a short-lived Lyria key at the
/// configured broker and injects it into the RealTime provider, so a signed-in
/// user gets live audio with no bring-your-own key (ADR-179). Silently a no-op
/// when no broker is configured or the user is not signed in; returns whether a
/// key was injected so callers can refresh provider status.
export async function activateCognitumLyria(): Promise<boolean> {
  if (!isTauri()) return false;
  try {
    const key = await invoke<string>("cognitum_lyria_credential");
    if (!key) return false;
    await invoke("lyria_realtime_configure_key", { key });
    return true;
  } catch {
    return false;
  }
}

export async function generateCognitumStylePack(description: string): Promise<CognitumStylePack> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<CognitumStylePack>("cognitum_style_pack", { description });
}

export interface SetArcFx {
  sweep?: number;
  reverb?: number;
  echo?: number;
  flanger?: number;
}

export interface SetArcStep {
  atMinute: number;
  styleId: string;
  visualScene: string;
  bpm: number;
  fx?: SetArcFx;
  note: string;
}

export interface SetArc {
  title: string;
  durationMinutes: number;
  steps: SetArcStep[];
}

export interface FxMove {
  effect: "flanger" | "phaser" | "drive" | "crush" | "sweep" | "reverb" | "echo";
  target: number;
  atBar: number;
}

export interface FxDirection {
  summary: string;
  moves: FxMove[];
}

export async function generateCognitumFxDirection(mood: string, bars: number): Promise<FxDirection> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<FxDirection>("cognitum_fx_direction", { mood, bars });
}

/// Keyword fallback when Cognitum is unavailable: a handful of curated
/// mood shapes over the bar budget, always resolving toward dry.
export function localFxDirection(mood: string, bars: number): FxDirection {
  const normalized = mood.toLowerCase();
  const half = Math.floor(bars / 2);
  const tail = Math.max(1, bars - 2);
  if (/underwater|submerge|deep|dive/.test(normalized)) {
    return {
      summary: "Dive under, then surface to dry",
      moves: [
        { effect: "sweep", target: 0.65, atBar: 0 },
        { effect: "reverb", target: 0.3, atBar: 1 },
        { effect: "sweep", target: 0.2, atBar: half },
        { effect: "sweep", target: 0, atBar: tail },
        { effect: "reverb", target: 0, atBar: tail },
      ],
    };
  }
  if (/space|air|wide|dream|float/.test(normalized)) {
    return {
      summary: "Open the space, drift, land dry",
      moves: [
        { effect: "reverb", target: 0.4, atBar: 0 },
        { effect: "echo", target: 0.25, atBar: 1 },
        { effect: "reverb", target: 0.15, atBar: half },
        { effect: "echo", target: 0, atBar: tail },
        { effect: "reverb", target: 0, atBar: tail },
      ],
    };
  }
  if (/aggress|hard|grit|dirty|heavy|rage/.test(normalized)) {
    return {
      summary: "Add grit, peak, clean out",
      moves: [
        { effect: "drive", target: 0.35, atBar: 0 },
        { effect: "crush", target: 0.2, atBar: half },
        { effect: "drive", target: 0.5, atBar: half },
        { effect: "crush", target: 0, atBar: tail },
        { effect: "drive", target: 0, atBar: tail },
      ],
    };
  }
  return {
    summary: "Gentle motion swell and release",
    moves: [
      { effect: "flanger", target: 0.25, atBar: 0 },
      { effect: "sweep", target: 0.3, atBar: half },
      { effect: "flanger", target: 0, atBar: tail },
      { effect: "sweep", target: 0, atBar: tail },
    ],
  };
}

export interface VisualDirection {
  scene: string;
  palette: string;
  hue: number;
  intensity: number;
  speed: number;
  trail: number;
  morph: number;
  camera: number;
  note: string;
}

export async function generateCognitumVisualDirection(
  mood: string,
  sceneIds: string[],
  paletteIds: string[],
): Promise<VisualDirection> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<VisualDirection>("cognitum_visual_direction", { mood, sceneIds, paletteIds });
}

/// Keyword fallback for AI visual direction when Cognitum is unavailable.
export function localVisualDirection(mood: string): VisualDirection {
  const normalized = mood.toLowerCase();
  if (/dark|night|shadow|tension|noir/.test(normalized)) {
    return { scene: "monolith", palette: "mono", hue: 0.78, intensity: 0.55, speed: 0.3, trail: 0.6, morph: 0.5, camera: 0.3, note: "Stark monolith, long shadows, slow drift" };
  }
  if (/retro|scope|analog|vintage|arcade/.test(normalized)) {
    return { scene: "oscilloscope", palette: "scene", hue: 0.4, intensity: 0.7, speed: 0.5, trail: 0.55, morph: 0.4, camera: 0.4, note: "Retro scope rings with phosphor trails" };
  }
  if (/euphoric|peak|rave|festival|explosive/.test(normalized)) {
    return { scene: "lasergrid", palette: "neon", hue: 0.9, intensity: 0.9, speed: 0.85, trail: 0.7, morph: 0.75, camera: 0.85, note: "Full laser peak, fast and bright" };
  }
  if (/calm|ambient|drift|water|dream|float/.test(normalized)) {
    return { scene: "aurora", palette: "ice", hue: 0.55, intensity: 0.45, speed: 0.2, trail: 0.85, morph: 0.45, camera: 0.2, note: "Slow aurora veil, icy and weightless" };
  }
  if (/warm|golden|sunset|soul|dusty/.test(normalized)) {
    return { scene: "terrain", palette: "ember", hue: 0.08, intensity: 0.65, speed: 0.4, trail: 0.5, morph: 0.55, camera: 0.4, note: "Warm spectral field in ember light" };
  }
  return { scene: "bloom", palette: "prism", hue: 0.6, intensity: 0.7, speed: 0.5, trail: 0.5, morph: 0.6, camera: 0.5, note: "Balanced bloom with prismatic color" };
}

export interface AutoDjBrief {
  brief: string;
  mood: string;
}

export async function generateCognitumAutoDjBrief(
  styleLabel: string,
  bpm: number,
  phrase: number,
  personalization: string,
  previousBrief: string,
): Promise<AutoDjBrief> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<AutoDjBrief>("cognitum_autodj_brief", { styleLabel, bpm, phrase, personalization, previousBrief });
}

export async function generateCognitumSetArc(
  durationMinutes: number,
  direction: string,
  styleIds: string[],
  sceneIds: string[],
): Promise<SetArc> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<SetArc>("cognitum_set_arc", { durationMinutes, direction, styleIds, sceneIds });
}

/// Deterministic fallback used when Cognitum is unavailable: a classic
/// establish → build → peak → breathe → second peak → resolve energy curve.
export function localSetArc(durationMinutes: number, styleIds: string[], sceneIds: string[]): SetArc {
  const duration = Math.max(30, Math.min(90, Math.round(durationMinutes)));
  const phases: Array<{ share: number; energy: number; note: string; fx?: SetArcFx }> = [
    { share: 0, energy: 0.35, note: "Establish the identity, low pressure" },
    { share: 0.14, energy: 0.5, note: "First build, introduce percussion drive" },
    { share: 0.3, energy: 0.75, note: "Open the floor, main groove" },
    { share: 0.45, energy: 0.95, note: "First peak, full energy", fx: { sweep: 0.2 } },
    { share: 0.58, energy: 0.45, note: "Breathe, strip back and add space", fx: { reverb: 0.3 } },
    { share: 0.7, energy: 0.8, note: "Rebuild toward the second peak" },
    { share: 0.82, energy: 1, note: "Second peak, strongest material", fx: { sweep: 0.15, echo: 0.15 } },
    { share: 0.93, energy: 0.3, note: "Resolve and land gently", fx: { reverb: 0.35 } },
  ];
  const orderedStyles = styleIds.length > 0 ? styleIds : ["rock"];
  const orderedScenes = sceneIds.length > 0 ? sceneIds : ["oscilloscope"];
  return {
    title: `${duration}-minute local arc`,
    durationMinutes: duration,
    steps: phases.map((phase, index) => ({
      atMinute: Math.round(phase.share * duration * 10) / 10,
      styleId: orderedStyles[index % orderedStyles.length]!,
      visualScene: orderedScenes[index % orderedScenes.length]!,
      bpm: Math.round(96 + phase.energy * 60),
      fx: phase.fx,
      note: phase.note,
    })),
  };
}

export async function generateCognitumVisualPlugin(description: string): Promise<VisualPluginSpec> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  const raw = await invoke<unknown>("cognitum_visual_plugin", { description });
  const spec = normalizeVisualPluginSpec(raw);
  if (!spec) throw new Error("Cognitum returned an invalid plugin");
  return spec;
}

export interface VocalGuidance {
  guidance: string;
  hook: string;
}

export async function generateCognitumVocalGuidance(styleLabel: string, hint: string): Promise<VocalGuidance> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<VocalGuidance>("cognitum_vocal_guidance", { styleLabel, hint });
}

/// Local fallback vocal guidance templates keyed by broad style family.
export function localVocalGuidance(styleLabel: string): VocalGuidance {
  const normalized = styleLabel.toLowerCase();
  if (/rock|blues|country/.test(normalized)) {
    return {
      guidance: "Raspy chest-voice lead; short powerful phrases answering the guitar hook; open ah and oh vowels; rest through verses, commit hard in the chorus",
      hook: "Leap up a fourth, hold with grit, fall back home in two steps",
    };
  }
  if (/house|techno|edm|dubstep|drum/.test(normalized)) {
    return {
      guidance: "Airy processed head voice; clipped rhythmic syllables locked to the offbeats; long sustained oo pad through breakdowns; silence during drops",
      hook: "Two-note call, octave echo answer, let it decay into the beat",
    };
  }
  if (/lofi|ambient|cinema|classical|jazz/.test(normalized)) {
    return {
      guidance: "Soft intimate hums and mm vowels close to the mic; behind-the-beat phrasing; leave two-bar rests; one gentle rising motif per sixteen bars",
      hook: "Slow stepwise rise of three notes, suspended, resolving down a third",
    };
  }
  return {
    guidance: "Expressive wordless lead with a memorable chorus contour; answer the main motif without competing; alternate breathy verses with full-voice chorus",
    hook: "Rise a fifth, hold, fall stepwise home",
  };
}
