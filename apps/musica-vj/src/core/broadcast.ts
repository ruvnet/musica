import { invoke, isTauri } from "@tauri-apps/api/core";
import {
  DEFAULT_MASTER_EFFECT_PARAMS,
  MASTER_EFFECT_IDS,
  type MasterEffectParams,
  type MasterEffectsState,
} from "../audio/AudioEngine";
import { DEFAULT_LYRIA_DECK_CONTROLS, type LyriaDeckControl } from "./lyriaDeckScenes";
import type {
  LyriaRealtimeConfig,
  LyriaRealtimeDeckId,
  LyriaRealtimeMode,
  LyriaRealtimeScale,
  LyriaWeightedPrompt,
} from "./lyriaRealtime";
import { clamp, MAX_BPM, MIN_BPM } from "./music";
import { VISUAL_SCENES } from "./presets";
import type { VisualColorControls, VisualPaletteId, VisualSceneId } from "./types";

/// Settings broadcast + live follow (ADR-182).
///
/// A snapshot is a few hundred bytes describing *how* a set is being played —
/// never its audio. Followers re-synthesize locally, so this module is the
/// trust boundary: everything arriving from another user is clamped, slewed,
/// whitelisted, and length-capped before it can reach the audio or visual
/// engine.

export const BROADCAST_SNAPSHOT_VERSION = 1;

/// How often a follower asks for the broadcaster's current state. This is a
/// product latency target, not a storage write cadence — the service skips the
/// presence write when it is still fresh and omits the body when `rev` matches.
export const BROADCAST_POLL_INTERVAL_MS = 2_000;
/// A broadcaster republishes only when the snapshot content actually changes,
/// and never faster than this. Local state re-emits on every fader move; that
/// frequency must not become the network's.
export const BROADCAST_PUBLISH_MIN_INTERVAL_MS = 2_000;
/// Past this age a broadcaster is shown as stale rather than live, so a
/// follower is never left staring at a zombie session that looks current.
export const BROADCAST_STALE_AFTER_SECONDS = 30;

const MAX_DISPLAY_NAME_LENGTH = 32;
const MAX_STYLE_LABEL_LENGTH = 48;
const MAX_ID_LENGTH = 64;
/// Prompts are free text that reaches the *follower's* Lyria credential, so
/// they are capped in both count and length on the way in.
const MAX_PROMPTS = 8;
const MAX_PROMPT_LENGTH = 200;

/// Photosensitivity is a trajectory problem, not a range problem: an intensity
/// that flips between extremes passes every clamp yet still pulses.
///
/// Two bounds apply, and it is worth being precise about which does the work.
/// The poll interval is the primary one — a follower's visuals cannot change
/// faster than once per `BROADCAST_POLL_INTERVAL_MS`, i.e. 0.5 Hz, far below
/// the range photosensitivity guidance concerns itself with. This budget bounds
/// the *magnitude* of each of those steps, so a full-range swing is spread over
/// more than one poll instead of snapping. It is deliberately small enough to
/// actually engage at the default poll rate.
export const MAX_REMOTE_INTENSITY_DELTA_PER_SECOND = 0.35;

const DECK_IDS = ["main", "sequence", "vocal"] as const satisfies readonly LyriaRealtimeDeckId[];
const PALETTE_IDS = ["scene", "neon", "ember", "ice", "prism", "mono"] as const satisfies readonly VisualPaletteId[];
const LYRIA_MODES = ["QUALITY", "DIVERSITY", "VOCALIZATION"] as const satisfies readonly LyriaRealtimeMode[];
const LYRIA_SCALES = [
  "C_MAJOR_A_MINOR",
  "D_FLAT_MAJOR_B_FLAT_MINOR",
  "D_MAJOR_B_MINOR",
  "E_FLAT_MAJOR_C_MINOR",
  "E_MAJOR_D_FLAT_MINOR",
  "F_MAJOR_D_MINOR",
  "G_FLAT_MAJOR_E_FLAT_MINOR",
  "G_MAJOR_E_MINOR",
  "A_FLAT_MAJOR_F_MINOR",
  "A_MAJOR_G_FLAT_MINOR",
  "B_FLAT_MAJOR_G_MINOR",
  "B_MAJOR_A_FLAT_MINOR",
  "SCALE_UNSPECIFIED",
] as const satisfies readonly LyriaRealtimeScale[];

// Compile-time guards: adding a deck, palette, mode, or scale to its union
// without adding it here would silently make that value un-followable, so the
// omission is made a type error instead.
type AssertCovered<Union, Listed> = Exclude<Union, Listed> extends never ? true : never;
const _decksCovered: AssertCovered<LyriaRealtimeDeckId, (typeof DECK_IDS)[number]> = true;
const _palettesCovered: AssertCovered<VisualPaletteId, (typeof PALETTE_IDS)[number]> = true;
const _modesCovered: AssertCovered<LyriaRealtimeMode, (typeof LYRIA_MODES)[number]> = true;
const _scalesCovered: AssertCovered<LyriaRealtimeScale, (typeof LYRIA_SCALES)[number]> = true;
void _decksCovered;
void _palettesCovered;
void _modesCovered;
void _scalesCovered;

/// The control surface: which decks are playing, how loud, at what pitch.
///
/// Saved deck-scene *presets* are deliberately not here. They are the
/// broadcaster's own library, and adopting them would overwrite the follower's
/// saved presets — which a follow must never do (ADR-182). Deck enablement and
/// controls already carry whatever scene the broadcaster has active.
export interface BroadcastPerformance {
  bpm: number;
  styleId: string;
  styleLabel: string;
  deckEnabled: Record<LyriaRealtimeDeckId, boolean>;
  deckControls: Record<LyriaRealtimeDeckId, LyriaDeckControl>;
}

/// The generative direction. Without this a follow is cosmetic — `styleId`
/// alone leaves density, brightness, guidance, temperature, and the prompts
/// themselves at the follower's own values (ADR-182).
///
/// Prompts travel *resolved* rather than as a style reference: a broadcaster's
/// custom style id means nothing on a follower's machine, but the prompts it
/// resolves to carry their own meaning.
export interface BroadcastLyria {
  config: LyriaRealtimeConfig;
  prompts: LyriaWeightedPrompt[];
}

export interface BroadcastVisual {
  scene: VisualSceneId;
  intensity: number;
  color: VisualColorControls;
}

export interface BroadcastFx {
  effects: MasterEffectsState;
  params: MasterEffectParams;
}

/// Deliberately absent: master volume (a follower's monitors are theirs), a
/// play command (following never starts audio — see `ADR-182` on consent), and
/// whole-workspace content such as visual plugins, set arcs, and the custom
/// style library.
export interface BroadcastSnapshot {
  v: typeof BROADCAST_SNAPSHOT_VERSION;
  /// Monotonic, broadcaster-assigned. Followers drop anything not newer than
  /// what they already applied, so retries and reordering cannot resurrect
  /// stale state.
  rev: number;
  performance: BroadcastPerformance;
  lyria: BroadcastLyria;
  visual: BroadcastVisual;
  fx: BroadcastFx;
}

export interface BroadcastListing {
  id: string;
  displayName: string;
  live: boolean;
  listeners: number;
  updatedAgoSeconds: number;
  styleLabel: string;
  bpm: number;
}

export interface BroadcastDirectory {
  /// Top 5 by concurrent listeners.
  top: BroadcastListing[];
  all: BroadcastListing[];
}

export interface BroadcastFollowUpdate {
  rev: number;
  /// Omitted when the follower's `since` rev already matches — an unchanged
  /// set costs a heartbeat, not a payload.
  snapshot?: BroadcastSnapshot;
  listeners: number;
  live: boolean;
  updatedAgoSeconds: number;
}

export interface BroadcastPublishState {
  live: boolean;
  id?: string;
  displayName?: string;
  listeners: number;
  reason?: string;
}

// ---------------------------------------------------------------------------
// Trust boundary
// ---------------------------------------------------------------------------

// Control characters plus the bidi overrides that let a name render as
// something other than what it stores.
const UNSAFE_TEXT = /[\u0000-\u001f\u007f-\u009f\u200e-\u200f\u202a-\u202e\u2066-\u2069]/g;

export function sanitizeBroadcastText(value: unknown, maxLength: number): string {
  if (typeof value !== "string") return "";
  return value.replace(UNSAFE_TEXT, "").trim().slice(0, maxLength);
}

function unit(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, 0, 1) : fallback;
}

function bounded(value: unknown, minimum: number, maximum: number, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, minimum, maximum) : fallback;
}

function enumValue<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
  return typeof value === "string" && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null ? (value as Record<string, unknown>) : {};
}

function normalizeDeckControl(value: unknown, fallback: LyriaDeckControl): LyriaDeckControl {
  const raw = record(value);
  return {
    volume: unit(raw.volume, fallback.volume),
    muted: raw.muted === true,
    pitchSemitones: Math.round(bounded(raw.pitchSemitones, -7, 7, fallback.pitchSemitones)),
    beatNudgeMs: Math.round(bounded(raw.beatNudgeMs, -250, 250, fallback.beatNudgeMs) / 5) * 5,
  };
}

function normalizeDeckMap<T>(value: unknown, read: (entry: unknown, deck: LyriaRealtimeDeckId) => T): Record<LyriaRealtimeDeckId, T> {
  const raw = record(value);
  return Object.fromEntries(DECK_IDS.map((deck) => [deck, read(raw[deck], deck)])) as Record<LyriaRealtimeDeckId, T>;
}

function normalizePrompts(value: unknown): LyriaWeightedPrompt[] {
  if (!Array.isArray(value)) return [];
  return value
    .slice(0, MAX_PROMPTS)
    .map((entry) => {
      const raw = record(entry);
      return {
        text: sanitizeBroadcastText(raw.text, MAX_PROMPT_LENGTH),
        weight: bounded(raw.weight, 0, 2, 1),
      };
    })
    .filter((prompt) => prompt.text.length > 0);
}

function normalizeLyriaConfig(value: unknown): LyriaRealtimeConfig {
  const raw = record(value);
  const muteBass = raw.muteBass === true;
  const muteDrums = raw.muteDrums === true;
  // The engine treats these as mutually exclusive; a hostile or buggy snapshot
  // must not be able to assert both at once.
  const onlyBassAndDrums = raw.onlyBassAndDrums === true && !muteBass && !muteDrums;
  return {
    bpm: Math.round(bounded(raw.bpm, MIN_BPM, MAX_BPM, 120)),
    guidance: bounded(raw.guidance, 0, 6, 4),
    density: unit(raw.density, 0.5),
    brightness: unit(raw.brightness, 0.5),
    temperature: bounded(raw.temperature, 0.1, 2, 1.1),
    topK: Math.round(bounded(raw.topK, 1, 100, 40)),
    seed: typeof raw.seed === "number" && Number.isFinite(raw.seed)
      ? Math.round(clamp(raw.seed, 0, 2_147_483_647))
      : undefined,
    scale: enumValue(raw.scale, LYRIA_SCALES, "SCALE_UNSPECIFIED"),
    muteBass,
    muteDrums,
    onlyBassAndDrums,
    musicGenerationMode: enumValue(raw.musicGenerationMode, LYRIA_MODES, "QUALITY"),
  };
}

const VISUAL_SCENE_IDS: readonly VisualSceneId[] = VISUAL_SCENES.map((scene) => scene.id);

function normalizeVisual(value: unknown): BroadcastVisual {
  const raw = record(value);
  const color = record(raw.color);
  return {
    scene: enumValue(raw.scene, VISUAL_SCENE_IDS, VISUAL_SCENE_IDS[0]),
    intensity: unit(raw.intensity, 0.6),
    color: {
      palette: enumValue(color.palette, PALETTE_IDS, "scene"),
      hue: unit(color.hue, 0.6),
      saturation: unit(color.saturation, 0.6),
      contrast: unit(color.contrast, 0.5),
      diversity: unit(color.diversity, 0.5),
    },
  };
}

function normalizeFx(value: unknown): BroadcastFx {
  const raw = record(value);
  const effects = record(raw.effects);
  const params = record(raw.params);
  return {
    effects: Object.fromEntries(
      MASTER_EFFECT_IDS.map((effect) => [effect, unit(effects[effect], 0)]),
    ) as unknown as MasterEffectsState,
    params: Object.fromEntries(
      (Object.keys(DEFAULT_MASTER_EFFECT_PARAMS) as Array<keyof MasterEffectParams>)
        .map((param) => [param, unit(params[param], DEFAULT_MASTER_EFFECT_PARAMS[param])]),
    ) as unknown as MasterEffectParams,
  };
}

/// Total and non-throwing: every hostile shape resolves to either a safe
/// snapshot or `undefined`. Unknown fields are ignored rather than rejected so
/// a newer broadcaster does not break an older follower.
export function normalizeBroadcastSnapshot(value: unknown): BroadcastSnapshot | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const raw = value as Record<string, unknown>;
  if (raw.v !== BROADCAST_SNAPSHOT_VERSION) return undefined;
  if (typeof raw.rev !== "number" || !Number.isFinite(raw.rev) || raw.rev < 0) return undefined;

  const performance = record(raw.performance);
  return {
    v: BROADCAST_SNAPSHOT_VERSION,
    rev: Math.floor(raw.rev),
    performance: {
      bpm: Math.round(bounded(performance.bpm, MIN_BPM, MAX_BPM, 120)),
      styleId: sanitizeBroadcastText(performance.styleId, MAX_ID_LENGTH),
      styleLabel: sanitizeBroadcastText(performance.styleLabel, MAX_STYLE_LABEL_LENGTH),
      deckEnabled: normalizeDeckMap(performance.deckEnabled, (entry) => entry === true),
      deckControls: normalizeDeckMap(performance.deckControls, (entry, deck) =>
        normalizeDeckControl(entry, DEFAULT_LYRIA_DECK_CONTROLS[deck])),
    },
    lyria: {
      config: normalizeLyriaConfig(record(raw.lyria).config),
      prompts: normalizePrompts(record(raw.lyria).prompts),
    },
    visual: normalizeVisual(raw.visual),
    fx: normalizeFx(raw.fx),
  };
}

export function normalizeBroadcastListing(value: unknown): BroadcastListing | undefined {
  const raw = record(value);
  const id = sanitizeBroadcastText(raw.id, MAX_ID_LENGTH);
  if (id.length === 0) return undefined;
  const displayName = sanitizeBroadcastText(raw.displayName, MAX_DISPLAY_NAME_LENGTH);
  return {
    id,
    displayName: displayName.length > 0 ? displayName : "Unnamed",
    live: raw.live === true,
    listeners: Math.max(0, Math.round(bounded(raw.listeners, 0, 1_000_000, 0))),
    updatedAgoSeconds: Math.max(0, Math.round(bounded(raw.updatedAgoSeconds, 0, 31_536_000, 0))),
    styleLabel: sanitizeBroadcastText(raw.styleLabel, MAX_STYLE_LABEL_LENGTH),
    bpm: Math.round(bounded(raw.bpm, MIN_BPM, MAX_BPM, 120)),
  };
}

export function normalizeBroadcastDirectory(value: unknown): BroadcastDirectory {
  const raw = record(value);
  const all = (Array.isArray(raw.all) ? raw.all : [])
    .map(normalizeBroadcastListing)
    .filter((listing): listing is BroadcastListing => listing !== undefined);
  const top = (Array.isArray(raw.top) ? raw.top : [])
    .map(normalizeBroadcastListing)
    .filter((listing): listing is BroadcastListing => listing !== undefined)
    .slice(0, 5);
  return { top, all };
}

export function normalizeBroadcastFollowUpdate(value: unknown): BroadcastFollowUpdate | undefined {
  const raw = record(value);
  if (typeof raw.rev !== "number" || !Number.isFinite(raw.rev)) return undefined;
  return {
    rev: Math.max(0, Math.floor(raw.rev)),
    snapshot: raw.snapshot === undefined || raw.snapshot === null
      ? undefined
      : normalizeBroadcastSnapshot(raw.snapshot),
    listeners: Math.max(0, Math.round(bounded(raw.listeners, 0, 1_000_000, 0))),
    live: raw.live === true,
    updatedAgoSeconds: Math.max(0, Math.round(bounded(raw.updatedAgoSeconds, 0, 31_536_000, 0))),
  };
}

/// True when an incoming snapshot should replace what is already applied.
/// Equal revisions are rejected too, so a repeated poll is a no-op.
export function isNewerBroadcastSnapshot(incoming: BroadcastSnapshot, appliedRev: number | undefined): boolean {
  return appliedRev === undefined || incoming.rev > appliedRev;
}

/// Rate-limits a remote visual intensity change. Range clamping alone permits
/// a 0 <-> 1 flip on every poll, which is a strobe; this bounds how fast the
/// follower's renderer is allowed to travel toward the broadcaster's value.
export function slewRemoteIntensity(current: number, target: number, elapsedMs: number): number {
  const from = clamp(current, 0, 1);
  const to = clamp(target, 0, 1);
  const budget = MAX_REMOTE_INTENSITY_DELTA_PER_SECOND * (Math.max(0, elapsedMs) / 1000);
  if (budget <= 0) return from;
  const delta = to - from;
  if (Math.abs(delta) <= budget) return to;
  return from + Math.sign(delta) * budget;
}

/// Stable content fingerprint used to skip republishing an unchanged set.
/// `rev` is excluded — it changes on every publish by definition.
export function broadcastSnapshotFingerprint(snapshot: BroadcastSnapshot): string {
  const { rev: _rev, ...content } = snapshot;
  return JSON.stringify(content);
}

export interface BroadcastCaptureInput {
  rev: number;
  bpm: number;
  styleId: string;
  styleLabel: string;
  deckEnabled: Record<LyriaRealtimeDeckId, boolean>;
  deckControls: Record<LyriaRealtimeDeckId, LyriaDeckControl>;
  lyriaConfig: LyriaRealtimeConfig;
  lyriaPrompts: LyriaWeightedPrompt[];
  visualScene: VisualSceneId;
  visualIntensity: number;
  visualColor: VisualColorControls;
  masterEffects: MasterEffectsState;
  masterEffectParams: MasterEffectParams;
}

/// Builds the outgoing snapshot from live state. Routed through the same
/// normalizer as inbound data so what is published is exactly what a follower
/// would accept — a field that would be clamped away on receipt is never sent.
export function captureBroadcastSnapshot(input: BroadcastCaptureInput): BroadcastSnapshot {
  const draft = {
    v: BROADCAST_SNAPSHOT_VERSION,
    rev: input.rev,
    performance: {
      bpm: input.bpm,
      styleId: input.styleId,
      styleLabel: input.styleLabel,
      deckEnabled: input.deckEnabled,
      deckControls: input.deckControls,
    },
    lyria: { config: input.lyriaConfig, prompts: input.lyriaPrompts },
    visual: { scene: input.visualScene, intensity: input.visualIntensity, color: input.visualColor },
    fx: { effects: input.masterEffects, params: input.masterEffectParams },
  };
  // The normalizer is total for a well-formed draft; the fallback keeps this
  // function non-throwing rather than asserting.
  return normalizeBroadcastSnapshot(draft) ?? {
    ...draft,
    rev: Math.max(0, Math.floor(input.rev)),
  } as BroadcastSnapshot;
}

// ---------------------------------------------------------------------------
// Transport — every call carries the Cognitum bearer, which stays in Rust
// ---------------------------------------------------------------------------

const OFFLINE_REASON = "Broadcasting requires the desktop app";

export function broadcastAvailable(): boolean {
  return isTauri();
}

export async function publishBroadcast(displayName: string, snapshot: BroadcastSnapshot): Promise<BroadcastPublishState> {
  if (!isTauri()) return { live: false, listeners: 0, reason: OFFLINE_REASON };
  const raw = await invoke<unknown>("broadcast_publish", {
    displayName: sanitizeBroadcastText(displayName, MAX_DISPLAY_NAME_LENGTH),
    snapshot,
  });
  const value = record(raw);
  return {
    live: value.live === true,
    id: sanitizeBroadcastText(value.id, MAX_ID_LENGTH) || undefined,
    displayName: sanitizeBroadcastText(value.displayName, MAX_DISPLAY_NAME_LENGTH) || undefined,
    listeners: Math.max(0, Math.round(bounded(value.listeners, 0, 1_000_000, 0))),
  };
}

export async function stopBroadcast(): Promise<void> {
  if (!isTauri()) return;
  await invoke("broadcast_stop");
}

export async function listBroadcasts(): Promise<BroadcastDirectory> {
  if (!isTauri()) return { top: [], all: [] };
  return normalizeBroadcastDirectory(await invoke<unknown>("broadcast_list"));
}

/// One authenticated round-trip that both renews the follower's presence and
/// returns the broadcaster's current state. `since` lets the service omit an
/// unchanged snapshot body.
export async function followBroadcast(id: string, since?: number): Promise<BroadcastFollowUpdate | undefined> {
  if (!isTauri()) return undefined;
  return normalizeBroadcastFollowUpdate(await invoke<unknown>("broadcast_listen", { id, since: since ?? null }));
}

export async function leaveBroadcast(id: string): Promise<void> {
  if (!isTauri()) return;
  await invoke("broadcast_leave", { id });
}
