import { DEFAULT_MASTER_EFFECT_PARAMS, MASTER_EFFECT_IDS, type MasterEffectParams, type MasterEffectsState } from "../audio/AudioEngine";
import { loadCustomLyriaStyles, type LyriaRealtimeStylePreset } from "./lyriaRealtime";
import { loadLyriaDeckScenes, type LyriaDeckScene } from "./lyriaDeckScenes";
import { normalizeVisualPluginList, type VisualPluginSpec } from "./visualPlugins";
import type { SetArc } from "./cognitum";
import { DEFAULT_ONBOARDING_PREFERENCES, normalizeOnboardingPreferences, type OnboardingPreferences } from "./onboarding";

const DB_NAME = "musica-settings";
const DB_VERSION = 1;
const STORE_NAME = "workspace";
const WORKSPACE_KEY = "workspace.v1";

export interface WorkspaceSettings {
  version: 1;
  savedAt?: string;
  onboarding: OnboardingPreferences;
  customStyles: LyriaRealtimeStylePreset[];
  deckScenes: LyriaDeckScene[];
  masterEffects: MasterEffectsState;
  masterEffectParams: MasterEffectParams;
  fxLocks: Record<string, boolean>;
  sfxLevel: number;
  plugins?: VisualPluginSpec[];
  setArc?: SetArc;
}

function boundedUnit(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : fallback;
}

export function normalizeWorkspaceSettings(value: unknown): WorkspaceSettings | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const raw = value as Partial<WorkspaceSettings>;
  if (raw.version !== 1) return undefined;
  const effects = (raw.masterEffects ?? {}) as Partial<MasterEffectsState>;
  const params = (raw.masterEffectParams ?? {}) as Partial<MasterEffectParams>;
  const locks = (typeof raw.fxLocks === "object" && raw.fxLocks !== null ? raw.fxLocks : {}) as Record<string, unknown>;
  return {
    version: 1,
    savedAt: typeof raw.savedAt === "string" ? raw.savedAt : undefined,
    onboarding: normalizeOnboardingPreferences({ ...DEFAULT_ONBOARDING_PREFERENCES, ...(typeof raw.onboarding === "object" && raw.onboarding !== null ? raw.onboarding : {}) }),
    customStyles: loadCustomLyriaStyles(JSON.stringify(raw.customStyles ?? [])),
    deckScenes: loadLyriaDeckScenes(JSON.stringify(raw.deckScenes ?? null)),
    masterEffects: Object.fromEntries(
      MASTER_EFFECT_IDS.map((effect) => [effect, boundedUnit(effects[effect], 0)]),
    ) as unknown as MasterEffectsState,
    masterEffectParams: Object.fromEntries(
      (Object.keys(DEFAULT_MASTER_EFFECT_PARAMS) as Array<keyof MasterEffectParams>)
        .map((param) => [param, boundedUnit(params[param], DEFAULT_MASTER_EFFECT_PARAMS[param])]),
    ) as unknown as MasterEffectParams,
    fxLocks: Object.fromEntries(MASTER_EFFECT_IDS.map((effect) => [effect, locks[effect] === true])),
    sfxLevel: boundedUnit(raw.sfxLevel, 0.5),
    plugins: normalizeVisualPluginList(raw.plugins),
    setArc: normalizeStoredSetArc(raw.setArc),
  };
}

function normalizeStoredSetArc(value: unknown): SetArc | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const raw = value as Partial<SetArc>;
  if (typeof raw.title !== "string" || typeof raw.durationMinutes !== "number" || !Array.isArray(raw.steps)) return undefined;
  const steps = raw.steps.filter((step) => (
    typeof step === "object" && step !== null
    && typeof (step as { atMinute?: unknown }).atMinute === "number"
    && typeof (step as { styleId?: unknown }).styleId === "string"
    && typeof (step as { visualScene?: unknown }).visualScene === "string"
    && typeof (step as { bpm?: unknown }).bpm === "number"
  ));
  if (steps.length === 0) return undefined;
  return { title: raw.title.slice(0, 40), durationMinutes: Math.max(30, Math.min(90, Math.round(raw.durationMinutes))), steps: steps as SetArc["steps"] };
}

export function serializeWorkspaceSettings(settings: WorkspaceSettings): string {
  return JSON.stringify({ ...settings, savedAt: new Date().toISOString() }, null, 2);
}

function openSettingsDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) request.result.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("IndexedDB is unavailable"));
  });
}

export async function saveWorkspaceSettings(settings: WorkspaceSettings): Promise<void> {
  const db = await openSettingsDb();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).put({ ...settings, savedAt: new Date().toISOString() }, WORKSPACE_KEY);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("Could not persist settings"));
    });
  } finally {
    db.close();
  }
}

export async function loadWorkspaceSettings(): Promise<WorkspaceSettings | undefined> {
  const db = await openSettingsDb();
  try {
    const stored = await new Promise<unknown>((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, "readonly");
      const request = transaction.objectStore(STORE_NAME).get(WORKSPACE_KEY);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("Could not read settings"));
    });
    return normalizeWorkspaceSettings(stored);
  } finally {
    db.close();
  }
}
