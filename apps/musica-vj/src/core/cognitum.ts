import { invoke, isTauri } from "@tauri-apps/api/core";
import type { LyriaRealtimeConfig, LyriaWeightedPrompt } from "./lyriaRealtime";

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

export async function generateCognitumStylePack(description: string): Promise<CognitumStylePack> {
  if (!isTauri()) throw new Error(OFFLINE_STATUS.reason);
  return invoke<CognitumStylePack>("cognitum_style_pack", { description });
}
