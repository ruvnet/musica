import { invoke, isTauri } from "@tauri-apps/api/core";
import type { AudioAnalysisResult, EncodedAudioMetadata } from "../audio/audioAnalysis";
import type { GenerationRequest, GenerationTask, ProviderStatus } from "./types";

export interface GenerationReceipt {
  taskId: string;
  provider: string;
  model?: string;
  promptHash: string;
  createdAt: string;
  outputUrlHash?: string;
  termsVersion?: string;
  modelVersion?: string;
  pricingVersion?: string;
  reservedCostUsd?: number;
  generationCostUsd?: number;
  providerBillingVerified?: boolean;
  synthidExpected?: boolean;
  c2paStatus?: string;
  encodedAudio?: EncodedAudioMetadata;
  analysis?: AnalysisReceiptSummary;
}

export type AnalysisReceiptSummary = Omit<AudioAnalysisResult, "waveform" | "onsetMap" | "beatGridSeconds"> & {
  waveformBucketCount: number;
  onsetCount: number;
  beatCount: number;
};

const RECEIPT_KEY = "musica-vj-generation-receipts-v1";
const MAX_RECEIPTS = 500;
const MAX_RECEIPT_STORAGE_BYTES = 2 * 1024 * 1024;

export async function getProviderStatus(): Promise<ProviderStatus> {
  if (!isTauri()) {
    return {
      available: false,
      provider: "offline",
      reason: "Creative providers are available only in the signed desktop application",
    };
  }
  return invoke<ProviderStatus>("creative_provider_status");
}

export async function generateMusic(request: GenerationRequest): Promise<GenerationTask> {
  if (!isTauri()) throw new Error("Creative providers require the desktop application");
  return invoke<GenerationTask>("creative_generate", { request });
}

export async function getGeneration(taskId: string): Promise<GenerationTask> {
  if (!isTauri()) throw new Error("Creative providers require the desktop application");
  return invoke<GenerationTask>("creative_generation_status", { taskId });
}

export async function downloadGeneratedAudio(taskId: string): Promise<ArrayBuffer> {
  if (!isTauri()) throw new Error("Creative providers require the desktop application");
  return invoke<ArrayBuffer>("creative_download_audio", { taskId });
}

export async function cancelGeneration(taskId: string): Promise<GenerationTask> {
  if (!isTauri()) throw new Error("Creative providers require the desktop application");
  return invoke<GenerationTask>("creative_cancel_generation", { taskId });
}

export async function saveGenerationReceipt(
  task: GenerationTask,
  prompt: string,
  details: { termsVersion?: string; encodedAudio?: EncodedAudioMetadata; analysis?: AudioAnalysisResult } = {},
): Promise<void> {
  const promptHash = await sha256(prompt);
  const outputUrlHash = task.audioUrl ? await sha256(task.audioUrl) : undefined;
  const receipt: GenerationReceipt = {
    taskId: task.id,
    provider: task.provider,
    model: task.model,
    promptHash,
    outputUrlHash,
    createdAt: new Date().toISOString(),
    termsVersion: details.termsVersion ?? task.provenance?.termsVersion,
    modelVersion: task.provenance?.modelVersion,
    pricingVersion: task.provenance?.pricingVersion,
    reservedCostUsd: task.reservedCostUsd,
    generationCostUsd: task.generationCostUsd,
    providerBillingVerified: task.provenance?.providerBillingVerified,
    synthidExpected: task.provenance?.synthidExpected,
    c2paStatus: task.provenance?.c2paStatus,
    encodedAudio: details.encodedAudio,
    analysis: details.analysis ? summarizeAnalysis(details.analysis) : undefined,
  };
  const receipts = readGenerationReceipts().filter((existing) => existing.taskId !== task.id);
  receipts.push(receipt);
  let bounded = receipts.slice(-MAX_RECEIPTS);
  let encoded = JSON.stringify(bounded);
  while (bounded.length > 1 && new TextEncoder().encode(encoded).byteLength > MAX_RECEIPT_STORAGE_BYTES) {
    bounded = bounded.slice(1);
    encoded = JSON.stringify(bounded);
  }
  if (new TextEncoder().encode(encoded).byteLength > MAX_RECEIPT_STORAGE_BYTES) {
    throw new Error("Generation receipt exceeds the local storage budget");
  }
  localStorage.setItem(RECEIPT_KEY, encoded);
}

function summarizeAnalysis(analysis: AudioAnalysisResult): AnalysisReceiptSummary {
  const { waveform, onsetMap, beatGridSeconds, ...summary } = analysis;
  return {
    ...summary,
    waveformBucketCount: waveform.length,
    onsetCount: onsetMap.length,
    beatCount: beatGridSeconds.length,
  };
}

export function readGenerationReceipts(): GenerationReceipt[] {
  try {
    const value = localStorage.getItem(RECEIPT_KEY);
    if (!value) return [];
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed) ? (parsed as GenerationReceipt[]) : [];
  } catch {
    return [];
  }
}

async function sha256(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
