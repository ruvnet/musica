import { beforeEach, describe, expect, it } from "vitest";
import { readGenerationReceipts, saveGenerationReceipt } from "../src/core/creativeProvider";
import type { AudioAnalysisResult } from "../src/audio/audioAnalysis";
import type { GenerationTask } from "../src/core/types";

class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();

  get length(): number {
    return this.values.size;
  }

  clear(): void {
    this.values.clear();
  }

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  key(index: number): string | null {
    return [...this.values.keys()][index] ?? null;
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

if (!globalThis.localStorage) {
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: new MemoryStorage(),
  });
}

describe("frontend generation receipt budget", () => {
  beforeEach(() => localStorage.clear());

  it("stores a bounded analysis summary instead of waveform and timing arrays", async () => {
    const task: GenerationTask = {
      id: "task-receipt-0001",
      status: "complete",
      provider: "lyria_3_pro",
      model: "lyria-3-pro-preview",
      hasAudio: true,
      cancellationRequested: false,
      providerCancelConfirmed: false,
      completedAfterCancel: false,
      generationCostUsd: 0.08,
    };
    const analysis: AudioAnalysisResult = {
      durationSeconds: 120,
      sampleRateHz: 48_000,
      channels: 2,
      loudnessLufs: -12,
      bpm: 128,
      key: "F minor",
      waveform: Array.from({ length: 256 }, (_, index) => ({
        startSeconds: index / 2,
        endSeconds: (index + 1) / 2,
        minimum: -0.5,
        maximum: 0.5,
        rms: 0.2,
      })),
      onsetMap: Array.from({ length: 128 }, (_, index) => ({ timeSeconds: index / 2, strength: 0.8 })),
      beatGridSeconds: Array.from({ length: 256 }, (_, index) => index * 0.46875),
      spectralProfile: { lowEnergy: 0.7, midEnergy: 0.5, highEnergy: 0.4, centroidHz: 2_400 },
      sections: [{ type: "intro", start: 0, end: 20, meanRms: 0.2 }],
      visualMapping: {
        bass: "camera_displacement",
        kick: "radial_pulse",
        highFrequencyEnergy: "particle_density",
        sectionChange: "scene_transition",
      },
      recommendedScene: "tunnel",
      visualIntensity: 0.8,
    };

    await saveGenerationReceipt(task, "confidential prompt", { analysis });
    const [receipt] = readGenerationReceipts();
    expect(receipt.analysis?.waveformBucketCount).toBe(256);
    expect(receipt.analysis?.onsetCount).toBe(128);
    expect(receipt.analysis?.beatCount).toBe(256);
    expect(receipt.analysis).not.toHaveProperty("waveform");
    expect(receipt.analysis).not.toHaveProperty("onsetMap");
    expect(receipt.analysis).not.toHaveProperty("beatGridSeconds");
    expect(new TextEncoder().encode(localStorage.getItem("musica-vj-generation-receipts-v1") ?? "").byteLength)
      .toBeLessThan(2 * 1024 * 1024);
  });
});
