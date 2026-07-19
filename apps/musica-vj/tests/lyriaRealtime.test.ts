import { describe, expect, it } from "vitest";
import {
  DEFAULT_LYRIA_REALTIME_CONFIG,
  DEFAULT_LYRIA_REALTIME_PROMPTS,
  type LyriaRealtimeConfig,
} from "../src/core/lyriaRealtime";

function validConfig(config: LyriaRealtimeConfig): boolean {
  return (
    config.bpm >= 60 &&
    config.bpm <= 200 &&
    config.density >= 0 &&
    config.density <= 1 &&
    config.brightness >= 0 &&
    config.brightness <= 1 &&
    config.guidance >= 0 &&
    config.guidance <= 6 &&
    config.temperature >= 0 &&
    config.temperature <= 3 &&
    config.topK >= 1 &&
    config.topK <= 1000 &&
    !(config.onlyBassAndDrums && (config.muteBass || config.muteDrums))
  );
}

describe("Lyria RealTime defaults", () => {
  it("ships a valid realtime music config envelope", () => {
    expect(validConfig(DEFAULT_LYRIA_REALTIME_CONFIG)).toBe(true);
    expect(DEFAULT_LYRIA_REALTIME_CONFIG.scale).toBe("E_FLAT_MAJOR_C_MINOR");
    expect(DEFAULT_LYRIA_REALTIME_CONFIG.musicGenerationMode).toBe("QUALITY");
  });

  it("starts from non-empty weighted prompts", () => {
    expect(DEFAULT_LYRIA_REALTIME_PROMPTS.length).toBeGreaterThanOrEqual(1);
    expect(DEFAULT_LYRIA_REALTIME_PROMPTS.length).toBeLessThanOrEqual(4);
    expect(DEFAULT_LYRIA_REALTIME_PROMPTS.every((prompt) => prompt.text.trim().length > 0 && prompt.weight !== 0)).toBe(true);
  });
});
