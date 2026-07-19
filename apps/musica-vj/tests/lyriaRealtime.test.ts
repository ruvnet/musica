import { describe, expect, it } from "vitest";
import {
  DEFAULT_LYRIA_REALTIME_CONFIG,
  DEFAULT_LYRIA_REALTIME_PROMPTS,
  LYRIA_REALTIME_STYLE_PRESETS,
  createLyriaRealtimeRequestForTemplate,
  createLyriaRealtimeRequestFromStyle,
  type LyriaRealtimeConfig,
} from "../src/core/lyriaRealtime";
import { PERFORMANCE_TEMPLATES } from "../src/core/presets";

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

  it("ships valid style buttons for common musical directions", () => {
    expect(LYRIA_REALTIME_STYLE_PRESETS.map((preset) => preset.id)).toEqual(
      expect.arrayContaining(["samba", "rock", "jazz", "techno", "classical", "cinematic"]),
    );
    for (const style of LYRIA_REALTIME_STYLE_PRESETS) {
      const request = createLyriaRealtimeRequestFromStyle(style);
      expect(request.weightedPrompts.length).toBeGreaterThanOrEqual(1);
      expect(request.weightedPrompts.length).toBeLessThanOrEqual(4);
      expect(request.weightedPrompts.every((prompt) => prompt.text.trim().length > 0 && prompt.weight !== 0)).toBe(true);
      expect(validConfig(request.config)).toBe(true);
    }
  });

  it("maps every performance template to a valid Lyria RealTime guide", () => {
    for (const template of PERFORMANCE_TEMPLATES) {
      const request = createLyriaRealtimeRequestForTemplate(template);
      expect(request.weightedPrompts.length).toBeGreaterThanOrEqual(1);
      expect(request.weightedPrompts.length).toBeLessThanOrEqual(4);
      expect(request.config.bpm).toBe(template.bpm);
      expect(validConfig(request.config)).toBe(true);
    }
  });
});
