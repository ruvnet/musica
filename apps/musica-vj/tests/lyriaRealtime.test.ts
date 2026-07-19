import { describe, expect, it } from "vitest";
import {
  DEFAULT_LYRIA_REALTIME_CONFIG,
  DEFAULT_LYRIA_REALTIME_PROMPTS,
  DEFAULT_LYRIA_REALTIME_STYLE_ID,
  LYRIA_REALTIME_STYLE_PRESETS,
  compensateLyriaBpmForPitch,
  createLyriaRealtimeRequestForTemplate,
  createLyriaRealtimeRequestFromStyle,
  lyriaRealtimeStyleForTemplate,
  type LyriaRealtimeConfig,
} from "../src/core/lyriaRealtime";
import { PERFORMANCE_TEMPLATES, performanceTemplateById } from "../src/core/presets";

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
  it("compensates source BPM so post-stream pitch changes remain beat locked", () => {
    expect(compensateLyriaBpmForPitch(120, 0)).toBe(120);
    expect(compensateLyriaBpmForPitch(120, 12)).toBe(60);
    expect(compensateLyriaBpmForPitch(120, -12)).toBe(200);
  });

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
    expect(DEFAULT_LYRIA_REALTIME_STYLE_ID).toBe("house");
    expect(LYRIA_REALTIME_STYLE_PRESETS.map((preset) => preset.id)).toEqual(
      expect.arrayContaining(["house", "techno", "cinematic", "drum-bass", "hiphop", "funk", "samba", "rock", "jazz", "classical", "ambient"]),
    );
    for (const style of LYRIA_REALTIME_STYLE_PRESETS) {
      expect(style.description.trim().length).toBeGreaterThan(12);
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
    expect(lyriaRealtimeStyleForTemplate(performanceTemplateById("afro-cosmic-house")).id).toBe("house");
  });
});
