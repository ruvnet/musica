import { describe, expect, it } from "vitest";
import {
  DEFAULT_LYRIA_REALTIME_CONFIG,
  DEFAULT_LYRIA_REALTIME_PROMPTS,
  DEFAULT_LYRIA_REALTIME_STYLE_ID,
  LYRIA_REALTIME_STYLE_PRESETS,
  compensateLyriaBpmForPitch,
  createLyriaSequenceConfig,
  createLyriaSequencePrompts,
  createLyriaVocalPrompts,
  createLyriaRealtimeRequestForTemplate,
  createLyriaRealtimeRequestFromStyle,
  lyriaRealtimeStyleForTemplate,
  type LyriaRealtimeConfig,
} from "../src/core/lyriaRealtime";
import { createEngineSnapshotFromTemplate } from "../src/audio/AudioEngine";
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

  it("encodes the editable 16-step grid into the dedicated Lyria sequence prompts", () => {
    const state = createEngineSnapshotFromTemplate(performanceTemplateById("warehouse-techno"));
    const drums = state.tracks.find((track) => track.id === "drums")!;
    const expectedPulse = drums.pattern.map((active) => (active ? "x" : "-")).join("");
    const style = LYRIA_REALTIME_STYLE_PRESETS.find((preset) => preset.id === "techno")!;
    const prompts = createLyriaSequencePrompts(state, style);

    expect(prompts.map((prompt) => prompt.text).join(" ")).toContain(`DR:${expectedPulse}`);
    expect(prompts.map((prompt) => prompt.text).join(" ")).toContain("Techno supporting beat layer");
    expect(prompts.map((prompt) => prompt.text).join(" ")).toContain("drums and bass only");
    expect(prompts.every((prompt) => prompt.text.length <= 240)).toBe(true);
    expect(prompts.length).toBeLessThanOrEqual(4);
  });

  it("gives the vocal deck a sparse style-matched role with negative guidance", () => {
    const style = LYRIA_REALTIME_STYLE_PRESETS.find((preset) => preset.id === "house")!;
    const prompts = createLyriaVocalPrompts(style);

    expect(prompts).toHaveLength(4);
    expect(prompts.some((prompt) => prompt.text.includes("House a cappella wordless vocalization"))).toBe(true);
    expect(prompts.some((prompt) => prompt.text.includes("isolated dry human voice stem"))).toBe(true);
    expect(prompts.some((prompt) => prompt.weight < 0 && prompt.text.includes("instrumental accompaniment"))).toBe(true);
    expect(prompts.every((prompt) => prompt.text.length <= 240)).toBe(true);
  });

  it("maps sequence lane activity into Lyria drum, bass, and density controls", () => {
    const state = createEngineSnapshotFromTemplate(performanceTemplateById("warehouse-techno"));
    const drums = state.tracks.find((track) => track.id === "drums")!;
    const bass = state.tracks.find((track) => track.id === "bass")!;
    drums.muted = true;
    bass.solo = true;
    const config = createLyriaSequenceConfig(state, DEFAULT_LYRIA_REALTIME_CONFIG, 0);

    expect(config.muteDrums).toBe(true);
    expect(config.muteBass).toBe(false);
    expect(config.onlyBassAndDrums).toBe(false);
    expect(config.density).toBeGreaterThan(0);
    expect(config.guidance).toBeGreaterThan(DEFAULT_LYRIA_REALTIME_CONFIG.guidance);
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
      expect(request.weightedPrompts.some((prompt) => prompt.weight < 0)).toBe(true);
      expect(request.weightedPrompts.every((prompt) => prompt.text.length <= 240)).toBe(true);
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
