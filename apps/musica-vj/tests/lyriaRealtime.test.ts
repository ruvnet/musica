import { describe, expect, it } from "vitest";
import {
  DEFAULT_LYRIA_REALTIME_CONFIG,
  DEFAULT_LYRIA_REALTIME_PROMPTS,
  DEFAULT_LYRIA_REALTIME_STYLE_ID,
  LYRIA_REALTIME_STYLE_PRESETS,
  AUTO_DJ_PHRASE_BARS,
  autoDjPhraseDurationMs,
  compensateLyriaBpmForPitch,
  createAutoDjRealtimeRequest,
  createLyriaSequenceConfig,
  createLyriaSequencePrompts,
  createLyriaVocalPrompts,
  createCustomLyriaStyle,
  createLyriaRealtimeRequestForTemplate,
  createLyriaRealtimeRequestFromStyle,
  loadCustomLyriaStyles,
  lyriaRealtimeStyleById,
  lyriaRealtimeStyleForTemplate,
  registerCustomLyriaStyles,
  type LyriaRealtimeConfig,
} from "../src/core/lyriaRealtime";
import { createEngineSnapshotFromTemplate } from "../src/audio/AudioEngine";
import { PERFORMANCE_TEMPLATES, defaultPerformanceTemplate, performanceTemplateById } from "../src/core/presets";

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
    const prompts = createLyriaSequencePrompts(state, style, {
      mainPrompts: [{ text: "Main techno identity with filtered metallic stabs and a dark two-bar motif", weight: 1.2 }],
      scale: "F_MAJOR_D_MINOR",
      customDirection: "Reinforce the main kick and follow its bass cadences",
    });

    expect(prompts.map((prompt) => prompt.text).join(" ")).toContain(`DR:${expectedPulse}`);
    expect(prompts.map((prompt) => prompt.text).join(" ")).toContain("Main techno identity");
    expect(prompts.map((prompt) => prompt.text).join(" ")).toContain("f major d minor");
    expect(prompts.map((prompt) => prompt.text).join(" ")).toContain("follow its bass cadences");
    expect(prompts.map((prompt) => prompt.text).join(" ")).toContain("drums and bass only");
    expect(prompts.every((prompt) => prompt.text.length <= 240)).toBe(true);
    expect(prompts.length).toBeLessThanOrEqual(4);
  });

  it("gives the vocal deck a main-linked 32-bar chorus form with negative guidance", () => {
    const style = LYRIA_REALTIME_STYLE_PRESETS.find((preset) => preset.id === "house")!;
    const prompts = createLyriaVocalPrompts(style, {
      mainPrompts: [{ text: "Warm elastic sub, glossy piano hook, and elegant restrained tension", weight: 1.2 }],
      scale: "C_MAJOR_A_MINOR",
      customDirection: "Answer the piano motif with a long wordless chorus hook",
    });

    expect(prompts).toHaveLength(4);
    expect(prompts.some((prompt) => prompt.text.includes("Warm elastic sub"))).toBe(true);
    expect(prompts.some((prompt) => prompt.text.includes("32-bar vocal form"))).toBe(true);
    expect(prompts.some((prompt) => prompt.text.includes("21-28 sustained chorus hook"))).toBe(true);
    expect(prompts.some((prompt) => prompt.text.includes("voice only"))).toBe(true);
    expect(prompts.some((prompt) => prompt.weight < 0 && prompt.text.includes("accompaniment"))).toBe(true);
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
    expect(DEFAULT_LYRIA_REALTIME_STYLE_ID).toBe("rock");
    expect(LYRIA_REALTIME_STYLE_PRESETS.map((preset) => preset.id)).toEqual(
      expect.arrayContaining(["house", "techno", "cinematic", "drum-bass", "hiphop", "funk", "samba", "rock", "jazz", "classical", "ambient"]),
    );
    for (const style of LYRIA_REALTIME_STYLE_PRESETS) {
      expect(style.description.trim().length).toBeGreaterThan(12);
      expect(style.prompts).toHaveLength(4);
      expect(style.prompts.every((prompt) => prompt.text.length <= 240)).toBe(true);
      expect(style.prompts[3].weight).toBeLessThan(0);
      const request = createLyriaRealtimeRequestFromStyle(style);
      expect(request.weightedPrompts.length).toBeGreaterThanOrEqual(1);
      expect(request.weightedPrompts.length).toBeLessThanOrEqual(4);
      expect(request.weightedPrompts.every((prompt) => prompt.text.trim().length > 0 && prompt.weight !== 0)).toBe(true);
      expect(request.weightedPrompts.some((prompt) => prompt.weight < 0)).toBe(true);
      expect(request.weightedPrompts.every((prompt) => prompt.text.length <= 240)).toBe(true);
      expect(validConfig(request.config)).toBe(true);
    }
  });

  it("builds detailed single-stream Auto DJ phrases at musical intervals", () => {
    const style = LYRIA_REALTIME_STYLE_PRESETS.find((preset) => preset.id === "techno")!;
    const request = createAutoDjRealtimeRequest(style, {
      personalization: "futuristic restrained warehouse set with a memorable metallic motif",
      step: 2,
      bpm: 128,
    });

    expect(AUTO_DJ_PHRASE_BARS).toBe(32);
    expect(autoDjPhraseDurationMs(128)).toBeGreaterThanOrEqual(60_000);
    expect(request.config.bpm).toBe(128);
    expect(request.weightedPrompts).toHaveLength(4);
    expect(request.weightedPrompts.map((prompt) => prompt.text).join(" ")).toContain("Single continuous main stereo stream");
    expect(request.weightedPrompts.map((prompt) => prompt.text).join(" ")).toContain("Beat design:");
    expect(request.weightedPrompts.map((prompt) => prompt.text).join(" ")).toContain("futuristic restrained warehouse set");
    expect(request.weightedPrompts.some((prompt) => prompt.weight < 0 && prompt.text.includes("multiple streams"))).toBe(true);
    expect(request.weightedPrompts.every((prompt) => prompt.text.length <= 240)).toBe(true);
    expect(validConfig(request.config)).toBe(true);

    const directed = createAutoDjRealtimeRequest(style, {
      personalization: "restrained warehouse identity",
      generatedBrief: "Precise kick and rolling percussion, metallic motif introduced over eight bars, mono-compatible sub and short room depth",
      step: 3,
      bpm: 128,
    });
    expect(directed.weightedPrompts.some((prompt) => prompt.text.startsWith("Director brief:"))).toBe(true);
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
    expect(lyriaRealtimeStyleForTemplate(defaultPerformanceTemplate()).id).toBe("rock");
  });

  it("creates, registers, and resolves user custom styles", () => {
    const base = LYRIA_REALTIME_STYLE_PRESETS[0];
    const custom = createCustomLyriaStyle("  Midnight Chant  ", base, [base.id, "custom-midnight-chant"]);
    expect(custom.id).toBe("custom-midnight-chant-2");
    expect(custom.label).toBe("Midnight Chant");
    expect(custom.prompts).toHaveLength(base.prompts.length);
    expect(custom.prompts[0]).not.toBe(base.prompts[0]);

    registerCustomLyriaStyles([custom]);
    expect(lyriaRealtimeStyleById(custom.id).label).toBe("Midnight Chant");
    registerCustomLyriaStyles([]);
    expect(lyriaRealtimeStyleById(custom.id).id).toBe(DEFAULT_LYRIA_REALTIME_STYLE_ID);
  });

  it("round-trips custom styles through serialization and rejects malformed entries", () => {
    const base = LYRIA_REALTIME_STYLE_PRESETS[0];
    const custom = createCustomLyriaStyle("Vapor", base, []);
    const restored = loadCustomLyriaStyles(JSON.stringify([custom]));
    expect(restored).toHaveLength(1);
    expect(restored[0].id).toBe(custom.id);

    expect(loadCustomLyriaStyles(undefined)).toEqual([]);
    expect(loadCustomLyriaStyles("not json")).toEqual([]);
    expect(loadCustomLyriaStyles(JSON.stringify({ id: "custom-x" }))).toEqual([]);
    expect(loadCustomLyriaStyles(JSON.stringify([{ id: "rock", label: "spoof", prompts: [], config: {} }]))).toEqual([]);
    expect(loadCustomLyriaStyles(JSON.stringify([{ id: "custom-bad", label: "x", prompts: [{ text: 1, weight: "y" }], config: {} }]))).toEqual([]);
  });
});

describe("workspace settings", () => {
  it("normalizes, serializes, and round-trips workspace settings", async () => {
    const { normalizeWorkspaceSettings, serializeWorkspaceSettings } = await import("../src/core/settingsStore");
    expect(normalizeWorkspaceSettings(undefined)).toBeUndefined();
    expect(normalizeWorkspaceSettings({ version: 2 })).toBeUndefined();

    const normalized = normalizeWorkspaceSettings({
      version: 1,
      masterEffects: { drive: 3, reverb: 0.4 },
      fxLocks: { drive: true, reverb: "yes" },
      sfxLevel: -2,
      onboarding: { styleId: "lofi" },
    });
    expect(normalized).toBeDefined();
    expect(normalized!.masterEffects.drive).toBe(1);
    expect(normalized!.masterEffects.reverb).toBe(0.4);
    expect(normalized!.masterEffects.flanger).toBe(0);
    expect(normalized!.fxLocks.drive).toBe(true);
    expect(normalized!.fxLocks.reverb).toBe(false);
    expect(normalized!.sfxLevel).toBe(0);
    expect(normalized!.onboarding.styleId).toBe("lofi");
    expect(normalized!.onboarding.visualScene).toBe("oscilloscope");

    const roundTripped = normalizeWorkspaceSettings(JSON.parse(serializeWorkspaceSettings(normalized!)));
    expect(roundTripped!.masterEffects).toEqual(normalized!.masterEffects);
    expect(roundTripped!.savedAt).toBeTruthy();
  });
});

describe("performance memory", () => {
  it("scores mood similarity by bounded token overlap", async () => {
    const { similarity, tokenize } = await import("../src/core/performanceMemory");
    expect(tokenize("Dark, RISING tension!")).toEqual(["dark", "rising", "tension"]);
    expect(similarity("dark rising tension", "dark rising tension")).toBe(1);
    expect(similarity("dark rising tension", "dark tension at night")).toBeGreaterThan(0.4);
    expect(similarity("dark rising tension", "golden sunset drift")).toBe(0);
    expect(similarity("", "anything")).toBe(0);
    expect(similarity("a b", "a b")).toBe(0);
  });
});

describe("auto dj style walks", () => {
  it("keeps every neighbor inside the preset vocabulary and walks deterministically", async () => {
    const { AUTO_DJ_STYLE_NEIGHBORS, nextAutoDjStyleId } = await import("../src/core/lyriaRealtime");
    const known = new Set(LYRIA_REALTIME_STYLE_PRESETS.map((style) => style.id));
    for (const [style, neighbors] of Object.entries(AUTO_DJ_STYLE_NEIGHBORS)) {
      expect(known.has(style)).toBe(true);
      expect(neighbors.length).toBeGreaterThanOrEqual(3);
      for (const neighbor of neighbors) {
        expect(known.has(neighbor)).toBe(true);
        expect(neighbor).not.toBe(style);
      }
    }
    for (const style of known) {
      expect(AUTO_DJ_STYLE_NEIGHBORS[style]).toBeDefined();
    }
    expect(nextAutoDjStyleId("rock", 0)).toBe("blues");
    expect(nextAutoDjStyleId("rock", 1)).toBe("dubstep");
    expect(nextAutoDjStyleId("custom-unknown", 0)).toBe("blues");
    expect(known.has(nextAutoDjStyleId("ambient", 7))).toBe(true);
  });
});
