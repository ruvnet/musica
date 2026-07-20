import { describe, expect, it } from "vitest";
import {
  clamp,
  createTrackDefinitions,
  effectiveTrackGain,
  hashSeed,
  midiToFrequency,
  mutatePattern,
  patternFromSteps,
  secondsPerStep,
} from "../src/core/music";
import { countLateSteps } from "../src/audio/AudioEngine";
import {
  DEFAULT_MIDI_SONG_BANK_IDS,
  DEFAULT_PERFORMANCE_TEMPLATE_ID,
  PERFORMANCE_TEMPLATES,
  VISUAL_SCENES,
  defaultPerformanceTemplate,
  midiSongBankTemplates,
} from "../src/core/presets";

describe("music invariants", () => {
  it("converts concert A to 440 Hz", () => {
    expect(midiToFrequency(69)).toBeCloseTo(440, 8);
  });

  it("uses sixteenth-note step timing", () => {
    expect(secondsPerStep(120)).toBeCloseTo(0.125, 8);
    expect(secondsPerStep(1)).toBeCloseTo(0.25, 8);
    expect(secondsPerStep(999)).toBeCloseTo(0.075, 8);
  });

  it("clamps values deterministically", () => {
    expect(clamp(-1, 0, 1)).toBe(0);
    expect(clamp(0.4, 0, 1)).toBe(0.4);
    expect(clamp(8, 0, 1)).toBe(1);
  });

  it("rejects invalid pattern indices", () => {
    const pattern = patternFromSteps([-1, 0, 4, 16, 1.5]);
    expect(pattern).toHaveLength(16);
    expect(pattern.filter(Boolean)).toHaveLength(2);
    expect(pattern[0]).toBe(true);
    expect(pattern[4]).toBe(true);
  });

  it("builds six reproducible, scale-constrained tracks", () => {
    const first = createTrackDefinitions("same seed");
    const second = createTrackDefinitions("same seed");
    expect(first).toEqual(second);
    expect(first).toHaveLength(6);
    expect(new Set(first.map((track) => track.id)).size).toBe(6);
    expect(first.every((track) => track.pattern.length === 16)).toBe(true);
    expect(first.every((track) => track.notes.length >= 16)).toBe(true);
    expect(first.find((track) => track.id === "lead")?.notes.length).toBeGreaterThan(16);
  });

  it("uses a stable prompt hash", () => {
    expect(hashSeed("liquid neon")).toBe(hashSeed("liquid neon"));
    expect(hashSeed("liquid neon")).not.toBe(hashSeed("liquid neon!"));
  });

  it("mutes tracks and applies an equal-power-style fader curve", () => {
    expect(effectiveTrackGain({ volume: 0.5, pan: 0, muted: false, solo: false }, false)).toBe(0.25);
    expect(effectiveTrackGain({ volume: 1, pan: 0, muted: true, solo: false }, false)).toBe(0);
    expect(effectiveTrackGain({ volume: 1, pan: 0, muted: false, solo: false }, true)).toBe(0);
    expect(effectiveTrackGain({ volume: 1, pan: 0, muted: false, solo: true }, true)).toBe(1);
  });

  it("mutates patterns reproducibly while keeping the downbeat stable", () => {
    const input = patternFromSteps([0, 4, 8, 12]);
    const first = mutatePattern(input, 42, 0.3);
    const second = mutatePattern(input, 42, 0.3);
    expect(first).toEqual(second);
    expect(first[0]).toBe(input[0]);
  });

  it("drops late scheduled steps instead of bursting them into the present", () => {
    expect(countLateSteps(10, 10.01, 0.125)).toBe(0);
    expect(countLateSteps(10, 10.5, 0.125)).toBe(5);
  });

  it("ships multiple complete performance templates", () => {
    expect(VISUAL_SCENES).toHaveLength(9);
    expect(PERFORMANCE_TEMPLATES.length).toBeGreaterThanOrEqual(10);
    for (const template of PERFORMANCE_TEMPLATES) {
      expect(template.bpm).toBeGreaterThanOrEqual(60);
      expect(template.bpm).toBeLessThanOrEqual(200);
      expect(VISUAL_SCENES.some((scene) => scene.id === template.scene)).toBe(true);
      expect(Object.values(template.tracks).every((track) => track.pattern.length > 0 && track.notes.length > 0)).toBe(true);
    }
  });

  it("includes longer melodic lanes for multi-bar variation", () => {
    const templatesWithLongLanes = PERFORMANCE_TEMPLATES.filter((template) =>
      Object.values(template.tracks).some((track) => track.notes.length > 16),
    );
    expect(templatesWithLongLanes.length).toBeGreaterThanOrEqual(4);
  });

  it("boots from the curated MIDI song bank instead of the base procedural pattern", () => {
    const defaultTemplate = defaultPerformanceTemplate();
    const bank = midiSongBankTemplates();

    expect(defaultTemplate.id).toBe(DEFAULT_PERFORMANCE_TEMPLATE_ID);
    expect(DEFAULT_MIDI_SONG_BANK_IDS).toContain(defaultTemplate.id as (typeof DEFAULT_MIDI_SONG_BANK_IDS)[number]);
    expect(bank.length).toBeGreaterThanOrEqual(6);
    expect(defaultTemplate.tracks.lead.notes.length).toBeGreaterThan(16);
    expect(bank.filter((template) => template.tracks.lead.notes.length > 16).length).toBeGreaterThanOrEqual(4);
  });
});
