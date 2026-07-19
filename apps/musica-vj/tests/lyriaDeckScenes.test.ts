import { describe, expect, it } from "vitest";
import {
  DEFAULT_LYRIA_DECK_SCENES,
  cloneLyriaDeckScene,
  loadLyriaDeckScenes,
  normalizeLyriaDeckScene,
} from "../src/core/lyriaDeckScenes";

describe("Lyria deck scenes", () => {
  it("normalizes live controls to the supported performance range", () => {
    const fallback = DEFAULT_LYRIA_DECK_SCENES[0];
    const scene = normalizeLyriaDeckScene({
      ...fallback,
      name: "  A very long custom performance preset  ",
      bpm: 260,
      controls: {
        ...fallback.controls,
        main: { volume: 2, muted: false, pitchSemitones: -20, beatNudgeMs: 999 },
      },
    }, fallback);

    expect(scene.name).toBe("A very long custom");
    expect(scene.bpm).toBe(200);
    expect(scene.controls.main).toMatchObject({ volume: 1, pitchSemitones: -7, beatNudgeMs: 250 });
  });

  it("merges persisted scenes by stable slot id and falls back on malformed storage", () => {
    const stored = JSON.stringify([{ id: "peak", name: "Warehouse", bpm: 138 }]);
    const scenes = loadLyriaDeckScenes(stored);

    expect(scenes).toHaveLength(4);
    expect(scenes[1]).toMatchObject({ id: "peak", name: "Warehouse", bpm: 138, styleId: "techno" });
    expect(loadLyriaDeckScenes("not-json")).toEqual(DEFAULT_LYRIA_DECK_SCENES);
  });

  it("clones controls so an editor cannot mutate the loaded slot", () => {
    const original = DEFAULT_LYRIA_DECK_SCENES[0];
    const copy = cloneLyriaDeckScene(original);
    copy.controls.main.volume = 0.1;

    expect(original.controls.main.volume).toBe(0.76);
  });
});
