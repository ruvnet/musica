import { describe, expect, it } from "vitest";
import {
  BROADCAST_POLL_INTERVAL_MS,
  BROADCAST_SNAPSHOT_VERSION,
  MAX_REMOTE_INTENSITY_DELTA_PER_SECOND,
  broadcastSnapshotFingerprint,
  captureBroadcastSnapshot,
  isNewerBroadcastSnapshot,
  normalizeBroadcastDirectory,
  normalizeBroadcastFollowUpdate,
  normalizeBroadcastSnapshot,
  sanitizeBroadcastText,
  slewRemoteIntensity,
  type BroadcastSnapshot,
} from "../src/core/broadcast";
import { DEFAULT_LYRIA_DECK_CONTROLS } from "../src/core/lyriaDeckScenes";
import { DEFAULT_MASTER_EFFECT_PARAMS } from "../src/audio/AudioEngine";

function validSnapshot(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    v: BROADCAST_SNAPSHOT_VERSION,
    rev: 7,
    performance: {
      bpm: 128,
      styleId: "techno",
      styleLabel: "Warehouse Techno",
      deckEnabled: { main: true, sequence: true, vocal: false },
      deckControls: {
        main: { volume: 0.8, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
        sequence: { volume: 0.4, muted: false, pitchSemitones: 2, beatNudgeMs: -20 },
        vocal: { volume: 0.3, muted: true, pitchSemitones: 0, beatNudgeMs: 0 },
      },
    },
    lyria: {
      config: {
        bpm: 128,
        guidance: 4,
        density: 0.7,
        brightness: 0.6,
        temperature: 1.1,
        topK: 40,
        scale: "C_MAJOR_A_MINOR",
        muteBass: false,
        muteDrums: false,
        onlyBassAndDrums: false,
        musicGenerationMode: "QUALITY",
      },
      prompts: [{ text: "warehouse techno, metallic stabs", weight: 1.2 }],
    },
    visual: {
      scene: "lasergrid",
      intensity: 0.8,
      color: { palette: "neon", hue: 0.9, saturation: 0.7, contrast: 0.5, diversity: 0.4 },
    },
    fx: { effects: { reverb: 0.2 }, params: { flangerRate: 0.3 } },
    ...overrides,
  };
}

describe("broadcast snapshot envelope", () => {
  it("accepts a well-formed snapshot and preserves the generative direction", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot());
    expect(snapshot).toBeDefined();
    // The generative fields are the whole point: a follow that drops these is
    // a remote fader panel, not a shared set (ADR-182).
    expect(snapshot?.lyria.config.density).toBe(0.7);
    expect(snapshot?.lyria.config.brightness).toBe(0.6);
    expect(snapshot?.lyria.config.guidance).toBe(4);
    expect(snapshot?.lyria.config.scale).toBe("C_MAJOR_A_MINOR");
    expect(snapshot?.lyria.prompts).toEqual([{ text: "warehouse techno, metallic stabs", weight: 1.2 }]);
  });

  it("rejects non-objects, wrong versions, and unusable revisions", () => {
    expect(normalizeBroadcastSnapshot(undefined)).toBeUndefined();
    expect(normalizeBroadcastSnapshot(null)).toBeUndefined();
    expect(normalizeBroadcastSnapshot("{}")).toBeUndefined();
    expect(normalizeBroadcastSnapshot(validSnapshot({ v: 2 }))).toBeUndefined();
    expect(normalizeBroadcastSnapshot(validSnapshot({ v: undefined }))).toBeUndefined();
    expect(normalizeBroadcastSnapshot(validSnapshot({ rev: "7" }))).toBeUndefined();
    expect(normalizeBroadcastSnapshot(validSnapshot({ rev: Number.NaN }))).toBeUndefined();
    expect(normalizeBroadcastSnapshot(validSnapshot({ rev: -1 }))).toBeUndefined();
  });

  it("ignores unknown fields so a newer broadcaster cannot break an older follower", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({ somethingNew: { nested: true } }));
    expect(snapshot).toBeDefined();
    expect(snapshot as unknown as Record<string, unknown>).not.toHaveProperty("somethingNew");
  });

  it("fills in a snapshot missing every optional group rather than throwing", () => {
    const snapshot = normalizeBroadcastSnapshot({ v: 1, rev: 0 });
    expect(snapshot).toBeDefined();
    expect(snapshot?.lyria.prompts).toEqual([]);
    expect(snapshot?.visual.intensity).toBeGreaterThanOrEqual(0);
    expect(snapshot?.fx.params.flangerRate).toBe(DEFAULT_MASTER_EFFECT_PARAMS.flangerRate);
  });
});

describe("broadcast snapshot clamping", () => {
  it("clamps out-of-range tempo, levels, and generative parameters", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({
      performance: { ...(validSnapshot().performance as object), bpm: 9_000 },
      lyria: {
        config: {
          bpm: -50,
          guidance: 999,
          density: 4,
          brightness: -2,
          temperature: 50,
          topK: 10_000,
          seed: -3,
          scale: "NOT_A_SCALE",
          musicGenerationMode: "TOTALLY_FAKE",
        },
        prompts: [],
      },
      visual: { scene: "lasergrid", intensity: 12, color: { palette: "neon", hue: -5 } },
      fx: { effects: { reverb: 8 }, params: { flangerRate: -1 } },
    }));

    expect(snapshot?.performance.bpm).toBe(200);
    expect(snapshot?.lyria.config.bpm).toBe(60);
    expect(snapshot?.lyria.config.guidance).toBe(6);
    expect(snapshot?.lyria.config.density).toBe(1);
    expect(snapshot?.lyria.config.brightness).toBe(0);
    expect(snapshot?.lyria.config.temperature).toBe(2);
    expect(snapshot?.lyria.config.topK).toBe(100);
    expect(snapshot?.lyria.config.seed).toBe(0);
    expect(snapshot?.visual.intensity).toBe(1);
    expect(snapshot?.visual.color.hue).toBe(0);
    expect(snapshot?.fx.effects.reverb).toBe(1);
    expect(snapshot?.fx.params.flangerRate).toBe(0);
  });

  it("falls back to known-safe values for unknown scene, palette, scale, and mode ids", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({
      visual: { scene: "../../etc/passwd", intensity: 0.5, color: { palette: "rainbow" } },
      lyria: { config: { scale: "Z_MAJOR", musicGenerationMode: "EXPENSIVE" }, prompts: [] },
    }));
    expect(snapshot?.visual.scene).toBe("tunnel");
    expect(snapshot?.visual.color.palette).toBe("scene");
    expect(snapshot?.lyria.config.scale).toBe("SCALE_UNSPECIFIED");
    expect(snapshot?.lyria.config.musicGenerationMode).toBe("QUALITY");
  });

  it("clamps per-deck pitch and nudge, and treats a missing deck as default", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({
      performance: {
        ...(validSnapshot().performance as object),
        deckControls: { main: { volume: 5, pitchSemitones: 96, beatNudgeMs: 10_000 } },
      },
    }));
    expect(snapshot?.performance.deckControls.main).toEqual({
      volume: 1,
      muted: false,
      pitchSemitones: 7,
      beatNudgeMs: 250,
    });
    expect(snapshot?.performance.deckControls.vocal).toEqual(DEFAULT_LYRIA_DECK_CONTROLS.vocal);
  });

  it("refuses to assert mutually exclusive deck mutes at once", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({
      lyria: {
        config: { muteBass: true, muteDrums: true, onlyBassAndDrums: true },
        prompts: [],
      },
    }));
    expect(snapshot?.lyria.config.muteBass).toBe(true);
    expect(snapshot?.lyria.config.muteDrums).toBe(true);
    expect(snapshot?.lyria.config.onlyBassAndDrums).toBe(false);
  });

  it("caps an oversized prompt list", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({
      lyria: {
        config: {},
        prompts: Array.from({ length: 500 }, (_, index) => ({ text: `prompt ${index}`, weight: 1 })),
      },
    }));
    expect(snapshot?.lyria.prompts).toHaveLength(8);
  });

  it("never carries the broadcaster's saved deck-scene presets", () => {
    // Adopting them would overwrite the follower's own saved presets, which a
    // follow must never do (ADR-182).
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({
      performance: {
        ...(validSnapshot().performance as object),
        deckScenes: [{ id: "theirs", name: "Theirs", bpm: 120 }],
        activeDeckSceneId: "theirs",
      },
    }));
    const performance = snapshot?.performance as unknown as Record<string, unknown>;
    expect(performance).not.toHaveProperty("deckScenes");
    expect(performance).not.toHaveProperty("activeDeckSceneId");
  });
});

describe("broadcast text safety", () => {
  it("strips control characters and bidi overrides, then caps length", () => {
    const RTL_OVERRIDE = String.fromCharCode(0x202e);
    const NUL_AND_ESC = String.fromCharCode(0x0000) + String.fromCharCode(0x001b);
    expect(sanitizeBroadcastText(`dj${NUL_AND_ESC}${RTL_OVERRIDE}name`, 32)).toBe("djname");
    expect(sanitizeBroadcastText(`${RTL_OVERRIDE}harmless-looking`, 32)).toBe("harmless-looking");
    expect(sanitizeBroadcastText("x".repeat(500), 32)).toHaveLength(32);
    expect(sanitizeBroadcastText(42, 32)).toBe("");
    expect(sanitizeBroadcastText(undefined, 32)).toBe("");
  });

  it("sanitizes prompt text, since it reaches the follower's own Lyria credential", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({
      lyria: {
        config: {},
        prompts: [
          { text: `bad${String.fromCharCode(0)}prompt${"!".repeat(400)}`, weight: 99 },
          { text: "   ", weight: 1 },
          { text: 12345, weight: 1 },
        ],
      },
    }));
    expect(snapshot?.lyria.prompts).toHaveLength(1);
    expect(snapshot?.lyria.prompts[0]?.text).toMatch(/^badprompt!+$/);
    expect(snapshot?.lyria.prompts[0]?.text).toHaveLength(200);
    expect(snapshot?.lyria.prompts[0]?.weight).toBe(2);
  });

  it("keeps hostile ids and names from reaching engine state intact", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({
      performance: {
        ...(validSnapshot().performance as object),
        styleId: "a".repeat(500),
        activeDeckSceneId: "   ",
      },
    }));
    expect(snapshot?.performance.styleId).toHaveLength(64);
  });
});

describe("broadcast revision ordering", () => {
  it("applies only strictly newer revisions", () => {
    const snapshot = normalizeBroadcastSnapshot(validSnapshot({ rev: 5 })) as BroadcastSnapshot;
    expect(isNewerBroadcastSnapshot(snapshot, undefined)).toBe(true);
    expect(isNewerBroadcastSnapshot(snapshot, 4)).toBe(true);
    // Equal revisions are rejected too, so a repeated poll is a no-op.
    expect(isNewerBroadcastSnapshot(snapshot, 5)).toBe(false);
    expect(isNewerBroadcastSnapshot(snapshot, 9)).toBe(false);
  });
});

describe("remote visual intensity slew", () => {
  it("bounds how fast a follower travels toward a remote value", () => {
    expect(slewRemoteIntensity(0, 1, 1_000)).toBeCloseTo(0.35, 5);
    expect(slewRemoteIntensity(1, 0, 1_000)).toBeCloseTo(0.65, 5);
  });

  it("actually engages at the default poll interval", () => {
    // The budget must be small enough that a full-range swing takes more than
    // one poll — otherwise the bound exists on paper and never fires.
    const step = slewRemoteIntensity(0, 1, BROADCAST_POLL_INTERVAL_MS);
    expect(step).toBeLessThan(1);
    expect(step).toBeCloseTo(MAX_REMOTE_INTENSITY_DELTA_PER_SECOND * 2, 5);
  });

  it("settles exactly on the target once it is within budget", () => {
    expect(slewRemoteIntensity(0.5, 0.55, 1_000)).toBe(0.55);
    expect(slewRemoteIntensity(0.5, 0.5, 1_000)).toBe(0.5);
  });

  it("never leaves the unit range, even from hostile inputs", () => {
    expect(slewRemoteIntensity(-5, 50, 10_000)).toBe(1);
    expect(slewRemoteIntensity(50, -5, 10_000)).toBe(0);
    expect(slewRemoteIntensity(0.4, 1, 0)).toBe(0.4);
    expect(slewRemoteIntensity(0.4, 1, -100)).toBe(0.4);
  });
});

describe("broadcast directory", () => {
  it("normalizes listings, drops unusable ones, and holds the leaderboard to five", () => {
    const directory = normalizeBroadcastDirectory({
      top: Array.from({ length: 20 }, (_, index) => ({ id: `dj${index}`, displayName: `DJ ${index}`, listeners: index })),
      all: [
        { id: "ok", displayName: "Fine", live: true, listeners: 3, bpm: 128 },
        { id: "", displayName: "No id" },
        { displayName: "Also no id" },
        "not an object",
      ],
    });
    expect(directory.top).toHaveLength(5);
    expect(directory.all).toHaveLength(1);
    expect(directory.all[0]).toMatchObject({ id: "ok", live: true, listeners: 3 });
  });

  it("gives an unnamed broadcaster a placeholder and clamps hostile counters", () => {
    const directory = normalizeBroadcastDirectory({
      all: [{ id: "x", displayName: "   ", listeners: -99, updatedAgoSeconds: -1, bpm: 10_000 }],
    });
    expect(directory.all[0]).toMatchObject({
      displayName: "Unnamed",
      listeners: 0,
      updatedAgoSeconds: 0,
      bpm: 200,
    });
  });

  it("tolerates a malformed directory payload", () => {
    expect(normalizeBroadcastDirectory(undefined)).toEqual({ top: [], all: [] });
    expect(normalizeBroadcastDirectory({ top: "nope", all: 7 })).toEqual({ top: [], all: [] });
  });
});

describe("broadcast follow updates", () => {
  it("carries a snapshot when present and stays valid when omitted", () => {
    const withBody = normalizeBroadcastFollowUpdate({ rev: 3, snapshot: validSnapshot({ rev: 3 }), listeners: 2, live: true });
    expect(withBody?.snapshot?.rev).toBe(3);
    expect(withBody?.listeners).toBe(2);

    // An unchanged set costs a heartbeat, not a payload.
    const unchanged = normalizeBroadcastFollowUpdate({ rev: 3, listeners: 2, live: true });
    expect(unchanged).toBeDefined();
    expect(unchanged?.snapshot).toBeUndefined();
  });

  it("drops an unusable update and survives a corrupt embedded snapshot", () => {
    expect(normalizeBroadcastFollowUpdate({ listeners: 1 })).toBeUndefined();
    expect(normalizeBroadcastFollowUpdate(undefined)).toBeUndefined();
    const corrupt = normalizeBroadcastFollowUpdate({ rev: 4, snapshot: { v: 99 } });
    expect(corrupt).toBeDefined();
    expect(corrupt?.snapshot).toBeUndefined();
  });
});

describe("capturing a snapshot to publish", () => {
  const capture = () => captureBroadcastSnapshot({
    rev: 3,
    bpm: 128,
    styleId: "techno",
    styleLabel: "Warehouse Techno",
    deckEnabled: { main: true, sequence: false, vocal: false },
    deckControls: { ...DEFAULT_LYRIA_DECK_CONTROLS },
    lyriaConfig: {
      bpm: 128,
      guidance: 4,
      density: 0.7,
      brightness: 0.6,
      temperature: 1.1,
      topK: 40,
      scale: "C_MAJOR_A_MINOR",
      muteBass: false,
      muteDrums: false,
      onlyBassAndDrums: false,
      musicGenerationMode: "QUALITY",
    },
    lyriaPrompts: [{ text: "warehouse techno", weight: 1 }],
    visualScene: "lasergrid",
    visualIntensity: 0.8,
    visualColor: { palette: "neon", hue: 0.9, saturation: 0.7, contrast: 0.5, diversity: 0.4 },
    masterEffects: { flanger: 0, phaser: 0, drive: 0, crush: 0, sweep: 0, reverb: 0.2, echo: 0 },
    masterEffectParams: { ...DEFAULT_MASTER_EFFECT_PARAMS },
  });

  it("produces a snapshot a follower accepts unchanged", () => {
    const snapshot = capture();
    // Outgoing data goes through the same normalizer as incoming, so a field
    // that would be clamped away on receipt is never sent in the first place.
    expect(normalizeBroadcastSnapshot(snapshot)).toEqual(snapshot);
  });

  it("fingerprints content independently of the revision, so an idle set stops republishing", () => {
    const first = capture();
    const second = { ...capture(), rev: 99 };
    expect(broadcastSnapshotFingerprint(first)).toBe(broadcastSnapshotFingerprint(second));

    const moved = { ...capture(), visual: { ...first.visual, intensity: 0.1 } };
    expect(broadcastSnapshotFingerprint(moved)).not.toBe(broadcastSnapshotFingerprint(first));
  });

  it("omits master volume and any transport command", () => {
    const snapshot = capture() as unknown as Record<string, unknown>;
    const performance = snapshot.performance as Record<string, unknown>;
    // A follower's monitor level is theirs, and following never starts audio.
    expect(performance).not.toHaveProperty("masterVolume");
    expect(performance).not.toHaveProperty("playing");
    expect(snapshot).not.toHaveProperty("playing");
  });
});
