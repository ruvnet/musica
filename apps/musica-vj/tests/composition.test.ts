import { describe, expect, it } from "vitest";
import {
  acceptedOutputCost,
  acceptedTrackCost,
  compileLyriaPrompt,
  LYRIA_PRO_MODEL,
  reserveGenerationCost,
  selectGenerationRoute,
  type StructuredComposition,
  validateCompositionSpec,
  validateProDuration,
} from "../src/core/composition";

function composition(overrides: Partial<StructuredComposition> = {}): StructuredComposition {
  return {
    durationSeconds: 165,
    genre: ["melodic techno", "cinematic electronica"],
    bpm: 128,
    timeSignature: "4/4",
    mood: ["hypnotic", "euphoric"],
    instruments: ["analog bass", "metallic percussion", "wide synthesizer pads"],
    vocals: { enabled: false },
    structure: [
      { time: "0:00", section: "intro" },
      { time: "0:20", section: "groove" },
      { time: "0:42", section: "drop", direction: "clear quarter-note transients" },
      { time: "1:12", section: "breakdown" },
      { time: "1:35", section: "second drop" },
    ],
    socialHook: { startSeconds: 0, durationSeconds: 12 },
    outputFormat: "wav",
    ...overrides,
  };
}

describe("Lyria composition contract", () => {
  it("compiles timestamps and user supplied lyrics deterministically", () => {
    const specification = composition({
      vocals: {
        enabled: true,
        type: "alto lead",
        language: "English",
        lyrics: "We turn the signal into light\nAnd make the midnight move",
      },
    });
    const first = compileLyriaPrompt(specification);
    const second = compileLyriaPrompt(structuredClone(specification));

    expect(first).toBe(second);
    expect(first).toContain("0:42 drop: clear quarter-note transients");
    expect(first).toContain("1:35 second drop");
    expect(first).toContain("User supplied lyrics:\nWe turn the signal into light\nAnd make the midnight move");
    expect(first).toContain("Social hook: optimize 0:00 to 0:12");
    expect(first).toContain("Requested response audio format: WAV");
  });

  it("compiles seamless loop and tonal controls for provider loop generation", () => {
    const prompt = compileLyriaPrompt(
      composition({
        durationSeconds: 32,
        productionStyle: "club master, tight sidechain, loud but clean",
        loop: { enabled: true, bars: 16, seamless: true },
        tonal: {
          key: "F minor",
          tonalCenter: "deep F sub bass, bright chord stab, metallic percussion",
          intensity: 0.82,
          negativePrompt: "muddy low end, weak kick, long intro, long fade out",
        },
        structure: [
          { time: "0:00", section: "bar 1 downbeat" },
          { time: "0:16", section: "midpoint lift" },
        ],
      }),
    );

    expect(prompt).toContain("Loop intent: create a 16 bar seamless, DJ-loopable phrase");
    expect(prompt).toContain("Key: F minor");
    expect(prompt).toContain("Tonal center: deep F sub bass");
    expect(prompt).toContain("Production intensity: 82%");
    expect(prompt).toContain("Avoid: muddy low end");
  });

  it("enforces the 31 to 180 second UI envelope and 184 second provider hard limit", () => {
    expect(validateProDuration(31, "ui").valid).toBe(true);
    expect(validateProDuration(180, "ui").valid).toBe(true);
    expect(validateProDuration(30, "ui").valid).toBe(false);
    expect(validateProDuration(181, "ui").valid).toBe(false);
    expect(validateProDuration(184, "provider").valid).toBe(true);
    expect(validateProDuration(185, "provider").valid).toBe(false);

    expect(validateCompositionSpec(composition({ durationSeconds: 184 }), "pro-provider").valid).toBe(true);
    expect(validateCompositionSpec(composition({ durationSeconds: 184 }), "pro-ui").valid).toBe(false);
  });

  it("routes complete V1 songs to Pro and leaves clip and realtime capabilities disabled in V1", () => {
    const pro = selectGenerationRoute(composition({ durationSeconds: 120 }));
    const clip = selectGenerationRoute(composition({ durationSeconds: 30 }), { loopOrPreview: true });
    const realtime = selectGenerationRoute(composition(), { interactive: true });

    expect(pro).toMatchObject({ route: "pro", model: LYRIA_PRO_MODEL, availableInV1: true });
    expect(clip).toMatchObject({ route: "clip", model: undefined, availableInV1: false });
    expect(realtime).toMatchObject({ route: "realtime", model: undefined, availableInV1: false });
  });

  it("reserves exactly one paid Pro attempt at eight cents", () => {
    expect(() => reserveGenerationCost("pro", 1)).toThrow(/explicit generation budget/i);
    expect(reserveGenerationCost("pro", 1, 0.08)).toEqual({
      route: "pro",
      unitCostUsd: 0.08,
      candidateCount: 1,
      reservedCostUsd: 0.08,
      maximumPaidAttempts: 1,
    });
    expect(reserveGenerationCost("pro", 4, 0.32).reservedCostUsd).toBe(0.32);
    expect(acceptedOutputCost(0.08, 1)).toBe(0.08);
    expect(acceptedTrackCost(0.08, 0.25)).toBe(0.32);
  });

  it("rejects timestamps that are unordered or outside the song", () => {
    const result = validateCompositionSpec(
      composition({
        structure: [
          { time: "0:40", section: "drop" },
          { time: "0:20", section: "intro" },
          { time: "2:50", section: "too late" },
        ],
      }),
    );
    expect(result.valid).toBe(false);
    expect(result.errors.map((entry) => entry.code)).toContain("order");
    expect(result.errors.map((entry) => entry.code)).toContain("range");

    const duplicate = validateCompositionSpec(
      composition({
        structure: [
          { time: "0:20", section: "build" },
          { time: "0:20", section: "drop" },
        ],
      }),
    );
    expect(duplicate.valid).toBe(false);
    expect(duplicate.errors).toContainEqual(expect.objectContaining({ path: "structure.1.time", code: "order" }));
  });
});
