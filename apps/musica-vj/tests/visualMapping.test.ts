import { describe, expect, it } from "vitest";
import {
  frequencyBandEnergy,
  mapVisualAudioResponse,
  normalizeAnimationStyle,
  normalizeArtDirection,
  normalizeTemporalControls,
  sceneForMeasuredSection,
  VISUAL_ANIMATION_STYLES,
} from "../src/visual/VisualEngine";

describe("measured section scene mapping", () => {
  it("maps locally detected sections to deterministic scene transitions", () => {
    expect(sceneForMeasuredSection("intro", 0)).toBe("terrain");
    expect(sceneForMeasuredSection("build", 1)).toBe("bloom");
    expect(sceneForMeasuredSection("drop", 2)).toBe("tunnel");
    expect(sceneForMeasuredSection("breakdown", 3)).toBe("bloom");
    expect(sceneForMeasuredSection("outro", 4)).toBe("terrain");
  });

  it("uses frequency-aware bass bins rather than a broad FFT percentage", () => {
    const frequency = new Uint8Array(1_024);
    const bassBin = Math.floor((80 / 24_000) * frequency.length);
    const midBin = Math.floor((1_000 / 24_000) * frequency.length);
    frequency[bassBin] = 255;
    frequency[midBin] = 255;
    expect(frequencyBandEnergy(frequency, 48_000, 30, 180)).toBeGreaterThan(0);
    frequency[bassBin] = 0;
    expect(frequencyBandEnergy(frequency, 48_000, 30, 180)).toBe(0);
  });

  it("maps bass, beats, and high-frequency energy to distinct visual controls", () => {
    const quiet = mapVisualAudioResponse(0, 0, 0, 0, 1);
    const active = mapVisualAudioResponse(1, 1, 1, 1, 1);
    expect(active.cameraDisplacement).toBeGreaterThan(quiet.cameraDisplacement);
    expect(active.radialPulse).toBeGreaterThan(quiet.radialPulse);
    expect(active.particleCount).toBeGreaterThan(quiet.particleCount);
    expect(active.particleCount).toBe(7_000);
    expect(active.spectralHeight).toBeGreaterThan(quiet.spectralHeight);
    expect(active.waveformAmplitude).toBeGreaterThan(quiet.waveformAmplitude);
    expect(active.hazeOpacity).toBeGreaterThan(quiet.hazeOpacity);
    expect(active.flowCurl).toBeGreaterThan(quiet.flowCurl);
    expect(active.beamIntensity).toBeGreaterThan(quiet.beamIntensity);
    expect(active.afterimageOpacity).toBeGreaterThan(quiet.afterimageOpacity);
  });

  it("bounds the Signal Bloom design response for hostile analyser values", () => {
    const response = mapVisualAudioResponse(Number.POSITIVE_INFINITY, -4, -4, 9, 5);
    expect(response.cameraDisplacement).toBe(1.15);
    expect(response.radialPulse).toBe(1.6);
    expect(response.particleCount).toBe(1_750);
    expect(response.spectralHeight).toBe(1.2);
    expect(response.waveformAmplitude).toBe(1);
    expect(response.hazeOpacity).toBe(0.1);
    expect(response.flowCurl).toBe(0.6);
    expect(response.beamIntensity).toBe(0.71);
    expect(response.afterimageOpacity).toBeCloseTo(0.305, 8);
  });

  it("keeps live artist macros inside performance-safe ranges", () => {
    expect(normalizeArtDirection({
      sculpture: 2,
      motion: -1,
      atmosphere: Number.NaN,
      ribbon: 0.7,
    })).toEqual({
      sculpture: 1,
      motion: 0,
      atmosphere: 0.5,
      ribbon: 0.7,
    });
  });

  it("keeps temporal VJ controls inside performance-safe ranges", () => {
    expect(normalizeTemporalControls({
      speed: 4,
      strobe: -1,
      trail: 0.8,
      morph: Number.NaN,
      camera: 0.25,
      phase: 2,
    })).toEqual({
      speed: 1,
      strobe: 0,
      trail: 0.8,
      morph: 0.5,
      camera: 0.25,
      phase: 1,
    });
  });

  it("normalizes visual animation styles to supported options", () => {
    expect(VISUAL_ANIMATION_STYLES.map((style) => style.id)).toEqual([
      "flow",
      "orbit",
      "warp",
      "shards",
      "scan",
      "minimal",
    ]);
    expect(normalizeAnimationStyle("warp")).toBe("warp");
    expect(normalizeAnimationStyle("unknown")).toBe("flow");
    expect(normalizeAnimationStyle(undefined)).toBe("flow");
  });
});
