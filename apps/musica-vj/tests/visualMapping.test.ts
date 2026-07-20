import { describe, expect, it } from "vitest";
import {
  computeSpectralFlux,
  followEnvelope,
  mapFeedbackResponse,
  frequencyBandEnergy,
  mapVisualAudioResponse,
  normalizeAnimationStyle,
  normalizeArtDirection,
  normalizeVisualColorControls,
  normalizeTemporalControls,
  SCENE_CHARACTERS,
  sceneCharacterById,
  sceneForMeasuredSection,
  VISUAL_ANIMATION_STYLES,
} from "../src/visual/VisualEngine";
import { VISUAL_SCENES } from "../src/core/presets";

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

  it("normalizes palette look controls", () => {
    expect(normalizeVisualColorControls({ palette: "prism", hue: -1, saturation: 2, contrast: 0.4, diversity: Number.NaN })).toEqual({
      palette: "prism",
      hue: 0,
      saturation: 1,
      contrast: 0.4,
      diversity: 0.5,
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

  it("detects onsets through positive spectral flux only", () => {
    const previous = new Uint8Array([100, 100, 100, 100]);
    const rising = new Uint8Array([180, 160, 140, 120]);
    const falling = new Uint8Array([20, 40, 60, 80]);
    expect(computeSpectralFlux(previous, previous)).toBe(0);
    expect(computeSpectralFlux(rising, previous)).toBeGreaterThan(0);
    expect(computeSpectralFlux(falling, previous)).toBe(0);
    expect(computeSpectralFlux(new Uint8Array(0), new Uint8Array(0))).toBe(0);
    expect(computeSpectralFlux(rising, new Uint8Array(2))).toBe(0);
    expect(computeSpectralFlux(new Uint8Array(64).fill(255), new Uint8Array(64))).toBe(1);
  });

  it("follows energy with a fast attack and a slower release", () => {
    const attacked = followEnvelope(0, 1, 0.6, 0.14);
    expect(attacked).toBeCloseTo(0.6, 8);
    const released = followEnvelope(attacked, 0, 0.6, 0.14);
    expect(released).toBeGreaterThan(attacked - attacked * 0.2);
    expect(released).toBeLessThan(attacked);
    expect(followEnvelope(0.5, 1, 4, 0.1)).toBe(1);
    expect(followEnvelope(0.5, 0, 0.6, -3)).toBe(0.5);
  });

  it("maps trail, motion, and beats into a bounded feedback echo", () => {
    const off = mapFeedbackResponse(0, 0, 0, 0.5);
    expect(off.damp).toBe(0);
    expect(off.zoom).toBeGreaterThan(1);
    expect(off.zoom).toBeLessThan(1.02);
    expect(off.rotate).toBe(0);
    const long = mapFeedbackResponse(1, 1, 1, 1);
    expect(long.damp).toBeGreaterThan(0.9);
    expect(long.damp).toBeLessThan(1);
    expect(long.zoom).toBeGreaterThan(off.zoom);
    expect(long.rotate).toBeGreaterThan(0);
    const hostile = mapFeedbackResponse(9, -3, Number.POSITIVE_INFINITY, -1);
    expect(hostile.damp).toBeLessThan(1);
    expect(hostile.zoom).toBeLessThan(1.05);
    expect(Math.abs(hostile.rotate)).toBeLessThan(0.02);
  });

  it("gives every visual scene a distinct rendering character", () => {
    for (const scene of VISUAL_SCENES) expect(SCENE_CHARACTERS[scene.id]).toBeDefined();
    expect(sceneCharacterById("lasergrid").travel).toBeGreaterThan(sceneCharacterById("aurora").travel);
    expect(sceneCharacterById("aurora").haze).toBeGreaterThan(sceneCharacterById("lasergrid").haze);
    expect(sceneCharacterById("monolith").terrainAmp).toBeLessThan(sceneCharacterById("terrain").terrainAmp);
    expect(sceneCharacterById("unknown-scene")).toBe(SCENE_CHARACTERS.bloom);
    const signatures = new Set(VISUAL_SCENES.map((scene) => JSON.stringify(SCENE_CHARACTERS[scene.id])));
    expect(signatures.size).toBe(VISUAL_SCENES.length);
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
