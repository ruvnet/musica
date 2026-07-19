import type { PerformanceTemplate, SocialPreset, VisualPreset, VisualSceneMeta, VisualTemporalControls } from "./types";

export const SOCIAL_PRESETS: SocialPreset[] = [
  { id: "reel-6", label: "6s clip", width: 1080, height: 1920, fps: 30, durationSeconds: 6, videoBitsPerSecond: 10_000_000 },
  { id: "reel-9", label: "9s hook", width: 1080, height: 1920, fps: 30, durationSeconds: 9, videoBitsPerSecond: 10_000_000 },
  { id: "reel-15", label: "15s reel", width: 1080, height: 1920, fps: 30, durationSeconds: 15, videoBitsPerSecond: 10_000_000 },
  { id: "reel-30", label: "30s reel", width: 1080, height: 1920, fps: 30, durationSeconds: 30, videoBitsPerSecond: 10_000_000 },
  { id: "square-15", label: "15s square", width: 1080, height: 1080, fps: 30, durationSeconds: 15, videoBitsPerSecond: 8_000_000 },
];

export const VISUAL_SCENES: VisualSceneMeta[] = [
  { id: "tunnel", mode: "tunnel", name: "Neon Fold", label: "01", color: "#9b52ff", accent: "#ff4f86" },
  { id: "bloom", mode: "bloom", name: "Signal Bloom", label: "02", color: "#ff31d2", accent: "#35dcff" },
  { id: "terrain", mode: "terrain", name: "Spectral Field", label: "03", color: "#35dcff", accent: "#75f4c5" },
  { id: "lasergrid", mode: "tunnel", name: "Laser Grid", label: "04", color: "#ff315f", accent: "#ffe066" },
  { id: "aurora", mode: "bloom", name: "Aurora Veil", label: "05", color: "#75f4c5", accent: "#70a9ff" },
  { id: "monolith", mode: "terrain", name: "Black Monolith", label: "06", color: "#f7f5ff", accent: "#b889ff" },
  { id: "pulsefield", mode: "tunnel", name: "Pulse Field", label: "07", color: "#ff9c42", accent: "#35dcff" },
  { id: "chromawave", mode: "bloom", name: "Chroma Wave", label: "08", color: "#70a9ff", accent: "#ffb7eb" },
];

export const DEFAULT_TEMPORAL_CONTROLS: Readonly<VisualTemporalControls> = Object.freeze({
  speed: 0.5,
  strobe: 0,
  trail: 0.42,
  morph: 0.62,
  camera: 0.48,
  phase: 0,
});

export const VISUAL_PRESETS: VisualPreset[] = [
  {
    id: "clean-sync",
    name: "Clean Sync",
    scene: "terrain",
    intensity: 0.62,
    artDirection: { sculpture: 0.7, motion: 0.36, atmosphere: 0.45, ribbon: 0.58 },
    temporal: { speed: 0.42, strobe: 0, trail: 0.28, morph: 0.54, camera: 0.28, phase: 0 },
  },
  {
    id: "peak-rave",
    name: "Peak Rave",
    scene: "lasergrid",
    intensity: 0.92,
    artDirection: { sculpture: 0.68, motion: 0.92, atmosphere: 0.34, ribbon: 0.86 },
    temporal: { speed: 0.9, strobe: 0.46, trail: 0.74, morph: 0.8, camera: 0.82, phase: 0.12 },
  },
  {
    id: "cinema-fog",
    name: "Cinema Fog",
    scene: "monolith",
    intensity: 0.66,
    artDirection: { sculpture: 0.94, motion: 0.26, atmosphere: 0.88, ribbon: 0.42 },
    temporal: { speed: 0.22, strobe: 0, trail: 0.58, morph: 0.68, camera: 0.34, phase: 0.33 },
  },
  {
    id: "hyperspace",
    name: "Hyperspace",
    scene: "pulsefield",
    intensity: 0.88,
    artDirection: { sculpture: 0.56, motion: 0.96, atmosphere: 0.5, ribbon: 0.72 },
    temporal: { speed: 1, strobe: 0.22, trail: 0.82, morph: 0.74, camera: 0.96, phase: 0.68 },
  },
  {
    id: "glass-ambient",
    name: "Glass Ambient",
    scene: "aurora",
    intensity: 0.56,
    artDirection: { sculpture: 0.82, motion: 0.18, atmosphere: 0.96, ribbon: 0.62 },
    temporal: { speed: 0.18, strobe: 0, trail: 0.88, morph: 0.48, camera: 0.18, phase: 0.52 },
  },
];

export const PERFORMANCE_TEMPLATES: PerformanceTemplate[] = [
  {
    id: "moonlight-sequencer",
    name: "Moonlight Sequencer",
    description: "Public-domain Beethoven arpeggios, Lyria-sourced synth tone bank",
    bpm: 72,
    prompt: "public-domain Beethoven Moonlight Sonata first movement, C-sharp minor arpeggios, soft piano grains, dark pad, restrained hall reverb",
    scene: "monolith",
    intensity: 0.54,
    artDirection: { sculpture: 0.92, motion: 0.18, atmosphere: 0.86, ribbon: 0.36 },
    temporal: { speed: 0.16, strobe: 0, trail: 0.9, morph: 0.42, camera: 0.2, phase: 0.14 },
    tracks: {
      drums: { pattern: [0], notes: [36, 42, 38, 42], volume: 0 },
      bass: { pattern: [0, 8], notes: [25, 32, 25, 32, 24, 31, 24, 31, 23, 30, 23, 30, 21, 28, 21, 28], volume: 0.48, pan: -0.08 },
      chords: { pattern: [0, 4, 8, 12], notes: [49, 49, 48, 48, 47, 47, 45, 45, 44, 44, 42, 42, 41, 41, 44, 44], volume: 0.52, pan: -0.18 },
      lead: {
        pattern: [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        notes: [68, 61, 64, 68, 61, 64, 68, 61, 66, 68, 61, 66, 68, 60, 64, 68, 60, 64, 68, 59, 63, 68, 59, 63, 68, 57, 61, 68, 57, 61, 68, 56],
        volume: 0.5,
        pan: 0.12,
      },
      voice: { pattern: [3, 7, 11, 15], notes: [64, 63, 61, 59, 57, 56, 57, 59, 61, 63, 64, 66, 64, 61, 59, 56], volume: 0.24, pan: 0.26 },
      texture: { pattern: [0, 8], notes: [37, 44, 49, 56, 61, 56, 49, 44, 36, 43, 48, 55, 60, 55, 48, 43], volume: 0.42, pan: 0.32 },
    },
  },
  {
    id: "warehouse-techno",
    name: "Warehouse Techno",
    description: "Four-on-floor pressure, clipped bass, bright stabs",
    bpm: 132,
    prompt: "warehouse techno, metallic stabs, dry punchy drums, late-night industrial energy",
    scene: "lasergrid",
    intensity: 0.86,
    artDirection: { sculpture: 0.72, motion: 0.78, atmosphere: 0.42, ribbon: 0.66 },
    tracks: {
      drums: { pattern: [0, 2, 4, 6, 8, 10, 12, 14], notes: [36, 38, 36, 42, 36, 38, 36, 46], volume: 0.86 },
      bass: { pattern: [0, 3, 6, 8, 11, 14], notes: [28, 28, 31, 28, 34, 31, 28, 26], volume: 0.84 },
      chords: { pattern: [0, 4, 8, 12], notes: [52, 55, 59, 62, 64, 62, 59, 55], volume: 0.5, pan: -0.12 },
      lead: { pattern: [3, 7, 10, 15], notes: [64, 67, 71, 76, 74, 71, 67, 64], volume: 0.58, pan: 0.16 },
      voice: { pattern: [4, 12], notes: [60, 62, 65, 67, 70, 67, 65, 62], volume: 0.42, pan: -0.24 },
      texture: { pattern: [0, 8, 11, 15], notes: [40, 47, 52, 59, 64, 59, 52, 47], volume: 0.44, pan: 0.28 },
    },
  },
  {
    id: "liquid-breaks",
    name: "Liquid Breaks",
    description: "Syncopated drums, rolling sub, glass lead",
    bpm: 164,
    prompt: "liquid drum and bass, rolling sub bass, airy pads, fast syncopated percussion",
    scene: "chromawave",
    intensity: 0.8,
    artDirection: { sculpture: 0.64, motion: 0.88, atmosphere: 0.58, ribbon: 0.9 },
    tracks: {
      drums: { pattern: [0, 3, 4, 7, 10, 11, 14, 15], notes: [36, 38, 42, 46, 36, 38, 42, 46], volume: 0.82 },
      bass: { pattern: [0, 2, 5, 7, 8, 10, 13, 15], notes: [31, 31, 38, 36, 34, 31, 29, 31], volume: 0.8 },
      chords: { pattern: [0, 6, 10, 14], notes: [55, 58, 62, 65, 67, 65, 62, 58], volume: 0.56, pan: -0.18 },
      lead: { pattern: [1, 4, 6, 9, 12, 14], notes: [70, 74, 77, 79, 82, 79, 77, 74], volume: 0.54, pan: 0.22 },
      voice: { pattern: [2, 8, 13], notes: [62, 65, 67, 70, 74, 70, 67, 65], volume: 0.38, pan: -0.32 },
      texture: { pattern: [0, 4, 8, 12, 15], notes: [43, 50, 55, 62, 67, 62, 55, 50], volume: 0.5, pan: 0.34 },
    },
  },
  {
    id: "ambient-dub",
    name: "Ambient Dub",
    description: "Wide space, slow chord swells, sparse pulse",
    bpm: 92,
    prompt: "ambient dub, tape delay chords, spacious sub pulses, smoky late-night texture",
    scene: "aurora",
    intensity: 0.58,
    artDirection: { sculpture: 0.86, motion: 0.28, atmosphere: 0.92, ribbon: 0.48 },
    tracks: {
      drums: { pattern: [0, 6, 10, 14], notes: [36, 42, 38, 46, 36, 42, 38, 46], volume: 0.52 },
      bass: { pattern: [0, 5, 8, 13], notes: [26, 31, 33, 31, 38, 33, 31, 26], volume: 0.76 },
      chords: { pattern: [0, 8], notes: [50, 53, 57, 60, 62, 60, 57, 53], volume: 0.68, pan: -0.2 },
      lead: { pattern: [7, 15], notes: [62, 65, 69, 72, 74, 72, 69, 65], volume: 0.32, pan: 0.26 },
      voice: { pattern: [4, 12], notes: [57, 60, 62, 65, 69, 65, 62, 60], volume: 0.44, pan: -0.36 },
      texture: { pattern: [0, 3, 8, 11], notes: [38, 45, 50, 57, 62, 57, 50, 45], volume: 0.62, pan: 0.38 },
    },
  },
  {
    id: "synthwave-drive",
    name: "Synthwave Drive",
    description: "Motorik kick, octave bass, cinematic neon chords",
    bpm: 118,
    prompt: "cinematic synthwave, neon highway, octave bass, glossy analog chords",
    scene: "tunnel",
    intensity: 0.74,
    artDirection: { sculpture: 0.58, motion: 0.58, atmosphere: 0.5, ribbon: 0.7 },
    tracks: {
      drums: { pattern: [0, 2, 4, 6, 8, 10, 12, 14], notes: [36, 42, 38, 42, 36, 42, 38, 46], volume: 0.76 },
      bass: { pattern: [0, 2, 4, 6, 8, 10, 12, 14], notes: [33, 45, 33, 45, 40, 52, 38, 50], volume: 0.78 },
      chords: { pattern: [0, 4, 8, 12], notes: [57, 60, 64, 67, 69, 67, 64, 60], volume: 0.62, pan: -0.18 },
      lead: { pattern: [2, 5, 9, 13, 15], notes: [69, 72, 76, 81, 79, 76, 72, 69], volume: 0.48, pan: 0.2 },
      voice: { pattern: [4, 11], notes: [64, 67, 69, 72, 76, 72, 69, 67], volume: 0.34, pan: -0.28 },
      texture: { pattern: [0, 8, 12], notes: [45, 52, 57, 64, 69, 64, 57, 52], volume: 0.46, pan: 0.32 },
    },
  },
  {
    id: "footwork-cuts",
    name: "Footwork Cuts",
    description: "Jittered percussion, clipped subs, chopped voice",
    bpm: 156,
    prompt: "footwork and juke, jittered percussion, chopped vocal breath, tense sub movement",
    scene: "pulsefield",
    intensity: 0.9,
    artDirection: { sculpture: 0.5, motion: 0.94, atmosphere: 0.36, ribbon: 0.82 },
    tracks: {
      drums: { pattern: [0, 1, 4, 6, 7, 10, 12, 13, 15], notes: [36, 38, 42, 46, 36, 38, 42, 46], volume: 0.86 },
      bass: { pattern: [0, 3, 5, 8, 9, 12, 15], notes: [29, 29, 36, 31, 38, 29, 34, 31], volume: 0.82 },
      chords: { pattern: [2, 10], notes: [53, 56, 60, 63, 65, 63, 60, 56], volume: 0.34, pan: -0.16 },
      lead: { pattern: [1, 5, 6, 9, 11, 14], notes: [65, 68, 72, 75, 77, 75, 72, 68], volume: 0.5, pan: 0.22 },
      voice: { pattern: [0, 2, 7, 8, 12, 15], notes: [60, 63, 65, 68, 72, 68, 65, 63], volume: 0.56, pan: -0.34 },
      texture: { pattern: [4, 12], notes: [41, 48, 53, 60, 65, 60, 53, 48], volume: 0.34, pan: 0.32 },
    },
  },
  {
    id: "cinematic-pulse",
    name: "Cinematic Pulse",
    description: "Slow arps, heavy downbeats, widescreen texture",
    bpm: 104,
    prompt: "cinematic electronica, massive slow pulses, shimmering arpeggios, dramatic widescreen build",
    scene: "monolith",
    intensity: 0.68,
    artDirection: { sculpture: 0.94, motion: 0.38, atmosphere: 0.78, ribbon: 0.58 },
    tracks: {
      drums: { pattern: [0, 4, 8, 12, 14], notes: [36, 38, 42, 46, 36, 38, 42, 46], volume: 0.72 },
      bass: { pattern: [0, 4, 7, 10, 12], notes: [24, 31, 36, 31, 29, 36, 31, 24], volume: 0.8 },
      chords: { pattern: [0, 6, 12], notes: [48, 52, 55, 60, 64, 60, 55, 52], volume: 0.72, pan: -0.14 },
      lead: { pattern: [3, 5, 9, 11, 15], notes: [67, 72, 76, 79, 84, 79, 76, 72], volume: 0.46, pan: 0.18 },
      voice: { pattern: [2, 10], notes: [60, 64, 67, 72, 76, 72, 67, 64], volume: 0.42, pan: -0.28 },
      texture: { pattern: [0, 4, 8, 12], notes: [36, 43, 48, 55, 60, 55, 48, 43], volume: 0.6, pan: 0.36 },
    },
  },
  {
    id: "uk-garage-neon",
    name: "UK Garage Neon",
    description: "Shuffled kicks, swung bass replies, clipped vocal sparks",
    bpm: 136,
    prompt: "future garage and UK bass, swung drums, rubbery sub, chopped vocal hooks, luminous night city",
    scene: "bloom",
    intensity: 0.82,
    artDirection: { sculpture: 0.58, motion: 0.84, atmosphere: 0.62, ribbon: 0.88 },
    temporal: { speed: 0.72, strobe: 0.12, trail: 0.66, morph: 0.7, camera: 0.62, phase: 0.26 },
    tracks: {
      drums: { pattern: [0, 3, 4, 6, 8, 10, 11, 13, 15], notes: [36, 42, 38, 42, 36, 46, 38, 42, 36, 42, 38, 46, 36, 42, 38, 49], volume: 0.84 },
      bass: { pattern: [0, 3, 5, 8, 10, 13, 15], notes: [30, 30, 37, 34, 42, 37, 34, 29, 30, 37, 41, 37, 34, 30, 27, 29], volume: 0.82 },
      chords: { pattern: [0, 5, 11, 14], notes: [54, 57, 61, 66, 59, 62, 66, 71, 52, 56, 59, 64, 57, 61, 64, 69], volume: 0.58, pan: -0.16 },
      lead: { pattern: [2, 4, 7, 9, 12, 15], notes: [69, 73, 76, 78, 81, 78, 76, 73, 71, 76, 78, 83, 81, 78, 76, 73, 69, 71, 76, 78, 85, 83, 78, 76], volume: 0.5, pan: 0.22 },
      voice: { pattern: [1, 6, 9, 13], notes: [61, 64, 66, 69, 73, 69, 66, 64, 59, 61, 66, 69, 73, 76, 73, 69], volume: 0.5, pan: -0.28 },
      texture: { pattern: [0, 4, 8, 12, 15], notes: [42, 49, 54, 61, 66, 61, 54, 49, 47, 54, 59, 66, 71, 66, 59, 54], volume: 0.44, pan: 0.32 },
    },
  },
  {
    id: "afro-cosmic-house",
    name: "Afro Cosmic House",
    description: "Polyrhythmic percussion, warm bass, call-response plucks",
    bpm: 122,
    prompt: "afro house, polyrhythmic percussion, warm analog bass, call and response plucks, cosmic club atmosphere",
    scene: "terrain",
    intensity: 0.78,
    artDirection: { sculpture: 0.7, motion: 0.7, atmosphere: 0.56, ribbon: 0.78 },
    temporal: { speed: 0.56, strobe: 0.04, trail: 0.52, morph: 0.66, camera: 0.48, phase: 0.44 },
    tracks: {
      drums: { pattern: [0, 2, 3, 5, 6, 8, 10, 11, 13, 14], notes: [36, 42, 46, 38, 42, 46, 36, 42, 38, 46, 42, 36, 46, 38, 42, 49], volume: 0.84 },
      bass: { pattern: [0, 3, 6, 8, 10, 13], notes: [33, 33, 40, 38, 36, 33, 45, 40, 31, 38, 43, 38, 36, 33, 31, 28], volume: 0.8 },
      chords: { pattern: [0, 6, 10, 14], notes: [57, 60, 64, 69, 64, 67, 72, 76, 55, 59, 62, 67, 52, 55, 60, 64], volume: 0.54, pan: -0.18 },
      lead: { pattern: [1, 4, 7, 9, 12, 14], notes: [72, 76, 79, 84, 81, 79, 76, 72, 74, 79, 83, 86, 84, 81, 79, 76, 72, 74, 76, 79, 84, 88, 86, 84], volume: 0.52, pan: 0.24 },
      voice: { pattern: [2, 7, 10, 15], notes: [64, 67, 72, 76, 79, 76, 72, 67, 62, 67, 71, 74, 79, 74, 71, 67], volume: 0.4, pan: -0.32 },
      texture: { pattern: [0, 4, 8, 11, 14], notes: [45, 52, 57, 64, 69, 64, 57, 52, 43, 50, 55, 62, 67, 62, 55, 50], volume: 0.5, pan: 0.36 },
    },
  },
  {
    id: "idm-crystalline",
    name: "IDM Crystalline",
    description: "Asymmetric percussion, icy FM tones, shifting micro-melody",
    bpm: 148,
    prompt: "crystalline IDM, asymmetric percussion, glass FM synths, glitch detail, emotional micro melodies",
    scene: "chromawave",
    intensity: 0.84,
    artDirection: { sculpture: 0.76, motion: 0.9, atmosphere: 0.52, ribbon: 0.86 },
    temporal: { speed: 0.78, strobe: 0.18, trail: 0.7, morph: 0.92, camera: 0.58, phase: 0.74 },
    tracks: {
      drums: { pattern: [0, 1, 4, 5, 7, 9, 10, 12, 14, 15], notes: [36, 42, 38, 46, 42, 36, 38, 42, 46, 42, 36, 49, 38, 42, 46, 42], volume: 0.78 },
      bass: { pattern: [0, 2, 5, 8, 11, 13, 15], notes: [25, 32, 37, 34, 29, 36, 41, 39, 27, 34, 39, 36, 32, 29, 37, 34], volume: 0.74 },
      chords: { pattern: [0, 3, 8, 13], notes: [49, 53, 56, 61, 58, 61, 65, 68, 54, 58, 61, 66, 51, 56, 60, 63], volume: 0.48, pan: -0.2 },
      lead: { pattern: [1, 2, 6, 7, 10, 13, 15], notes: [73, 80, 78, 85, 82, 80, 75, 78, 70, 77, 75, 82, 80, 77, 73, 75, 68, 75, 80, 82, 87, 85, 80, 77], volume: 0.54, pan: 0.18 },
      voice: { pattern: [4, 9, 12], notes: [61, 65, 68, 73, 77, 73, 68, 65, 58, 61, 66, 70, 73, 70, 66, 61], volume: 0.34, pan: -0.3 },
      texture: { pattern: [0, 2, 6, 8, 12, 15], notes: [37, 44, 49, 56, 61, 56, 49, 44, 42, 49, 54, 61, 66, 61, 54, 49], volume: 0.48, pan: 0.34 },
    },
  },
  {
    id: "hyperpop-rush",
    name: "Hyperpop Rush",
    description: "Glossy supersaw hooks, pitched voice cuts, restless club drums",
    bpm: 174,
    prompt: "hyperpop club rush, glossy supersaw hooks, pitched vocal cuts, huge bright drums, maximal electronic energy",
    scene: "pulsefield",
    intensity: 0.94,
    artDirection: { sculpture: 0.52, motion: 0.98, atmosphere: 0.44, ribbon: 0.96 },
    temporal: { speed: 0.96, strobe: 0.34, trail: 0.8, morph: 0.82, camera: 0.9, phase: 0.62 },
    tracks: {
      drums: { pattern: [0, 2, 4, 6, 7, 8, 10, 12, 13, 14, 15], notes: [36, 42, 38, 46, 36, 42, 38, 49, 36, 42, 38, 46, 36, 46, 38, 49], volume: 0.88 },
      bass: { pattern: [0, 2, 4, 7, 8, 10, 12, 15], notes: [32, 44, 39, 47, 36, 48, 43, 51, 34, 46, 41, 53, 39, 51, 44, 56], volume: 0.78 },
      chords: { pattern: [0, 4, 8, 12], notes: [56, 60, 63, 68, 63, 67, 70, 75, 58, 62, 65, 70, 61, 65, 68, 73], volume: 0.62, pan: -0.16 },
      lead: { pattern: [1, 3, 5, 7, 9, 11, 13, 15], notes: [80, 84, 87, 92, 99, 96, 92, 87, 82, 87, 91, 94, 101, 99, 94, 91, 84, 87, 92, 96, 104, 101, 96, 92], volume: 0.58, pan: 0.2 },
      voice: { pattern: [0, 3, 6, 9, 12, 15], notes: [68, 72, 75, 80, 87, 84, 80, 75, 70, 75, 79, 82, 89, 87, 82, 79], volume: 0.52, pan: -0.34 },
      texture: { pattern: [0, 4, 6, 8, 12, 14], notes: [44, 51, 56, 63, 68, 63, 56, 51, 46, 53, 58, 65, 70, 65, 58, 53], volume: 0.4, pan: 0.3 },
    },
  },
];

export function visualSceneById(id: string): VisualSceneMeta {
  return VISUAL_SCENES.find((scene) => scene.id === id) ?? VISUAL_SCENES[0];
}

export function performanceTemplateById(id: string): PerformanceTemplate {
  return PERFORMANCE_TEMPLATES.find((template) => template.id === id) ?? PERFORMANCE_TEMPLATES[0];
}

export const DEFAULT_PERFORMANCE_TEMPLATE_ID = "afro-cosmic-house";

export const DEFAULT_MIDI_SONG_BANK_IDS = [
  "afro-cosmic-house",
  "uk-garage-neon",
  "moonlight-sequencer",
  "idm-crystalline",
  "liquid-breaks",
  "synthwave-drive",
  "hyperpop-rush",
] as const;

export function defaultPerformanceTemplate(): PerformanceTemplate {
  return performanceTemplateById(DEFAULT_PERFORMANCE_TEMPLATE_ID);
}

export function midiSongBankTemplates(): PerformanceTemplate[] {
  return DEFAULT_MIDI_SONG_BANK_IDS.map(performanceTemplateById);
}
