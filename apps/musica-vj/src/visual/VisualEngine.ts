import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { Pass, FullScreenQuad } from "three/examples/jsm/postprocessing/Pass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { AudioEngine } from "../audio/AudioEngine";
import type { AudioAnalysisResult, MeasuredSectionType } from "../audio/audioAnalysis";
import { DEFAULT_TEMPORAL_CONTROLS, visualSceneById } from "../core/presets";
import type { AudioMetrics, TrackId, VisualColorControls, VisualPaletteId, VisualSceneId, VisualTemporalControls } from "../core/types";

export interface RenderStats {
  fps: number;
  frameTimeMs: number;
  pixelRatio: number;
  quality: "adaptive" | "export";
}

type StatsListener = (stats: RenderStats) => void;
type SceneListener = (scene: VisualSceneId) => void;
const PARTICLE_COUNT = 7_000;
const FLOW_PARTICLE_COUNT = 12_000;
const VOLUMETRIC_BEAM_COUNT = 28;
const AFTERIMAGE_PANEL_COUNT = 10;
const TERRAIN_SEGMENTS = 64;
const SPECTRAL_TRAIL_COUNT = 18;
const SPECTRAL_POINTS = 128;
const WAVEFORM_POINTS = 256;
const ATMOSPHERE_PARTICLES = 1_600;
const FLOOR_Y = -3.45;

export interface VisualAudioResponse {
  cameraDisplacement: number;
  radialPulse: number;
  particleCount: number;
  spectralHeight: number;
  waveformAmplitude: number;
  hazeOpacity: number;
  flowCurl: number;
  beamIntensity: number;
  afterimageOpacity: number;
}

export interface VisualArtDirection {
  sculpture: number;
  motion: number;
  atmosphere: number;
  ribbon: number;
}

export const VISUAL_ANIMATION_STYLES = [
  { id: "flow", label: "FLOW", description: "liquid particle streams" },
  { id: "orbit", label: "ORBIT", description: "rotating camera sculpture" },
  { id: "warp", label: "WARP", description: "forward tunnel thrust" },
  { id: "shards", label: "SHARDS", description: "fragmented crystalline hits" },
  { id: "scan", label: "SCAN", description: "raster sweeps and wave scans" },
  { id: "minimal", label: "MIN", description: "restrained architectural motion" },
] as const;

export type VisualAnimationStyle = (typeof VISUAL_ANIMATION_STYLES)[number]["id"];

interface AnimationStyleFactors {
  styleIndex: number;
  cameraOrbit: number;
  cameraPush: number;
  tunnelTravel: number;
  tunnelTwist: number;
  bloomScale: number;
  bloomSpin: number;
  flowScale: number;
  flowDraw: number;
  beamScale: number;
  afterimageScale: number;
  terrainWave: number;
  waveformScale: number;
  coreSpin: number;
  opacityScale: number;
}

export const DEFAULT_ART_DIRECTION: Readonly<VisualArtDirection> = Object.freeze({
  sculpture: 0.78,
  motion: 0.46,
  atmosphere: 0.62,
  ribbon: 0.72,
});

export const VISUAL_COLOR_PALETTES: ReadonlyArray<{ id: VisualPaletteId; label: string; color?: string; accent?: string }> = [
  { id: "scene", label: "SCENE" },
  { id: "neon", label: "NEON", color: "#d64ba8", accent: "#39bdd0" },
  { id: "ember", label: "EMBER", color: "#e96d4b", accent: "#d9c45d" },
  { id: "ice", label: "ICE", color: "#4b9fc8", accent: "#9bd8c5" },
  { id: "prism", label: "PRISM", color: "#a95bd5", accent: "#49c893" },
  { id: "mono", label: "MONO", color: "#c6ccd5", accent: "#697583" },
];

export const DEFAULT_VISUAL_COLOR_CONTROLS: Readonly<VisualColorControls> = Object.freeze({
  palette: "scene",
  hue: 0.5,
  saturation: 0.64,
  contrast: 0.48,
  diversity: 0.58,
});

export function normalizeVisualColorControls(controls: VisualColorControls): VisualColorControls {
  const bounded = (value: number) => Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.5;
  return {
    palette: VISUAL_COLOR_PALETTES.some((palette) => palette.id === controls.palette) ? controls.palette : "scene",
    hue: bounded(controls.hue),
    saturation: bounded(controls.saturation),
    contrast: bounded(controls.contrast),
    diversity: bounded(controls.diversity),
  };
}

export function normalizeArtDirection(direction: VisualArtDirection): VisualArtDirection {
  const bounded = (value: number) => Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.5;
  return {
    sculpture: bounded(direction.sculpture),
    motion: bounded(direction.motion),
    atmosphere: bounded(direction.atmosphere),
    ribbon: bounded(direction.ribbon),
  };
}

export function normalizeTemporalControls(controls: VisualTemporalControls): VisualTemporalControls {
  const bounded = (value: number) => Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0.5;
  return {
    speed: bounded(controls.speed),
    strobe: bounded(controls.strobe),
    trail: bounded(controls.trail),
    morph: bounded(controls.morph),
    camera: bounded(controls.camera),
    phase: bounded(controls.phase),
  };
}

export function normalizeAnimationStyle(style: string | undefined): VisualAnimationStyle {
  return VISUAL_ANIMATION_STYLES.some((candidate) => candidate.id === style) ? style as VisualAnimationStyle : "flow";
}

export interface FeedbackResponse {
  damp: number;
  zoom: number;
  rotate: number;
}

export interface BloomSettings {
  strength: number;
  radius: number;
  threshold: number;
}

export const DEFAULT_BLOOM_SETTINGS: Readonly<BloomSettings> = Object.freeze({
  strength: 0.72,
  radius: 0.86,
  threshold: 0.18,
});

export function normalizeBloomSettings(settings: Partial<BloomSettings>): BloomSettings {
  const bounded = (value: number | undefined, fallback: number, maximum: number) =>
    typeof value === "number" && Number.isFinite(value) ? Math.max(0, Math.min(maximum, value)) : fallback;
  return {
    strength: bounded(settings.strength, DEFAULT_BLOOM_SETTINGS.strength, 2.5),
    radius: bounded(settings.radius, DEFAULT_BLOOM_SETTINGS.radius, 1.5),
    threshold: bounded(settings.threshold, DEFAULT_BLOOM_SETTINGS.threshold, 1),
  };
}

export function mapFeedbackResponse(trail: number, motion: number, beatPulse: number, morph: number): FeedbackResponse {
  const boundedTrail = Math.max(0, Math.min(1, trail));
  const boundedMotion = Math.max(0, Math.min(1, motion));
  const boundedBeat = Math.max(0, Math.min(1, beatPulse));
  const boundedMorph = Math.max(0, Math.min(1, morph));
  return {
    damp: boundedTrail * (0.62 + boundedTrail * 0.32),
    zoom: 1 + (0.0035 + boundedBeat * 0.014) * (0.3 + boundedMotion),
    rotate: (boundedMorph - 0.5) * 0.011 * (0.25 + boundedMotion),
  };
}

class FeedbackPass extends Pass {
  private readonly quad: FullScreenQuad;
  private readonly composeMaterial: THREE.ShaderMaterial;
  private readonly copyMaterial: THREE.ShaderMaterial;
  private historyTarget: THREE.WebGLRenderTarget;
  private scratchTarget: THREE.WebGLRenderTarget;
  damp = 0;
  zoom = 1;
  rotate = 0;

  constructor(width: number, height: number) {
    super();
    this.historyTarget = new THREE.WebGLRenderTarget(width, height, { type: THREE.HalfFloatType });
    this.scratchTarget = new THREE.WebGLRenderTarget(width, height, { type: THREE.HalfFloatType });
    this.composeMaterial = new THREE.ShaderMaterial({
      uniforms: {
        tDiffuse: { value: null },
        tHistory: { value: null },
        uDamp: { value: 0 },
        uZoom: { value: 1 },
        uRotate: { value: 0 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        uniform sampler2D tHistory;
        uniform float uDamp;
        uniform float uZoom;
        uniform float uRotate;
        varying vec2 vUv;
        void main() {
          vec2 centered = vUv - 0.5;
          float c = cos(uRotate);
          float s = sin(uRotate);
          vec2 warped = mat2(c, -s, s, c) * centered / uZoom + 0.5;
          vec4 current = texture2D(tDiffuse, vUv);
          vec4 echo = texture2D(tHistory, warped) * uDamp;
          gl_FragColor = max(current, echo);
        }
      `,
      depthTest: false,
      depthWrite: false,
    });
    this.copyMaterial = new THREE.ShaderMaterial({
      uniforms: { tDiffuse: { value: null } },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D tDiffuse;
        varying vec2 vUv;
        void main() {
          gl_FragColor = texture2D(tDiffuse, vUv);
        }
      `,
      depthTest: false,
      depthWrite: false,
    });
    this.quad = new FullScreenQuad(this.composeMaterial);
  }

  override setSize(width: number, height: number): void {
    this.historyTarget.setSize(width, height);
    this.scratchTarget.setSize(width, height);
  }

  override render(
    renderer: THREE.WebGLRenderer,
    writeBuffer: THREE.WebGLRenderTarget,
    readBuffer: THREE.WebGLRenderTarget,
  ): void {
    this.composeMaterial.uniforms.tDiffuse.value = readBuffer.texture;
    this.composeMaterial.uniforms.tHistory.value = this.historyTarget.texture;
    this.composeMaterial.uniforms.uDamp.value = this.damp;
    this.composeMaterial.uniforms.uZoom.value = this.zoom;
    this.composeMaterial.uniforms.uRotate.value = this.rotate;
    this.quad.material = this.composeMaterial;
    renderer.setRenderTarget(this.scratchTarget);
    this.quad.render(renderer);

    this.copyMaterial.uniforms.tDiffuse.value = this.scratchTarget.texture;
    this.quad.material = this.copyMaterial;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    this.quad.render(renderer);

    const previousHistory = this.historyTarget;
    this.historyTarget = this.scratchTarget;
    this.scratchTarget = previousHistory;
  }

  override dispose(): void {
    this.historyTarget.dispose();
    this.scratchTarget.dispose();
    this.composeMaterial.dispose();
    this.copyMaterial.dispose();
    this.quad.dispose();
  }
}

export interface VisualSceneCharacter {
  travel: number;
  twist: number;
  swirl: number;
  beams: number;
  haze: number;
  terrainAmp: number;
  fogDensity: number;
  exposure: number;
  coreScale: number;
}

export const SCENE_CHARACTERS: Readonly<Record<VisualSceneId, VisualSceneCharacter>> = Object.freeze({
  tunnel: { travel: 1, twist: 1.2, swirl: 0.9, beams: 1, haze: 0.9, terrainAmp: 1, fogDensity: 0.021, exposure: 1, coreScale: 1 },
  bloom: { travel: 0.9, twist: 0.9, swirl: 1, beams: 0.95, haze: 1, terrainAmp: 1, fogDensity: 0.021, exposure: 1, coreScale: 1 },
  terrain: { travel: 0.85, twist: 0.8, swirl: 0.85, beams: 0.8, haze: 0.95, terrainAmp: 1, fogDensity: 0.023, exposure: 0.97, coreScale: 0.9 },
  lasergrid: { travel: 1.55, twist: 0.5, swirl: 0.75, beams: 1.5, haze: 0.45, terrainAmp: 1, fogDensity: 0.015, exposure: 1.08, coreScale: 0.85 },
  aurora: { travel: 0.6, twist: 0.65, swirl: 0.55, beams: 0.5, haze: 1.55, terrainAmp: 0.8, fogDensity: 0.03, exposure: 0.9, coreScale: 0.7 },
  monolith: { travel: 0.55, twist: 0.6, swirl: 0.6, beams: 0.6, haze: 1.35, terrainAmp: 0.55, fogDensity: 0.031, exposure: 0.84, coreScale: 0.5 },
  pulsefield: { travel: 1.3, twist: 1.55, swirl: 1.15, beams: 1.25, haze: 0.75, terrainAmp: 1.15, fogDensity: 0.018, exposure: 1.05, coreScale: 1.35 },
  chromawave: { travel: 0.95, twist: 1.1, swirl: 1.5, beams: 0.85, haze: 1.1, terrainAmp: 1.05, fogDensity: 0.024, exposure: 0.98, coreScale: 1.1 },
  oscilloscope: { travel: 0.7, twist: 0.7, swirl: 0.8, beams: 0.35, haze: 0.4, terrainAmp: 0.8, fogDensity: 0.012, exposure: 1.02, coreScale: 0.6 },
});

export function sceneCharacterById(id: string): VisualSceneCharacter {
  return SCENE_CHARACTERS[id as VisualSceneId] ?? SCENE_CHARACTERS.bloom;
}

export function computeSpectralFlux(current: Uint8Array, previous: Uint8Array): number {
  if (current.length === 0 || current.length !== previous.length) return 0;
  let flux = 0;
  for (let index = 0; index < current.length; index += 1) {
    const rise = current[index] - previous[index];
    if (rise > 0) flux += rise;
  }
  return Math.min(1, (flux / (current.length * 255)) * 6);
}

export function followEnvelope(previous: number, target: number, attack: number, release: number): number {
  const coefficient = target > previous ? attack : release;
  const bounded = Math.max(0, Math.min(1, coefficient));
  return previous + (target - previous) * bounded;
}

export function frequencyBandEnergy(
  frequency: Uint8Array,
  sampleRateHz: number,
  lowerHz: number,
  upperHz: number,
): number {
  if (frequency.length === 0 || sampleRateHz <= 0 || upperHz <= lowerHz) return 0;
  const nyquist = sampleRateHz / 2;
  const from = Math.max(0, Math.min(frequency.length - 1, Math.floor((lowerHz / nyquist) * frequency.length)));
  const to = Math.max(from + 1, Math.min(frequency.length, Math.ceil((upperHz / nyquist) * frequency.length)));
  let sum = 0;
  for (let index = from; index < to; index += 1) sum += frequency[index];
  return sum / (to - from) / 255;
}

export function mapVisualAudioResponse(
  bassEnergy: number,
  midEnergy: number,
  highEnergy: number,
  beatPulse: number,
  intensity: number,
): VisualAudioResponse {
  const boundedIntensity = Math.max(0.05, Math.min(1, intensity));
  const bass = Math.max(0, Math.min(1, bassEnergy));
  const mid = Math.max(0, Math.min(1, midEnergy));
  const high = Math.max(0, Math.min(1, highEnergy));
  const beat = Math.max(0, Math.min(1, beatPulse));
  return {
    cameraDisplacement: bass * 1.15 * boundedIntensity,
    radialPulse: beat * 1.6 * boundedIntensity,
    particleCount: Math.round(PARTICLE_COUNT * (0.25 + high * 0.75)),
    spectralHeight: (0.32 + bass * 0.88) * boundedIntensity,
    waveformAmplitude: (0.16 + beat * 0.84) * boundedIntensity,
    hazeOpacity: (0.1 + high * 0.42) * boundedIntensity,
    flowCurl: (0.18 + mid * 0.72 + beat * 0.42) * boundedIntensity,
    beamIntensity: (0.05 + bass * 0.32 + high * 0.24 + beat * 0.34) * boundedIntensity,
    afterimageOpacity: (0.025 + beat * 0.28 + high * 0.08) * boundedIntensity,
  };
}

export function sceneForMeasuredSection(type: MeasuredSectionType, sectionIndex: number): VisualSceneId {
  switch (type) {
    case "intro":
    case "outro":
      return "terrain";
    case "build":
    case "breakdown":
      return "bloom";
    case "drop":
      return "tunnel";
    case "groove":
      return sectionIndex % 2 === 0 ? "tunnel" : "bloom";
  }
}

export class VisualEngine {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly composer: EffectComposer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(55, 1, 0.1, 120);
  private readonly overlayScene = new THREE.Scene();
  private readonly overlayCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10);
  private readonly clock = new THREE.Clock();
  private readonly tunnelGroup = new THREE.Group();
  private readonly bloomGroup = new THREE.Group();
  private readonly terrainGroup = new THREE.Group();
  private readonly signalBloomGroup = new THREE.Group();
  private readonly atmosphereGroup = new THREE.Group();
  private readonly kineticFieldGroup = new THREE.Group();
  private readonly volumetricGroup = new THREE.Group();
  private readonly afterimageGroup = new THREE.Group();
  private readonly floorGroup = new THREE.Group();
  private readonly bloomPass: UnrealBloomPass;
  private feedbackBoost = 0;
  private readonly scopeGroup = new THREE.Group();
  private readonly scopeLines: Array<THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>> = [];
  private scopeSpikes?: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  private readonly feedbackPass: FeedbackPass;
  private readonly tunnelRings: THREE.Mesh[] = [];
  private readonly spectralTrails: Array<THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>> = [];
  private readonly spectralReflections: Array<THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>> = [];
  private readonly bloomMaterial: THREE.ShaderMaterial;
  private readonly flowMaterial: THREE.ShaderMaterial;
  private readonly beamMaterial: THREE.ShaderMaterial;
  private readonly afterimageMaterials: THREE.ShaderMaterial[] = [];
  private readonly hazeMaterial: THREE.ShaderMaterial;
  private readonly atmosphereMaterial: THREE.PointsMaterial;
  private bloomPoints?: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>;
  private flowPoints?: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>;
  private readonly volumetricBeams: Array<THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>> = [];
  private readonly waveformRibbon: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  private readonly waveformGlow: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  private readonly waveformReflection: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  private readonly terrain: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  private readonly core: THREE.Mesh<THREE.IcosahedronGeometry, THREE.MeshStandardMaterial>;
  private readonly brandingTexture: THREE.CanvasTexture;
  private readonly branding: THREE.Sprite;
  private readonly resizeObserver: ResizeObserver;
  private readonly statsListeners = new Set<StatsListener>();
  private readonly sceneListeners = new Set<SceneListener>();
  private animationFrame?: number;
  private currentScene: VisualSceneId = "bloom";
  private intensity = 0.72;
  private artDirection: VisualArtDirection = { ...DEFAULT_ART_DIRECTION };
  private colorControls: VisualColorControls = { ...DEFAULT_VISUAL_COLOR_CONTROLS };
  private temporal: VisualTemporalControls = { ...DEFAULT_TEMPORAL_CONTROLS };
  private animationStyle: VisualAnimationStyle = "flow";
  private pixelRatio = Math.min(window.devicePixelRatio || 1, 1.75);
  private sceneCharacter: VisualSceneCharacter = SCENE_CHARACTERS.bloom;
  private previousFrequency = new Uint8Array(0);
  private fluxEnvelope = 0;
  private lowEnvelope = 0;
  private midEnvelope = 0;
  private highEnvelope = 0;
  private frameTimeEma = 16.67;
  private lastFrameAt = performance.now();
  private statsAt = performance.now();
  private framesSinceStats = 0;
  private stableFrames = 0;
  private exportLock?: { width: number; height: number };
  private analyzedTrack?: { trackId: TrackId; analysis: AudioAnalysisResult; sectionIndex: number };

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly audio: AudioEngine,
  ) {
    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.86;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.renderer.autoClear = false;

    const branding = this.createBranding();
    this.brandingTexture = branding.texture;
    this.branding = branding.sprite;
    this.branding.visible = false;
    this.overlayCamera.position.z = 1;
    this.overlayScene.add(this.branding);

    this.scene.background = new THREE.Color(0x020107);
    this.scene.fog = new THREE.FogExp2(0x05010d, 0.021);
    this.camera.position.set(0, 0.35, 9);
    this.camera.lookAt(0, 0, -10);

    this.scene.add(new THREE.AmbientLight(0x755dff, 0.28));
    const key = new THREE.PointLight(0xff3f91, 45, 45);
    key.position.set(4, 5, 4);
    this.scene.add(key);
    const fill = new THREE.PointLight(0x44ffd1, 35, 40);
    fill.position.set(-5, -2, 1);
    this.scene.add(fill);

    this.bloomMaterial = this.createBloomMaterial();
    this.flowMaterial = this.createFlowMaterial();
    this.beamMaterial = this.createBeamMaterial();
    this.hazeMaterial = this.createHazeMaterial();
    this.atmosphereMaterial = this.createAtmosphere();
    const waveform = this.createWaveformRibbon();
    this.waveformRibbon = waveform.ribbon;
    this.waveformGlow = waveform.glow;
    this.waveformReflection = waveform.reflection;
    this.terrain = this.createTerrain();
    this.core = this.createCore();
    this.createTunnel();
    this.createBloom();
    this.createKineticField();
    this.createVolumetricBeams();
    this.createAfterimagePanels();
    this.createSignalBloom();
    this.createFloor();
    this.createScope();
    this.terrainGroup.add(this.terrain);
    this.scene.add(
      this.floorGroup,
      this.tunnelGroup,
      this.bloomGroup,
      this.terrainGroup,
      this.signalBloomGroup,
      this.kineticFieldGroup,
      this.volumetricGroup,
      this.afterimageGroup,
      this.atmosphereGroup,
      this.scopeGroup,
      this.core,
    );

    const size = new THREE.Vector2(Math.max(1, canvas.clientWidth), Math.max(1, canvas.clientHeight));
    this.composer = new EffectComposer(this.renderer);
    this.composer.setPixelRatio(this.pixelRatio);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.feedbackPass = new FeedbackPass(Math.max(1, size.x), Math.max(1, size.y));
    this.composer.addPass(this.feedbackPass);
    this.bloomPass = new UnrealBloomPass(size, 0.72, 0.86, 0.18);
    this.composer.addPass(this.bloomPass);

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement ?? canvas);
    this.setScene(this.currentScene);
    this.resize();
  }

  start(): void {
    if (this.animationFrame !== undefined) return;
    this.clock.start();
    this.lastFrameAt = performance.now();
    const frame = () => {
      this.animationFrame = requestAnimationFrame(frame);
      this.render();
    };
    frame();
  }

  stop(): void {
    if (this.animationFrame !== undefined) cancelAnimationFrame(this.animationFrame);
    this.animationFrame = undefined;
  }

  dispose(): void {
    this.stop();
    this.resizeObserver.disconnect();
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh || object instanceof THREE.Points || object instanceof THREE.Line) {
        object.geometry?.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        for (const material of materials) material?.dispose();
      }
    });
    this.composer.dispose();
    this.feedbackPass.dispose();
    this.beamMaterial.dispose();
    this.branding.material.dispose();
    this.brandingTexture.dispose();
    this.renderer.dispose();
  }

  setScene(scene: VisualSceneId): void {
    const changed = scene !== this.currentScene;
    this.currentScene = scene;
    this.sceneCharacter = sceneCharacterById(scene);
    const theme = visualSceneById(scene);
    this.tunnelGroup.visible = theme.mode === "tunnel";
    this.bloomGroup.visible = theme.mode === "bloom";
    this.terrainGroup.visible = theme.mode === "terrain";
    this.scopeGroup.visible = theme.mode === "scope";
    this.core.visible = theme.mode !== "terrain" && theme.mode !== "scope";
    this.signalBloomGroup.visible = theme.mode !== "scope";
    this.kineticFieldGroup.visible = theme.mode !== "scope";
    this.volumetricGroup.visible = theme.mode !== "scope";
    this.afterimageGroup.visible = theme.mode !== "scope";
    this.atmosphereGroup.visible = true;
    this.floorGroup.visible = theme.mode !== "scope";
    for (const trail of this.spectralTrails) {
      trail.visible = theme.mode === "bloom";
      trail.material.opacity = 0.46;
    }
    for (const reflection of this.spectralReflections) reflection.visible = theme.mode === "bloom";
    this.applySceneTheme(scene);
    this.drawBranding(scene);
    if (changed) for (const listener of this.sceneListeners) listener(scene);
  }

  getScene(): VisualSceneId {
    return this.currentScene;
  }

  setIntensity(value: number): void {
    this.intensity = Math.min(1, Math.max(0.05, value));
  }

  getIntensity(): number {
    return this.intensity;
  }

  setArtDirection(direction: VisualArtDirection): void {
    this.artDirection = normalizeArtDirection(direction);
  }

  getArtDirection(): VisualArtDirection {
    return { ...this.artDirection };
  }

  setColorControls(controls: VisualColorControls): void {
    this.colorControls = normalizeVisualColorControls(controls);
    this.applySceneTheme(this.currentScene);
  }

  getColorControls(): VisualColorControls {
    return { ...this.colorControls };
  }

  setTemporalControls(controls: VisualTemporalControls): void {
    this.temporal = normalizeTemporalControls(controls);
  }

  getTemporalControls(): VisualTemporalControls {
    return { ...this.temporal };
  }

  setBloomSettings(settings: Partial<BloomSettings>): void {
    const normalized = normalizeBloomSettings({ ...this.getBloomSettings(), ...settings });
    this.bloomPass.strength = normalized.strength;
    this.bloomPass.radius = normalized.radius;
    this.bloomPass.threshold = normalized.threshold;
  }

  getBloomSettings(): BloomSettings {
    return {
      strength: this.bloomPass.strength,
      radius: this.bloomPass.radius,
      threshold: this.bloomPass.threshold,
    };
  }

  setFeedbackBoost(value: number): void {
    this.feedbackBoost = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  }

  getFeedbackBoost(): number {
    return this.feedbackBoost;
  }

  setAnimationStyle(style: VisualAnimationStyle): void {
    this.animationStyle = normalizeAnimationStyle(style);
    const styleFactors = this.animationStyleFactors();
    this.flowMaterial.uniforms.uStyle.value = styleFactors.styleIndex;
    this.bloomMaterial.uniforms.uStyle.value = styleFactors.styleIndex;
  }

  getAnimationStyle(): VisualAnimationStyle {
    return this.animationStyle;
  }

  lockResolution(width: number, height: number): void {
    this.exportLock = { width, height };
    this.branding.visible = true;
    this.renderer.setPixelRatio(1);
    this.composer.setPixelRatio(1);
    this.renderer.setSize(width, height, false);
    this.composer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  unlockResolution(): void {
    this.exportLock = undefined;
    this.branding.visible = false;
    this.renderer.setPixelRatio(this.pixelRatio);
    this.composer.setPixelRatio(this.pixelRatio);
    this.resize();
  }

  subscribeStats(listener: StatsListener): () => void {
    this.statsListeners.add(listener);
    return () => this.statsListeners.delete(listener);
  }

  subscribeScene(listener: SceneListener): () => void {
    this.sceneListeners.add(listener);
    return () => this.sceneListeners.delete(listener);
  }

  setAudioAnalysis(trackId: TrackId, analysis: AudioAnalysisResult): void {
    this.analyzedTrack = { trackId, analysis, sectionIndex: -1 };
  }

  clearAudioAnalysis(trackId?: TrackId): void {
    if (trackId === undefined || this.analyzedTrack?.trackId === trackId) this.analyzedTrack = undefined;
  }

  private render(): void {
    const now = performance.now();
    const frameTime = Math.max(1, now - this.lastFrameAt);
    this.lastFrameAt = now;
    this.frameTimeEma = this.frameTimeEma * 0.94 + frameTime * 0.06;
    this.framesSinceStats += 1;

    const elapsed = this.clock.getElapsedTime();
    const time = elapsed * (0.35 + this.temporal.speed * 1.65) + this.temporal.phase * Math.PI * 2;
    this.updateMeasuredSectionScene();
    const metrics = this.audio.getMetrics();
    const flux = computeSpectralFlux(metrics.frequency, this.previousFrequency);
    if (this.previousFrequency.length !== metrics.frequency.length) this.previousFrequency = new Uint8Array(metrics.frequency.length);
    this.previousFrequency.set(metrics.frequency);
    this.fluxEnvelope = followEnvelope(this.fluxEnvelope, flux, 0.85, 0.16);
    this.lowEnvelope = followEnvelope(this.lowEnvelope, frequencyBandEnergy(metrics.frequency, 48_000, 30, 180), 0.6, 0.14);
    this.midEnvelope = followEnvelope(this.midEnvelope, frequencyBandEnergy(metrics.frequency, 48_000, 180, 2_000), 0.55, 0.12);
    this.highEnvelope = followEnvelope(this.highEnvelope, frequencyBandEnergy(metrics.frequency, 48_000, 6_000, 16_000), 0.65, 0.16);
    const low = this.lowEnvelope;
    const mid = this.midEnvelope;
    const high = this.highEnvelope;
    const transportPulse = Math.pow(Math.max(0, 1 - metrics.beatPhase), 8) * metrics.masterLevel;
    const beatPulse = Math.max(transportPulse, this.measuredBeatPulse(), this.fluxEnvelope * 0.9);
    const response = mapVisualAudioResponse(low, mid, high, beatPulse, this.intensity);
    const pulse = Math.max(metrics.masterLevel, low * 0.9) * this.intensity;
    const style = this.animationStyleFactors();

    const performedMotion = (0.45 + this.artDirection.motion * 0.9) * (0.4 + this.temporal.camera * 1.2) * style.cameraOrbit;
    this.camera.position.x = Math.sin(time * (0.35 + this.artDirection.motion * 0.7)) * response.cameraDisplacement * 0.18 * performedMotion;
    this.camera.position.y = 0.35 + Math.cos(time * (0.5 + this.artDirection.motion * 0.7)) * response.cameraDisplacement * 0.12 * performedMotion;
    this.camera.position.z = 9 - response.cameraDisplacement * performedMotion * style.cameraPush;
    this.bloomPoints?.geometry.setDrawRange(0, response.particleCount);
    this.flowPoints?.geometry.setDrawRange(0, Math.round(FLOW_PARTICLE_COUNT * (0.24 + high * 0.76) * style.flowDraw));

    this.updateTunnel(time, low, mid, high, style);
    this.updateBloom(time, pulse, high, style);
    this.updateTerrain(time, metrics, style);
    this.updateSignalBloom(time, metrics, response, low, mid, high, style);
    this.updateKineticField(time, response, low, mid, high, style);
    this.updateScope(time, metrics, response, style);
    const feedback = mapFeedbackResponse(this.temporal.trail, this.artDirection.motion, beatPulse, this.temporal.morph);
    this.feedbackPass.damp = Math.max(feedback.damp, this.feedbackBoost * 0.94);
    this.feedbackPass.zoom = feedback.zoom;
    this.feedbackPass.rotate = feedback.rotate;
    this.core.rotation.x = time * 0.21 * style.coreSpin + mid * 0.4;
    this.core.rotation.y = time * 0.34 * style.coreSpin + low * 0.6;
    const strobeGate = this.temporal.strobe <= 0.01 ? 1 : Math.pow(Math.max(0, Math.sin(time * (12 + this.temporal.strobe * 44))), 0.22);
    this.core.scale.setScalar((0.85 + pulse * 1.35 + response.radialPulse) * (0.94 + strobeGate * 0.06) * this.sceneCharacter.coreScale);
    this.core.material.emissiveIntensity = 0.45 + high * 2.15 * this.intensity;

    this.composer.render();
    if (this.branding.visible) {
      this.renderer.clearDepth();
      this.renderer.render(this.overlayScene, this.overlayCamera);
    }
    this.adaptQuality();

    if (now - this.statsAt >= 1_000) {
      const elapsedStats = now - this.statsAt;
      const stats: RenderStats = {
        fps: Math.round((this.framesSinceStats * 1_000) / elapsedStats),
        frameTimeMs: Math.round(this.frameTimeEma * 10) / 10,
        pixelRatio: this.exportLock ? 1 : Math.round(this.pixelRatio * 100) / 100,
        quality: this.exportLock ? "export" : "adaptive",
      };
      for (const listener of this.statsListeners) listener(stats);
      this.framesSinceStats = 0;
      this.statsAt = now;
    }
  }

  private updateMeasuredSectionScene(): void {
    const analyzed = this.analyzedTrack;
    if (!analyzed) return;
    const position = this.audio.getLoadedTrackPosition(analyzed.trackId);
    if (position === undefined) return;
    const sectionIndex = analyzed.analysis.sections.findIndex(
      (section) => position >= section.start && position < section.end,
    );
    if (sectionIndex < 0 || sectionIndex === analyzed.sectionIndex) return;
    analyzed.sectionIndex = sectionIndex;
    const section = analyzed.analysis.sections[sectionIndex];
    if (section) this.setScene(sceneForMeasuredSection(section.type, sectionIndex));
  }

  private measuredBeatPulse(): number {
    const analyzed = this.analyzedTrack;
    if (!analyzed || analyzed.analysis.beatGridSeconds.length === 0) return 0;
    const position = this.audio.getLoadedTrackPosition(analyzed.trackId);
    if (position === undefined) return 0;
    const beats = analyzed.analysis.beatGridSeconds;
    let low = 0;
    let high = beats.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if ((beats[middle] ?? Number.POSITIVE_INFINITY) < position) low = middle + 1;
      else high = middle;
    }
    const previous = beats[Math.max(0, low - 1)] ?? Number.NEGATIVE_INFINITY;
    const next = beats[Math.min(beats.length - 1, low)] ?? Number.POSITIVE_INFINITY;
    const distance = Math.min(Math.abs(position - previous), Math.abs(next - position));
    return Math.exp(-distance * 24);
  }

  private animationStyleFactors(): AnimationStyleFactors {
    switch (this.animationStyle) {
      case "orbit":
        return { styleIndex: 1, cameraOrbit: 1.45, cameraPush: 0.82, tunnelTravel: 0.82, tunnelTwist: 1.55, bloomScale: 1.08, bloomSpin: 1.85, flowScale: 0.82, flowDraw: 0.8, beamScale: 0.82, afterimageScale: 0.72, terrainWave: 0.9, waveformScale: 0.86, coreSpin: 1.65, opacityScale: 0.92 };
      case "warp":
        return { styleIndex: 2, cameraOrbit: 1.12, cameraPush: 1.55, tunnelTravel: 1.72, tunnelTwist: 1.15, bloomScale: 0.9, bloomSpin: 1.08, flowScale: 1.42, flowDraw: 1, beamScale: 1.18, afterimageScale: 0.98, terrainWave: 1.04, waveformScale: 1, coreSpin: 1.2, opacityScale: 0.96 };
      case "shards":
        return { styleIndex: 3, cameraOrbit: 0.98, cameraPush: 1.08, tunnelTravel: 1.03, tunnelTwist: 0.72, bloomScale: 1.25, bloomSpin: 0.78, flowScale: 1.08, flowDraw: 0.72, beamScale: 1.32, afterimageScale: 0.62, terrainWave: 1.18, waveformScale: 0.92, coreSpin: 2.05, opacityScale: 0.9 };
      case "scan":
        return { styleIndex: 4, cameraOrbit: 0.7, cameraPush: 0.7, tunnelTravel: 0.68, tunnelTwist: 0.45, bloomScale: 0.72, bloomSpin: 0.48, flowScale: 0.62, flowDraw: 0.66, beamScale: 0.58, afterimageScale: 1.18, terrainWave: 1.5, waveformScale: 1.55, coreSpin: 0.52, opacityScale: 0.86 };
      case "minimal":
        return { styleIndex: 5, cameraOrbit: 0.42, cameraPush: 0.38, tunnelTravel: 0.42, tunnelTwist: 0.32, bloomScale: 0.48, bloomSpin: 0.26, flowScale: 0.36, flowDraw: 0.38, beamScale: 0.24, afterimageScale: 0.3, terrainWave: 0.5, waveformScale: 0.58, coreSpin: 0.35, opacityScale: 0.52 };
      case "flow":
      default:
        return { styleIndex: 0, cameraOrbit: 1, cameraPush: 1, tunnelTravel: 1, tunnelTwist: 1, bloomScale: 1, bloomSpin: 1, flowScale: 1, flowDraw: 1, beamScale: 1, afterimageScale: 1, terrainWave: 1, waveformScale: 1, coreSpin: 1, opacityScale: 1 };
    }
  }

  private updateTunnel(time: number, low: number, mid: number, high: number, style: AnimationStyleFactors): void {
    for (let index = 0; index < this.tunnelRings.length; index += 1) {
      const ring = this.tunnelRings[index];
      const travel = (index * 1.85 + time * (1.8 + this.artDirection.motion * 4.4 + this.temporal.speed * 3.2 + low * 8) * style.tunnelTravel * this.sceneCharacter.travel) % 52;
      ring.position.z = 7 - travel;
      ring.rotation.z = time * (index % 2 === 0 ? 0.09 : -0.07) * (0.5 + this.temporal.morph) * style.tunnelTwist * this.sceneCharacter.twist + index * 0.32;
      const scale = 0.75 + Math.sin(time * (1.1 + this.temporal.morph * 1.8) * style.tunnelTwist + index * 0.7) * 0.09 + mid * 0.22 * style.opacityScale;
      ring.scale.setScalar(scale);
      const material = ring.material as THREE.MeshBasicMaterial;
      const strobe = this.temporal.strobe <= 0.01 ? 1 : Math.pow(Math.max(0, Math.sin(time * (16 + this.temporal.strobe * 50) + index)), 0.18);
      material.opacity = Math.min(0.68, (0.12 + this.temporal.trail * 0.17 + high * 0.42 + (1 - travel / 52) * 0.18) * strobe * style.opacityScale);
    }
  }

  private updateBloom(time: number, energy: number, high: number, style: AnimationStyleFactors): void {
    this.bloomMaterial.uniforms.uTime.value = time;
    this.bloomMaterial.uniforms.uEnergy.value = energy;
    this.bloomMaterial.uniforms.uIntensity.value = this.intensity * 0.78 * style.bloomScale;
    this.bloomMaterial.uniforms.uPointSize.value = (1 + high * 2.3 + this.temporal.trail * 0.85) * style.bloomScale;
    this.bloomMaterial.uniforms.uStyle.value = style.styleIndex;
    this.bloomGroup.rotation.z = time * (0.012 + this.artDirection.motion * 0.075 + this.temporal.morph * 0.035) * style.bloomSpin * this.sceneCharacter.swirl;
  }

  private updateKineticField(
    time: number,
    response: VisualAudioResponse,
    low: number,
    mid: number,
    high: number,
    style: AnimationStyleFactors,
  ): void {
    const theme = visualSceneById(this.currentScene);
    const motion = 0.28 + this.artDirection.motion * 1.3;
    const atmosphere = 0.18 + this.artDirection.atmosphere * 1.15;
    const sculpture = 0.24 + this.artDirection.sculpture * 1.1;
    this.flowMaterial.uniforms.uTime.value = time;
    this.flowMaterial.uniforms.uBass.value = low;
    this.flowMaterial.uniforms.uMid.value = mid;
    this.flowMaterial.uniforms.uHigh.value = high;
    this.flowMaterial.uniforms.uIntensity.value = this.intensity * 0.72 * style.flowScale;
    this.flowMaterial.uniforms.uFlow.value = response.flowCurl * (0.65 + motion) * style.flowScale;
    this.flowMaterial.uniforms.uPointSize.value = (0.75 + high * 1.85 + this.temporal.trail * 1.15) * Math.max(0.45, style.flowScale);
    this.flowMaterial.uniforms.uStyle.value = style.styleIndex;
    this.kineticFieldGroup.rotation.y = time * 0.018 * motion * style.tunnelTwist;
    this.kineticFieldGroup.rotation.z = Math.sin(time * 0.04) * 0.08 * this.temporal.morph;
    this.kineticFieldGroup.position.z = theme.mode === "terrain" ? -2.8 : -1.2;

    const beamIntensity = Math.min(0.68, response.beamIntensity * atmosphere * style.beamScale * this.sceneCharacter.beams);
    this.volumetricGroup.rotation.z = time * (0.006 + this.temporal.camera * 0.02) * style.coreSpin;
    this.volumetricGroup.rotation.y = Math.sin(time * 0.05) * 0.12 * this.temporal.camera;
    for (let index = 0; index < this.volumetricBeams.length; index += 1) {
      const beam = this.volumetricBeams[index]!;
      const phase = index / Math.max(1, this.volumetricBeams.length - 1);
      const spread = 1.4 + sculpture * 1.8 + response.radialPulse * 0.32 * style.beamScale;
      beam.scale.set((0.55 + high * 0.9 + phase * 0.38) * Math.max(0.42, style.beamScale), spread, 1);
      beam.rotation.z = phase * Math.PI * 2 + time * (0.025 + motion * 0.035) * style.coreSpin;
      beam.position.z = -8.8 - Math.sin(time * 0.25 * style.tunnelTravel + phase * 7.0) * (0.9 + response.cameraDisplacement * style.cameraPush);
      beam.material.uniforms.uTime.value = time;
      beam.material.uniforms.uIntensity.value = beamIntensity;
      beam.material.uniforms.uBass.value = low;
      beam.material.uniforms.uPhase.value = phase;
    }

    for (let index = 0; index < this.afterimageMaterials.length; index += 1) {
      const material = this.afterimageMaterials[index]!;
      const delay = index / Math.max(1, this.afterimageMaterials.length - 1);
      material.uniforms.uTime.value = time - delay * (0.18 + this.temporal.trail * 0.72);
      material.uniforms.uOpacity.value = response.afterimageOpacity * Math.pow(1 - delay, 1.5) * (0.2 + this.temporal.trail * 0.45) * style.afterimageScale;
      material.uniforms.uWarp.value = response.flowCurl * (0.6 + delay * 1.2) * Math.max(0.35, style.flowScale);
    }
  }

  private updateSignalBloom(
    time: number,
    metrics: AudioMetrics,
    response: VisualAudioResponse,
    low: number,
    mid: number,
    high: number,
    style: AnimationStyleFactors,
  ): void {
    const frequency = metrics.frequency;
    const frequencyMaxIndex = Math.max(0, frequency.length - 1);
    const sculpture = 0.48 + this.artDirection.sculpture * 0.92;
    const motion = 0.22 + this.artDirection.motion * 1.18;
    const ribbon = 0.35 + this.artDirection.ribbon * 1.05;
    const atmosphere = 0.2 + this.artDirection.atmosphere * 1.15;
    if (visualSceneById(this.currentScene).mode === "bloom") {
      for (let trailIndex = 0; trailIndex < this.spectralTrails.length; trailIndex += 1) {
        const trail = this.spectralTrails[trailIndex];
        const reflection = this.spectralReflections[trailIndex];
        const positions = trail.geometry.attributes.position as THREE.BufferAttribute;
        const reflected = reflection.geometry.attributes.position as THREE.BufferAttribute;
        const trailPhase = trailIndex * 0.63;
        const depth = -4.2 - trailIndex * 0.075;
        for (let point = 0; point < SPECTRAL_POINTS; point += 1) {
          const progress = point / (SPECTRAL_POINTS - 1);
          const normalizedX = progress * 2 - 1;
          const envelope = Math.pow(Math.max(0, Math.sin(progress * Math.PI)), 0.48 + this.temporal.morph * 0.42);
          const frequencyBin = Math.min(
            frequencyMaxIndex,
            Math.floor(Math.pow(progress, 1.72) * frequencyMaxIndex * 0.78),
          );
          const spectrum = frequency.length > 0 ? (frequency[frequencyBin] ?? 0) / 255 : 0;
          const drift = Math.sin(progress * 15 + time * 0.72 * motion * style.waveformScale + trailPhase)
            * (0.05 + mid * 0.2 * motion * style.waveformScale);
          const fan = 4.65 - envelope * (1.3 + response.spectralHeight * 0.78 * sculpture);
          const x = normalizedX * fan
            + Math.sin(progress * 8.4 + trailPhase) * (0.08 + sculpture * 0.1)
            + Math.sin(time * 0.18 * motion + trailPhase) * (0.04 + motion * 0.07);
          const ridge = envelope * (3.5 + response.spectralHeight * 2.55 * sculpture)
            + spectrum * (0.26 + response.spectralHeight * 2.8 * sculpture)
            + drift;
          const y = FLOOR_Y + ridge;
          const z = depth + Math.cos(progress * 10 + trailPhase + time * 0.22 * motion * style.waveformScale)
            * (0.05 + high * 0.2 * motion * style.waveformScale);
          positions.setXYZ(point, x, y, z);
          reflected.setXYZ(point, x, FLOOR_Y - (y - FLOOR_Y) * 0.22, z + 0.16);
        }
        positions.needsUpdate = true;
        reflected.needsUpdate = true;
        trail.material.opacity = (0.22 + this.temporal.trail * 0.34) * style.opacityScale;
        reflection.material.opacity = 0.015 + this.temporal.trail * 0.045 + low * 0.075;
      }
    }

    const waveform = metrics.waveform;
    const ribbonPositions = this.waveformRibbon.geometry.attributes.position as THREE.BufferAttribute;
    const glowPositions = this.waveformGlow.geometry.attributes.position as THREE.BufferAttribute;
    const reflectionPositions = this.waveformReflection.geometry.attributes.position as THREE.BufferAttribute;
    for (let point = 0; point < WAVEFORM_POINTS; point += 1) {
      const progress = point / (WAVEFORM_POINTS - 1);
      const waveformIndex = Math.min(
        Math.max(0, waveform.length - 1),
        Math.floor(progress * Math.max(0, waveform.length - 1)),
      );
      const sample = waveform.length > 0 ? ((waveform[waveformIndex] ?? 128) - 128) / 128 : 0;
      const x = -7.15 + progress * 14.3;
      const y = FLOOR_Y + 0.72 + sample * (0.22 + response.waveformAmplitude * 1.35 * ribbon * style.waveformScale);
      const z = -0.64 + Math.sin(progress * Math.PI * 4 + time * 0.9 * motion * style.waveformScale) * (0.015 + ribbon * 0.025 * style.waveformScale);
      ribbonPositions.setXYZ(point, x, y, z);
      glowPositions.setXYZ(point, x, y, z + 0.025);
      reflectionPositions.setXYZ(point, x, FLOOR_Y - (y - FLOOR_Y) * 0.34, z + 0.12);
    }
    ribbonPositions.needsUpdate = true;
    glowPositions.needsUpdate = true;
    reflectionPositions.needsUpdate = true;
    this.waveformRibbon.material.opacity = Math.min(0.72, 0.24 + ribbon * 0.18 + this.temporal.trail * 0.12 + metrics.masterLevel * 0.12) * style.opacityScale;
    this.waveformGlow.material.opacity = (0.02 + response.waveformAmplitude * (0.1 + this.temporal.trail * 0.14) * ribbon) * style.opacityScale;
    this.waveformReflection.material.opacity = 0.01 + low * (0.04 + this.temporal.trail * 0.08) * ribbon;

    this.hazeMaterial.uniforms.uTime.value = time;
    this.hazeMaterial.uniforms.uEnergy.value = Math.min(1, high * 0.72 + metrics.masterLevel * 0.5);
    this.hazeMaterial.uniforms.uIntensity.value = Math.min(0.48, response.hazeOpacity * atmosphere * 0.72 * style.opacityScale * this.sceneCharacter.haze);
    this.atmosphereMaterial.opacity = Math.min(0.46, 0.02 + response.hazeOpacity * 0.42 * atmosphere * style.opacityScale * this.sceneCharacter.haze);
    this.atmosphereGroup.rotation.y = time * 0.006 * motion;
    this.atmosphereGroup.rotation.z = Math.sin(time * 0.05 * motion) * 0.022 * atmosphere;
    this.floorGroup.position.x = Math.sin(time * 0.08 * motion) * 0.06 * motion;
  }

  private updateTerrain(time: number, metrics: AudioMetrics, style: AnimationStyleFactors): void {
    if (!this.terrainGroup.visible) return;
    const positions = this.terrain.geometry.attributes.position as THREE.BufferAttribute;
    const frequency = metrics.frequency;
    for (let index = 0; index < positions.count; index += 1) {
      const x = index % (TERRAIN_SEGMENTS + 1);
      const y = Math.floor(index / (TERRAIN_SEGMENTS + 1));
      const normalized = y / TERRAIN_SEGMENTS;
      const bin = Math.min(frequency.length - 1, Math.floor(normalized * normalized * frequency.length * 0.75));
      const spectrum = frequency[bin] / 255;
      const wave = Math.sin(x * (0.26 + this.temporal.morph * 0.22) + time * 1.8 * style.terrainWave) * 0.18
        + Math.cos(y * (0.18 + this.temporal.morph * 0.18) - time * style.terrainWave) * 0.12;
      positions.setZ(index, (wave * style.terrainWave + spectrum * (1.7 + this.temporal.morph * 1.8) * this.intensity * style.terrainWave) * this.sceneCharacter.terrainAmp);
    }
    positions.needsUpdate = true;
    this.terrain.material.opacity = (0.38 + metrics.masterLevel * 0.24) * style.opacityScale;
  }

  private applySceneTheme(scene: VisualSceneId): void {
    const theme = visualSceneById(scene);
    const palette = VISUAL_COLOR_PALETTES.find((candidate) => candidate.id === this.colorControls.palette);
    const hueShift = (this.colorControls.hue - 0.5) * 0.9;
    const color = new THREE.Color(palette?.color ?? theme.color).offsetHSL(hueShift, 0, 0);
    const accent = new THREE.Color(palette?.accent ?? theme.accent).offsetHSL(hueShift, 0, 0);
    for (const value of [color, accent]) {
      const hsl = { h: 0, s: 0, l: 0 };
      value.getHSL(hsl);
      value.setHSL(hsl.h, Math.min(1, hsl.s * (0.35 + this.colorControls.saturation * 1.15)), hsl.l);
    }
    const third = color.clone().lerp(accent, 0.5).offsetHSL((this.colorControls.diversity - 0.5) * 0.55, 0.08, 0.08);
    const character = sceneCharacterById(scene);
    this.renderer.toneMappingExposure = (0.68 + this.colorControls.contrast * 0.38) * character.exposure;
    if (this.scene.background instanceof THREE.Color) this.scene.background.copy(color).multiplyScalar(0.015 + this.colorControls.contrast * 0.018);
    if (this.scene.fog instanceof THREE.FogExp2) {
      this.scene.fog.color.copy(accent).multiplyScalar(0.025 + this.colorControls.contrast * 0.02);
      this.scene.fog.density = character.fogDensity;
    }
    this.bloomMaterial.uniforms.uColorA.value = color;
    this.bloomMaterial.uniforms.uColorB.value = accent;
    this.flowMaterial.uniforms.uColorA.value = color;
    this.flowMaterial.uniforms.uColorB.value = accent;
    this.beamMaterial.uniforms.uColorA.value = color;
    this.beamMaterial.uniforms.uColorB.value = accent;
    for (const beam of this.volumetricBeams) {
      beam.material.uniforms.uColorA.value = color;
      beam.material.uniforms.uColorB.value = accent;
    }
    for (const material of this.afterimageMaterials) {
      material.uniforms.uColorA.value = color;
      material.uniforms.uColorB.value = accent;
    }
    this.atmosphereMaterial.color = color;
    this.waveformGlow.material.color = accent;
    this.waveformReflection.material.color = color;
    this.terrain.material.color = color;
    this.core.material.emissive = accent;
    for (let index = 0; index < this.tunnelRings.length; index += 1) {
      const material = this.tunnelRings[index]?.material as THREE.MeshBasicMaterial | undefined;
      if (!material) continue;
      material.color = index % 3 === 0 ? color : index % 3 === 1 ? accent : third;
    }
    for (let index = 0; index < this.spectralTrails.length; index += 1) {
      this.spectralTrails[index]!.material.color = index % 2 === 0 ? color : accent;
      this.spectralReflections[index]!.material.color = index % 2 === 0 ? accent : color;
    }
    if (this.scopeLines.length >= 2) {
      this.scopeLines[0]!.material.color = color;
      this.scopeLines[1]!.material.color = accent;
    }
    if (this.scopeSpikes) this.scopeSpikes.material.color = accent;
  }

  private adaptQuality(): void {
    if (this.exportLock) return;
    this.stableFrames += 1;
    if (this.frameTimeEma > 20 && this.stableFrames > 90 && this.pixelRatio > 1) {
      this.pixelRatio = Math.max(1, this.pixelRatio - 0.18);
      this.renderer.setPixelRatio(this.pixelRatio);
      this.composer.setPixelRatio(this.pixelRatio);
      this.resize();
      this.stableFrames = 0;
    } else if (this.frameTimeEma < 13.8 && this.stableFrames > 300 && this.pixelRatio < Math.min(window.devicePixelRatio || 1, 1.75)) {
      this.pixelRatio = Math.min(Math.min(window.devicePixelRatio || 1, 1.75), this.pixelRatio + 0.12);
      this.renderer.setPixelRatio(this.pixelRatio);
      this.composer.setPixelRatio(this.pixelRatio);
      this.resize();
      this.stableFrames = 0;
    }
  }

  private resize(): void {
    if (this.exportLock) return;
    const parent = this.canvas.parentElement;
    const width = Math.max(1, Math.floor(parent?.clientWidth ?? this.canvas.clientWidth));
    const height = Math.max(1, Math.floor(parent?.clientHeight ?? this.canvas.clientHeight));
    this.renderer.setSize(width, height, false);
    this.composer.setSize(width, height);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  private createTunnel(): void {
    const geometry = new THREE.TorusGeometry(7.2, 0.055, 6, 72);
    const colors = [0xff3f91, 0xffb443, 0x55ffd4, 0x6f8cff, 0xb269ff];
    for (let index = 0; index < 30; index += 1) {
      const material = new THREE.MeshBasicMaterial({
        color: colors[index % colors.length],
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const ring = new THREE.Mesh(geometry, material);
      ring.position.z = 7 - index * 1.85;
      ring.rotation.z = index * 0.32;
      this.tunnelGroup.add(ring);
      this.tunnelRings.push(ring);
    }
  }

  private createBloom(): void {
    const positions = new Float32Array(PARTICLE_COUNT * 3);
    const seeds = new Float32Array(PARTICLE_COUNT);
    let randomState = 0xabc123;
    const random = () => {
      randomState = Math.imul(randomState ^ (randomState >>> 15), 1 | randomState);
      randomState ^= randomState + Math.imul(randomState ^ (randomState >>> 7), 61 | randomState);
      return ((randomState ^ (randomState >>> 14)) >>> 0) / 4294967296;
    };
    for (let index = 0; index < PARTICLE_COUNT; index += 1) {
      const radius = Math.sqrt(random()) * 6.8;
      const angle = random() * Math.PI * 2;
      const depth = (random() - 0.5) * 5;
      positions[index * 3] = Math.cos(angle) * radius;
      positions[index * 3 + 1] = Math.sin(angle) * radius;
      positions[index * 3 + 2] = depth - 5;
      seeds[index] = random();
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
    const points = new THREE.Points(geometry, this.bloomMaterial);
    this.bloomPoints = points;
    this.bloomGroup.add(points);
  }

  private createBloomMaterial(): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uEnergy: { value: 0 },
        uIntensity: { value: 0.7 },
        uPointSize: { value: 1.5 },
        uStyle: { value: 0 },
        uColorA: { value: new THREE.Color(0xff3f91) },
        uColorB: { value: new THREE.Color(0x58ffe0) },
      },
      vertexShader: `
        attribute float aSeed;
        uniform float uTime;
        uniform float uEnergy;
        uniform float uIntensity;
        uniform float uPointSize;
        uniform float uStyle;
        varying float vSeed;
        void main() {
          vSeed = aSeed;
          vec3 p = position;
          float angle = uTime * (0.08 + aSeed * 0.18) + p.z * 0.06;
          mat2 rotation = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
          p.xy = rotation * p.xy;
          float shard = step(2.5, uStyle) * (1.0 - step(3.5, uStyle));
          float scan = step(3.5, uStyle) * (1.0 - step(4.5, uStyle));
          float minimal = step(4.5, uStyle);
          p.xy *= 1.0 + uEnergy * (0.15 + aSeed * 0.48) * uIntensity;
          p.xy += shard * vec2(sign(sin(aSeed * 81.0)), sign(cos(aSeed * 53.0))) * uEnergy * 0.72;
          p.y += scan * sin(position.x * 3.5 + uTime * 2.4) * (0.08 + uEnergy * 0.35);
          p.z += sin(uTime * 0.8 + aSeed * 18.0) * (0.1 + uEnergy * 0.6) * (1.0 - minimal * 0.55);
          vec4 view = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * view;
          gl_PointSize = (uPointSize + aSeed * 2.2) * (180.0 / max(1.0, -view.z)) * (1.0 + shard * 0.32 - minimal * 0.4);
        }
      `,
      fragmentShader: `
        uniform vec3 uColorA;
        uniform vec3 uColorB;
        uniform float uEnergy;
        uniform float uIntensity;
        varying float vSeed;
        void main() {
          vec2 centered = gl_PointCoord - 0.5;
          float distanceToCenter = length(centered);
          float alpha = smoothstep(0.5, 0.06, distanceToCenter);
          vec3 color = mix(uColorA, uColorB, vSeed + uEnergy * 0.2);
          gl_FragColor = vec4(color, alpha * (0.035 + uEnergy * 0.16) * uIntensity);
        }
      `,
    });
  }

  private createKineticField(): void {
    const positions = new Float32Array(FLOW_PARTICLE_COUNT * 3);
    const seeds = new Float32Array(FLOW_PARTICLE_COUNT);
    const lanes = new Float32Array(FLOW_PARTICLE_COUNT);
    let randomState = 0x44ddee;
    const random = () => {
      randomState = Math.imul(randomState ^ (randomState >>> 15), 1 | randomState);
      randomState ^= randomState + Math.imul(randomState ^ (randomState >>> 7), 61 | randomState);
      return ((randomState ^ (randomState >>> 14)) >>> 0) / 4294967296;
    };
    for (let index = 0; index < FLOW_PARTICLE_COUNT; index += 1) {
      const lane = Math.floor(random() * 9);
      const radius = 1.1 + random() * 7.8;
      const angle = random() * Math.PI * 2;
      positions[index * 3] = Math.cos(angle) * radius;
      positions[index * 3 + 1] = (random() - 0.5) * 7.8 + Math.sin(lane * 1.73) * 0.32;
      positions[index * 3 + 2] = -3 - random() * 13;
      seeds[index] = random();
      lanes[index] = lane / 8;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("aSeed", new THREE.BufferAttribute(seeds, 1));
    geometry.setAttribute("aLane", new THREE.BufferAttribute(lanes, 1));
    const points = new THREE.Points(geometry, this.flowMaterial);
    points.renderOrder = 16;
    this.flowPoints = points;
    this.kineticFieldGroup.add(points);
  }

  private createFlowMaterial(): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uBass: { value: 0 },
        uMid: { value: 0 },
        uHigh: { value: 0 },
        uIntensity: { value: 0.7 },
        uFlow: { value: 0.2 },
        uPointSize: { value: 1.3 },
        uStyle: { value: 0 },
        uColorA: { value: new THREE.Color(0xff3f91) },
        uColorB: { value: new THREE.Color(0x58ffe0) },
      },
      vertexShader: `
        attribute float aSeed;
        attribute float aLane;
        uniform float uTime;
        uniform float uBass;
        uniform float uMid;
        uniform float uHigh;
        uniform float uIntensity;
        uniform float uFlow;
        uniform float uPointSize;
        uniform float uStyle;
        varying float vSeed;
        varying float vEnergy;
        void main() {
          vSeed = aSeed;
          vEnergy = clamp(uBass * 0.45 + uMid * 0.35 + uHigh * 0.5, 0.0, 1.0);
          vec3 p = position;
          float lane = aLane * 6.2831853;
          float orbit = step(0.5, uStyle) * (1.0 - step(1.5, uStyle));
          float warp = step(1.5, uStyle) * (1.0 - step(2.5, uStyle));
          float shard = step(2.5, uStyle) * (1.0 - step(3.5, uStyle));
          float scan = step(3.5, uStyle) * (1.0 - step(4.5, uStyle));
          float minimal = step(4.5, uStyle);
          float stream = fract(aSeed + uTime * (0.025 + uFlow * 0.05 + warp * 0.06));
          p.z = mix(5.8, -18.0, stream);
          float curlA = sin(p.z * 0.42 + lane + uTime * (0.65 + uFlow));
          float curlB = cos(p.z * 0.31 + aSeed * 17.0 - uTime * (0.4 + uFlow * 0.7));
          float radius = length(p.xy) * (0.82 + uBass * 0.22) + curlB * uFlow * 0.8;
          float angle = atan(p.y, p.x) + curlA * uFlow * 0.3 + uTime * (0.035 + aLane * 0.05 + orbit * 0.08);
          p.x = cos(angle) * radius + curlB * 0.28 * uIntensity;
          p.y = sin(angle) * radius * (0.62 + uMid * 0.28) + curlA * 0.5 * uIntensity;
          p.xy += shard * vec2(sign(sin(aSeed * 94.0)), sign(cos(aSeed * 67.0))) * (0.18 + uHigh * 0.72);
          p.x += scan * sin((position.y + uTime) * 8.0 + aSeed * 11.0) * (0.1 + uMid * 0.42);
          p.y = mix(p.y, floor((p.y + 4.0) * 5.0) / 5.0 - 4.0, scan * 0.72);
          p.xy *= 0.8 + uIntensity * 0.32;
          vec4 view = modelViewMatrix * vec4(p, 1.0);
          gl_Position = projectionMatrix * view;
          gl_PointSize = (uPointSize + aSeed * 1.8 + vEnergy * 2.2) * (150.0 / max(1.0, -view.z)) * (1.0 + shard * 0.4 - minimal * 0.45);
        }
      `,
      fragmentShader: `
        uniform vec3 uColorA;
        uniform vec3 uColorB;
        uniform float uHigh;
        uniform float uIntensity;
        varying float vSeed;
        varying float vEnergy;
        void main() {
          vec2 p = gl_PointCoord - 0.5;
          float d = length(p);
          float core = smoothstep(0.48, 0.03, d);
          float filament = smoothstep(0.5, 0.16, abs(p.x * p.y) * 8.0 + d * 0.55);
          vec3 color = mix(uColorA, uColorB, fract(vSeed * 1.7 + vEnergy + uHigh * 0.35));
          gl_FragColor = vec4(color, core * filament * (0.055 + vEnergy * 0.22) * uIntensity);
        }
      `,
    });
  }

  private createVolumetricBeams(): void {
    const geometry = new THREE.PlaneGeometry(1, 8, 1, 16);
    for (let index = 0; index < VOLUMETRIC_BEAM_COUNT; index += 1) {
      const material = this.beamMaterial.clone();
      const progress = index / VOLUMETRIC_BEAM_COUNT;
      material.uniforms.uPhase.value = progress;
      const beam = new THREE.Mesh(geometry, material);
      const angle = progress * Math.PI * 2;
      beam.position.set(Math.cos(angle) * 2.4, 0.2 + Math.sin(angle * 2.0) * 0.4, -7.8 - progress * 4.5);
      beam.rotation.z = angle;
      beam.rotation.x = Math.PI * 0.5;
      beam.renderOrder = -10;
      this.volumetricBeams.push(beam);
      this.volumetricGroup.add(beam);
    }
  }

  private createBeamMaterial(): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uIntensity: { value: 0.2 },
        uBass: { value: 0 },
        uPhase: { value: 0 },
        uColorA: { value: new THREE.Color(0xff3f91) },
        uColorB: { value: new THREE.Color(0x58ffe0) },
      },
      vertexShader: `
        varying vec2 vUv;
        uniform float uTime;
        uniform float uPhase;
        uniform float uBass;
        void main() {
          vUv = uv;
          vec3 p = position;
          float taper = 1.0 - abs(uv.y - 0.5) * 1.6;
          p.x *= 0.2 + max(0.0, taper) * (1.8 + uBass * 2.2);
          p.z += sin(uv.y * 9.0 + uTime * 0.45 + uPhase * 6.2831853) * 0.2;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform float uTime;
        uniform float uIntensity;
        uniform float uPhase;
        uniform vec3 uColorA;
        uniform vec3 uColorB;
        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(41.17, 289.13))) * 43758.5453);
        }
        void main() {
          vec2 p = vUv - 0.5;
          float shaft = smoothstep(0.5, 0.02, abs(p.x)) * smoothstep(0.52, 0.0, abs(p.y));
          float scan = sin(vUv.y * 26.0 - uTime * 1.7 + uPhase * 6.2831853) * 0.5 + 0.5;
          float grain = hash(floor(vUv * 160.0) + floor(uTime * 0.35));
          vec3 color = mix(uColorA, uColorB, vUv.y + scan * 0.18);
          gl_FragColor = vec4(color, shaft * (0.08 + scan * 0.14 + grain * 0.04) * uIntensity);
        }
      `,
    });
  }

  private createAfterimagePanels(): void {
    const geometry = new THREE.PlaneGeometry(15.4, 8.8, 1, 1);
    for (let index = 0; index < AFTERIMAGE_PANEL_COUNT; index += 1) {
      const material = this.createAfterimageMaterial();
      const depth = -2.2 - index * 0.58;
      const panel = new THREE.Mesh(geometry, material);
      panel.position.set(0, 0.22, depth);
      panel.renderOrder = -6 + index;
      this.afterimageMaterials.push(material);
      this.afterimageGroup.add(panel);
    }
  }

  private createAfterimageMaterial(): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uOpacity: { value: 0.08 },
        uWarp: { value: 0.2 },
        uColorA: { value: new THREE.Color(0xff3f91) },
        uColorB: { value: new THREE.Color(0x58ffe0) },
      },
      vertexShader: `
        varying vec2 vUv;
        uniform float uTime;
        uniform float uWarp;
        void main() {
          vUv = uv;
          vec3 p = position;
          p.x += sin(uv.y * 8.0 + uTime * 0.72) * 0.1 * uWarp;
          p.y += cos(uv.x * 7.0 - uTime * 0.5) * 0.08 * uWarp;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform float uTime;
        uniform float uOpacity;
        uniform float uWarp;
        uniform vec3 uColorA;
        uniform vec3 uColorB;
        void main() {
          vec2 p = vUv - 0.5;
          float rings = abs(sin((length(p * vec2(1.25, 0.78)) * 18.0) - uTime * (1.8 + uWarp)));
          float grid = smoothstep(0.98, 1.0, sin(vUv.x * 58.0 + uTime) * sin(vUv.y * 34.0 - uTime * 0.7));
          float vignette = smoothstep(0.72, 0.08, length(p));
          vec3 color = mix(uColorA, uColorB, vUv.x + rings * 0.22);
          gl_FragColor = vec4(color, vignette * (0.035 + rings * 0.055 + grid * 0.16) * uOpacity);
        }
      `,
    });
  }

  private createSignalBloom(): void {
    const hazeGeometry = new THREE.PlaneGeometry(32, 24);
    const haze = new THREE.Mesh(hazeGeometry, this.hazeMaterial);
    haze.position.set(0, 0.8, -16);
    haze.renderOrder = -20;
    this.signalBloomGroup.add(haze);

    const colors = [0xff27d8, 0xff4ff2, 0xc33cff, 0x7e5cff, 0x27d7ff];
    for (let trailIndex = 0; trailIndex < SPECTRAL_TRAIL_COUNT; trailIndex += 1) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(SPECTRAL_POINTS * 3), 3));
      const material = new THREE.LineBasicMaterial({
        color: colors[trailIndex % colors.length],
        transparent: true,
        opacity: 0.68,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const trail = new THREE.Line(geometry, material);
      trail.renderOrder = 4 + trailIndex;
      this.spectralTrails.push(trail);
      this.signalBloomGroup.add(trail);

      const reflectionGeometry = new THREE.BufferGeometry();
      reflectionGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(new Float32Array(SPECTRAL_POINTS * 3), 3),
      );
      const reflectionMaterial = new THREE.LineBasicMaterial({
        color: colors[trailIndex % colors.length],
        transparent: true,
        opacity: 0.08,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const reflection = new THREE.Line(reflectionGeometry, reflectionMaterial);
      reflection.renderOrder = 2;
      this.spectralReflections.push(reflection);
      this.signalBloomGroup.add(reflection);
    }

    this.waveformRibbon.renderOrder = 80;
    this.waveformGlow.renderOrder = 79;
    this.waveformReflection.renderOrder = 3;
    this.signalBloomGroup.add(this.waveformGlow, this.waveformRibbon, this.waveformReflection);

    const railMaterial = new THREE.LineBasicMaterial({
      color: 0x2eeaff,
      transparent: true,
      opacity: 0.22,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const railPositions = new Float32Array([
      -7.1, -3.2, -2.8, -7.1, 4.9, -2.8,
      7.1, -3.2, -2.8, 7.1, 4.9, -2.8,
      -6.75, 4.55, -3.2, -5.4, 4.55, -3.2,
      5.4, -3.05, -3.2, 6.75, -3.05, -3.2,
    ]);
    const railGeometry = new THREE.BufferGeometry();
    railGeometry.setAttribute("position", new THREE.BufferAttribute(railPositions, 3));
    this.signalBloomGroup.add(new THREE.LineSegments(railGeometry, railMaterial));
  }

  private createWaveformRibbon(): {
    ribbon: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
    glow: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
    reflection: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  } {
    const geometry = () => {
      const value = new THREE.BufferGeometry();
      value.setAttribute("position", new THREE.BufferAttribute(new Float32Array(WAVEFORM_POINTS * 3), 3));
      return value;
    };
    const ribbon = new THREE.Line(
      geometry(),
      new THREE.LineBasicMaterial({
        color: 0xf7f5ff,
        transparent: true,
        opacity: 0.86,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    const glow = new THREE.Line(
      geometry(),
      new THREE.LineBasicMaterial({
        color: 0xff29d7,
        transparent: true,
        opacity: 0.24,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    const reflection = new THREE.Line(
      geometry(),
      new THREE.LineBasicMaterial({
        color: 0x32ddff,
        transparent: true,
        opacity: 0.08,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    );
    return { ribbon, glow, reflection };
  }

  private createAtmosphere(): THREE.PointsMaterial {
    const positions = new Float32Array(ATMOSPHERE_PARTICLES * 3);
    let state = 0x51a9e2;
    const random = () => {
      state = Math.imul(state ^ (state >>> 15), 1 | state);
      state ^= state + Math.imul(state ^ (state >>> 7), 61 | state);
      return ((state ^ (state >>> 14)) >>> 0) / 4294967296;
    };
    for (let index = 0; index < ATMOSPHERE_PARTICLES; index += 1) {
      positions[index * 3] = (random() - 0.5) * 22;
      positions[index * 3 + 1] = (random() - 0.42) * 15;
      positions[index * 3 + 2] = -2 - random() * 20;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0xff4cda,
      size: 0.035,
      transparent: true,
      opacity: 0.22,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });
    this.atmosphereGroup.add(new THREE.Points(geometry, material));
    return material;
  }

  private createScope(): void {
    const line = (color: number, opacity: number) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(WAVEFORM_POINTS * 3), 3));
      const scopeLine = new THREE.Line(geometry, new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }));
      this.scopeLines.push(scopeLine);
      this.scopeGroup.add(scopeLine);
      return scopeLine;
    };
    line(0x5dff8a, 0.92);
    line(0x35dcff, 0.66);
    line(0xf7f5ff, 0.8);

    const spikeGeometry = new THREE.BufferGeometry();
    spikeGeometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(96 * 2 * 3), 3));
    this.scopeSpikes = new THREE.LineSegments(spikeGeometry, new THREE.LineBasicMaterial({
      color: 0x35dcff,
      transparent: true,
      opacity: 0.42,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }));
    this.scopeGroup.add(this.scopeSpikes);
    this.scopeGroup.position.z = -1.5;
    this.scopeGroup.visible = false;
  }

  private updateScope(time: number, metrics: AudioMetrics, response: VisualAudioResponse, style: AnimationStyleFactors): void {
    if (!this.scopeGroup.visible) return;
    const waveform = metrics.waveform;
    const sampleAt = (progress: number) => {
      if (waveform.length === 0) return 0;
      const index = Math.min(waveform.length - 1, Math.floor(progress * (waveform.length - 1)));
      return ((waveform[index] ?? 128) - 128) / 128;
    };
    const [outerRing, innerRing, centerLine] = this.scopeLines;
    const amp = (0.5 + response.waveformAmplitude * 1.4) * (0.4 + this.artDirection.ribbon);
    const spin = time * 0.05 * (0.3 + this.artDirection.motion) * style.coreSpin;
    for (let point = 0; point < WAVEFORM_POINTS; point += 1) {
      const progress = point / (WAVEFORM_POINTS - 1);
      const closing = point === WAVEFORM_POINTS - 1 ? 0 : progress;
      const angle = closing * Math.PI * 2 + spin;
      const outerRadius = 2.55 + sampleAt(closing) * amp;
      outerRing?.geometry.attributes.position.setXYZ(point, Math.cos(angle) * outerRadius, Math.sin(angle) * outerRadius, 0);
      const innerSample = sampleAt((closing + 0.25) % 1);
      const innerRadius = 1.55 + innerSample * amp * 0.62;
      innerRing?.geometry.attributes.position.setXYZ(point, Math.cos(-angle * 2 + spin) * innerRadius, Math.sin(-angle * 2 + spin) * innerRadius, 0.1);
      centerLine?.geometry.attributes.position.setXYZ(point, -4.6 + progress * 9.2, sampleAt(progress) * amp * 1.15, 0.2);
    }
    for (const scopeLine of this.scopeLines) {
      const positions = scopeLine.geometry.attributes.position as THREE.BufferAttribute;
      positions.needsUpdate = true;
    }
    if (this.scopeSpikes) {
      const spikes = this.scopeSpikes.geometry.attributes.position as THREE.BufferAttribute;
      const frequency = metrics.frequency;
      for (let spike = 0; spike < 96; spike += 1) {
        const progress = spike / 96;
        const angle = progress * Math.PI * 2 - spin * 0.5;
        const bin = Math.min(Math.max(0, frequency.length - 1), Math.floor(Math.pow(progress, 1.6) * frequency.length * 0.8));
        const level = frequency.length > 0 ? (frequency[bin] ?? 0) / 255 : 0;
        const baseRadius = 3.35;
        const tipRadius = baseRadius + 0.06 + level * (0.5 + this.intensity * 1.35);
        spikes.setXYZ(spike * 2, Math.cos(angle) * baseRadius, Math.sin(angle) * baseRadius, -0.1);
        spikes.setXYZ(spike * 2 + 1, Math.cos(angle) * tipRadius, Math.sin(angle) * tipRadius, -0.1);
      }
      spikes.needsUpdate = true;
      this.scopeSpikes.material.opacity = 0.2 + response.beamIntensity * 0.5;
    }
  }

  private createFloor(): void {
    const grid = new THREE.GridHelper(24, 36, 0x512065, 0x101d31);
    grid.position.set(0, FLOOR_Y, -5.5);
    const materials = Array.isArray(grid.material) ? grid.material : [grid.material];
    for (const material of materials) {
      material.transparent = true;
      material.opacity = 0.22;
      material.blending = THREE.AdditiveBlending;
      material.depthWrite = false;
    }
    this.floorGroup.add(grid);
  }

  private createHazeMaterial(): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      uniforms: {
        uTime: { value: 0 },
        uEnergy: { value: 0 },
        uIntensity: { value: 0.2 },
      },
      vertexShader: `
        varying vec2 vUv;
        void main() {
          vUv = uv;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        varying vec2 vUv;
        uniform float uTime;
        uniform float uEnergy;
        uniform float uIntensity;
        float hash(vec2 p) {
          return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }
        void main() {
          vec2 p = vUv - 0.5;
          float radial = max(0.0, 1.0 - length(p * vec2(1.05, 0.82)) * 1.75);
          float cloudA = sin(p.x * 8.0 + uTime * 0.045) * sin(p.y * 9.0 - uTime * 0.038);
          float cloudB = sin((p.x + p.y) * 13.0 - uTime * 0.026);
          float grain = hash(floor(vUv * 220.0) + floor(uTime * 0.2));
          float haze = radial * (0.16 + cloudA * 0.06 + cloudB * 0.035 + grain * 0.025);
          vec3 violet = vec3(0.34, 0.01, 0.48);
          vec3 cyan = vec3(0.01, 0.2, 0.34);
          vec3 color = mix(violet, cyan, clamp(vUv.x + uEnergy * 0.22, 0.0, 1.0));
          gl_FragColor = vec4(color, haze * (0.55 + uEnergy) * (0.6 + uIntensity));
        }
      `,
    });
  }

  private createTerrain(): THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial> {
    const geometry = new THREE.PlaneGeometry(18, 24, TERRAIN_SEGMENTS, TERRAIN_SEGMENTS);
    const material = new THREE.MeshBasicMaterial({
      color: 0x69fbd0,
      wireframe: true,
      transparent: true,
      opacity: 0.68,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI * 0.48;
    mesh.position.set(0, -4.5, -10);
    return mesh;
  }

  private createCore(): THREE.Mesh<THREE.IcosahedronGeometry, THREE.MeshStandardMaterial> {
    const geometry = new THREE.IcosahedronGeometry(1.35, 3);
    const material = new THREE.MeshStandardMaterial({
      color: 0x0b0e16,
      emissive: 0xff377f,
      emissiveIntensity: 1.2,
      metalness: 0.82,
      roughness: 0.24,
      wireframe: true,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.z = -1.5;
    return mesh;
  }

  private createBranding(): { texture: THREE.CanvasTexture; sprite: THREE.Sprite } {
    const canvas = document.createElement("canvas");
    canvas.width = 1024;
    canvas.height = 256;
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      toneMapped: false,
    });
    const sprite = new THREE.Sprite(material);
    sprite.position.set(-0.1, 0.76, 0);
    sprite.scale.set(1.7, 0.425, 1);
    sprite.renderOrder = 10_000;
    return { texture, sprite };
  }

  private drawBranding(scene: VisualSceneId): void {
    const canvas = this.brandingTexture.image as HTMLCanvasElement;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const theme = visualSceneById(scene);
    const sceneName = theme.name.toUpperCase();
    context.fillStyle = "rgba(2, 1, 7, 0.66)";
    context.fillRect(18, 22, 760, 168);
    context.fillStyle = "#ff34db";
    context.fillRect(52, 48, 82, 5);
    context.fillStyle = "#32dfff";
    context.fillRect(141, 48, 14, 5);
    context.fillStyle = "#f7f7f4";
    context.font = "600 42px -apple-system, BlinkMacSystemFont, sans-serif";
    context.fillText(`MUSICA / ${sceneName}`, 52, 112);
    context.fillStyle = "#ff35d8";
    context.font = "500 22px Menlo, monospace";
    context.fillText("LIVE GENERATIVE SET  /  48 KHZ", 54, 160);
    this.brandingTexture.needsUpdate = true;
  }

}
