import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { AudioEngine, createEngineSnapshotFromTemplate, type EngineSnapshot } from "./audio/AudioEngine";
import { ControlRouter, TapTempo, type ControllerStatus } from "./controllers/ControlRouter";
import { createAgentPlan, getAgentStatus, type AgentPlan, type AgentStatus } from "./core/agentProvider";
import {
  compileLyriaPrompt,
  LYRIA_PRO_PRICE_USD,
  LYRIA_VOCAL_LANGUAGES,
  reserveGenerationCost,
  selectGenerationRoute,
  timestampToSeconds,
  type AudioOutputFormat,
  type CompositionSection,
  type StructuredComposition,
} from "./core/composition";
import { DEFAULT_TEMPORAL_CONTROLS, PERFORMANCE_TEMPLATES, SOCIAL_PRESETS, VISUAL_PRESETS, VISUAL_SCENES, defaultPerformanceTemplate, performanceTemplateById } from "./core/presets";
import {
  cancelGeneration,
  downloadGeneratedAudio,
  generateMusic,
  getGeneration,
  getProviderStatus,
  saveGenerationReceipt,
} from "./core/creativeProvider";
import {
  LYRIA_REALTIME_STYLE_PRESETS,
  DEFAULT_LYRIA_REALTIME_STYLE_ID,
  compensateLyriaBpmForPitch,
  createLyriaSequenceConfig,
  createLyriaSequencePrompts,
  createLyriaVocalPrompts,
  createLyriaRealtimeRequestForTemplate,
  createLyriaRealtimeRequestFromStyle,
  getLyriaRealtimeStatus,
  lyriaRealtimeStyleById,
  lyriaRealtimeStyleForTemplate,
  pollLyriaRealtimeAudio,
  startLyriaRealtime,
  stopLyriaRealtime,
  updateLyriaRealtime,
  type LyriaRealtimeConfig,
  type LyriaRealtimeDeckId,
  type LyriaRealtimeRequest,
  type LyriaRealtimeSession,
  type LyriaRealtimeStylePreset,
  type LyriaRealtimeStatus,
  type LyriaWeightedPrompt,
} from "./core/lyriaRealtime";
import { importMidiPerformance } from "./core/midiImport";
import {
  DEFAULT_LYRIA_DECK_CONTROLS,
  LYRIA_DECK_SCENE_STORAGE_KEY,
  cloneLyriaDeckScene,
  loadLyriaDeckScenes,
  normalizeLyriaDeckScene,
  type LyriaDeckControl,
  type LyriaDeckScene,
} from "./core/lyriaDeckScenes";
import { TRACK_IDS, type ControlMessage, type GenerationTask, type ProviderStatus, type SocialPreset, type TrackId, type VisualSceneId, type VisualTemporalControls } from "./core/types";
import { SocialRecorder, type RecordingResult } from "./export/SocialRecorder";
import {
  VISUAL_ANIMATION_STYLES,
  VisualEngine,
  normalizeAnimationStyle,
  type RenderStats,
  type VisualAnimationStyle,
  type VisualArtDirection,
} from "./visual/VisualEngine";

const DEFAULT_TEMPLATE = defaultPerformanceTemplate();
const INITIAL_SNAPSHOT: EngineSnapshot = createEngineSnapshotFromTemplate(DEFAULT_TEMPLATE);
const DEFAULT_REALTIME_REQUEST = createLyriaRealtimeRequestForTemplate(DEFAULT_TEMPLATE);
const DEFAULT_REALTIME_STYLE = lyriaRealtimeStyleById(DEFAULT_LYRIA_REALTIME_STYLE_ID);
const LYRIA_STREAM_STARTUP_TIMEOUT_MS = 15_000;
const LYRIA_STREAM_POLL_MS = 80;
const LYRIA_PREBUFFER_SECONDS = 1.25;
const LYRIA_PREBUFFER_BYTES = 48_000 * 2 * 2 * LYRIA_PREBUFFER_SECONDS;
const LYRIA_MIN_START_BYTES = 48_000 * 2 * 2 * 0.75;
const LYRIA_PLAYBACK_LEAD_SECONDS = 1;
const LYRIA_LIVE_UPDATE_DEBOUNCE_MS = 420;
const LYRIA_DECKS: LyriaRealtimeDeckId[] = ["main", "sequence", "vocal"];

interface LyriaBufferingState {
  active: boolean;
  message: string;
  bytes: number;
}

interface LyriaStyleGuidance {
  text: string;
  weight: number;
}

interface LyriaGuidanceDialogState extends LyriaStyleGuidance {
  styleId: string;
}

interface BufferedRealtimeChunk {
  bytes: Uint8Array;
  sampleRateHz: number;
  channels: number;
}

function createRealtimePrebuffer(): Record<LyriaRealtimeDeckId, BufferedRealtimeChunk[]> {
  return { main: [], sequence: [], vocal: [] };
}

function applyPrimaryGuidance(
  style: LyriaRealtimeStylePreset,
  guidance?: LyriaStyleGuidance,
): LyriaRealtimeStylePreset {
  if (!guidance) return style;
  return {
    ...style,
    prompts: style.prompts.map((prompt, index) => (
      index === 0 ? { text: guidance.text, weight: guidance.weight } : { ...prompt }
    )),
  };
}

type StudioPanelId =
  | "visual-scenes"
  | "visual-presets"
  | "visual-animation"
  | "visual-reactivity"
  | "visual-macros"
  | "visual-temporal"
  | "audio-lyria"
  | "audio-templates"
  | "audio-agent"
  | "audio-generation"
  | "av-output";

const INITIAL_CONTROLLER_STATUS: ControllerStatus = {
  keyboard: true,
  globalShortcuts: false,
  logitechBridge: false,
  midi: false,
  midiInputs: [],
};

const WAIT = (milliseconds: number) => new Promise<void>((resolve) => setTimeout(resolve, milliseconds));
const GENERATION_POLL_INTERVAL_MS = 2_000;
const DEFAULT_STRUCTURE = `0:00 atmospheric intro
0:20 groove enters
0:42 first drop
1:12 breakdown
1:35 larger second drop
2:15 short outro`;

interface SceneVisualSettings {
  intensity: number;
  artDirection: VisualArtDirection;
  temporal: VisualTemporalControls;
  animationStyle: VisualAnimationStyle;
}

type SceneVisualSettingsMap = Record<VisualSceneId, SceneVisualSettings>;

const DEFAULT_SCENE_VISUAL_SETTINGS: SceneVisualSettings = {
  intensity: DEFAULT_TEMPLATE.intensity,
  artDirection: DEFAULT_TEMPLATE.artDirection,
  temporal: DEFAULT_TEMPLATE.temporal ?? { ...DEFAULT_TEMPORAL_CONTROLS },
  animationStyle: defaultAnimationStyleForScene(DEFAULT_TEMPLATE.scene),
};

function defaultAnimationStyleForScene(scene: VisualSceneId): VisualAnimationStyle {
  switch (scene) {
    case "tunnel":
    case "lasergrid":
      return "warp";
    case "terrain":
    case "aurora":
      return "scan";
    case "monolith":
      return "minimal";
    case "pulsefield":
      return "shards";
    case "bloom":
    case "chromawave":
    default:
      return "flow";
  }
}

function sequencerSignature(snapshot: EngineSnapshot): string {
  return [
    `bpm:${snapshot.bpm}`,
    ...snapshot.tracks.map((track) => [
      track.id,
      track.pattern.map((active) => (active ? "1" : "0")).join(""),
      Math.round(track.volume * 100),
      track.muted ? "m" : "-",
      track.solo ? "s" : "-",
    ].join(":")),
  ].join("|");
}

function cloneVisualSettings(settings: SceneVisualSettings): SceneVisualSettings {
  return {
    intensity: Math.max(0.05, Math.min(1, settings.intensity)),
    artDirection: { ...settings.artDirection },
    temporal: { ...settings.temporal },
    animationStyle: normalizeAnimationStyle(settings.animationStyle),
  };
}

function createInitialSceneVisualSettings(): SceneVisualSettingsMap {
  return Object.fromEntries(
    VISUAL_SCENES.map((scene) => {
      const visualPreset = VISUAL_PRESETS.find((preset) => preset.scene === scene.id);
      const performanceTemplate = PERFORMANCE_TEMPLATES.find((template) => template.scene === scene.id);
      const settings = visualPreset ?? performanceTemplate ?? DEFAULT_SCENE_VISUAL_SETTINGS;
      return [
        scene.id,
        cloneVisualSettings({
          intensity: settings.intensity,
          artDirection: settings.artDirection,
          temporal: settings.temporal ?? { ...DEFAULT_TEMPORAL_CONTROLS },
          animationStyle: defaultAnimationStyleForScene(scene.id),
        }),
      ];
    }),
  ) as SceneVisualSettingsMap;
}

function parseStructure(value: string, durationSeconds: number): CompositionSection[] {
  const lines = value
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length > 32) throw new Error("Song structure is limited to 32 timed sections");

  let previousSeconds = -1;
  return lines.map((line, index) => {
    const match = /^(\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?)\s+(.+)$/.exec(line);
    if (!match) throw new Error(`Structure line ${index + 1} must use “MM:SS section name”`);
    const seconds = timestampToSeconds(match[1]);
    if (seconds === undefined) throw new Error(`Structure line ${index + 1} has an invalid timestamp`);
    if (seconds <= previousSeconds) throw new Error("Structure timestamps must be strictly increasing");
    if (seconds >= durationSeconds) throw new Error(`Structure line ${index + 1} must start before ${durationSeconds} seconds`);
    previousSeconds = seconds;
    return { time: match[1], section: match[2].trim() };
  });
}

export function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engineRef = useRef(new AudioEngine());
  const routerRef = useRef(new ControlRouter());
  const tapTempoRef = useRef(new TapTempo());
  const visualRef = useRef<VisualEngine | null>(null);
  const recorderRef = useRef<SocialRecorder | null>(null);
  const selectedTrackRef = useRef<TrackId>("drums");
  const selectedSceneRef = useRef<VisualSceneId>(DEFAULT_TEMPLATE.scene);
  const intensityRef = useRef(DEFAULT_TEMPLATE.intensity);
  const snapshotRef = useRef(INITIAL_SNAPSHOT);
  const selectedPresetRef = useRef<SocialPreset>(SOCIAL_PRESETS[2]);
  const generationLockRef = useRef(false);
  const cancellationLockRef = useRef(false);
  const activeGenerationIdRef = useRef<string | undefined>(undefined);
  const submissionRequestIdRef = useRef<string | undefined>(undefined);
  const stepDragRef = useRef<{ active: boolean } | undefined>(undefined);
  const lyriaBufferCancelRef = useRef(false);
  const liveUpdateSignatureRef = useRef<Record<LyriaRealtimeDeckId, string>>({ main: "", sequence: "", vocal: "" });
  const realtimePrebufferRef = useRef<Record<LyriaRealtimeDeckId, BufferedRealtimeChunk[]>>(createRealtimePrebuffer());
  const realtimePrebufferStartedRef = useRef(false);
  const realtimePollInFlightRef = useRef(false);
  const sceneVisualSettingsRef = useRef<SceneVisualSettingsMap>({
    ...createInitialSceneVisualSettings(),
    [DEFAULT_TEMPLATE.scene]: cloneVisualSettings(DEFAULT_SCENE_VISUAL_SETTINGS),
  });

  const [snapshot, setSnapshot] = useState(INITIAL_SNAPSHOT);
  const [selectedTrack, setSelectedTrack] = useState<TrackId>("drums");
  const [selectedScene, setSelectedScene] = useState<VisualSceneId>(DEFAULT_TEMPLATE.scene);
  const [intensity, setIntensity] = useState(DEFAULT_TEMPLATE.intensity);
  const [artDirection, setArtDirection] = useState<VisualArtDirection>(DEFAULT_TEMPLATE.artDirection);
  const [temporalControls, setTemporalControls] = useState<VisualTemporalControls>(DEFAULT_TEMPLATE.temporal ?? { ...DEFAULT_TEMPORAL_CONTROLS });
  const [animationStyle, setAnimationStyle] = useState<VisualAnimationStyle>(defaultAnimationStyleForScene(DEFAULT_TEMPLATE.scene));
  const [sceneVisualSettings, setSceneVisualSettings] = useState<SceneVisualSettingsMap>(() => sceneVisualSettingsRef.current);
  const [controllerStatus, setControllerStatus] = useState(INITIAL_CONTROLLER_STATUS);
  const [renderStats, setRenderStats] = useState<RenderStats>({ fps: 0, frameTimeMs: 0, pixelRatio: 1, quality: "adaptive" });
  const [providerStatus, setProviderStatus] = useState<ProviderStatus>({ available: false, provider: "checking" });
  const [lyriaRealtimeStatus, setLyriaRealtimeStatus] = useState<LyriaRealtimeStatus>({
    deck: "main",
    available: false,
    provider: "checking",
    model: "models/lyria-realtime-exp",
    sampleRateHz: 48_000,
    channels: 2,
    audioFormat: "pcm16",
    instrumentalOnly: true,
    bufferedAudioBytes: 0,
    streamedAudioBytes: 0,
  });
  const [lyriaRealtimeConfig, setLyriaRealtimeConfig] = useState<LyriaRealtimeConfig>({ ...DEFAULT_REALTIME_REQUEST.config });
  const [lyriaPrompts, setLyriaPrompts] = useState<LyriaWeightedPrompt[]>(DEFAULT_REALTIME_REQUEST.weightedPrompts);
  const [lyriaStyleId, setLyriaStyleId] = useState(DEFAULT_REALTIME_STYLE.id);
  const [lyriaStyleGuidance, setLyriaStyleGuidance] = useState<Record<string, LyriaStyleGuidance>>({});
  const [lyriaGuidanceDialog, setLyriaGuidanceDialog] = useState<LyriaGuidanceDialogState>();
  const [lyriaSession, setLyriaSession] = useState<LyriaRealtimeSession>();
  const [sequenceLyriaSession, setSequenceLyriaSession] = useState<LyriaRealtimeSession>();
  const [vocalLyriaSession, setVocalLyriaSession] = useState<LyriaRealtimeSession>();
  const [lyriaStreamBytes, setLyriaStreamBytes] = useState(0);
  const [sequenceLyriaStreamBytes, setSequenceLyriaStreamBytes] = useState(0);
  const [vocalLyriaStreamBytes, setVocalLyriaStreamBytes] = useState(0);
  const [lyriaDeckControls, setLyriaDeckControls] = useState<Record<LyriaRealtimeDeckId, LyriaDeckControl>>(() => ({
    main: { ...DEFAULT_LYRIA_DECK_CONTROLS.main },
    sequence: { ...DEFAULT_LYRIA_DECK_CONTROLS.sequence },
    vocal: { ...DEFAULT_LYRIA_DECK_CONTROLS.vocal },
  }));
  const [lyriaDeckScenes, setLyriaDeckScenes] = useState<LyriaDeckScene[]>(() => {
    try {
      return loadLyriaDeckScenes(window.localStorage.getItem(LYRIA_DECK_SCENE_STORAGE_KEY));
    } catch {
      return loadLyriaDeckScenes();
    }
  });
  const [activeLyriaDeckSceneId, setActiveLyriaDeckSceneId] = useState<string>();
  const [lyriaDeckSceneDialog, setLyriaDeckSceneDialog] = useState<LyriaDeckScene>();
  const [lyriaRealtimeBusy, setLyriaRealtimeBusy] = useState(false);
  const [lyriaBuffering, setLyriaBuffering] = useState<LyriaBufferingState>({ active: false, message: "", bytes: 0 });
  const [autoDjMode, setAutoDjMode] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [autoDjStep, setAutoDjStep] = useState(0);
  const [agentStatus, setAgentStatus] = useState<AgentStatus>({ available: false, provider: "checking" });
  const [agentGoal, setAgentGoal] = useState("Make this set evolve into a darker peak-time system with sharper drums and a clearer visual hook.");
  const [agentPlan, setAgentPlan] = useState<AgentPlan>();
  const [agentBusy, setAgentBusy] = useState(false);
  const [prompt, setPrompt] = useState(DEFAULT_TEMPLATE.prompt);
  const [generationDuration, setGenerationDuration] = useState(150);
  const [generationBpm, setGenerationBpm] = useState(DEFAULT_TEMPLATE.bpm);
  const [generationKey, setGenerationKey] = useState("F minor");
  const [tonalCenter, setTonalCenter] = useState("deep F sub bass, bright minor chord stabs, crisp metallic top loop");
  const [productionIntensity, setProductionIntensity] = useState(0.82);
  const [negativePrompt, setNegativePrompt] = useState("muddy low end, weak kick, random fills, long intro, long fade out, washed out transients");
  const [instrumental, setInstrumental] = useState(true);
  const [generationLanguage, setGenerationLanguage] = useState<(typeof LYRIA_VOCAL_LANGUAGES)[number]>("English");
  const [lyrics, setLyrics] = useState("");
  const [structureText, setStructureText] = useState(DEFAULT_STRUCTURE);
  const [outputFormat, setOutputFormat] = useState<AudioOutputFormat>("mp3");
  const [budgetConfirmed, setBudgetConfirmed] = useState(false);
  const [rightsDeclared, setRightsDeclared] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [generation, setGeneration] = useState<GenerationTask>();
  const [selectedPreset, setSelectedPreset] = useState<SocialPreset>(SOCIAL_PRESETS[2]);
  const [recording, setRecording] = useState(false);
  const [recordProgress, setRecordProgress] = useState(0);
  const [lastRecording, setLastRecording] = useState<RecordingResult>();
  const [collapsedPanels, setCollapsedPanels] = useState<Set<StudioPanelId>>(() => new Set([
    "audio-templates",
    "audio-agent",
    "audio-generation",
    "av-output",
  ]));
  const [notice, setNotice] = useState(`${DEFAULT_TEMPLATE.name} loaded. Press play to buffer Lyria RealTime as the primary output.`);

  const selectedSceneMeta = VISUAL_SCENES.find((scene) => scene.id === selectedScene) ?? VISUAL_SCENES[0];
  const lyriaAvailable = providerStatus.available && providerStatus.provider === "lyria_3_pro";
  const hasUserSuppliedLyrics = !instrumental && lyrics.trim().length > 0;
  const generationIsActive = generation !== undefined && ["queued", "processing"].includes(generation.status);
  const sequencerGuideSignature = useMemo(() => sequencerSignature(snapshot), [snapshot]);
  const paidGenerationReady =
    lyriaAvailable &&
    prompt.trim().length > 0 &&
    budgetConfirmed &&
    (!hasUserSuppliedLyrics || rightsDeclared) &&
    !generating &&
    !generationIsActive;

  const activeLyriaStyle = useMemo(() => applyPrimaryGuidance(
    lyriaRealtimeStyleById(lyriaStyleId),
    lyriaStyleGuidance[lyriaStyleId],
  ), [lyriaStyleGuidance, lyriaStyleId]);
  const realtimeRequest = useMemo<LyriaRealtimeRequest>(
    () => ({
      weightedPrompts: lyriaPrompts.filter((prompt) => prompt.text.trim().length > 0).slice(0, 4),
      config: {
        ...lyriaRealtimeConfig,
        bpm: compensateLyriaBpmForPitch(snapshot.bpm, lyriaDeckControls.main.pitchSemitones),
      },
    }),
    [lyriaDeckControls.main.pitchSemitones, lyriaPrompts, lyriaRealtimeConfig, snapshot.bpm],
  );
  const sequenceRealtimeRequest = useMemo<LyriaRealtimeRequest>(() => ({
    weightedPrompts: createLyriaSequencePrompts(snapshot, activeLyriaStyle),
    config: createLyriaSequenceConfig(snapshot, lyriaRealtimeConfig, lyriaDeckControls.sequence.pitchSemitones),
  }), [activeLyriaStyle, lyriaDeckControls.sequence.pitchSemitones, lyriaRealtimeConfig, sequencerGuideSignature]);
  const vocalRealtimeRequest = useMemo<LyriaRealtimeRequest>(() => ({
    weightedPrompts: createLyriaVocalPrompts(activeLyriaStyle),
    config: {
      ...lyriaRealtimeConfig,
      bpm: compensateLyriaBpmForPitch(snapshot.bpm, lyriaDeckControls.vocal.pitchSemitones),
      guidance: Math.min(6, lyriaRealtimeConfig.guidance + 0.7),
      density: Math.max(0.12, Math.min(0.52, lyriaRealtimeConfig.density * 0.62)),
      brightness: Math.max(0.3, Math.min(0.82, lyriaRealtimeConfig.brightness + 0.1)),
      muteBass: true,
      muteDrums: true,
      onlyBassAndDrums: false,
      musicGenerationMode: "VOCALIZATION",
    },
  }), [activeLyriaStyle, lyriaDeckControls.vocal.pitchSemitones, lyriaRealtimeConfig, snapshot.bpm]);

  const stopTransportAndRealtime = useCallback(async (announce = false) => {
    lyriaBufferCancelRef.current = true;
    realtimePrebufferRef.current = createRealtimePrebuffer();
    engineRef.current.stop();
    engineRef.current.setRealtimeStreamPrimary(false);
    setLyriaBuffering((current) => ({ active: false, message: "", bytes: current.bytes }));
    if (!lyriaSession && !sequenceLyriaSession && !vocalLyriaSession) {
      if (announce) setNotice("Transport stopped.");
      return;
    }
    setLyriaRealtimeBusy(true);
    try {
      await Promise.all(LYRIA_DECKS.map((deck) => stopLyriaRealtime(deck)));
      setLyriaSession(undefined);
      setSequenceLyriaSession(undefined);
      setVocalLyriaSession(undefined);
      LYRIA_DECKS.forEach((deck) => engineRef.current.resetRealtimeDeckClock(deck));
      if (announce) setNotice("Transport and all Lyria RealTime decks stopped.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not stop Lyria RealTime stream");
    } finally {
      setLyriaRealtimeBusy(false);
    }
  }, [lyriaSession, sequenceLyriaSession, vocalLyriaSession]);

  const ingestRealtimeAudioPoll = useCallback(async (deck: LyriaRealtimeDeckId, schedule = true): Promise<number> => {
    const poll = await pollLyriaRealtimeAudio(deck);
    if (poll.warning) setLyriaRealtimeStatus((current) => ({ ...current, warning: poll.warning }));
    if (deck === "main") setLyriaStreamBytes(poll.streamedAudioBytes);
    if (deck === "sequence") setSequenceLyriaStreamBytes(poll.streamedAudioBytes);
    if (deck === "vocal") setVocalLyriaStreamBytes(poll.streamedAudioBytes);
    setLyriaBuffering((current) => current.active ? { ...current, bytes: current.bytes + poll.chunks.reduce((sum, chunk) => sum + chunk.length, 0) } : current);
    let bytesScheduled = 0;
    for (const chunk of poll.chunks) {
      bytesScheduled += chunk.length;
      const bytes = new Uint8Array(chunk);
      if (schedule) {
        await engineRef.current.playRealtimePcm16(bytes, poll.sampleRateHz, poll.channels, deck);
      } else {
        realtimePrebufferRef.current[deck].push({ bytes, sampleRateHz: poll.sampleRateHz, channels: poll.channels });
      }
    }
    return bytesScheduled;
  }, []);

  const waitForRealtimeAudioFrames = useCallback(async (deck: LyriaRealtimeDeckId, schedule = true): Promise<number> => {
    const deadline = performance.now() + LYRIA_STREAM_STARTUP_TIMEOUT_MS;
    let receivedBytes = 0;
    while (receivedBytes < LYRIA_PREBUFFER_BYTES && performance.now() < deadline) {
      if (lyriaBufferCancelRef.current) return 0;
      receivedBytes += await ingestRealtimeAudioPoll(deck, schedule);
      if (receivedBytes < LYRIA_PREBUFFER_BYTES) await WAIT(LYRIA_STREAM_POLL_MS);
    }
    return receivedBytes;
  }, [ingestRealtimeAudioPoll]);

  const flushSynchronizedRealtimePrebuffer = useCallback(async () => {
    await engineRef.current.synchronizeRealtimeDeckClocks(LYRIA_PLAYBACK_LEAD_SECONDS);
    const prebuffer = realtimePrebufferRef.current;
    const chunkCount = Math.max(...LYRIA_DECKS.map((deck) => prebuffer[deck].length));
    for (let index = 0; index < chunkCount; index += 1) {
      await Promise.all(LYRIA_DECKS.map(async (deck) => {
        const chunk = prebuffer[deck][index];
        if (chunk) await engineRef.current.playRealtimePcm16(chunk.bytes, chunk.sampleRateHz, chunk.channels, deck);
      }));
    }
    realtimePrebufferRef.current = createRealtimePrebuffer();
  }, []);

  const startAllRealtimeDecks = useCallback(async () => {
    const [main, sequence, vocal] = await Promise.all([
      startLyriaRealtime(realtimeRequest, "main"),
      startLyriaRealtime(sequenceRealtimeRequest, "sequence"),
      startLyriaRealtime(vocalRealtimeRequest, "vocal"),
    ]);
    setLyriaSession(main);
    setSequenceLyriaSession(sequence);
    setVocalLyriaSession(vocal);
    liveUpdateSignatureRef.current = {
      main: JSON.stringify(realtimeRequest),
      sequence: JSON.stringify(sequenceRealtimeRequest),
      vocal: JSON.stringify(vocalRealtimeRequest),
    };
    return { main, sequence, vocal };
  }, [realtimeRequest, sequenceRealtimeRequest, vocalRealtimeRequest]);

  const handleTransportToggle = useCallback(async () => {
    if (snapshotRef.current.playing) {
      await stopTransportAndRealtime();
      return;
    }
    if (!snapshotRef.current.playing) {
      if (lyriaRealtimeStatus.available && lyriaRealtimeBusy) {
        setNotice("Lyria decks are still preparing. Wait for the synchronized buffer.");
        return;
      }
      if (lyriaRealtimeStatus.available && !lyriaRealtimeBusy) {
        setLyriaRealtimeBusy(true);
        lyriaBufferCancelRef.current = false;
        setLyriaBuffering({ active: true, message: "Buffering 3 synchronized Lyria decks", bytes: 0 });
        try {
          await Promise.all(LYRIA_DECKS.map((deck) => stopLyriaRealtime(deck)));
          await startAllRealtimeDecks();
          engineRef.current.setRealtimeStreamPrimary(true);
          realtimePrebufferRef.current = createRealtimePrebuffer();
          const [mainBytes, sequenceBytes, vocalBytes] = await Promise.all(LYRIA_DECKS.map((deck) => waitForRealtimeAudioFrames(deck, false)));
          if (mainBytes < LYRIA_MIN_START_BYTES) {
            setLyriaBuffering({ active: false, message: "", bytes: 0 });
            setNotice("The main Lyria deck did not build a stable audio buffer before the startup deadline.");
            return;
          }
          await flushSynchronizedRealtimePrebuffer();
          const readyDecks = [mainBytes, sequenceBytes, vocalBytes].filter((bytes) => bytes >= LYRIA_MIN_START_BYTES).length;
          setLyriaBuffering({ active: false, message: "", bytes: mainBytes + sequenceBytes + vocalBytes });
          setNotice(`${readyDecks}/3 Lyria decks beat-locked to one audio clock · main, beat, vocal muted.`);
        } catch (error) {
          await Promise.all(LYRIA_DECKS.map((deck) => stopLyriaRealtime(deck).catch(() => undefined)));
          setLyriaSession(undefined);
          setSequenceLyriaSession(undefined);
          setVocalLyriaSession(undefined);
          engineRef.current.setRealtimeStreamPrimary(false);
          setLyriaBuffering({ active: false, message: "", bytes: 0 });
          setNotice(error instanceof Error ? error.message : "Lyria RealTime start failed");
          return;
        } finally {
          setLyriaRealtimeBusy(false);
        }
      }
    }
    await engineRef.current.toggle();
  }, [flushSynchronizedRealtimePrebuffer, lyriaRealtimeBusy, lyriaRealtimeStatus.available, startAllRealtimeDecks, stopTransportAndRealtime, waitForRealtimeAudioFrames]);

  const toggleDemoMode = useCallback(async () => {
    if (demoMode) {
      setDemoMode(false);
      setAutoDjMode(false);
      setNotice("Demo automation stopped. Lyria transport remains under manual control.");
      return;
    }
    setDemoMode(true);
    setAutoDjMode(true);
    setNotice("Demo mode is starting synchronized Lyria audio and visual automation.");
    if (!snapshotRef.current.playing) await handleTransportToggle();
  }, [demoMode, handleTransportToggle]);

  const refreshLyriaRealtimeStatus = useCallback(async () => {
    try {
      setLyriaRealtimeStatus(await getLyriaRealtimeStatus("main"));
    } catch (error) {
      setLyriaRealtimeStatus((current) => ({
        ...current,
        available: false,
        reason: error instanceof Error ? error.message : "Lyria RealTime status unavailable",
      }));
    }
  }, []);

  const startOrUpdateLyriaRealtime = useCallback(async () => {
    setLyriaRealtimeBusy(true);
    try {
      const [session, sequence, vocal] = lyriaSession && sequenceLyriaSession && vocalLyriaSession
        ? await Promise.all([
          updateLyriaRealtime(realtimeRequest, "main"),
          updateLyriaRealtime(sequenceRealtimeRequest, "sequence"),
          updateLyriaRealtime(vocalRealtimeRequest, "vocal"),
        ])
        : Object.values(await startAllRealtimeDecks());
      setLyriaSession(session);
      setSequenceLyriaSession(sequence);
      setVocalLyriaSession(vocal);
      liveUpdateSignatureRef.current = {
        main: JSON.stringify(realtimeRequest),
        sequence: JSON.stringify(sequenceRealtimeRequest),
        vocal: JSON.stringify(vocalRealtimeRequest),
      };
      engineRef.current.setRealtimeStreamPrimary(true);
      await refreshLyriaRealtimeStatus();
      setNotice(`${session.model} 3-deck system ${lyriaSession ? "updated" : "armed"}: main · sequence · vocalization.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Lyria RealTime command failed");
    } finally {
      setLyriaRealtimeBusy(false);
    }
  }, [lyriaSession, realtimeRequest, refreshLyriaRealtimeStatus, sequenceLyriaSession, sequenceRealtimeRequest, startAllRealtimeDecks, vocalLyriaSession, vocalRealtimeRequest]);

  const applyRealtimeRequest = useCallback(async (request: typeof realtimeRequest, label: string, styleId?: string) => {
    if (styleId) setLyriaStyleId(styleId);
    setLyriaPrompts(request.weightedPrompts);
    setLyriaRealtimeConfig(request.config);
    if (!lyriaSession) {
      setNotice(`${label} loaded into Lyria RealTime controls.`);
      return;
    }
    setLyriaRealtimeBusy(true);
    try {
      const session = await updateLyriaRealtime(request);
      setLyriaSession(session);
      liveUpdateSignatureRef.current.main = JSON.stringify(request);
      setNotice(`${label} sent to Lyria RealTime: ${session.config.bpm} BPM · density ${Math.round(session.config.density * 100)}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Lyria RealTime update failed");
    } finally {
      setLyriaRealtimeBusy(false);
    }
  }, [lyriaSession]);

  const applyRealtimeStyle = useCallback(async (styleId: string) => {
    setActiveLyriaDeckSceneId(undefined);
    const style = applyPrimaryGuidance(lyriaRealtimeStyleById(styleId), lyriaStyleGuidance[styleId]);
    await applyRealtimeRequest(createLyriaRealtimeRequestFromStyle(style), `${style.label} style`, style.id);
  }, [applyRealtimeRequest, lyriaStyleGuidance]);

  const updateLyriaDeckControl = useCallback((deck: LyriaRealtimeDeckId, update: Partial<LyriaDeckControl>) => {
    setActiveLyriaDeckSceneId(undefined);
    setLyriaDeckControls((current) => ({
      ...current,
      [deck]: { ...current[deck], ...update },
    }));
  }, []);

  const applyLyriaDeckScene = useCallback(async (scene: LyriaDeckScene) => {
    const normalized = normalizeLyriaDeckScene(scene, scene);
    const style = applyPrimaryGuidance(
      lyriaRealtimeStyleById(normalized.styleId),
      lyriaStyleGuidance[normalized.styleId],
    );
    const request = createLyriaRealtimeRequestFromStyle(style);
    engineRef.current.setBpm(normalized.bpm);
    setLyriaDeckControls({
      main: { ...normalized.controls.main },
      sequence: { ...normalized.controls.sequence },
      vocal: { ...normalized.controls.vocal },
    });
    setActiveLyriaDeckSceneId(normalized.id);
    await applyRealtimeRequest({
      ...request,
      config: {
        ...request.config,
        bpm: compensateLyriaBpmForPitch(normalized.bpm, normalized.controls.main.pitchSemitones),
      },
    }, `${normalized.name} deck scene`, style.id);
  }, [applyRealtimeRequest, lyriaStyleGuidance]);

  const applyLyriaDeckSceneByIndex = useCallback(async (index: number) => {
    const scene = lyriaDeckScenes[index];
    if (scene) await applyLyriaDeckScene(scene);
  }, [applyLyriaDeckScene, lyriaDeckScenes]);

  const saveLyriaDeckSceneDialog = useCallback(async (loadAfterSave: boolean) => {
    if (!lyriaDeckSceneDialog) return;
    const fallback = lyriaDeckScenes.find((scene) => scene.id === lyriaDeckSceneDialog.id) ?? lyriaDeckSceneDialog;
    const scene = normalizeLyriaDeckScene(lyriaDeckSceneDialog, fallback);
    setLyriaDeckScenes((current) => current.map((candidate) => candidate.id === scene.id ? scene : candidate));
    setLyriaDeckSceneDialog(undefined);
    if (loadAfterSave) await applyLyriaDeckScene(scene);
    else setNotice(`${scene.name} deck scene saved. Recall it with Shift+${lyriaDeckScenes.findIndex((candidate) => candidate.id === scene.id) + 1}.`);
  }, [applyLyriaDeckScene, lyriaDeckSceneDialog, lyriaDeckScenes]);

  const openLyriaGuidanceDialog = useCallback((styleId: string) => {
    const style = lyriaRealtimeStyleById(styleId);
    const primary = lyriaStyleGuidance[styleId] ?? style.prompts[0];
    setLyriaGuidanceDialog({ styleId, text: primary.text, weight: primary.weight });
  }, [lyriaStyleGuidance]);

  const applyLyriaGuidanceDialog = useCallback(async () => {
    if (!lyriaGuidanceDialog) return;
    const text = lyriaGuidanceDialog.text.trim();
    if (!text) {
      setNotice("Primary Lyria guidance cannot be empty.");
      return;
    }
    const guidance = { text: text.slice(0, 240), weight: lyriaGuidanceDialog.weight };
    const style = applyPrimaryGuidance(lyriaRealtimeStyleById(lyriaGuidanceDialog.styleId), guidance);
    setLyriaStyleGuidance((current) => ({ ...current, [style.id]: guidance }));
    setLyriaGuidanceDialog(undefined);
    if (style.id === lyriaStyleId) {
      await applyRealtimeRequest(createLyriaRealtimeRequestFromStyle(style), `${style.label} primary guidance`, style.id);
    } else {
      setNotice(`${style.label} primary guidance saved for the next switch.`);
    }
  }, [applyRealtimeRequest, lyriaGuidanceDialog, lyriaStyleId]);

  const stopRealtimeSession = useCallback(async () => {
    setLyriaRealtimeBusy(true);
    try {
      lyriaBufferCancelRef.current = true;
      await Promise.all(LYRIA_DECKS.map((deck) => stopLyriaRealtime(deck)));
      setLyriaSession(undefined);
      setSequenceLyriaSession(undefined);
      setVocalLyriaSession(undefined);
      engineRef.current.setRealtimeStreamPrimary(false);
      setLyriaBuffering((current) => ({ active: false, message: "", bytes: current.bytes }));
      await refreshLyriaRealtimeStatus();
      setNotice("All Lyria RealTime decks stopped.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Lyria RealTime stop failed");
    } finally {
      setLyriaRealtimeBusy(false);
    }
  }, [refreshLyriaRealtimeStatus]);

  const cancelLyriaBuffering = useCallback(async () => {
    lyriaBufferCancelRef.current = true;
    setLyriaBuffering((current) => ({ active: false, message: "", bytes: current.bytes }));
    setLyriaRealtimeBusy(true);
    try {
      await Promise.all(LYRIA_DECKS.map((deck) => stopLyriaRealtime(deck)));
      setLyriaSession(undefined);
      setSequenceLyriaSession(undefined);
      setVocalLyriaSession(undefined);
      engineRef.current.setRealtimeStreamPrimary(false);
      setNotice("Lyria RealTime deck buffering cancelled.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not cancel Lyria buffering");
    } finally {
      setLyriaRealtimeBusy(false);
    }
  }, []);

  const pollRealtimeAudio = useCallback(async () => {
    if (realtimePollInFlightRef.current) return;
    realtimePollInFlightRef.current = true;
    try {
      const activeDecks = LYRIA_DECKS.filter((deck) => (
        deck === "main" ? lyriaSession : deck === "sequence" ? sequenceLyriaSession : vocalLyriaSession
      ));
      await Promise.all(activeDecks.map((deck) => ingestRealtimeAudioPoll(deck)));
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Lyria RealTime audio polling failed");
    } finally {
      realtimePollInFlightRef.current = false;
    }
  }, [ingestRealtimeAudioPoll, lyriaSession, sequenceLyriaSession, vocalLyriaSession]);

  useEffect(() => {
    selectedTrackRef.current = selectedTrack;
  }, [selectedTrack]);

  useEffect(() => {
    selectedSceneRef.current = selectedScene;
  }, [selectedScene]);

  useEffect(() => {
    intensityRef.current = intensity;
  }, [intensity]);

  useEffect(() => {
    visualRef.current?.setArtDirection(artDirection);
  }, [artDirection]);

  useEffect(() => {
    visualRef.current?.setTemporalControls(temporalControls);
  }, [temporalControls]);

  useEffect(() => {
    visualRef.current?.setAnimationStyle(animationStyle);
  }, [animationStyle]);

  useEffect(() => {
    snapshotRef.current = snapshot;
  }, [snapshot]);

  useEffect(() => {
    for (const deck of LYRIA_DECKS) {
      engineRef.current.setRealtimeDeckControl(deck, lyriaDeckControls[deck]);
    }
  }, [lyriaDeckControls]);

  useEffect(() => {
    try {
      window.localStorage.setItem(LYRIA_DECK_SCENE_STORAGE_KEY, JSON.stringify(lyriaDeckScenes));
    } catch {
      // Presets remain available for this session when webview storage is disabled.
    }
  }, [lyriaDeckScenes]);

  useEffect(() => {
    if (!("mediaSession" in navigator)) return;
    navigator.mediaSession.playbackState = snapshot.playing ? "playing" : "paused";
  }, [snapshot.playing]);

  useEffect(() => {
    selectedPresetRef.current = selectedPreset;
  }, [selectedPreset]);

  useEffect(() => {
    sceneVisualSettingsRef.current = sceneVisualSettings;
  }, [sceneVisualSettings]);

  useEffect(() => engineRef.current.subscribe(setSnapshot), []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const visual = new VisualEngine(canvas, engineRef.current);
    visualRef.current = visual;
    visual.start();
    visual.setScene(DEFAULT_TEMPLATE.scene);
    visual.setIntensity(DEFAULT_TEMPLATE.intensity);
    visual.setArtDirection(DEFAULT_TEMPLATE.artDirection);
    visual.setTemporalControls(DEFAULT_TEMPLATE.temporal ?? { ...DEFAULT_TEMPORAL_CONTROLS });
    visual.setAnimationStyle(defaultAnimationStyleForScene(DEFAULT_TEMPLATE.scene));
    const recorder = new SocialRecorder(canvas, engineRef.current, visual);
    recorderRef.current = recorder;
    const unlistenStats = visual.subscribeStats(setRenderStats);
    const unlistenScene = visual.subscribeScene((scene) => {
      selectedSceneRef.current = scene;
      setSelectedScene(scene);
      const settings = sceneVisualSettingsRef.current[scene] ?? DEFAULT_SCENE_VISUAL_SETTINGS;
      setIntensity(settings.intensity);
      intensityRef.current = settings.intensity;
      setArtDirection(settings.artDirection);
      setTemporalControls(settings.temporal);
      setAnimationStyle(settings.animationStyle);
      visual.setIntensity(settings.intensity);
      visual.setArtDirection(settings.artDirection);
      visual.setTemporalControls(settings.temporal);
      visual.setAnimationStyle(settings.animationStyle);
    });
    const unlistenResults = recorder.subscribeResults((result) => {
      setRecording(false);
      setRecordProgress(1);
      setLastRecording(result);
      setNotice(`Captured ${(result.bytes / 1_000_000).toFixed(1)} MB. Ready to save.`);
    });
    const unlistenErrors = recorder.subscribeErrors((error) => {
      setRecording(false);
      setNotice(error.message);
    });
    return () => {
      unlistenStats();
      unlistenScene();
      unlistenResults();
      unlistenErrors();
      visual.dispose();
      visualRef.current = null;
      recorderRef.current = null;
    };
  }, []);

  const applySceneSettings = useCallback((settings: SceneVisualSettings) => {
    const next = cloneVisualSettings(settings);
    setIntensity(next.intensity);
    intensityRef.current = next.intensity;
    visualRef.current?.setIntensity(next.intensity);
    setArtDirection(next.artDirection);
    visualRef.current?.setArtDirection(next.artDirection);
    setTemporalControls(next.temporal);
    visualRef.current?.setTemporalControls(next.temporal);
    setAnimationStyle(next.animationStyle);
    visualRef.current?.setAnimationStyle(next.animationStyle);
  }, []);

  const saveSceneSettings = useCallback((scene: VisualSceneId, settings: Partial<SceneVisualSettings>) => {
    setSceneVisualSettings((current) => {
      const existing = current[scene] ?? DEFAULT_SCENE_VISUAL_SETTINGS;
      return {
        ...current,
        [scene]: cloneVisualSettings({
          intensity: settings.intensity ?? existing.intensity,
          artDirection: settings.artDirection ?? existing.artDirection,
          temporal: settings.temporal ?? existing.temporal,
          animationStyle: settings.animationStyle ?? existing.animationStyle,
        }),
      };
    });
  }, []);

  const changeScene = useCallback((scene: VisualSceneId) => {
    setSelectedScene(scene);
    selectedSceneRef.current = scene;
    visualRef.current?.setScene(scene);
    applySceneSettings(sceneVisualSettings[scene] ?? DEFAULT_SCENE_VISUAL_SETTINGS);
  }, [applySceneSettings, sceneVisualSettings]);

  const changeTrack = useCallback((track: TrackId) => {
    selectedTrackRef.current = track;
    setSelectedTrack(track);
  }, []);

  const changeIntensity = useCallback((next: number) => {
    const value = Math.max(0.05, Math.min(1, next));
    setIntensity(value);
    intensityRef.current = value;
    visualRef.current?.setIntensity(value);
    saveSceneSettings(selectedSceneRef.current, { intensity: value });
  }, [saveSceneSettings]);

  const changeArtDirection = useCallback((key: keyof VisualArtDirection, value: number) => {
    setArtDirection((current) => {
      const next = { ...current, [key]: Math.max(0, Math.min(1, value)) };
      saveSceneSettings(selectedSceneRef.current, { artDirection: next });
      return next;
    });
  }, [saveSceneSettings]);

  const changeTemporalControl = useCallback((key: keyof VisualTemporalControls, value: number) => {
    setTemporalControls((current) => {
      const next = { ...current, [key]: Math.max(0, Math.min(1, value)) };
      saveSceneSettings(selectedSceneRef.current, { temporal: next });
      return next;
    });
  }, [saveSceneSettings]);

  const changeAnimationStyle = useCallback((style: VisualAnimationStyle) => {
    const next = normalizeAnimationStyle(style);
    setAnimationStyle(next);
    visualRef.current?.setAnimationStyle(next);
    saveSceneSettings(selectedSceneRef.current, { animationStyle: next });
  }, [saveSceneSettings]);

  const applyVisualPreset = useCallback((presetId: string) => {
    const preset = VISUAL_PRESETS.find((candidate) => candidate.id === presetId) ?? VISUAL_PRESETS[0];
    const settings = {
      intensity: preset.intensity,
      artDirection: preset.artDirection,
      temporal: preset.temporal,
      animationStyle: sceneVisualSettings[preset.scene]?.animationStyle ?? defaultAnimationStyleForScene(preset.scene),
    };
    saveSceneSettings(preset.scene, settings);
    changeScene(preset.scene);
    applySceneSettings(settings);
    setNotice(`${preset.name} visual preset applied.`);
  }, [applySceneSettings, changeScene, saveSceneSettings, sceneVisualSettings]);

  const applyTemplate = useCallback((templateId: string) => {
    const template = performanceTemplateById(templateId);
    const baseStyle = lyriaRealtimeStyleForTemplate(template);
    const style = applyPrimaryGuidance(baseStyle, lyriaStyleGuidance[baseStyle.id]);
    engineRef.current.applyPerformanceTemplate(template);
    void applyRealtimeRequest(createLyriaRealtimeRequestForTemplate(template, style), `${template.name} realtime guide`, style.id);
    setPrompt(template.prompt);
    setGenerationBpm(template.bpm);
    const settings = {
      intensity: template.intensity,
      artDirection: template.artDirection,
      temporal: template.temporal ?? sceneVisualSettings[template.scene]?.temporal ?? { ...DEFAULT_TEMPORAL_CONTROLS },
      animationStyle: sceneVisualSettings[template.scene]?.animationStyle ?? defaultAnimationStyleForScene(template.scene),
    };
    saveSceneSettings(template.scene, settings);
    changeScene(template.scene);
    applySceneSettings(settings);
    setNotice(`${template.name} template applied across rhythm, mix, and visuals.`);
  }, [applyRealtimeRequest, applySceneSettings, changeScene, lyriaStyleGuidance, saveSceneSettings, sceneVisualSettings]);

  const applyAgentPlan = useCallback((plan: AgentPlan) => {
    const template = performanceTemplateById(plan.templateId);
    engineRef.current.applyPerformanceTemplate({ ...template, bpm: plan.bpm, scene: plan.scene, intensity: plan.intensity, artDirection: plan.artDirection });
    setPrompt(plan.prompt);
    setGenerationBpm(plan.bpm);
    const settings = {
      intensity: plan.intensity,
      artDirection: plan.artDirection,
      temporal: plan.temporal ?? template.temporal ?? sceneVisualSettings[plan.scene]?.temporal ?? { ...DEFAULT_TEMPORAL_CONTROLS },
      animationStyle: sceneVisualSettings[plan.scene]?.animationStyle ?? defaultAnimationStyleForScene(plan.scene),
    };
    saveSceneSettings(plan.scene, settings);
    changeScene(plan.scene);
    applySceneSettings(settings);
    setAgentPlan(plan);
    setNotice(`Agent applied: ${plan.title}.`);
  }, [applySceneSettings, changeScene, saveSceneSettings, sceneVisualSettings]);

  const stopRecording = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.getState() !== "recording") return;
    setNotice("Finalizing audio and video…");
    try {
      await recorder.stop();
    } catch (error) {
      setRecording(false);
      setNotice(error instanceof Error ? error.message : "Recording failed");
    }
  }, []);

  const startRecording = useCallback(async () => {
    const recorder = recorderRef.current;
    if (!recorder) return;
    const recorderState = recorder.getState();
    if (recorderState === "recording") {
      await stopRecording();
      return;
    }
    if (recorderState !== "idle") return;
    try {
      await engineRef.current.initialize();
      if (!snapshotRef.current.playing) {
        await engineRef.current.start();
      }
      setLastRecording(undefined);
      setRecordProgress(0);
      await recorder.start(selectedPresetRef.current);
      setRecording(true);
      setNotice(`Recording ${selectedPresetRef.current.label} at ${selectedPresetRef.current.width} × ${selectedPresetRef.current.height}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Recording is unavailable");
    }
  }, [stopRecording]);

  useEffect(() => {
    if (!recording) return;
    const timer = window.setInterval(() => {
      const recorder = recorderRef.current;
      if (!recorder) return;
      setRecordProgress(recorder.getProgress());
      if (recorder.getState() === "idle") setRecording(false);
    }, 100);
    return () => window.clearInterval(timer);
  }, [recording]);

  const handleControl = useCallback(
    async (message: ControlMessage) => {
      const engine = engineRef.current;
      const tracks = TRACK_IDS;
      const trackIndex = tracks.indexOf(selectedTrackRef.current);
      const scenes = VISUAL_SCENES.map((scene) => scene.id);
      const sceneIndex = scenes.indexOf(selectedSceneRef.current);
      switch (message.action) {
        case "transport.toggle":
          await handleTransportToggle();
          break;
        case "transport.stop":
          engine.stop();
          break;
        case "transport.record":
          await startRecording();
          break;
        case "tempo.tap": {
          const bpm = tapTempoRef.current.tap();
          if (bpm) engine.setBpm(bpm);
          break;
        }
        case "tempo.delta":
          engine.setBpm(snapshotRef.current.bpm + Math.sign(message.value ?? 0));
          break;
        case "track.next":
          changeTrack(tracks[(trackIndex + 1) % tracks.length]);
          break;
        case "track.previous":
          changeTrack(tracks[(trackIndex - 1 + tracks.length) % tracks.length]);
          break;
        case "track.mute":
          engine.toggleMute(selectedTrackRef.current);
          break;
        case "track.solo":
          engine.toggleSolo(selectedTrackRef.current);
          break;
        case "track.trigger": {
          const mapped = Number.isInteger(message.value) && (message.value ?? -1) >= 0 ? tracks[Math.min(tracks.length - 1, message.value ?? 0)] : selectedTrackRef.current;
          changeTrack(mapped);
          break;
        }
        case "master.delta":
          engine.setMasterVolume(snapshotRef.current.masterVolume + (message.value ?? 0) * 0.035);
          break;
        case "visual.next":
          changeScene(scenes[(sceneIndex + 1) % scenes.length]);
          break;
        case "visual.previous":
          changeScene(scenes[(sceneIndex - 1 + scenes.length) % scenes.length]);
          break;
        case "visual.scene.select": {
          const index = Math.round(message.value ?? -1);
          const scene = scenes[index];
          if (scene) changeScene(scene);
          break;
        }
        case "visual.intensity.delta":
          changeIntensity(intensityRef.current + (message.value ?? 0) * 0.04);
          break;
        case "visual.sculpture.delta":
          setArtDirection((current) => ({ ...current, sculpture: Math.max(0, Math.min(1, current.sculpture + (message.value ?? 0) * 0.04)) }));
          break;
        case "visual.motion.delta":
          setArtDirection((current) => ({ ...current, motion: Math.max(0, Math.min(1, current.motion + (message.value ?? 0) * 0.04)) }));
          break;
        case "visual.atmosphere.delta":
          setArtDirection((current) => ({ ...current, atmosphere: Math.max(0, Math.min(1, current.atmosphere + (message.value ?? 0) * 0.04)) }));
          break;
        case "visual.ribbon.delta":
          setArtDirection((current) => ({ ...current, ribbon: Math.max(0, Math.min(1, current.ribbon + (message.value ?? 0) * 0.04)) }));
          break;
        case "visual.temporal.speed.delta":
          changeTemporalControl("speed", temporalControls.speed + (message.value ?? 0) * 0.04);
          break;
        case "visual.temporal.strobe.delta":
          changeTemporalControl("strobe", temporalControls.strobe + (message.value ?? 0) * 0.04);
          break;
        case "visual.temporal.trail.delta":
          changeTemporalControl("trail", temporalControls.trail + (message.value ?? 0) * 0.04);
          break;
        case "visual.temporal.morph.delta":
          changeTemporalControl("morph", temporalControls.morph + (message.value ?? 0) * 0.04);
          break;
        case "visual.temporal.camera.delta":
          changeTemporalControl("camera", temporalControls.camera + (message.value ?? 0) * 0.04);
          break;
        case "visual.temporal.phase.delta":
          changeTemporalControl("phase", temporalControls.phase + (message.value ?? 0) * 0.04);
          break;
        case "lyria.deck-scene.select":
          await applyLyriaDeckSceneByIndex(Math.round(message.value ?? -1));
          break;
        case "performance.template.select": {
          const index = Math.round(message.value ?? -1);
          const template = PERFORMANCE_TEMPLATES[index % PERFORMANCE_TEMPLATES.length];
          if (template) applyTemplate(template.id);
          break;
        }
      }
    },
    [applyLyriaDeckSceneByIndex, applyTemplate, changeIntensity, changeScene, changeTemporalControl, changeTrack, handleTransportToggle, startRecording, temporalControls],
  );

  useEffect(() => {
    const router = routerRef.current;
    const unsubscribeControl = router.subscribe((message) => void handleControl(message));
    const unsubscribeStatus = router.subscribeStatus(setControllerStatus);
    void router.start();
    return () => {
      unsubscribeControl();
      unsubscribeStatus();
      void router.stop();
    };
  }, [handleControl]);

  useEffect(() => {
    void getProviderStatus().then(setProviderStatus).catch((error) => {
      setProviderStatus({ available: false, provider: "unavailable", reason: error instanceof Error ? error.message : String(error) });
    });
  }, []);

  useEffect(() => {
    void refreshLyriaRealtimeStatus();
  }, [refreshLyriaRealtimeStatus]);

  useEffect(() => {
    if (
      realtimePrebufferStartedRef.current ||
      !lyriaRealtimeStatus.available ||
      (lyriaSession && sequenceLyriaSession && vocalLyriaSession) ||
      lyriaRealtimeBusy
    ) {
      return;
    }
    realtimePrebufferStartedRef.current = true;
    setLyriaRealtimeBusy(true);
    void startAllRealtimeDecks()
      .then(() => {
        engineRef.current.setRealtimeStreamPrimary(true);
        setNotice("Three Lyria decks are prebuffering: main, sequence, and vocalization.");
      })
      .catch((error) => {
        engineRef.current.setRealtimeStreamPrimary(false);
        setNotice(error instanceof Error ? error.message : "Lyria RealTime multistream prebuffer failed");
      })
      .finally(() => setLyriaRealtimeBusy(false));
  }, [lyriaRealtimeBusy, lyriaRealtimeStatus.available, lyriaSession, sequenceLyriaSession, startAllRealtimeDecks, vocalLyriaSession]);

  useEffect(() => {
    if (!lyriaSession || snapshot.playing || lyriaBuffering.active) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      void getLyriaRealtimeStatus("main")
        .then((status) => {
          if (cancelled) return;
          setLyriaRealtimeStatus(status);
          setLyriaStreamBytes(status.streamedAudioBytes);
        })
        .catch((error) => {
          if (!cancelled) setNotice(error instanceof Error ? error.message : "Lyria RealTime status update failed");
        });
    }, 1_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [lyriaBuffering.active, lyriaSession, snapshot.playing]);

  useEffect(() => {
    const stopStepDrag = () => {
      stepDragRef.current = undefined;
    };
    window.addEventListener("pointerup", stopStepDrag);
    window.addEventListener("pointercancel", stopStepDrag);
    return () => {
      window.removeEventListener("pointerup", stopStepDrag);
      window.removeEventListener("pointercancel", stopStepDrag);
    };
  }, []);

  useEffect(() => {
    void getAgentStatus().then(setAgentStatus).catch((error) => {
      setAgentStatus({ available: false, provider: "unavailable", reason: error instanceof Error ? error.message : String(error) });
    });
  }, []);

  useEffect(() => {
    if (!autoDjMode) return;
    const timer = window.setInterval(() => {
      setAutoDjStep((step) => {
        const nextStep = step + 1;
        const baseStyle = LYRIA_REALTIME_STYLE_PRESETS[nextStep % LYRIA_REALTIME_STYLE_PRESETS.length];
        const style = applyPrimaryGuidance(baseStyle, lyriaStyleGuidance[baseStyle.id]);
        const styleRequest = createLyriaRealtimeRequestFromStyle(style);
        const guidedRequest = {
          ...styleRequest,
          config: {
            ...styleRequest.config,
            bpm: Math.max(60, Math.min(200, styleRequest.config.bpm + (nextStep % 3 === 0 ? 4 : 0))),
            density: Math.max(0, Math.min(1, styleRequest.config.density + ((nextStep % 4) - 1) * 0.04)),
            brightness: Math.max(0, Math.min(1, styleRequest.config.brightness + ((nextStep % 5) - 2) * 0.035)),
          },
        };
        setLyriaStyleId(style.id);
        setLyriaPrompts(styleRequest.weightedPrompts);
        setLyriaRealtimeConfig(guidedRequest.config);
        if (lyriaSession) {
          void updateLyriaRealtime(guidedRequest, "main")
            .then(setLyriaSession)
            .catch((error) => {
              setNotice(error instanceof Error ? error.message : "Auto DJ Lyria update failed");
            });
        }
        return nextStep;
      });
    }, demoMode ? 16_000 : 3_200);
    return () => window.clearInterval(timer);
  }, [autoDjMode, demoMode, lyriaSession, lyriaStyleGuidance]);

  useEffect(() => {
    if (!demoMode) return;
    let step = 0;
    const timer = window.setInterval(() => {
      step += 1;
      const preset = VISUAL_PRESETS[step % VISUAL_PRESETS.length];
      applyVisualPreset(preset.id);
    }, 12_000);
    return () => window.clearInterval(timer);
  }, [applyVisualPreset, demoMode]);

  useEffect(() => {
    if (!lyriaSession || !sequenceLyriaSession || !vocalLyriaSession || lyriaBuffering.active) return;
    const requests: Record<LyriaRealtimeDeckId, LyriaRealtimeRequest> = {
      main: realtimeRequest,
      sequence: sequenceRealtimeRequest,
      vocal: vocalRealtimeRequest,
    };
    const changedDecks = LYRIA_DECKS.filter((deck) => JSON.stringify(requests[deck]) !== liveUpdateSignatureRef.current[deck]);
    if (changedDecks.length === 0) return;
    const timer = window.setTimeout(() => {
      void Promise.all(changedDecks.map(async (deck) => {
        const session = await updateLyriaRealtime(requests[deck], deck);
        liveUpdateSignatureRef.current[deck] = JSON.stringify(requests[deck]);
        if (deck === "main") setLyriaSession(session);
        if (deck === "sequence") setSequenceLyriaSession(session);
        if (deck === "vocal") setVocalLyriaSession(session);
      }))
        .then(() => setNotice(`Lyria decks locked to ${snapshot.bpm} BPM · ${changedDecks.join(", ")} updated.`))
        .catch((error) => setNotice(error instanceof Error ? error.message : "Live Lyria deck update failed"));
    }, LYRIA_LIVE_UPDATE_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [lyriaBuffering.active, lyriaSession, realtimeRequest, sequenceLyriaSession, sequenceRealtimeRequest, snapshot.bpm, vocalLyriaSession, vocalRealtimeRequest]);

  useEffect(() => {
    if (!lyriaSession && !sequenceLyriaSession && !vocalLyriaSession) return;
    if (!snapshot.playing) return;
    let cancelled = false;
    const timer = window.setInterval(() => {
      if (!cancelled) void pollRealtimeAudio();
    }, LYRIA_STREAM_POLL_MS);
    void pollRealtimeAudio();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [lyriaSession, pollRealtimeAudio, sequenceLyriaSession, snapshot.playing, vocalLyriaSession]);

  const handleAgentPlan = async () => {
    const goal = agentGoal.trim();
    if (!goal || agentBusy) return;
    setAgentBusy(true);
    setNotice(agentStatus.provider === "meta_llm" ? "Meta-LLM director is planning the next performance state…" : "Local agent is planning the next performance state…");
    try {
      const plan = await createAgentPlan({
        goal,
        currentPrompt: prompt,
        bpm: snapshotRef.current.bpm,
        scene: selectedSceneRef.current,
        selectedTrack: selectedTrackRef.current,
      });
      applyAgentPlan(plan);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Agent director failed");
    } finally {
      setAgentBusy(false);
    }
  };

  const handleLocalMutate = () => {
    const normalized = prompt.trim();
    if (!normalized) return;
    engineRef.current.mutate(normalized);
    setNotice("Lyria beat-control pattern mutated from the prompt.");
  };

  const handleGenerate = async (loopMode = false) => {
    if (generationLockRef.current) return;
    if (activeGenerationIdRef.current) {
      setNotice("The current paid generation must finish or be cancelled before another candidate is submitted.");
      return;
    }
    generationLockRef.current = true;
    setGenerating(true);
    setCancelling(false);
    setGeneration(undefined);
    setNotice("Lyria request acknowledged. Validating the paid generation budget…");

    let task: GenerationTask | undefined;
    let compiledPrompt = prompt.trim();
    let receiptDetails: Parameters<typeof saveGenerationReceipt>[2] = {};
    let outcomeNotice = "Creative generation failed";
    const targetTrackId = selectedTrackRef.current;

    try {
      if (!lyriaAvailable) throw new Error(providerStatus.reason ?? "Lyria 3 Pro is not configured in the Rust backend");
      if (!budgetConfirmed) throw new Error(`Confirm the $${LYRIA_PRO_PRICE_USD.toFixed(2)} paid generation budget`);
      if (hasUserSuppliedLyrics && !rightsDeclared) {
        throw new Error("Declare that you own or may use the supplied lyrics before generating");
      }

      const normalized = prompt.trim();
      if (!normalized) throw new Error("Creative direction is required");
      const targetDuration = loopMode ? 32 : generationDuration;
      const targetOutputFormat = loopMode ? "wav" : outputFormat;
      const structure = loopMode
        ? [
            { time: "0:00", section: "bar 1 downbeat" },
            { time: "0:08", section: "variation enters" },
            { time: "0:16", section: "midpoint lift" },
            { time: "0:24", section: "return phrase" },
          ]
        : parseStructure(structureText, targetDuration);
      const specification: StructuredComposition = {
        durationSeconds: targetDuration,
        genre: [normalized],
        bpm: generationBpm,
        timeSignature: "4/4",
        loop: loopMode ? { enabled: true, bars: 16, seamless: true } : undefined,
        tonal: {
          key: generationKey.trim() || undefined,
          tonalCenter: tonalCenter.trim() || undefined,
          intensity: productionIntensity,
          negativePrompt: negativePrompt.trim() || undefined,
        },
        vocals: {
          enabled: loopMode ? false : !instrumental,
          language: loopMode || instrumental ? undefined : generationLanguage,
          lyrics: !loopMode && !instrumental && lyrics.trim() ? lyrics.trim() : undefined,
        },
        structure,
        socialHook: { startSeconds: 0, durationSeconds: Math.min(12, targetDuration) },
        visualSyncCues: [
          "clear beat transients",
          "audible section changes",
          "dynamic contrast for synchronized Three.js scenes",
          loopMode ? "bar-accurate downbeat for seamless visual loop capture" : "song-scale arrangement changes",
        ],
        outputFormat: targetOutputFormat,
      };
      const route = selectGenerationRoute(specification, loopMode ? {} : undefined);
      if (!route.availableInV1 || route.route !== "pro") throw new Error(route.reason);
      const reservation = reserveGenerationCost("pro", 1, LYRIA_PRO_PRICE_USD);
      compiledPrompt = compileLyriaPrompt(specification);
      const clientRequestId = submissionRequestIdRef.current ?? crypto.randomUUID();
      submissionRequestIdRef.current = clientRequestId;

      task = await generateMusic({
        prompt: compiledPrompt,
        durationSeconds: specification.durationSeconds,
        instrumental: loopMode ? true : instrumental,
        language: loopMode || instrumental ? undefined : generationLanguage,
        bpm: generationBpm,
        lyrics: !loopMode && !instrumental && lyrics.trim() ? lyrics.trim() : undefined,
        structure: structure.map((section) => ({
          timeSeconds: timestampToSeconds(section.time) ?? 0,
          section: section.section,
        })),
        outputFormat: targetOutputFormat,
        referenceAssets: [],
        seamlessLoop: loopMode,
        key: generationKey.trim() || undefined,
        tonalCenter: tonalCenter.trim() || undefined,
        negativePrompt: negativePrompt.trim() || undefined,
        productionIntensity,
        maxCostUsd: reservation.reservedCostUsd,
        candidateCount: 1,
        maxAttempts: 1,
        rightsDeclared: hasUserSuppliedLyrics && rightsDeclared,
        clientRequestId,
      });
      submissionRequestIdRef.current = undefined;
      activeGenerationIdRef.current = task.id;
      setGeneration(task);
      setBudgetConfirmed(false);
      setNotice(`Reserved $${(task.reservedCostUsd ?? reservation.reservedCostUsd).toFixed(2)}. Lyria is generating ${loopMode ? "a seamless loop" : "asynchronously"}…`);

      let consecutiveStatusErrors = 0;
      while (["queued", "processing"].includes(task.status)) {
        await WAIT(GENERATION_POLL_INTERVAL_MS);
        try {
          const hadStatusErrors = consecutiveStatusErrors > 0;
          task = await getGeneration(task.id);
          consecutiveStatusErrors = 0;
          setGeneration(task);
          if (hadStatusErrors && ["queued", "processing"].includes(task.status)) {
            setNotice("Lyria status connection restored. Generation is still active…");
          }
        } catch (error) {
          consecutiveStatusErrors += 1;
          if (consecutiveStatusErrors === 3) {
            setNotice("Lyria is still active, but status polling is temporarily unavailable. Musica will keep trying…");
          }
        }
      }

      if (task.status === "complete") {
        if (!task.hasAudio) throw new Error("Lyria completed without a downloadable audio asset");
        if (task.completedAfterCancel || task.cancellationRequested) {
          outcomeNotice = "Lyria completed after cancellation. The immutable asset and cost receipt were retained, but audio was not loaded automatically.";
        } else {
          const bytes = await downloadGeneratedAudio(task.id);
          const loaded = await engineRef.current.loadAudioFile(
            targetTrackId,
            bytes,
            `${task.title ?? task.id}.${targetOutputFormat}`,
            { declaredMimeType: task.audioMimeType, loop: loopMode, requireEncodedValidation: true },
          );
          if (loaded.analysis.bpm !== null) engineRef.current.setBpm(loaded.analysis.bpm);
          changeScene(loaded.analysis.recommendedScene);
          changeIntensity(loaded.analysis.visualIntensity);
          visualRef.current?.setAudioAnalysis(targetTrackId, loaded.analysis);
          receiptDetails = { encodedAudio: loaded.encoded, analysis: loaded.analysis };
          const measuredBpm = loaded.analysis.bpm === null ? "tempo unconfirmed" : `${loaded.analysis.bpm.toFixed(1)} BPM`;
          const sourceRate = loaded.encoded?.sampleRateHz ?? task.sampleRateHz ?? loaded.analysis.sampleRateHz;
          const sourceChannels = loaded.encoded?.channels ?? task.channels ?? loaded.analysis.channels;
          const musicalKey = loaded.analysis.key ?? "key unconfirmed";
          const outputWarning = task.errorCode === "output_shorter_than_requested"
            ? " Provider output was materially shorter than requested."
            : "";
          outcomeNotice = `${(loaded.encoded?.codec ?? targetOutputFormat).toUpperCase()} loaded ${loopMode ? "as a bar-quantized loop" : "one-shot"} into ${targetTrackId}: ${loaded.analysis.durationSeconds.toFixed(1)}s · ${(sourceRate / 1000).toFixed(1)} kHz source · ${sourceChannels} ch · ${measuredBpm} · ${musicalKey}.${outputWarning}`;
        }
      } else if (task.status === "failed") {
        throw new Error(task.errorCode ? `Lyria generation failed: ${task.errorCode}` : "Lyria generation failed");
      } else if (task.status === "cancelled") {
        outcomeNotice = task.providerCancelConfirmed
          ? "Lyria generation cancelled before provider dispatch."
          : "Cancellation recorded locally. Provider cancellation and charge remain unconfirmed.";
      }
    } catch (error) {
      outcomeNotice = error instanceof Error ? error.message : "Creative generation failed";
    } finally {
      if (task) {
        try {
          await saveGenerationReceipt(task, compiledPrompt, receiptDetails);
        } catch (error) {
          const receiptError = error instanceof Error ? error.message : "unknown receipt error";
          outcomeNotice = `${outcomeNotice} Receipt persistence failed: ${receiptError}`;
        }
        if (!["queued", "processing"].includes(task.status)) activeGenerationIdRef.current = undefined;
      }
      setNotice(outcomeNotice);
      setGenerating(false);
      setCancelling(false);
      generationLockRef.current = false;
    }
  };

  const handleCancelGeneration = async () => {
    const taskId = activeGenerationIdRef.current ?? generation?.id;
    if (!taskId || cancellationLockRef.current || generation?.cancellationRequested) return;
    cancellationLockRef.current = true;
    setCancelling(true);
    setNotice("Requesting cancellation…");
    try {
      const task = await cancelGeneration(taskId);
      setGeneration(task);
      if (!["queued", "processing"].includes(task.status)) activeGenerationIdRef.current = undefined;
      setNotice(
        task.providerCancelConfirmed
          ? "Generation cancelled before provider dispatch."
          : "Cancellation recorded locally; the provider may still finish and charge this request.",
      );
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not cancel generation");
    } finally {
      setCancelling(false);
      cancellationLockRef.current = false;
    }
  };

  const handleMidiFile = async (file?: File) => {
    if (!file) return;
    try {
      if (!/\.(?:mid|midi)$/i.test(file.name)) throw new Error("Choose a .mid or .midi file");
      const imported = importMidiPerformance(await file.arrayBuffer(), file.name);
      engineRef.current.applyImportedMidi(imported.tracks, imported.bpm);
      setNotice(`${imported.name} imported as Lyria beat-control patterns${imported.bpm ? ` · ${imported.bpm} BPM` : ""}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not import MIDI");
    }
  };

  const paintStep = (trackId: TrackId, step: number, active: boolean) => {
    engineRef.current.setStep(trackId, step, active);
  };

  const saveLastRecording = async () => {
    if (!lastRecording || !recorderRef.current) return;
    try {
      const path = await recorderRef.current.save(lastRecording);
      if (path) setNotice(`Saved ${path}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Could not save recording");
    }
  };

  const copyProgramSourceUrl = async () => {
    const url = `${window.location.origin}${window.location.pathname}`;
    try {
      await navigator.clipboard.writeText(url);
      setNotice(`Copied program source URL: ${url}`);
    } catch {
      setNotice(`Program source URL: ${url}`);
    }
  };

  const toggleStudioPanel = (panelId: StudioPanelId) => {
    setCollapsedPanels((current) => {
      const next = new Set(current);
      if (next.has(panelId)) next.delete(panelId);
      else next.add(panelId);
      return next;
    });
  };

  const renderStudioPanel = (panelId: StudioPanelId, title: string, meta: string, children: ReactNode, className = "") => {
    const collapsed = collapsedPanels.has(panelId);
    return (
      <section className={`studio-panel ${className} ${collapsed ? "collapsed" : ""}`}>
        <button
          className="studio-panel-toggle"
          type="button"
          aria-expanded={!collapsed}
          onClick={() => toggleStudioPanel(panelId)}
        >
          <span>{title}</span>
          <b>{meta}</b>
          <i aria-hidden="true">{collapsed ? "+" : "-"}</i>
        </button>
        {!collapsed && <div className="studio-panel-body">{children}</div>}
      </section>
    );
  };

  return (
    <main className="app-shell">
      {lyriaGuidanceDialog && (
        <div
          className="lyria-guidance-overlay"
          role="presentation"
          onPointerDown={(event) => {
            if (event.currentTarget === event.target) setLyriaGuidanceDialog(undefined);
          }}
        >
          <section className="lyria-guidance-dialog" role="dialog" aria-modal="true" aria-labelledby="lyria-guidance-title">
            <header>
              <span>PRIMARY GUIDANCE</span>
              <button type="button" onClick={() => setLyriaGuidanceDialog(undefined)} aria-label="Close guidance dialog">X</button>
            </header>
            <h2 id="lyria-guidance-title">{lyriaRealtimeStyleById(lyriaGuidanceDialog.styleId).label}</h2>
            <label className="guidance-copy">
              <span>DIRECTION</span>
              <textarea
                autoFocus
                maxLength={240}
                value={lyriaGuidanceDialog.text}
                onChange={(event) => setLyriaGuidanceDialog((current) => current ? { ...current, text: event.target.value } : current)}
              />
              <b>{lyriaGuidanceDialog.text.length}/240</b>
            </label>
            <label className="guidance-weight">
              <span>WEIGHT</span>
              <input
                type="range"
                min="0.1"
                max="3"
                step="0.05"
                value={lyriaGuidanceDialog.weight}
                onChange={(event) => setLyriaGuidanceDialog((current) => current ? { ...current, weight: Number(event.target.value) } : current)}
              />
              <b>{lyriaGuidanceDialog.weight.toFixed(2)}</b>
            </label>
            <div className="guidance-scope">
              <span>MAIN ARRANGEMENT</span>
            </div>
            <footer>
              <button
                type="button"
                onClick={() => {
                  const primary = lyriaRealtimeStyleById(lyriaGuidanceDialog.styleId).prompts[0];
                  setLyriaGuidanceDialog({ ...lyriaGuidanceDialog, text: primary.text, weight: primary.weight });
                }}
              >RESET</button>
              <button type="button" onClick={() => setLyriaGuidanceDialog(undefined)}>CANCEL</button>
              <button type="button" className="primary" onClick={() => void applyLyriaGuidanceDialog()} disabled={!lyriaGuidanceDialog.text.trim()}>APPLY</button>
            </footer>
          </section>
        </div>
      )}
      {lyriaDeckSceneDialog && (
        <div
          className="lyria-guidance-overlay"
          role="presentation"
          onPointerDown={(event) => {
            if (event.currentTarget === event.target) setLyriaDeckSceneDialog(undefined);
          }}
        >
          <section className="lyria-guidance-dialog deck-scene-dialog" role="dialog" aria-modal="true" aria-labelledby="deck-scene-title">
            <header>
              <span>MULTI-TRACK PRESET</span>
              <button type="button" onClick={() => setLyriaDeckSceneDialog(undefined)} aria-label="Close deck scene editor">X</button>
            </header>
            <h2 id="deck-scene-title">Edit deck scene</h2>
            <div className="deck-scene-identity">
              <label>
                <span>NAME</span>
                <input autoFocus maxLength={18} value={lyriaDeckSceneDialog.name} onChange={(event) => setLyriaDeckSceneDialog((current) => current ? { ...current, name: event.target.value } : current)} />
              </label>
              <label>
                <span>STYLE</span>
                <select value={lyriaDeckSceneDialog.styleId} onChange={(event) => setLyriaDeckSceneDialog((current) => current ? { ...current, styleId: event.target.value } : current)}>
                  {LYRIA_REALTIME_STYLE_PRESETS.map((style) => <option key={style.id} value={style.id}>{style.label}</option>)}
                </select>
              </label>
              <label>
                <span>BPM</span>
                <input type="number" min="60" max="200" value={lyriaDeckSceneDialog.bpm} onChange={(event) => setLyriaDeckSceneDialog((current) => current ? { ...current, bpm: Number(event.target.value) } : current)} />
              </label>
            </div>
            <div className="deck-scene-tracks">
              {LYRIA_DECKS.map((deck) => {
                const control = lyriaDeckSceneDialog.controls[deck];
                const updateControl = (update: Partial<LyriaDeckControl>) => setLyriaDeckSceneDialog((current) => current ? {
                  ...current,
                  controls: { ...current.controls, [deck]: { ...current.controls[deck], ...update } },
                } : current);
                return (
                  <article key={deck}>
                    <header>
                      <strong>{deck === "vocal" ? "VOCALIZE" : deck.toUpperCase()}</strong>
                      <label><input type="checkbox" checked={control.muted} onChange={(event) => updateControl({ muted: event.target.checked })} /> MUTE</label>
                    </header>
                    <label><span>VOL</span><input type="range" min="0" max="1" step="0.01" value={control.volume} onChange={(event) => updateControl({ volume: Number(event.target.value) })} /><b>{Math.round(control.volume * 100)}</b></label>
                    <label><span>PITCH</span><input type="range" min="-7" max="7" step="1" value={control.pitchSemitones} onChange={(event) => updateControl({ pitchSemitones: Number(event.target.value) })} /><b>{control.pitchSemitones > 0 ? `+${control.pitchSemitones}` : control.pitchSemitones}</b></label>
                    <label><span>BEAT</span><input type="range" min="-250" max="250" step="5" value={control.beatNudgeMs} onChange={(event) => updateControl({ beatNudgeMs: Number(event.target.value) })} /><b>{control.beatNudgeMs > 0 ? `+${control.beatNudgeMs}` : control.beatNudgeMs}</b></label>
                  </article>
                );
              })}
            </div>
            <footer className="deck-scene-footer">
              <button type="button" onClick={() => setLyriaDeckSceneDialog(undefined)}>CANCEL</button>
              <button type="button" onClick={() => void saveLyriaDeckSceneDialog(false)} disabled={!lyriaDeckSceneDialog.name.trim()}>SAVE</button>
              <button type="button" className="primary" onClick={() => void saveLyriaDeckSceneDialog(true)} disabled={!lyriaDeckSceneDialog.name.trim()}>SAVE + LOAD</button>
            </footer>
          </section>
        </div>
      )}
      {lyriaBuffering.active && (
        <div className="lyria-buffer-overlay" role="dialog" aria-modal="true" aria-live="assertive" aria-label="Buffering Lyria RealTime stream">
          <div className="lyria-buffer-dialog">
            <div className="buffer-orbit" aria-hidden="true"><i /><i /><i /></div>
            <span>LYRIA REALTIME</span>
            <h2>{lyriaBuffering.message}</h2>
            <p>Holding transport until each live PCM queue has stable headroom. Playback starts from a shared clock with Lyria as the exclusive audio output.</p>
            {(lyriaRealtimeStatus.warning || lyriaRealtimeStatus.reason) && (
              <p className="buffer-warning">{lyriaRealtimeStatus.warning ?? lyriaRealtimeStatus.reason}</p>
            )}
            <div className="buffer-meter">
              <b>{Math.round(Math.max(lyriaBuffering.bytes, lyriaStreamBytes) / 1024)} KB</b>
              <em>{lyriaSession?.model ?? lyriaRealtimeStatus.model}</em>
            </div>
            <button onClick={() => void cancelLyriaBuffering()}>CANCEL</button>
          </div>
        </div>
      )}
      <header className="topbar">
        <div className="brand" aria-label="Musica VJ">
          <span className="brand-mark">M</span>
          <div><strong>MUSICA</strong><span>VJ STUDIO</span></div>
        </div>

        <div className="transport" aria-label="Transport controls">
          <button className="icon-button" onClick={() => void stopTransportAndRealtime(true)} aria-label="Stop">■</button>
          <button
            className={`play-button ${snapshot.playing ? "is-playing" : ""}`}
            onClick={() => void handleTransportToggle()}
            aria-label={snapshot.playing ? "Pause" : "Play"}
            data-testid="transport-toggle"
          >
            {snapshot.playing ? "Ⅱ" : "▶"}
          </button>
          <label className="tempo-control">
            <span>BPM</span>
            <input type="number" min={60} max={200} value={snapshot.bpm} onChange={(event) => engineRef.current.setBpm(Number(event.target.value))} />
          </label>
          <button className="text-button" onClick={() => routerRef.current.dispatch("tempo.tap")}>TAP</button>
        </div>

        <div className="top-actions">
          <span className={`device-pill ${controllerStatus.midi ? "online" : ""}`} title={controllerStatus.midiInputs.join(", ") || "No MIDI input detected"}>
            <i /> MIDI {controllerStatus.midi ? `${controllerStatus.midiInputs.length} IN` : "READY"}
          </span>
          <span className={`device-pill ${controllerStatus.logitechBridge ? "online" : ""}`}>
            <i /> MX CONSOLE {controllerStatus.logitechBridge ? "LIVE" : "READY"}
          </span>
          <button className={`record-button ${recording ? "active" : ""}`} onClick={() => void startRecording()}>
            <span /> {recording ? "STOP" : "CAPTURE"}
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="left-panel panel">
          {renderStudioPanel("visual-scenes", "VISUAL BANK", `${VISUAL_SCENES.length} SCENES`, (
            <div className="scene-list">
              {VISUAL_SCENES.map((scene) => (
                <button key={scene.id} className={`scene-card ${scene.id === selectedScene ? "selected" : ""}`} onClick={() => changeScene(scene.id)}>
                  <span>{scene.label}</span><strong>{scene.name}</strong><i style={{ "--scene-color": scene.color } as React.CSSProperties} />
                </button>
              ))}
            </div>
          ))}

          {renderStudioPanel("visual-presets", "VJ PRESETS", `${VISUAL_PRESETS.length} LOOKS`, (
            <section className="template-bank visual-preset-bank" aria-label="VJ visual presets">
              <div className="template-grid">
                {VISUAL_PRESETS.map((preset) => (
                  <button key={preset.id} onClick={() => applyVisualPreset(preset.id)}>
                    <strong>{preset.name}</strong>
                    <span>{VISUAL_SCENES.find((scene) => scene.id === preset.scene)?.name ?? preset.scene}</span>
                  </button>
                ))}
              </div>
            </section>
          ))}

          {renderStudioPanel("visual-animation", "ANIMATION", selectedSceneMeta.label, (
            <section className="template-bank animation-style-bank" aria-label="Scene animation style">
              <div className="animation-style-grid">
                {VISUAL_ANIMATION_STYLES.map((style) => (
                  <button
                    key={style.id}
                    className={style.id === animationStyle ? "selected" : ""}
                    title={style.description}
                    onClick={() => changeAnimationStyle(style.id)}
                  >
                    <strong>{style.label}</strong>
                    <span>{style.description}</span>
                  </button>
                ))}
              </div>
            </section>
          ))}

          {renderStudioPanel("visual-reactivity", "REACTIVITY", `${Math.round(intensity * 100)}%`, (
            <label className="control-block">
              <span><b>SCENE</b><em>{Math.round(intensity * 100)}%</em></span>
              <input type="range" min="0.05" max="1" step="0.01" value={intensity} onChange={(event) => changeIntensity(Number(event.target.value))} />
            </label>
          ))}

          {renderStudioPanel("visual-macros", "ARTIST MACROS", selectedSceneMeta.label, (
            <section className="artist-macros" aria-label="Live visual instrument controls">
              {([
                ["sculpture", "SCULPTURE"],
                ["motion", "MOTION"],
                ["atmosphere", "ATMOSPHERE"],
                ["ribbon", "RIBBON"],
              ] as const).map(([key, label]) => (
                <label className={`artist-macro macro-${key}`} key={key}>
                  <span><b>{label}</b><em>{Math.round(artDirection[key] * 100)}</em></span>
                  <input
                    aria-label={`${label.toLowerCase()} macro`}
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={artDirection[key]}
                    onChange={(event) => changeArtDirection(key, Number(event.target.value))}
                  />
                </label>
              ))}
            </section>
          ))}

          {renderStudioPanel("visual-temporal", "TEMPORAL", selectedSceneMeta.label, (
            <section className="artist-macros temporal-controls" aria-label="Temporal visual controls">
              {([
                ["speed", "SPEED"],
                ["strobe", "STROBE"],
                ["trail", "TRAIL"],
                ["morph", "MORPH"],
                ["camera", "CAMERA"],
                ["phase", "PHASE"],
              ] as const).map(([key, label]) => (
                <label className={`artist-macro temporal-${key}`} key={key}>
                  <span><b>{label}</b><em>{Math.round(temporalControls[key] * 100)}</em></span>
                  <input
                    aria-label={`${label.toLowerCase()} temporal control`}
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={temporalControls[key]}
                    onChange={(event) => changeTemporalControl(key, Number(event.target.value))}
                  />
                </label>
              ))}
            </section>
          ))}
        </aside>

        <section className="performance-column">
          <div className={`visual-stage ${recording ? "is-recording" : ""}`}>
            <canvas ref={canvasRef} data-testid="visual-canvas" />
            <div className="stage-grid" />
            <div className="stage-topline">
              <span className="stage-brandline"><b>MUSICA</b><i />{selectedSceneMeta.name.toUpperCase()}</span>
              <span>{renderStats.fps || "—"} FPS · {renderStats.frameTimeMs || "—"} MS{snapshot.droppedLateSteps > 0 ? ` · ${snapshot.droppedLateSteps} LATE STEP${snapshot.droppedLateSteps === 1 ? "" : "S"} DROPPED` : ""}</span>
            </div>
            <div className="stage-side-readout" aria-hidden="true">
              <i /><span>{snapshot.bpm}</span><small>BPM</small><i /><small>48K</small>
            </div>
            <div className="stage-color-key" aria-hidden="true"><i /><i /><i /><span /></div>
            <div className="stage-title">
              <span>AUDIO REACTIVE / LIVE GENERATIVE SET</span>
              <strong>{selectedSceneMeta.name}</strong>
              <small>REALTIME SPECTRAL SCULPTURE · LOCAL ANALYSIS</small>
            </div>
            <div className="stage-footer">
              <span>{String(snapshot.currentStep + 1).padStart(2, "0")} / 16</span>
              <div className="beat-line"><i style={{ width: `${((snapshot.currentStep + 1) / 16) * 100}%` }} /></div>
              <span>{snapshot.bpm} BPM</span>
            </div>
            {recording && (
              <div className="record-overlay">
                <span>REC</span>
                <div><i style={{ width: `${recordProgress * 100}%` }} /></div>
                <b>{Math.ceil(selectedPreset.durationSeconds * (1 - recordProgress))}s</b>
              </div>
            )}
          </div>

          <div className="sequencer panel">
            <div className="panel-heading sequence-heading">
              <span>LYRIA BEAT / 16 STEPS</span>
              <strong className={sequenceLyriaSession ? "online" : ""}><i /> LYRIA</strong>
              <label>
                <em>VOL</em>
                <input type="range" min="0" max="1" step="0.01" value={lyriaDeckControls.sequence.volume} onChange={(event) => updateLyriaDeckControl("sequence", { volume: Number(event.target.value) })} />
                <b>{Math.round(lyriaDeckControls.sequence.volume * 100)}</b>
              </label>
              <label>
                <em>PITCH</em>
                <input type="range" min="-7" max="7" step="1" value={lyriaDeckControls.sequence.pitchSemitones} onChange={(event) => updateLyriaDeckControl("sequence", { pitchSemitones: Number(event.target.value) })} />
                <b>{lyriaDeckControls.sequence.pitchSemitones > 0 ? `+${lyriaDeckControls.sequence.pitchSemitones}` : lyriaDeckControls.sequence.pitchSemitones}</b>
              </label>
              <label>
                <em>BEAT</em>
                <input type="range" min="-250" max="250" step="5" value={lyriaDeckControls.sequence.beatNudgeMs} onChange={(event) => updateLyriaDeckControl("sequence", { beatNudgeMs: Number(event.target.value) })} />
                <b>{lyriaDeckControls.sequence.beatNudgeMs}</b>
              </label>
              <button
                className={lyriaDeckControls.sequence.muted ? "active" : ""}
                onClick={() => updateLyriaDeckControl("sequence", { muted: !lyriaDeckControls.sequence.muted })}
                title="Mute Lyria sequence stream"
              >M</button>
            </div>
            <label className="beat-midi-import">
              IMPORT MIDI
              <input type="file" accept=".mid,.midi" onChange={(event) => void handleMidiFile(event.target.files?.[0])} />
            </label>
            <div className="step-grid" data-testid="step-grid">
              {snapshot.tracks.filter((track) => track.id === "drums" || track.id === "bass").map((track) => (
                <div className={`step-row ${track.id === selectedTrack ? "selected" : ""}`} key={track.id}>
                  <button className="step-label" style={{ color: track.color }} onClick={() => changeTrack(track.id)}>{track.shortName}</button>
                  {track.pattern.map((active, step) => (
                    <button
                      key={step}
                      aria-label={`${track.name} step ${step + 1}`}
                      className={`${active ? "active" : ""} ${snapshot.playing && snapshot.currentStep === step ? "playhead" : ""}`}
                      style={active ? { "--track-color": track.color } as React.CSSProperties : undefined}
                      onPointerDown={(event) => {
                        event.preventDefault();
                        const nextActive = !active;
                        stepDragRef.current = { active: nextActive };
                        paintStep(track.id, step, nextActive);
                      }}
                      onPointerEnter={() => {
                        const drag = stepDragRef.current;
                        if (drag) paintStep(track.id, step, drag.active);
                      }}
                    />
                  ))}
                </div>
              ))}
            </div>
          </div>
        </section>

        <aside className="right-panel panel">
          {renderStudioPanel("audio-lyria", "LYRIA REALTIME", lyriaRealtimeStatus.available ? "READY" : "OFFLINE", (
            <section className="lyria-realtime-deck live-deck" aria-label="Lyria RealTime controls">
              <div className="realtime-status">
                <i className={lyriaRealtimeStatus.available ? "online" : ""} />
                <span>{lyriaSession ? "3-DECK STREAM" : lyriaRealtimeStatus.model}</span>
                <b>{Math.round((lyriaStreamBytes + sequenceLyriaStreamBytes + vocalLyriaStreamBytes) / 1024)} KB</b>
              </div>
              <div className="deck-scene-bank" aria-label="Multi-track deck scenes">
                <header><span>DECK SCENES</span><b>SHIFT + 1-4</b></header>
                <div>
                  {lyriaDeckScenes.map((scene, index) => (
                    <span className="deck-scene-slot" key={scene.id}>
                      <button
                        type="button"
                        className={scene.id === activeLyriaDeckSceneId ? "active" : ""}
                        onClick={() => void applyLyriaDeckScene(scene)}
                        disabled={lyriaRealtimeBusy}
                        title={`Load ${scene.name}: ${scene.bpm} BPM, ${lyriaRealtimeStyleById(scene.styleId).label}. Shift+${index + 1}`}
                      >
                        <em>{index + 1}</em>
                        <strong>{scene.name}</strong>
                        <small>{scene.bpm}</small>
                      </button>
                      <button type="button" className="deck-scene-edit" onClick={() => setLyriaDeckSceneDialog(cloneLyriaDeckScene(scene))} aria-label={`Edit ${scene.name} deck scene`}>E</button>
                    </span>
                  ))}
                </div>
              </div>
              <div className="lyria-stream-mixer" aria-label="Lyria stream mixer">
                {LYRIA_DECKS.map((deck) => {
                  const control = lyriaDeckControls[deck];
                  const session = deck === "main" ? lyriaSession : deck === "sequence" ? sequenceLyriaSession : vocalLyriaSession;
                  const bytes = deck === "main" ? lyriaStreamBytes : deck === "sequence" ? sequenceLyriaStreamBytes : vocalLyriaStreamBytes;
                  return (
                    <article className="lyria-stream-strip" key={deck}>
                      <header>
                        <i className={session ? "online" : ""} />
                        <strong>{deck === "vocal" ? "VOCALIZE" : deck.toUpperCase()}</strong>
                        <span>{session ? `${Math.round(bytes / 1024)}K` : "IDLE"}</span>
                        <button
                          className={control.muted ? "active" : ""}
                          onClick={() => updateLyriaDeckControl(deck, { muted: !control.muted })}
                          title={`Mute ${deck} Lyria stream`}
                        >M</button>
                      </header>
                      <label>
                        <span>VOL</span>
                        <input type="range" min="0" max="1" step="0.01" value={control.volume} onChange={(event) => updateLyriaDeckControl(deck, { volume: Number(event.target.value) })} />
                        <b>{Math.round(control.volume * 100)}</b>
                      </label>
                      <label>
                        <span>PITCH</span>
                        <input type="range" min="-7" max="7" step="1" value={control.pitchSemitones} onChange={(event) => updateLyriaDeckControl(deck, { pitchSemitones: Number(event.target.value) })} />
                        <b>{control.pitchSemitones > 0 ? `+${control.pitchSemitones}` : control.pitchSemitones}</b>
                      </label>
                      <label>
                        <span>BEAT</span>
                        <input type="range" min="-250" max="250" step="5" value={control.beatNudgeMs} onChange={(event) => updateLyriaDeckControl(deck, { beatNudgeMs: Number(event.target.value) })} />
                        <b>{control.beatNudgeMs > 0 ? `+${control.beatNudgeMs}` : control.beatNudgeMs}</b>
                      </label>
                    </article>
                  );
                })}
              </div>
              <label className="realtime-style-select">
                <span>STYLE</span>
                <select value={lyriaStyleId} onChange={(event) => void applyRealtimeStyle(event.target.value)}>
                  {LYRIA_REALTIME_STYLE_PRESETS.map((style) => (
                    <option key={style.id} value={style.id}>{style.label}</option>
                  ))}
                </select>
              </label>
              <small className="realtime-style-description">{activeLyriaStyle.description}</small>
              <div className="realtime-style-buttons">
                {LYRIA_REALTIME_STYLE_PRESETS.map((style) => (
                  <button
                    key={style.id}
                    className={style.id === lyriaStyleId ? "active" : ""}
                    onClick={() => void applyRealtimeStyle(style.id)}
                    onContextMenu={(event) => {
                      event.preventDefault();
                      openLyriaGuidanceDialog(style.id);
                    }}
                    disabled={lyriaRealtimeBusy}
                    aria-haspopup="dialog"
                    title={`${style.description} Right-click to edit primary guidance.`}
                  >
                    {style.label}
                  </button>
                ))}
              </div>
              <div className="realtime-grid">
                <label>
                  <span>BPM</span>
                  <input
                    type="number"
                    min={60}
                    max={200}
                    value={snapshot.bpm}
                    onChange={(event) => {
                      const bpm = Number(event.target.value);
                      setActiveLyriaDeckSceneId(undefined);
                      engineRef.current.setBpm(bpm);
                      setLyriaRealtimeConfig((current) => ({ ...current, bpm }));
                    }}
                  />
                </label>
                <label>
                  <span>DENS</span>
                  <input type="range" min="0" max="1" step="0.01" value={lyriaRealtimeConfig.density} onChange={(event) => setLyriaRealtimeConfig((current) => ({ ...current, density: Number(event.target.value) }))} />
                </label>
                <label>
                  <span>BRITE</span>
                  <input type="range" min="0" max="1" step="0.01" value={lyriaRealtimeConfig.brightness} onChange={(event) => setLyriaRealtimeConfig((current) => ({ ...current, brightness: Number(event.target.value) }))} />
                </label>
                <label>
                  <span>GUIDE</span>
                  <input type="range" min="0" max="6" step="0.05" value={lyriaRealtimeConfig.guidance} onChange={(event) => setLyriaRealtimeConfig((current) => ({ ...current, guidance: Number(event.target.value) }))} />
                </label>
              </div>
              <div className="realtime-toggles">
                <label><input type="checkbox" checked={lyriaRealtimeConfig.muteBass} onChange={(event) => setLyriaRealtimeConfig((current) => ({ ...current, muteBass: event.target.checked, onlyBassAndDrums: event.target.checked ? false : current.onlyBassAndDrums }))} /> BASS MUTE</label>
                <label><input type="checkbox" checked={lyriaRealtimeConfig.muteDrums} onChange={(event) => setLyriaRealtimeConfig((current) => ({ ...current, muteDrums: event.target.checked, onlyBassAndDrums: event.target.checked ? false : current.onlyBassAndDrums }))} /> DRUM MUTE</label>
                <label><input type="checkbox" checked={lyriaRealtimeConfig.onlyBassAndDrums} onChange={(event) => setLyriaRealtimeConfig((current) => ({ ...current, onlyBassAndDrums: event.target.checked, muteBass: event.target.checked ? false : current.muteBass, muteDrums: event.target.checked ? false : current.muteDrums }))} /> BASS+DRUMS</label>
              </div>
              <details className="advanced-prompts">
                <summary>LIVE PROMPTS</summary>
                <div className="realtime-prompts">
                  {lyriaPrompts.map((weightedPrompt, index) => (
                    <label key={index}>
                      <span>P{index + 1}</span>
                      <input
                        value={weightedPrompt.text}
                        maxLength={240}
                        onChange={(event) => setLyriaPrompts((current) => current.map((prompt, promptIndex) => (
                          promptIndex === index ? { ...prompt, text: event.target.value } : prompt
                        )))}
                      />
                      <input
                        aria-label={`Lyria prompt ${index + 1} weight`}
                        type="range"
                        min="-3"
                        max="3"
                        step="0.05"
                        value={weightedPrompt.weight}
                        onChange={(event) => setLyriaPrompts((current) => current.map((prompt, promptIndex) => (
                          promptIndex === index ? { ...prompt, weight: Number(event.target.value) || 1 } : prompt
                        )))}
                      />
                    </label>
                  ))}
                </div>
              </details>
              <div className="realtime-actions">
                <button onClick={() => void startOrUpdateLyriaRealtime()} disabled={lyriaRealtimeBusy}>
                  {lyriaSession ? "UPDATE 3 DECKS" : "START 3 DECKS"}
                </button>
                <button className={autoDjMode ? "active" : ""} onClick={() => setAutoDjMode((active) => !active)}>
                  AUTO DJ
                </button>
                <button className={demoMode ? "active" : ""} onClick={() => void toggleDemoMode()} disabled={lyriaRealtimeBusy}>
                  {demoMode ? "EXIT DEMO" : "DEMO"}
                </button>
                {lyriaSession && <button onClick={() => void stopRealtimeSession()} disabled={lyriaRealtimeBusy}>STOP</button>}
              </div>
              {lyriaRealtimeStatus.warning && <small>{lyriaRealtimeStatus.warning}</small>}
              {!lyriaRealtimeStatus.available && <small>{lyriaRealtimeStatus.reason ?? "Desktop Lyria RealTime bridge is not configured"}</small>}
            </section>
          ), "primary-panel")}

          {renderStudioPanel("audio-templates", "MUSICAL STYLES", `${PERFORMANCE_TEMPLATES.length} SETS`, (
            <section className="template-bank" aria-label="Performance templates">
              <div className="template-grid">
                {PERFORMANCE_TEMPLATES.map((template) => (
                  <button key={template.id} onClick={() => applyTemplate(template.id)}>
                    <strong>{template.name}</strong>
                    <span>{template.bpm} BPM</span>
                  </button>
                ))}
              </div>
            </section>
          ))}

          {renderStudioPanel("audio-agent", "AGENT DIRECTOR", agentStatus.available ? agentStatus.provider.toUpperCase() : "LOCAL", (
            <section className="creative-panel compact-creative">
              <textarea className="direction-input" value={prompt} maxLength={1000} onChange={(event) => setPrompt(event.target.value)} aria-label="Creative direction" />
              <button className="local-mutate-button" onClick={handleLocalMutate} disabled={!prompt.trim()}>
                MUTATE LYRIA BEAT
              </button>
              <section className="agent-director">
                <textarea value={agentGoal} maxLength={1500} onChange={(event) => setAgentGoal(event.target.value)} aria-label="Agent director goal" />
                <button className="agent-button" onClick={() => void handleAgentPlan()} disabled={!agentGoal.trim() || agentBusy}>
                  {agentBusy ? "PLANNING..." : "PLAN + APPLY SET"}
                </button>
                {agentPlan && (
                  <div className="agent-plan">
                    <strong>{agentPlan.title}</strong>
                    <span>{agentPlan.rationale}</span>
                  </div>
                )}
              </section>
            </section>
          ))}

          {renderStudioPanel("audio-generation", "LYRIA 3 EXPORT", lyriaAvailable ? "ONLINE" : "BATCH OFF", (
            <section className="creative-panel compact-creative">
              <div className="generation-grid">
                <label>
                  <span>DURATION</span>
                  <input aria-label="Song duration in seconds" type="number" min={31} max={180} step={1} value={generationDuration} onChange={(event) => setGenerationDuration(Number(event.target.value))} />
                  <em>SEC</em>
                </label>
                <label>
                  <span>TEMPO</span>
                  <input aria-label="Requested song BPM" type="number" min={60} max={200} step={1} value={generationBpm} onChange={(event) => setGenerationBpm(Number(event.target.value))} />
                  <em>BPM</em>
                </label>
                <label>
                  <span>KEY</span>
                  <input aria-label="Generated music key" maxLength={40} value={generationKey} onChange={(event) => setGenerationKey(event.target.value)} />
                </label>
                <label>
                  <span>INTENSITY</span>
                  <input aria-label="Production intensity" type="range" min={0} max={1} step={0.01} value={productionIntensity} onChange={(event) => setProductionIntensity(Number(event.target.value))} />
                  <em>{Math.round(productionIntensity * 100)}</em>
                </label>
                <label>
                  <span>FORMAT</span>
                  <select aria-label="Generated audio format" value={outputFormat} onChange={(event) => setOutputFormat(event.target.value as AudioOutputFormat)}>
                    <option value="mp3">MP3</option>
                    <option value="wav">WAV</option>
                  </select>
                </label>
                <label>
                  <span>LANGUAGE</span>
                  <select aria-label="Vocal language" value={generationLanguage} disabled={instrumental} onChange={(event) => setGenerationLanguage(event.target.value as (typeof LYRIA_VOCAL_LANGUAGES)[number])}>
                    {LYRIA_VOCAL_LANGUAGES.map((language) => <option value={language} key={language}>{language}</option>)}
                  </select>
                </label>
              </div>
              <label className="generation-check">
                <input type="checkbox" checked={instrumental} onChange={(event) => setInstrumental(event.target.checked)} />
                <span>Instrumental only</span>
              </label>
              <details className="advanced-prompts">
                <summary>EXPORT PROMPTS</summary>
                <label className="generation-textarea">
                  <span>LYRICS</span>
                  <textarea aria-label="User supplied lyrics" value={lyrics} maxLength={12_000} disabled={instrumental} placeholder={instrumental ? "Enable vocals to supply lyrics" : "Optional user supplied lyrics"} onChange={(event) => { setLyrics(event.target.value); setRightsDeclared(false); }} />
                </label>
                <label className="generation-textarea">
                  <span>TONAL CENTER / SOUND DESIGN</span>
                  <textarea aria-label="Tonal center and sound design" value={tonalCenter} maxLength={800} onChange={(event) => setTonalCenter(event.target.value)} />
                </label>
                <label className="generation-textarea">
                  <span>AVOID</span>
                  <textarea aria-label="Negative music prompt" value={negativePrompt} maxLength={800} onChange={(event) => setNegativePrompt(event.target.value)} />
                </label>
                <label className="generation-textarea structure-field">
                  <span>STRUCTURE</span>
                  <textarea aria-label="Timed song structure" value={structureText} maxLength={1600} onChange={(event) => setStructureText(event.target.value)} />
                </label>
              </details>
              <label className="generation-check consent-check">
                <input type="checkbox" checked={budgetConfirmed} onChange={(event) => setBudgetConfirmed(event.target.checked)} />
                <span>I approve one paid candidate, maximum $0.08</span>
              </label>
              <label className={`generation-check consent-check ${hasUserSuppliedLyrics ? "" : "is-optional"}`}>
                <input type="checkbox" checked={rightsDeclared} disabled={!hasUserSuppliedLyrics} onChange={(event) => setRightsDeclared(event.target.checked)} />
                <span>I own or may use supplied lyrics or reference assets</span>
              </label>
              <button className="generate-button" onClick={() => void handleGenerate()} disabled={!paidGenerationReady}>
                {generating ? "GENERATING..." : lyriaAvailable ? "GENERATE EXPORT" : "BATCH EXPORT OFFLINE"}
              </button>
              <button className="generate-button loop-generate-button" onClick={() => void handleGenerate(true)} disabled={!paidGenerationReady}>
                {generating ? "GENERATING LOOP..." : lyriaAvailable ? "GENERATE LOOP" : "GCP LOOP OFFLINE"}
              </button>
              {generationIsActive && (
                <button className="cancel-generation-button" onClick={() => void handleCancelGeneration()} disabled={cancelling || generation.cancellationRequested}>
                  {generation.cancellationRequested ? "CANCELLATION REQUESTED" : cancelling ? "CANCELLING..." : "CANCEL GENERATION"}
                </button>
              )}
              <p className="provider-note">
                <i className={lyriaAvailable ? "online" : ""} /> {lyriaAvailable ? `${providerStatus.model ?? "lyria-3-pro-preview"} batch export` : "Live playback uses Lyria RealTime"}
              </p>
              {generation && (
                <small className="generation-status" aria-live="polite">
                  {generation.id.slice(0, 12)} · {generation.status} · ${(generation.generationCostUsd ?? generation.reservedCostUsd ?? 0).toFixed(2)}
                </small>
              )}
            </section>
          ))}

          {renderStudioPanel("av-output", "AV OUTPUT", recording ? "REC" : selectedPreset.label.toUpperCase(), (
            <section className="av-output-panel">
              <div className="av-output-grid">
                <button className={recording ? "active" : ""} onClick={() => void startRecording()}>
                  {recording ? "STOP CAPTURE" : "START CAPTURE"}
                </button>
                <button onClick={() => void copyProgramSourceUrl()}>COPY OBS URL</button>
              </div>
              <label>
                <span>FORMAT</span>
                <select value={selectedPreset.id} onChange={(event) => setSelectedPreset(SOCIAL_PRESETS.find((preset) => preset.id === event.target.value) ?? SOCIAL_PRESETS[2])}>
                  {SOCIAL_PRESETS.map((preset) => <option value={preset.id} key={preset.id}>{preset.label} · {preset.width}x{preset.height}</option>)}
                </select>
              </label>
              {lastRecording && <button className="save-button av-save-button" onClick={() => void saveLastRecording()}>SAVE LAST CAPTURE</button>}
            </section>
          ))}
        </aside>
      </section>

      <footer className="footerbar">
        <div className="notice"><i /> {notice}</div>
        <div className="capture-settings">
          <label>
            FORMAT
            <select value={selectedPreset.id} onChange={(event) => setSelectedPreset(SOCIAL_PRESETS.find((preset) => preset.id === event.target.value) ?? SOCIAL_PRESETS[2])}>
              {SOCIAL_PRESETS.map((preset) => <option value={preset.id} key={preset.id}>{preset.label} · {preset.width}×{preset.height}</option>)}
            </select>
          </label>
          {lastRecording && <button className="save-button" onClick={() => void saveLastRecording()}>SAVE LAST CAPTURE</button>}
          <span className="shortcut-hint">SPACE/MEDIA PLAY · R RECORD · ARROWS NAVIGATE</span>
        </div>
      </footer>
    </main>
  );
}
