import { invoke, isTauri } from "@tauri-apps/api/core";
import type { PerformanceTemplate, TrackId, TrackSnapshot } from "./types";

export type LyriaRealtimeScale =
  | "C_MAJOR_A_MINOR"
  | "D_FLAT_MAJOR_B_FLAT_MINOR"
  | "D_MAJOR_B_MINOR"
  | "E_FLAT_MAJOR_C_MINOR"
  | "E_MAJOR_D_FLAT_MINOR"
  | "F_MAJOR_D_MINOR"
  | "G_FLAT_MAJOR_E_FLAT_MINOR"
  | "G_MAJOR_E_MINOR"
  | "A_FLAT_MAJOR_F_MINOR"
  | "A_MAJOR_G_FLAT_MINOR"
  | "B_FLAT_MAJOR_G_MINOR"
  | "B_MAJOR_A_FLAT_MINOR"
  | "SCALE_UNSPECIFIED";

export type LyriaRealtimeMode = "QUALITY" | "DIVERSITY" | "VOCALIZATION";
export type LyriaRealtimeDeckId = "main" | "sequence" | "vocal";

export interface LyriaWeightedPrompt {
  text: string;
  weight: number;
}

export interface LyriaRealtimeConfig {
  bpm: number;
  guidance: number;
  density: number;
  brightness: number;
  temperature: number;
  topK: number;
  seed?: number;
  scale: LyriaRealtimeScale;
  muteBass: boolean;
  muteDrums: boolean;
  onlyBassAndDrums: boolean;
  musicGenerationMode: LyriaRealtimeMode;
}

export interface LyriaRealtimeRequest {
  weightedPrompts: LyriaWeightedPrompt[];
  config: LyriaRealtimeConfig;
}

export interface LyriaRealtimeStylePreset {
  id: string;
  label: string;
  description: string;
  prompts: LyriaWeightedPrompt[];
  config: Partial<LyriaRealtimeConfig>;
}

export interface AutoDjDirection {
  personalization: string;
  generatedBrief?: string;
  step: number;
  bpm: number;
  bars?: number;
}

export interface LyriaRealtimeStatus {
  deck: LyriaRealtimeDeckId;
  available: boolean;
  provider: "lyria_realtime" | string;
  model: string;
  sampleRateHz: number;
  channels: number;
  audioFormat: "pcm16" | string;
  instrumentalOnly: boolean;
  reason?: string;
  activeSessionId?: string;
  bufferedAudioBytes: number;
  streamedAudioBytes: number;
  warning?: string;
}

export interface LyriaRealtimeSession {
  deck: LyriaRealtimeDeckId;
  id: string;
  provider: string;
  model: string;
  state: string;
  weightedPrompts: LyriaWeightedPrompt[];
  config: LyriaRealtimeConfig;
  sampleRateHz: number;
  channels: number;
  audioFormat: string;
}

export interface LyriaRealtimeAudioPoll {
  deck: LyriaRealtimeDeckId;
  sessionId?: string;
  sampleRateHz: number;
  channels: number;
  audioFormat: string;
  chunks: number[][];
  bufferedAudioBytes: number;
  streamedAudioBytes: number;
  warning?: string;
}

export const DEFAULT_LYRIA_REALTIME_CONFIG: LyriaRealtimeConfig = {
  bpm: 118,
  guidance: 4,
  density: 0.52,
  brightness: 0.42,
  temperature: 1.1,
  topK: 40,
  scale: "E_FLAT_MAJOR_C_MINOR",
  muteBass: false,
  muteDrums: false,
  onlyBassAndDrums: false,
  musicGenerationMode: "QUALITY",
};

export const DEFAULT_LYRIA_REALTIME_PROMPTS: LyriaWeightedPrompt[] = [
  { text: "Deep House, Rhodes Piano, Precision Bass, TR-909 Drum Machine, warm analog synth pads", weight: 1.15 },
  { text: "Tight Groove, Live Performance, memorable motif, clear eight-bar phrases, controlled transitions, polished stereo mix", weight: 0.82 },
  { text: "primary arrangement bed with restrained lead lines and space for a supporting pulse and short vocalization responses", weight: 0.68 },
  { text: "free tempo, random genre changes, clashing harmony, overbusy arrangement, long intro, abrupt fills, muddy mix, harsh master", weight: -0.62 },
];

export const DEFAULT_LYRIA_REALTIME_STYLE_ID = "house";

export const AUTO_DJ_PHRASE_BARS = 32;

const AUTO_DJ_BEAT_DIRECTIONS: Record<string, string> = {
  house: "four-on-the-floor kick; clap on 2 and 4; shuffled closed hats; restrained open hat on offbeats; syncopated bass answering the kick",
  techno: "solid quarter-note kick; rolling sixteenth-note percussion; tight offbeat hats; hypnotic one-bar bass pulse; sparse fills only at phrase boundaries",
  cinematic: "measured low pulse; restrained hybrid percussion; half-time accents; tension risers over eight bars; decisive downbeats without trailer cliches",
  "drum-bass": "clean two-step breakbeat; snare on 2 and 4; detailed ghost notes; deep sub following a two-bar motif; controlled fills every eight bars",
  hiphop: "laid-back kick and snare pocket; swung hats; selective ghost hits; deep sub with intentional rests; no trap roll clutter",
  funk: "dry syncopated drum pocket; ghost-note snare; sixteenth-note hats; bass and clav interlock; short guitar answers leaving clear rests",
  samba: "surdo downbeats; caixa drive; tamborim syncopation; hand percussion in a stable two-bar pattern; bass locked beneath the ensemble",
  rock: "punchy acoustic kick and snare backbeat; driving eighth-note hats; bass locked to kick; guitar accents around a memorable two-bar hook",
  jazz: "human swing ride pattern; feathered kick; brushed snare comping; walking bass with breathing room; piano answers across four-bar phrases",
  classical: "steady arpeggiated inner pulse; expressive rubato within the master tempo; clear harmonic rhythm; restrained dynamic swells every eight bars",
  ambient: "subtle low pulse; sparse textural ticks; long breathing rests; evolving pad rhythm; no conventional drum groove unless nearly subliminal",
};

const AUTO_DJ_SOUND_DIRECTIONS: Record<string, string> = {
  house: "warm Rhodes and piano hook, rounded analog bass, TR-909 character, soft chord pads, one concise synth motif",
  techno: "weighty mono bass, precise drum machine transients, metallic percussion, filtered chord stab, distant industrial texture",
  cinematic: "felt piano motif, low analog pulse, chamber strings, granular air, restrained brass weight, wide but focused orchestral electronics",
  "drum-bass": "clean break layers, deep sine sub, glassy pads, short Rhodes fragments, precise stereo percussion, no abrasive reese wall",
  hiphop: "dusty drum break, modern controlled sub, warm electric keys, pitched texture fragments, understated melodic sample-like motif",
  funk: "live dry kit, articulate electric bass, clavinet, muted rhythm guitar, compact brass punctuation, organic room tone",
  samba: "surdo, caixa, tamborim, pandeiro, cavaquinho, nylon guitar, warm bass, vivid ensemble dynamics without tourist pastiche",
  rock: "tight live drum kit, defined electric bass, layered rhythm guitars, one singable instrumental lead, subtle analog keyboard support",
  jazz: "acoustic piano, upright bass, brushed kit, occasional muted horn color, intimate room sound, conversational improvisation",
  classical: "concert grand piano, chamber strings, soft woodwind color, natural hall depth, coherent motif development, nuanced dynamics",
  ambient: "slow analog pads, tape echo, felt piano fragments, low sine foundation, granular field texture, spacious high-frequency detail",
};

export function autoDjPhraseDurationMs(bpm: number, bars = AUTO_DJ_PHRASE_BARS): number {
  const boundedBpm = Math.max(60, Math.min(200, bpm));
  const boundedBars = Math.max(8, Math.min(64, Math.round(bars)));
  return Math.max(30_000, Math.round((60_000 / boundedBpm) * 4 * boundedBars));
}

export function createAutoDjRealtimeRequest(
  style: LyriaRealtimeStylePreset,
  direction: AutoDjDirection,
): LyriaRealtimeRequest {
  const base = createLyriaRealtimeRequestFromStyle(style, direction.bpm);
  const bars = Math.max(8, Math.min(64, Math.round(direction.bars ?? AUTO_DJ_PHRASE_BARS)));
  const phase = Math.abs(Math.round(direction.step)) % 4;
  const energy = ["restrained opening", "confident groove", "controlled lift", "resolved peak"][phase];
  const personalization = direction.personalization.trim() || "futuristic live set with a memorable musical identity and disciplined club-ready dynamics";
  const generatedBrief = direction.generatedBrief?.trim();
  const beat = AUTO_DJ_BEAT_DIRECTIONS[style.id] ?? "stable two-bar beat pattern; clear downbeat; intentional syncopation; fills only at phrase boundaries";
  const sound = AUTO_DJ_SOUND_DIRECTIONS[style.id] ?? style.description;

  return {
    weightedPrompts: [
      {
        text: `${style.label} instrumental; ${style.description} Single continuous main stereo stream at ${direction.bpm} BPM; ${energy}; preserve pulse, key center, and sonic identity through the transition.`.slice(0, 240),
        weight: 1.28,
      },
      {
        text: `Beat design: ${beat}. Build in coherent 8-bar phrases inside one ${bars}-bar section; introduce one change at a time; make every transition land on bar 1.`.slice(0, 240),
        weight: 1.12,
      },
      {
        text: (generatedBrief
          ? `Director brief: ${generatedBrief}`
          : `Sound and personalization: ${sound}. Creative identity: ${personalization}. Polished low end, audible motif, deliberate contrast, headroom for live mixing.`).slice(0, 240),
        weight: 1.04,
      },
      {
        text: "multiple songs, multiple streams, vocals, genre roulette, tempo drift, key clash, random fills, constant solos, overbusy layers, muddy bass, harsh highs, long intro, fade out, abrupt ending",
        weight: -1.05,
      },
    ],
    config: {
      ...base.config,
      bpm: Math.max(60, Math.min(200, Math.round(direction.bpm))),
      density: Math.max(0.12, Math.min(0.88, base.config.density + [-0.08, 0, 0.06, 0.1][phase])),
      brightness: Math.max(0.16, Math.min(0.78, base.config.brightness + [-0.06, 0, 0.04, 0.07][phase])),
      guidance: Math.max(4.4, Math.min(6, base.config.guidance + 0.45)),
      temperature: Math.max(0.75, Math.min(1.25, base.config.temperature)),
      topK: Math.min(48, base.config.topK),
      musicGenerationMode: "QUALITY",
    },
  };
}

export function compensateLyriaBpmForPitch(masterBpm: number, semitones: number): number {
  return Math.max(60, Math.min(200, Math.round(masterBpm / (2 ** (semitones / 12)))));
}

export interface LyriaSequenceState {
  bpm: number;
  tracks: TrackSnapshot[];
}

const SEQUENCE_TRACK_CODES: Record<TrackId, string> = {
  drums: "DR",
  bass: "BS",
  chords: "CH",
  lead: "LD",
  voice: "VO",
  texture: "TX",
};

function effectiveSequenceTracks(state: LyriaSequenceState): TrackSnapshot[] {
  const hasSolo = state.tracks.some((track) => track.solo);
  return state.tracks.filter((track) => (
    !track.muted
    && (!hasSolo || track.solo)
    && track.volume > 0.03
    && track.pattern.some(Boolean)
  ));
}

function pulseString(track: TrackSnapshot): string {
  return track.pattern.slice(0, 16).map((active) => (active ? "x" : "-")).join("");
}

function lanePrompt(tracks: TrackSnapshot[], label: string): string | undefined {
  if (tracks.length === 0) return undefined;
  const lanes = tracks.map((track) => {
    const notes = track.id === "drums" || track.notes.length === 0
      ? ""
      : ` notes:${[...new Set(track.notes)].slice(0, 8).join(".")}`;
    return `${SEQUENCE_TRACK_CODES[track.id]}:${pulseString(track)} vol:${Math.round(track.volume * 100)}${notes}`;
  });
  return `${label}; ${lanes.join("; ")}; x is hit, - is rest`;
}

export function createLyriaSequencePrompts(
  state: LyriaSequenceState,
  style?: LyriaRealtimeStylePreset,
): LyriaWeightedPrompt[] {
  const active = effectiveSequenceTracks(state).filter((track) => track.id === "drums" || track.id === "bass");
  const prompts = [
    lanePrompt(active, `repeat this exact 16-step drum and bass rhythm at ${state.bpm} BPM`),
  ].filter((prompt): prompt is string => Boolean(prompt));
  if (prompts.length === 0) {
    return [
      { text: `minimal breakdown at ${state.bpm} BPM, no drums, no bass, near silence, instrumental only`, weight: 1.5 },
      ...(style ? [{ text: `${style.label} rhythmic character, silent beat stem`.slice(0, 240), weight: 0.72 }] : []),
    ];
  }
  return [
    ...prompts.map((text, index) => ({ text: text.slice(0, 240), weight: index === 0 ? 1.7 : 1.45 })),
    ...(style ? [{ text: `${style.label} supporting beat layer, match the main deck's groove and production character, percussion and bass only`.slice(0, 240), weight: 0.86 }] : []),
    { text: "repeating one-bar beat stem, drums and bass only, locked tempo, identical pulse each bar, immediate downbeat, no intro, no fills, no melody, no vocals", weight: 1.35 },
    { text: "chords, lead melody, pads, strings, piano, guitar, vocals, breakdown, free tempo, evolving arrangement", weight: -0.92 },
  ].slice(0, 4);
}

export function createLyriaVocalPrompts(style: LyriaRealtimeStylePreset): LyriaWeightedPrompt[] {
  return [
    { text: `${style.label} a cappella wordless vocalization, isolated dry human voice stem, expressive vowels, precise rhythmic syllables, voice only`.slice(0, 240), weight: 1.35 },
    { text: "match the main deck's key, tempo, groove, and emotional tone; unaccompanied vocal performance with complete silence between phrases", weight: 0.92 },
    { text: "sparse call-and-response vocal phrases, singable contour, short two-bar answers, intentional rests, supporting background role", weight: 0.9 },
    { text: "drums, percussion, bass, synths, pads, piano, guitar, strings, brass, sound effects, instrumental accompaniment, continuous vocal wall", weight: -1.15 },
  ];
}

export function createLyriaSequenceConfig(
  state: LyriaSequenceState,
  base: LyriaRealtimeConfig,
  pitchSemitones: number,
): LyriaRealtimeConfig {
  const active = effectiveSequenceTracks(state).filter((track) => track.id === "drums" || track.id === "bass");
  const drumsActive = active.some((track) => track.id === "drums");
  const bassActive = active.some((track) => track.id === "bass");
  const activeStepCount = active.reduce((sum, track) => sum + track.pattern.filter(Boolean).length, 0);
  const availableSteps = Math.max(16, active.length * 16);
  const patternDensity = activeStepCount / availableSteps;
  return {
    ...base,
    bpm: compensateLyriaBpmForPitch(state.bpm, pitchSemitones),
    guidance: Math.min(6, base.guidance + 1.15),
    density: Math.max(0.08, Math.min(0.9, patternDensity * 1.35)),
    brightness: Math.max(0.18, Math.min(0.68, base.brightness - 0.08)),
    temperature: Math.min(base.temperature, 0.95),
    topK: Math.min(base.topK, 32),
    muteBass: !bassActive,
    muteDrums: !drumsActive,
    onlyBassAndDrums: drumsActive && bassActive,
    musicGenerationMode: "QUALITY",
  };
}

export const LYRIA_REALTIME_STYLE_PRESETS: LyriaRealtimeStylePreset[] = [
  {
    id: "house",
    label: "House",
    description: "Polished club groove with piano chords, warm bass, and clean four-on-the-floor drums.",
    prompts: [
      { text: "modern deep house instrumental, warm bass, clean four on the floor drums, piano chord hooks, polished club mix", weight: 1 },
      { text: "uplifting but restrained dance groove, no vocals, no latin percussion, crisp master", weight: 0.78 },
    ],
    config: { bpm: 122, density: 0.58, brightness: 0.48, guidance: 4.3, scale: "C_MAJOR_A_MINOR" },
  },
  {
    id: "techno",
    label: "Techno",
    description: "Dark warehouse pulse with solid kick, rolling bass, crisp hats, and hypnotic stabs.",
    prompts: [
      { text: "deep warehouse techno, solid kick, rolling bass, hypnotic stabs, club ready instrumental", weight: 1 },
      { text: "minimal arrangement, powerful low end, crisp hats, dark atmosphere, no vocals", weight: 0.76 },
    ],
    config: { bpm: 132, density: 0.62, brightness: 0.46, guidance: 4.1, scale: "F_MAJOR_D_MINOR" },
  },
  {
    id: "cinematic",
    label: "Cinema",
    description: "Wide electronic score with piano shimmer, low pulses, and orchestral pad movement.",
    prompts: [
      { text: "cinematic electronic score, low pulses, shimmering piano, orchestral pads, widescreen drama", weight: 1 },
      { text: "clear emotional arc, polished trailer-quality instrumental bed, no vocals", weight: 0.72 },
    ],
    config: { bpm: 104, density: 0.52, brightness: 0.44, guidance: 4.5, scale: "C_MAJOR_A_MINOR" },
  },
  {
    id: "drum-bass",
    label: "D+B",
    description: "Liquid drum and bass with fast breaks, deep sub, glass pads, and melodic fragments.",
    prompts: [
      { text: "liquid drum and bass instrumental, fast breakbeats, deep sub bass, glass pads, clean melodic fragments", weight: 1 },
      { text: "rolling energy, polished mix, atmospheric but rhythmic, no vocals", weight: 0.72 },
    ],
    config: { bpm: 174, density: 0.7, brightness: 0.5, guidance: 4.2, scale: "D_MAJOR_B_MINOR" },
  },
  {
    id: "hiphop",
    label: "Hip Hop",
    description: "Instrumental beat tape feel with dusty drums, sub bass, keys, and chopped texture.",
    prompts: [
      { text: "instrumental hip hop beat, dusty drums, deep sub bass, warm electric piano, chopped texture", weight: 1 },
      { text: "head nod groove, clean modern low end, spacious instrumental mix, no vocals", weight: 0.74 },
    ],
    config: { bpm: 92, density: 0.48, brightness: 0.34, guidance: 4.1, scale: "E_FLAT_MAJOR_C_MINOR" },
  },
  {
    id: "funk",
    label: "Funk",
    description: "Tight pocket with slap bass, clav, dry drums, and short guitar accents.",
    prompts: [
      { text: "tight instrumental funk band, slap bass, clavinet, dry drums, short guitar accents, strong pocket", weight: 1 },
      { text: "clean stage mix, syncopated groove, energetic but controlled, no vocals", weight: 0.72 },
    ],
    config: { bpm: 108, density: 0.66, brightness: 0.56, guidance: 4.2, scale: "G_MAJOR_E_MINOR" },
  },
  {
    id: "samba",
    label: "Samba",
    description: "Brazilian samba-house percussion and nylon-string colors for intentional Latin sets.",
    prompts: [
      { text: "Brazilian samba groove, nylon guitar, cavaquinho, surdo, tamborim, warm percussion", weight: 1 },
      { text: "sunny dance energy, syncopated acoustic rhythm section, polished instrumental mix", weight: 0.75 },
    ],
    config: { bpm: 104, density: 0.72, brightness: 0.64, guidance: 4.4, scale: "C_MAJOR_A_MINOR" },
  },
  {
    id: "rock",
    label: "Rock",
    description: "Live rock band energy with electric guitars, bass, drums, and a strong instrumental hook.",
    prompts: [
      { text: "tight live rock band, electric guitars, punchy bass, real drum kit, anthemic instrumental hook", weight: 1 },
      { text: "clean production, strong backbeat, energetic but not noisy", weight: 0.72 },
    ],
    config: { bpm: 126, density: 0.68, brightness: 0.58, guidance: 4.2, scale: "E_MAJOR_D_FLAT_MINOR" },
  },
  {
    id: "jazz",
    label: "Jazz",
    description: "Modern trio feel with piano harmony, upright bass, brushed drums, and human swing.",
    prompts: [
      { text: "modern jazz trio, piano, upright bass, brushed drums, tasteful harmonic movement", weight: 1 },
      { text: "blue note color, human swing feel, intimate club recording", weight: 0.68 },
    ],
    config: { bpm: 112, density: 0.5, brightness: 0.38, guidance: 4.6, scale: "B_FLAT_MAJOR_G_MINOR", muteDrums: false },
  },
  {
    id: "classical",
    label: "Classical",
    description: "Concert-hall piano and chamber strings with restrained romantic movement.",
    prompts: [
      { text: "classical piano and chamber strings, public domain romantic harmony, expressive arpeggios", weight: 1 },
      { text: "cinematic concert hall sound, elegant phrasing, restrained dynamics", weight: 0.7 },
    ],
    config: { bpm: 76, density: 0.36, brightness: 0.32, guidance: 5.0, scale: "E_FLAT_MAJOR_C_MINOR", muteDrums: true },
  },
  {
    id: "ambient",
    label: "Ambient",
    description: "Slow evolving pads, tape echoes, soft piano fragments, and spacious texture.",
    prompts: [
      { text: "ambient electronic instrumental, slow evolving pads, tape echo, soft piano fragments, spacious texture", weight: 1 },
      { text: "calm immersive atmosphere, gentle low pulse, no drums unless subtle, no vocals", weight: 0.72 },
    ],
    config: { bpm: 78, density: 0.26, brightness: 0.3, guidance: 4.4, scale: "C_MAJOR_A_MINOR", muteDrums: true },
  },
];

const TEMPLATE_STYLE_MAP: Record<string, string> = {
  "moonlight-sequencer": "classical",
  "warehouse-techno": "techno",
  "liquid-breaks": "drum-bass",
  "ambient-dub": "ambient",
  "synthwave-drive": "cinematic",
  "footwork-cuts": "techno",
  "cinematic-pulse": "cinematic",
  "uk-garage-neon": "techno",
  "afro-cosmic-house": "house",
  "idm-crystalline": "techno",
  "hyperpop-rush": "rock",
};

export async function getLyriaRealtimeStatus(deck: LyriaRealtimeDeckId = "main"): Promise<LyriaRealtimeStatus> {
  if (!isTauri()) {
    return {
      deck,
      available: false,
      provider: "browser_preview",
      model: "models/lyria-realtime-exp",
      sampleRateHz: 48_000,
      channels: 2,
      audioFormat: "pcm16",
      instrumentalOnly: true,
      reason: "Lyria RealTime requires the desktop app so the Gemini key stays out of React",
      bufferedAudioBytes: 0,
      streamedAudioBytes: 0,
    };
  }
  return invoke<LyriaRealtimeStatus>("lyria_realtime_status", { deck });
}

export function lyriaRealtimeStyleById(id: string): LyriaRealtimeStylePreset {
  return LYRIA_REALTIME_STYLE_PRESETS.find((preset) => preset.id === id)
    ?? LYRIA_REALTIME_STYLE_PRESETS.find((preset) => preset.id === DEFAULT_LYRIA_REALTIME_STYLE_ID)
    ?? LYRIA_REALTIME_STYLE_PRESETS[0];
}

export function lyriaRealtimeStyleForTemplate(template: PerformanceTemplate): LyriaRealtimeStylePreset {
  return lyriaRealtimeStyleById(TEMPLATE_STYLE_MAP[template.id] ?? DEFAULT_LYRIA_REALTIME_STYLE_ID);
}

export function createLyriaRealtimeRequestFromStyle(style: LyriaRealtimeStylePreset, bpm?: number): LyriaRealtimeRequest {
  return {
    weightedPrompts: [
      ...style.prompts.slice(0, 2).map((prompt) => ({ ...prompt })),
      {
        text: "Tight Groove, Live Performance, memorable motif, clear eight-bar phrases, controlled transitions, balanced dynamics, polished stereo mix",
        weight: 0.78,
      },
      {
        text: "free tempo, random genre changes, clashing harmony, overbusy arrangement, long intro, abrupt fills, muddy mix, harsh master",
        weight: -0.62,
      },
    ].slice(0, 4),
    config: {
      ...DEFAULT_LYRIA_REALTIME_CONFIG,
      ...style.config,
      bpm: bpm ?? style.config.bpm ?? DEFAULT_LYRIA_REALTIME_CONFIG.bpm,
      onlyBassAndDrums: style.config.onlyBassAndDrums ?? false,
      muteBass: style.config.muteBass ?? false,
      muteDrums: style.config.muteDrums ?? false,
    },
  };
}

export function createLyriaRealtimeRequestForTemplate(
  template: PerformanceTemplate,
  style: LyriaRealtimeStylePreset = lyriaRealtimeStyleForTemplate(template),
): LyriaRealtimeRequest {
  const request = createLyriaRealtimeRequestFromStyle(style, template.bpm);
  return {
    weightedPrompts: [
      ...request.weightedPrompts.slice(0, 2),
      {
        text: `${template.name} arrangement: ${template.description}; coherent eight-bar phrasing; reserve space for a supporting pulse and wordless responses`.slice(0, 240),
        weight: 0.78,
      },
      request.weightedPrompts[request.weightedPrompts.length - 1],
    ],
    config: request.config,
  };
}

export async function startLyriaRealtime(
  request: LyriaRealtimeRequest,
  deck: LyriaRealtimeDeckId = "main",
): Promise<LyriaRealtimeSession> {
  if (!isTauri()) throw new Error("Lyria RealTime requires the desktop app");
  return invoke<LyriaRealtimeSession>("lyria_realtime_start", { deck, request });
}

export async function updateLyriaRealtime(
  request: LyriaRealtimeRequest,
  deck: LyriaRealtimeDeckId = "main",
): Promise<LyriaRealtimeSession> {
  if (!isTauri()) throw new Error("Lyria RealTime requires the desktop app");
  return invoke<LyriaRealtimeSession>("lyria_realtime_update", { deck, request });
}

export async function stopLyriaRealtime(deck: LyriaRealtimeDeckId = "main"): Promise<void> {
  if (!isTauri()) return;
  await invoke<void>("lyria_realtime_stop", { deck });
}

export async function pollLyriaRealtimeAudio(deck: LyriaRealtimeDeckId = "main"): Promise<LyriaRealtimeAudioPoll> {
  if (!isTauri()) {
    return {
      deck,
      sampleRateHz: 48_000,
      channels: 2,
      audioFormat: "pcm16",
      chunks: [],
      bufferedAudioBytes: 0,
      streamedAudioBytes: 0,
    };
  }
  return invoke<LyriaRealtimeAudioPoll>("lyria_realtime_poll_audio", { deck });
}
