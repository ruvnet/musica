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

export interface LyriaCompanionPromptContext {
  mainPrompts: LyriaWeightedPrompt[];
  scale: LyriaRealtimeScale;
  customDirection?: string;
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
  const lanes = tracks.map((track) => `${SEQUENCE_TRACK_CODES[track.id]}:${pulseString(track)} vol:${Math.round(track.volume * 100)}`);
  return `${label}; ${lanes.join("; ")}; x is hit, - is rest`;
}

function companionIdentity(style: LyriaRealtimeStylePreset | undefined, context?: LyriaCompanionPromptContext): string {
  const main = context?.mainPrompts.find((prompt) => prompt.weight > 0)?.text.trim();
  return (main || style?.description || "the current main arrangement").slice(0, 112);
}

function readableScale(scale?: LyriaRealtimeScale): string {
  return (scale ?? "SCALE_UNSPECIFIED").replaceAll("_", " ").toLowerCase();
}

export function createLyriaSequencePrompts(
  state: LyriaSequenceState,
  style?: LyriaRealtimeStylePreset,
  context?: LyriaCompanionPromptContext,
): LyriaWeightedPrompt[] {
  const active = effectiveSequenceTracks(state).filter((track) => track.id === "drums" || track.id === "bass");
  const identity = companionIdentity(style, context);
  const scale = readableScale(context?.scale);
  const custom = context?.customDirection?.trim() || "Reinforce the main kick and pocket; bass mirrors the main harmonic roots and cadences; leave deliberate space for its hooks";
  const prompts = [
    lanePrompt(active, `repeat this exact 16-step drum and bass rhythm at ${state.bpm} BPM`),
  ].filter((prompt): prompt is string => Boolean(prompt));
  if (prompts.length === 0) {
    return [
      { text: `minimal breakdown at ${state.bpm} BPM, no drums, no bass, near silence, instrumental only`, weight: 1.5 },
      { text: `Companion to main: ${identity}; preserve ${scale}; remain silent except for phrase-boundary texture`.slice(0, 240), weight: 0.82 },
    ];
  }
  return [
    ...prompts.map((text, index) => ({ text: text.slice(0, 240), weight: index === 0 ? 1.7 : 1.45 })),
    { text: `${style?.label ?? "Style-matched"} companion beat for main identity: ${identity}; use ${scale}; bass follows only the main root motion and resolves with its cadence; no independent harmony`.slice(0, 240), weight: 1.2 },
    { text: `${custom}; drums and bass only; same downbeats and eight-bar boundaries as main; stable one-bar pulse with tiny phrase-end variation; immediate start, no intro`.slice(0, 240), weight: 1.35 },
    { text: "independent song, independent chord progression, off-key bass, countermelody, chords, lead, pads, strings, piano, guitar, vocals, tempo drift, free-time fill, breakdown, long transition", weight: -1.15 },
  ].slice(0, 4);
}

export function createLyriaVocalPrompts(
  style: LyriaRealtimeStylePreset,
  context?: LyriaCompanionPromptContext,
): LyriaWeightedPrompt[] {
  const identity = companionIdentity(style, context);
  const scale = readableScale(context?.scale);
  const custom = context?.customDirection?.trim() || "Expressive wordless lead with a memorable chorus contour; answer the main motif without competing with it";
  return [
    { text: `${style.label} a cappella companion vocal in ${scale}; main identity: ${identity}; ${custom}; isolated human voice, expressive vowels and rhythmic syllables, no intelligible words`.slice(0, 240), weight: 1.42 },
    { text: "32-bar vocal form: bars 1-4 rest; 5-8 introduce short motif; 9-16 sparse verse answers; 17-20 pre-chorus lift; 21-28 sustained chorus hook; 29-32 resolve and leave space on final downbeat", weight: 1.3 },
    { text: "Match the main tempo, scale, emotional arc, eight-bar boundaries, and cadences; favor root, third, fifth and consonant extensions; reuse one singable motif; voice only with complete silence between phrases", weight: 1.18 },
    { text: "independent song, different key, new chord progression, lyrics, intelligible words, drums, percussion, bass, synths, pads, piano, guitar, strings, brass, sound effects, accompaniment, continuous vocal wall", weight: -1.3 },
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
    description: "Forward-looking 2026 deep house with elastic sub bass, shuffled percussion, glossy piano hooks, and elegant club dynamics.",
    prompts: [
      { text: "Forward-looking 2026 deep house instrumental at 123 BPM; warm elastic sub bass, punchy four-on-the-floor kick, tight low end, crisp shuffled percussion, glossy piano chord hooks, subtly evolving synth textures", weight: 1.3 },
      { text: "Blend deep house with melodic, organic, and minimal club influences; use vocal-like synth chops without voices, granular ambience, filtered rhythmic detail, spatial ear candy, restrained tension builds", weight: 1.08 },
      { text: "Hypnotic elegant DJ-friendly arc; brief cinematic opening, confident groove, spacious breakdown, controlled final drop; premium wide club mix, clean transients, analog warmth, modern loudness, precise bass management", weight: 0.96 },
      { text: "actual vocals, lyrics, cheesy melody, retro imitation, festival EDM drop, supersaw wall, latin percussion, muddy sub, weak kick, harsh master, random fills, tempo drift, long intro, fade out", weight: -1.08 },
    ],
    config: { bpm: 123, density: 0.6, brightness: 0.5, guidance: 4.8, scale: "C_MAJOR_A_MINOR" },
  },
  {
    id: "techno",
    label: "Techno",
    description: "Forward 2026 hypnotic techno with physical low-end pressure, surgical percussion, evolving machine detail, and disciplined tension.",
    prompts: [
      { text: "Forward-looking 2026 hypnotic techno instrumental at 132 BPM; weighty mono kick, rolling controlled sub, crisp offbeat hats, intricate sixteenth-note percussion, filtered chord stab, metallic machine motif", weight: 1.3 },
      { text: "Blend deep warehouse, minimal, dub techno, and modern sound design; evolving polyrhythmic details, short delays, granular industrial air, restrained acid inflections, fills only at eight-bar boundaries", weight: 1.08 },
      { text: "Long-form DJ-friendly pressure curve; immediate groove, subtle layer swaps, spacious tension break, forceful controlled return; mono-compatible bass, sharp transients, dark stereo depth, loud clean club master", weight: 0.96 },
      { text: "vocals, lyrics, big-room EDM, trance supersaws, cheesy acid riff, random breakdowns, constant risers, distorted kick wash, muddy rumble, harsh hats, tempo drift, excessive reverb, long intro", weight: -1.1 },
    ],
    config: { bpm: 132, density: 0.64, brightness: 0.44, guidance: 4.8, scale: "F_MAJOR_D_MINOR" },
  },
  {
    id: "cinematic",
    label: "Cinema",
    description: "Modern hybrid electronic score with intimate motifs, organic orchestral movement, precise low pulses, and immersive spatial detail.",
    prompts: [
      { text: "Forward-looking 2026 cinematic electronic instrumental at 104 BPM; felt-piano motif, deep analog pulse, chamber strings, restrained hybrid percussion, granular air, soft woodwind color, controlled low brass weight", weight: 1.28 },
      { text: "Blend contemporary classical, ambient electronics, and detailed film scoring; develop one memorable motif through register, harmony, and orchestration; measured impacts and transitions on phrase boundaries", weight: 1.06 },
      { text: "Clear emotional 32-bar arc from intimate suspense to a wide resolved peak; natural dynamics, deep front-to-back staging, focused center, luminous width, clean low end, premium theatrical mix without trailer excess", weight: 0.95 },
      { text: "vocals, choir words, generic trailer braams, superhero ostinato, sentimental piano cliche, constant impacts, melodrama, random key changes, boomy lows, washed-out reverb, abrupt edits", weight: -1.05 },
    ],
    config: { bpm: 104, density: 0.5, brightness: 0.44, guidance: 5, scale: "C_MAJOR_A_MINOR" },
  },
  {
    id: "drum-bass",
    label: "D+B",
    description: "2026 liquid and minimal drum-and-bass with articulate break science, deep clean sub, glass harmonies, and controlled rolling energy.",
    prompts: [
      { text: "Forward-looking 2026 liquid-minimal drum and bass instrumental at 174 BPM; crisp two-step break, detailed ghost snares, articulate shuffled tops, deep sine sub, glass pads, concise Rhodes motif", weight: 1.3 },
      { text: "Blend liquid warmth, autonomic space, and modern break design; alternate clean break layers every eight bars, let the sub answer the kick, add granular atmosphere and precise stereo percussion", weight: 1.08 },
      { text: "Rolling DJ-ready 32-bar arc with immediate rhythm, melodic breath, tension subtraction, and decisive return; huge controlled depth, clear snare, mono sub, smooth highs, competitive loudness with transient headroom", weight: 0.96 },
      { text: "vocals, lyrics, jump-up wobble cliche, abrasive reese wall, neurofunk overload, random amen edits, trap hats, weak snare, distorted sub, piercing tops, washed pads, tempo drift, long intro", weight: -1.1 },
    ],
    config: { bpm: 174, density: 0.72, brightness: 0.5, guidance: 4.8, scale: "D_MAJOR_B_MINOR" },
  },
  {
    id: "hiphop",
    label: "Hip Hop",
    description: "Future-facing instrumental hip-hop with a heavy human pocket, tactile drums, modern sub control, and cinematic sample-like detail.",
    prompts: [
      { text: "Forward-looking 2026 instrumental hip-hop at 92 BPM; heavy human kick-snare pocket, dusty layered drums, swung hats, deep controlled sub, warm electric keys, pitched texture fragments, memorable two-bar motif", weight: 1.28 },
      { text: "Blend progressive beat music, soulful harmony, minimal trap detail, and cinematic sampling aesthetics; intentional silence, selective ghost hits, microtiming, granular ambience, evolving ear candy without vocals", weight: 1.06 },
      { text: "Head-nod 32-bar structure with clear A/B sections, sparse breakdown, confident final variation; close tactile drums, centered sub, warm depth, wide textures, clean transients, modern loudness without flattening swing", weight: 0.94 },
      { text: "rapping, singing, vocal samples, generic trap loop, nonstop hi-hat rolls, drill cliche, boom-bap imitation, cheesy jazz sample, random fills, muddy 808, crushed master, overbusy melody, long intro", weight: -1.08 },
    ],
    config: { bpm: 92, density: 0.5, brightness: 0.36, guidance: 4.8, scale: "E_FLAT_MAJOR_C_MINOR" },
  },
  {
    id: "funk",
    label: "Funk",
    description: "Contemporary future-funk with disciplined live pocket, elastic bass, dry drums, sharp clavinet, and modern electronic polish.",
    prompts: [
      { text: "Forward-looking 2026 instrumental future-funk at 110 BPM; elastic electric bass, dry punchy kit, ghost-note snare, crisp sixteenth hats, clavinet syncopation, muted guitar answers, compact brass accents", weight: 1.28 },
      { text: "Blend live funk pocket, broken-beat sophistication, subtle electronic processing, and modern R&B harmony without vocals; interlock bass, kick, clav, and guitar while preserving intentional rests", weight: 1.05 },
      { text: "DJ-friendly 32-bar groove with hook introduction, call-and-response development, stripped pocket break, and tight final lift; tactile center, short room, articulate lows, sparkling detail, warm analog saturation", weight: 0.94 },
      { text: "vocals, disco pastiche, slap-bass comedy, retro imitation, busy brass solos, rock distortion, quantized stiffness, weak pocket, boomy bass, harsh clav, endless fills, key drift, long intro", weight: -1.05 },
    ],
    config: { bpm: 110, density: 0.66, brightness: 0.56, guidance: 4.8, scale: "G_MAJOR_E_MINOR" },
  },
  {
    id: "samba",
    label: "Samba",
    description: "Contemporary Brazilian electronic samba with authentic interlocking percussion, nylon-string detail, fluid bass, and sophisticated club architecture.",
    prompts: [
      { text: "Forward-looking 2026 Brazilian electronic samba instrumental at 108 BPM; grounded surdo pulse, articulate caixa, tamborim syncopation, pandeiro detail, cavaquinho, nylon guitar, warm fluid bass", weight: 1.3 },
      { text: "Blend authentic samba ensemble interplay with restrained deep-house architecture, granular ambience, filtered percussion, and elegant harmonic color; every rhythm has a clear role and stable two-bar cycle", weight: 1.08 },
      { text: "Joyful but sophisticated 32-bar DJ arc with percussion reveal, full groove, spacious string-led break, and controlled ensemble return; natural transients, deep bass, vivid width, warm acoustic-electronic mix", weight: 0.95 },
      { text: "vocals, chants, tourist latin cliche, generic salsa, reggaeton dembow, carnival overload, cheesy brass, random percussion, quantized stiffness, muddy surdo, brittle highs, festival drop, long intro", weight: -1.08 },
    ],
    config: { bpm: 108, density: 0.72, brightness: 0.62, guidance: 5, scale: "C_MAJOR_A_MINOR" },
  },
  {
    id: "rock",
    label: "Rock",
    description: "Forward-leaning 2026 instrumental rock with a tight live band, wide expressive guitars, acoustic drum impact, and cinematic dynamics.",
    prompts: [
      { text: "Forward-leaning 2026 instrumental rock at 126 BPM built around a tight live band; expressive electric guitars, wide rhythm parts, sharp melodic accents, controlled feedback, bold anthemic lead hook", weight: 1.3 },
      { text: "Punchy bass locked closely to a real acoustic drum kit; deep kick, hard snare hits, energetic tom fills, natural cymbal dynamics; muscular immediate performance with human timing and live-room energy", weight: 1.1 },
      { text: "Cinematic 32-bar arc moving from restrained tension into a huge chorus-sized instrumental payoff; modern polished mix, strong transient impact, wide guitars, focused low end, minimal studio gloss", weight: 0.98 },
      { text: "vocals, lyrics, programmed drums, drum machine, generic arena-rock cliche, retro imitation, endless soloing, metal wall, pop-punk melody, fizzy guitars, muddy bass, crushed dynamics, synthetic room", weight: -1.12 },
    ],
    config: { bpm: 126, density: 0.68, brightness: 0.56, guidance: 4.8, scale: "E_MAJOR_D_FLAT_MINOR" },
  },
  {
    id: "jazz",
    label: "Jazz",
    description: "Contemporary electro-acoustic jazz with deep human swing, conversational harmony, tactile trio detail, and restrained spatial electronics.",
    prompts: [
      { text: "Forward-looking 2026 instrumental jazz at 112 BPM; acoustic piano, articulate upright bass, brushed kit, human ride swing, selective muted-horn color, subtle granular room texture, memorable harmonic motif", weight: 1.28 },
      { text: "Blend modern piano-trio conversation, broken-beat nuance, modal harmony, and restrained electro-acoustic sound design; responsive comping, breathing bass movement, microtiming, silence between phrases", weight: 1.06 },
      { text: "Coherent 32-bar club-set form with theme, conversational variation, sparse bass-piano break, and elegant return; intimate front image, natural room depth, centered lows, soft transients, high dynamic clarity", weight: 0.94 },
      { text: "vocals, scat, lounge cliche, smooth-jazz sax, endless solos, bebop imitation, random chord substitutions, stiff quantization, busy drums, boomy upright bass, washy room, cocktail background music", weight: -1.05 },
    ],
    config: { bpm: 112, density: 0.5, brightness: 0.4, guidance: 5, scale: "B_FLAT_MAJOR_G_MINOR", muteDrums: false },
  },
  {
    id: "classical",
    label: "Classical",
    description: "Contemporary chamber-classical performance with expressive piano, intimate strings, precise motif development, and modern cinematic space.",
    prompts: [
      { text: "Forward-looking contemporary classical instrumental at 76 BPM; concert grand piano, intimate chamber strings, soft woodwind color, expressive arpeggiated pulse, clear minor-key motif, nuanced human dynamics", weight: 1.3 },
      { text: "Blend romantic harmonic depth, contemporary minimalism, and restrained cinematic sound; develop one motif through inversion, register, counterline, and orchestral color while preserving natural rubato", weight: 1.08 },
      { text: "32-bar dramatic arc with exposed piano opening, gradual string dialogue, spacious central suspension, and resolved ensemble peak; realistic hall depth, detailed bow texture, warm lows, wide natural dynamics", weight: 0.96 },
      { text: "vocals, choir, direct imitation of a named recording, generic trailer music, sentimental cliche, constant arpeggios, oversized percussion, synthetic strings, boomy hall, harsh piano, abrupt key changes", weight: -1.08 },
    ],
    config: { bpm: 76, density: 0.38, brightness: 0.34, guidance: 5.2, scale: "E_FLAT_MAJOR_C_MINOR", muteDrums: true },
  },
  {
    id: "ambient",
    label: "Ambient",
    description: "Immersive 2026 ambient electronics with evolving harmonic depth, tactile acoustic fragments, granular motion, and disciplined low-frequency space.",
    prompts: [
      { text: "Forward-looking 2026 ambient electronic instrumental at 78 BPM; slow analog pads, low sine foundation, felt-piano fragments, granular field textures, tape echoes, delicate high-frequency particles", weight: 1.28 },
      { text: "Blend deep ambient, electro-acoustic detail, dub space, and contemporary minimalism; gentle subliminal pulse, evolving voicings, long breathing rests, microscopic motion, one recognizable tonal motif", weight: 1.06 },
      { text: "32-bar immersive arc with near-silent emergence, layered harmonic bloom, open suspended center, and luminous controlled resolution; vast front-to-back depth, clean sub, soft transients, high dynamic range", weight: 0.94 },
      { text: "vocals, nature-sound cliche, meditation stock music, new-age melody, conventional drum beat, constant drone, random notes, excessive shimmer, muddy reverb, sub rumble, harsh particles, sudden climax", weight: -1.05 },
    ],
    config: { bpm: 78, density: 0.28, brightness: 0.32, guidance: 4.9, scale: "C_MAJOR_A_MINOR", muteDrums: true },
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
  const detailedPrompts = style.prompts.slice(0, 4).map((prompt) => ({ ...prompt, text: prompt.text.slice(0, 240) }));
  return {
    weightedPrompts: detailedPrompts.length >= 3 ? detailedPrompts : [
      ...detailedPrompts.slice(0, 2),
      { text: "Tight groove, memorable motif, clear eight-bar phrases, controlled transitions, balanced dynamics, polished stereo mix", weight: 0.78 },
      { text: "free tempo, random genre changes, clashing harmony, overbusy arrangement, long intro, abrupt fills, muddy mix, harsh master", weight: -0.62 },
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
