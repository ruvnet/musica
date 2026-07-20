import { existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawn } from "node:child_process";

const MODEL = "models/lyria-realtime-exp";
const WS_ENDPOINT = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic";
const SAMPLE_RATE_HZ = 48_000;
const CHANNELS = 2;
const CAPTURE_SECONDS = 32;
const OUTPUT_SECONDS = 30;
const TARGET_CAPTURE_BYTES = SAMPLE_RATE_HZ * CHANNELS * 2 * CAPTURE_SECONDS;
const MIN_CAPTURE_BYTES = SAMPLE_RATE_HZ * CHANNELS * 2 * 31;

const styles = [
  ["rock", 126, "G_MAJOR_E_MINOR",
    "Hard-driving 2026 instrumental rock at 126 BPM; two dominant overdriven electric rhythm guitars, palm-muted eighth-note riff, huge acoustic kick and snare, picked bass, anthemic octave lead hook",
    "Real four-piece band performance with relentless forward motion, powerful crash accents, energetic tom fills, tight bass-and-kick lock, muscular human timing, physical live-room impact",
    "acoustic folk, country, latin rhythm, funk, jazz, orchestral score, electronic dance beat, programmed drums, soft rock, retro pastiche, endless solo, fizzy guitar, muddy room, weak snare"],
  ["8bit", 140, "C_MAJOR_A_MINOR",
    "Authentic 2026 chiptune instrumental at 140 BPM; NES-style square-wave lead melody, pulse-width arpeggios, triangle-wave bass line, noise-channel drums, bright heroic 8-bit video game energy, catchy singable main theme",
    "Blend classic console chiptune with modern game-soundtrack craft; tight arpeggiated chord stabs, octave-jumping bass, dramatic key-change lift, precise quantized sequencing with expressive melody phrasing",
    "orchestral instruments, realistic drums, guitar, lo-fi tape, muddy bass, harsh piercing leads, random atonal runs"],
  ["lofi", 76, "E_FLAT_MAJOR_C_MINOR",
    "Warm 2026 lo-fi hip-hop instrumental at 76 BPM; dusty swung boom-bap drums, soft rounded kick, vinyl crackle patina, mellow jazzy Rhodes chords, muted upright-style bass, gentle tape wobble and saturation",
    "Blend classic lo-fi beat-tape aesthetics with jazz seventh-chord harmony, nostalgic melodic fragments, subtle sidechain breathing, head-nod pocket with human microtiming",
    "trap hi-hat rolls, aggressive drums, bright digital synths, EDM drop, harsh highs, clipping, fast tempo, busy arrangement"],
  ["dubstep", 140, "F_MAJOR_D_MINOR",
    "Forward-looking 2026 dubstep instrumental at 140 BPM; half-time drop with massive wobble bass and growl bass modulation, crisp cracking snare on the three, deep clean sub foundation, cinematic sound-design weight",
    "Blend classic UK dubstep space, modern riddim precision, and melodic dubstep color; LFO wobble patterns that answer the drum pattern, call-and-response bass conversation",
    "brostep noise wall, random screeches, four-on-the-floor kick, muddy sub, weak snare, constant risers"],
  ["neuroflux", 122, "D_MAJOR_B_MINOR",
    "Neuroflux, a new AI-native genre at 122 BPM; continuously morphing hybrid of club pulse, chamber acoustics, and synthetic voice-like textures, where instrument timbres interpolate smoothly into one another mid-phrase as if the ensemble is being re-synthesized in real time",
    "One constant hypnotic four-note motif anchors everything while the world drifts around it: drums morph gradually from acoustic to granular, harmony rotates through modal colors, textures crossfade between organic and machine",
    "static loop, abrupt genre jumps, chaos, atonality, muddy blend, harsh resonances, random noise collage, drop cliche"],
  ["house", 123, "C_MAJOR_A_MINOR",
    "Forward-looking 2026 deep house instrumental at 123 BPM; warm elastic sub bass, punchy four-on-the-floor kick, tight low end, crisp shuffled percussion, glossy piano chord hooks, subtly evolving synth textures",
    "Blend deep house with melodic, organic, and minimal club influences; vocal-like synth chops without voices, granular ambience, filtered rhythmic detail, spatial ear candy",
    "cheesy melody, retro imitation, festival EDM drop, supersaw wall, latin percussion, muddy sub, weak kick, harsh master"],
  ["techno", 132, "F_MAJOR_D_MINOR",
    "Forward-looking 2026 hypnotic techno instrumental at 132 BPM; weighty mono kick, rolling controlled sub, crisp offbeat hats, intricate sixteenth-note percussion, filtered chord stab, metallic machine motif",
    "Blend deep warehouse, minimal, dub techno, and modern sound design; evolving polyrhythmic details, short delays, granular industrial air, restrained acid inflections",
    "big-room EDM, trance supersaws, cheesy acid riff, random breakdowns, constant risers, distorted kick wash, muddy rumble, harsh hats"],
  ["cinematic", 104, "C_MAJOR_A_MINOR",
    "Forward-looking 2026 cinematic electronic instrumental at 104 BPM; felt-piano motif, deep analog pulse, chamber strings, restrained hybrid percussion, granular air, soft woodwind color, controlled low brass weight",
    "Blend contemporary classical, ambient electronics, and detailed film scoring; develop one memorable motif through register, harmony, and orchestration",
    "generic trailer braams, superhero ostinato, sentimental piano cliche, constant impacts, melodrama, random key changes, boomy lows, washed-out reverb"],
  ["drum-bass", 174, "D_MAJOR_B_MINOR",
    "Forward-looking 2026 liquid-minimal drum and bass instrumental at 174 BPM; crisp two-step break, detailed ghost snares, articulate shuffled tops, deep sine sub, glass pads, concise Rhodes motif",
    "Blend liquid warmth, autonomic space, and modern break design; the sub answers the kick, granular atmosphere, precise stereo percussion",
    "jump-up wobble cliche, abrasive reese wall, neurofunk overload, random amen edits, trap hats, weak snare, distorted sub, piercing tops"],
  ["hiphop", 92, "E_FLAT_MAJOR_C_MINOR",
    "Forward-looking 2026 instrumental hip-hop at 92 BPM; heavy human kick-snare pocket, dusty layered drums, swung hats, deep controlled sub, warm electric keys, pitched texture fragments, memorable two-bar motif",
    "Blend progressive beat music, soulful harmony, minimal trap detail, and cinematic sampling aesthetics; intentional silence, selective ghost hits, microtiming, granular ambience",
    "rapping, vocal samples, generic trap loop, nonstop hi-hat rolls, drill cliche, boom-bap imitation, cheesy jazz sample, muddy 808, crushed master"],
  ["funk", 110, "G_MAJOR_E_MINOR",
    "Forward-looking 2026 instrumental future-funk at 110 BPM; elastic electric bass, dry punchy kit, ghost-note snare, crisp sixteenth hats, clavinet syncopation, muted guitar answers, compact brass accents",
    "Blend live funk pocket, broken-beat sophistication, subtle electronic processing, and modern R&B harmony; interlock bass, kick, clav, and guitar while preserving intentional rests",
    "disco pastiche, slap-bass comedy, retro imitation, busy brass solos, rock distortion, quantized stiffness, weak pocket, boomy bass, harsh clav"],
  ["samba", 108, "C_MAJOR_A_MINOR",
    "Forward-looking 2026 Brazilian electronic samba instrumental at 108 BPM; grounded surdo pulse, articulate caixa, tamborim syncopation, pandeiro detail, cavaquinho, nylon guitar, warm fluid bass",
    "Blend authentic samba ensemble interplay with restrained deep-house architecture, granular ambience, filtered percussion, and elegant harmonic color",
    "chants, tourist latin cliche, generic salsa, reggaeton dembow, carnival overload, cheesy brass, quantized stiffness, muddy surdo, brittle highs, festival drop"],
  ["country", 106, "G_MAJOR_E_MINOR",
    "Forward-looking 2026 instrumental country at 106 BPM; tight acoustic drum kit, rounded electric bass, articulate acoustic strum, warm Telecaster accents, pedal steel color, concise fiddle hook",
    "Blend modern roots, Americana, and restrained Nashville production; human pocket, clean picking, honest chord movement, short call-and-response phrases",
    "bro-country cliche, novelty banjo, pop drum machine, arena-rock wall, endless guitar solo, fake southern pastiche, stiff timing, harsh fiddle, muddy low mids"],
  ["edm", 128, "D_MAJOR_B_MINOR",
    "Forward-looking 2026 instrumental electronic dance music at 128 BPM; physical four-on-floor kick, controlled sub, crisp percussion, dimensional chord stack, distinctive synth hook, detailed transitions",
    "Blend progressive, melodic, bass, and left-field club production with one strong identity; tension through subtraction, automation, granular ear candy",
    "generic festival supersaws, cheesy melody, predictable white-noise riser, big-room cliche, constant drop, muddy sub, harsh limiter, retro imitation"],
  ["rnb", 88, "E_FLAT_MAJOR_C_MINOR",
    "Forward-looking 2026 instrumental R&B at 88 BPM; deep human drum pocket, soft punchy kick, rim and ghost-note detail, rounded sub bass, expressive Rhodes voicings, muted guitar, concise synth motif",
    "Blend alternative R&B, neo-soul harmony, progressive beat craft, and subtle electronic sound design; microtiming, intentional silence, rich extensions, responsive bass movement",
    "generic trap beat, nonstop hi-hat rolls, smooth-jazz cliche, pop ballad melody, overplayed runs, muddy sub, washed reverb, stiff quantization"],
  ["blues", 96, "A_MAJOR_G_FLAT_MINOR",
    "Forward-looking 2026 instrumental electric blues at 96 BPM; deep live drum pocket, warm fingered bass, expressive tube-amp guitar, concise organ responses, memorable bent-note motif",
    "Authentic call-and-response, tasteful dominant harmony, human microtiming, dynamic touch; guitar, organ, and rhythm section leave deliberate room",
    "blues-rock cliche, endless guitar solo, bar-band shuffle parody, synthetic drums, excessive distortion, stiff timing, muddy bass, harsh organ"],
  ["experimental", 118, "D_MAJOR_B_MINOR",
    "Forward-looking 2026 experimental instrumental at 118 BPM; prepared piano attacks, physical-model percussion, elastic sub pulse, spectral synth fragments, resonant metal, one clear three-note identity",
    "Electro-acoustic detail, generative rhythm, microtonal color around a stable tonal center, granular transformations, asymmetric accents, controlled negative space without losing the downbeat",
    "random noise collage, beatless drift, genre roulette, constant glitching, key collapse, academic abstraction, novelty sounds, harsh resonances, no motif"],
  ["jazz", 112, "B_FLAT_MAJOR_G_MINOR",
    "Forward-looking 2026 instrumental jazz at 112 BPM; acoustic piano, articulate upright bass, brushed kit, human ride swing, selective muted-horn color, subtle granular room texture, memorable harmonic motif",
    "Blend modern piano-trio conversation, broken-beat nuance, modal harmony, and restrained electro-acoustic sound design; responsive comping, breathing bass movement, silence between phrases",
    "scat, lounge cliche, smooth-jazz sax, endless solos, bebop imitation, random chord substitutions, stiff quantization, busy drums, boomy upright bass, cocktail background music"],
  ["classical", 84, "C_MAJOR_A_MINOR",
    "Forward-looking contemporary classical instrumental at 84 BPM; concert grand piano, intimate chamber strings, soft woodwind color, expressive arpeggiated pulse, clear minor-key motif, nuanced human dynamics",
    "Blend romantic harmonic depth, contemporary minimalism, and restrained cinematic sound; develop one motif through inversion, register, counterline, and orchestral color",
    "choir, generic trailer music, sentimental cliche, constant arpeggios, oversized percussion, synthetic strings, boomy hall, harsh piano, abrupt key changes"],
  ["ambient", 76, "E_FLAT_MAJOR_C_MINOR",
    "Forward-looking 2026 ambient electronic instrumental at 76 BPM; slow analog pads, low sine foundation, felt-piano fragments, granular field textures, tape echoes, delicate high-frequency particles",
    "Blend deep ambient, electro-acoustic detail, dub space, and contemporary minimalism; gentle subliminal pulse, evolving voicings, one recognizable tonal motif",
    "nature-sound cliche, meditation stock music, new-age melody, conventional drum beat, constant drone, random notes, excessive shimmer, muddy reverb, sudden climax"],
].map(([id, bpm, scale, identity, blend, negative]) => ({
  id,
  prompts: [
    { text: identity, weight: 1.3 },
    { text: blend, weight: 1.06 },
    { text: "Complete 30-second miniature composition in four phrases: 0-7s full groove and hook; 7-15s develop it; 15-22s subtract into a short breakdown; 22-30s deliver the strongest resolved payoff", weight: 1.14 },
    { text: "Begin on beat 1 with no intro silence; use authored phrase-boundary fills and one memorable motif, not a random jam; make the final two seconds harmonically compatible with the opening downbeat", weight: 1.0 },
    { text: `vocals, lyrics, incomplete sketch, random jam, weak hook, tempo drift, key clash, genre changes, long intro, fade out, abrupt cadence, ${negative}`, weight: -1.18 },
  ],
  config: { bpm, scale, density: id === "rock" ? 0.78 : id === "ambient" ? 0.3 : id === "drum-bass" ? 0.72 : 0.58, brightness: id === "rock" ? 0.62 : id === "techno" ? 0.44 : 0.52, guidance: id === "rock" ? 5.9 : 5.5, temperature: id === "rock" ? 0.74 : 0.86, topK: id === "rock" ? 24 : 32, muteBass: false, muteDrums: id === "ambient" || id === "classical", onlyBassAndDrums: false, musicGenerationMode: id === "experimental" ? "DIVERSITY" : "QUALITY" },
}));

styles.push(...[
  ["vocal-male", "Low masculine wordless lead vocalization; resonant chest voice, controlled grit, sustained vowel hook, rhythmic breaths, expressive dynamics, voice only, no lyrics"],
  ["vocal-female", "Powerful feminine wordless lead vocalization; clear chest-to-head register, focused sustained vowel hook, precise rhythmic phrases, expressive dynamics, voice only, no lyrics"],
  ["vocal-other", "Androgynous experimental wordless lead vocalization; flexible register, layered vowel colors, controlled extended texture, memorable melodic contour, voice only, no lyrics"],
].map(([id, prompt]) => ({
  id,
  prompts: [
    { text: prompt, weight: 1.35 },
    { text: "Complete 30-second voice-only miniature in four phrases: introduce motif, develop contour, leave a short breath, then deliver a sustained resolved final hook compatible with looping", weight: 1.15 },
    { text: "dry intimate center with tasteful short room and clean stereo doubles only on the final phrase; human breath and articulation; no backing music", weight: 1.02 },
    { text: "lyrics, intelligible words, instruments, drums, bass, guitar, piano, synth, pads, accompaniment, random choir wall, harsh sibilance, muddy reverb, constant singing", weight: -1.25 },
  ],
  config: { bpm: 96, scale: "G_MAJOR_E_MINOR", density: 0.36, brightness: id === "vocal-male" ? 0.42 : 0.62, guidance: 5.9, temperature: 0.78, topK: 24, muteBass: true, muteDrums: true, onlyBassAndDrums: false, musicGenerationMode: "VOCALIZATION" },
})));

styles.push({
  id: "welcome",
  prompts: [
    { text: "Futuristic Musica loading sound; warm low sine foundation, four-note glass-and-synth sonic logo, slow spectral wave, soft spatial pulses, premium restrained digital identity", weight: 1.35 },
    { text: "Complete seamless 30-second ambient loading loop; introduce the four-note identity immediately, let it echo through evolving harmonics, then resolve gently into its opening tone", weight: 1.15 },
    { text: "Elegant dark stereo depth, smooth transients, subtle cyan-like brightness, low listening level, no dramatic climax, compatible underneath a realtime connection screen", weight: 1.02 },
    { text: "vocals, lyrics, drums, beat, bass groove, full song, melody solo, trailer impact, alarm, notification beep, harsh highs, loud master, silence, fade out, abrupt ending", weight: -1.25 },
  ],
  config: { bpm: 72, scale: "C_MAJOR_A_MINOR", density: 0.24, brightness: 0.52, guidance: 5.8, temperature: 0.74, topK: 24, muteBass: true, muteDrums: true, onlyBassAndDrums: false, musicGenerationMode: "QUALITY" },
});

styles.push({
  id: "intro-chime",
  oneShot: true,
  skipSeconds: 1.4,
  clipSeconds: 6,
  fadeOutSeconds: 1.2,
  postFilter: "flanger=depth=5:delay=2:regen=18:width=55:speed=0.35,apad=pad_dur=0.9,aecho=0.72:0.45:34|58:0.2|0.13",
  prompts: [
    { text: "Slow building corporate boot tone resolving into an iconic Intel-bong-style chime; one continuous warm synthesized tone that swells gradually in volume, pitch, and brightness like a machine slowly powering on, patient and cinematic", weight: 1.45 },
    { text: "The swell builds without interruption for four seconds, tension rising the whole time, then blooms into a struck glass-and-bell five-note mnemonic that rings out warmly and decays to silence", weight: 1.3 },
    { text: "Woven faintly beneath the building swell, a ghostly music-box whisper of Beethoven's Fur Elise opening motif (E, D-sharp, E, D-sharp, E, B, D, C, A) in A minor, barely audible, more texture and memory than melody, never dominant", weight: 0.8 },
    { text: "Single short one-shot brand identity sound, not a loop; wide clean stereo image, precise mallet transients, warm sub-tone under each bell strike, quick natural decay to silence", weight: 1.05 },
    { text: "vocals, lyrics, drums, beat, bass groove, full song, loop, repeating phrase, melody solo, harsh distortion, loud master, muddy low end, alarm klaxon, abrupt cutoff, loud piano, full Fur Elise performance, orchestra, strings, slow ambient drift", weight: -1.3 },
  ],
  config: { bpm: 60, scale: "C_MAJOR_A_MINOR", density: 0.18, brightness: 0.58, guidance: 6, temperature: 0.68, topK: 20, muteBass: true, muteDrums: true, onlyBassAndDrums: false, musicGenerationMode: "QUALITY" },
});

function parseEnvFile(path) {
  const values = {};
  for (const rawLine of readFileSync(path, "utf8").split(/\r?\n/)) {
    const match = /^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(rawLine.trim());
    if (!match) continue;
    let value = match[2].trim();
    const singleQuote = String.fromCharCode(39);
    if ((value.startsWith(String.fromCharCode(34)) && value.endsWith(String.fromCharCode(34))) || (value.startsWith(singleQuote) && value.endsWith(singleQuote))) value = value.slice(1, -1);
    values[match[1]] = value;
  }
  return values;
}

function findEnvFiles() {
  const files = [];
  let current = resolve(process.cwd());
  while (true) {
    const candidate = join(current, ".env");
    if (existsSync(candidate)) files.unshift(candidate);
    const parent = dirname(current);
    if (parent === current) return files;
    current = parent;
  }
}

function run(command, args) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(command, args, { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => { stderr += chunk.toString(); });
    child.on("error", reject);
    child.on("exit", (code) => code === 0 ? resolvePromise() : reject(new Error(`${command} failed with exit ${code}: ${stderr.trim()}`)));
  });
}

async function messageDataToBuffer(data) {
  if (typeof data === "string") return Buffer.from(data, "utf8");
  if (data instanceof Buffer) return data;
  if (data instanceof ArrayBuffer) return Buffer.from(data);
  if (ArrayBuffer.isView(data)) return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  if (typeof Blob !== "undefined" && data instanceof Blob) return Buffer.from(await data.arrayBuffer());
  throw new Error("Unsupported Lyria WebSocket payload");
}

async function captureStyle(style, apiKey) {
  const chunks = [];
  let capturedBytes = 0;
  let completed = false;
  const socket = new WebSocket(`${WS_ENDPOINT}?key=${encodeURIComponent(apiKey)}`);
  const timeout = setTimeout(() => socket.close(), 75_000);
  await new Promise((resolvePromise, reject) => {
    let setupComplete = false;
    socket.addEventListener("open", () => socket.send(JSON.stringify({ setup: { model: MODEL } })));
    socket.addEventListener("error", () => reject(new Error(`Lyria WebSocket failed for ${style.id}`)));
    socket.addEventListener("message", async (event) => {
      const payload = await messageDataToBuffer(event.data);
      let value;
      try { value = JSON.parse(payload.toString("utf8")); } catch {
        chunks.push(payload);
        capturedBytes += payload.length;
        return;
      }
      if (!setupComplete && (value.setupComplete ?? value.setup_complete) !== undefined) {
        setupComplete = true;
        socket.send(JSON.stringify({ client_content: { weightedPrompts: style.prompts } }));
        socket.send(JSON.stringify({ music_generation_config: style.config }));
        setTimeout(() => socket.send(JSON.stringify({ playback_control: "PLAY" })), 100);
        return;
      }
      const audioChunks = value.serverContent?.audioChunks ?? value.server_content?.audio_chunks ?? [];
      for (const chunk of audioChunks) {
        if (typeof chunk.data !== "string") continue;
        const bytes = Buffer.from(chunk.data, "base64");
        chunks.push(bytes);
        capturedBytes += bytes.length;
        if (!completed && capturedBytes >= TARGET_CAPTURE_BYTES) {
          completed = true;
          socket.send(JSON.stringify({ playback_control: "STOP" }));
          setTimeout(() => socket.close(), 250);
        }
      }
    });
    socket.addEventListener("close", () => { clearTimeout(timeout); resolvePromise(); });
  });
  const requiredBytes = style.oneShot
    ? SAMPLE_RATE_HZ * CHANNELS * 2 * ((style.skipSeconds ?? 1.2) + (style.clipSeconds ?? 5) + 1)
    : MIN_CAPTURE_BYTES;
  if (capturedBytes < requiredBytes) throw new Error(`${style.id} returned only ${Math.round(capturedBytes / 1024)} KB of PCM`);
  return Buffer.concat(chunks).subarray(0, TARGET_CAPTURE_BYTES);
}

const env = Object.assign({}, ...findEnvFiles().map(parseEnvFile));
const apiKey = process.env.GEMINI_API_KEY ?? env.GEMINI_API_KEY;
if (!apiKey || apiKey.trim().length < 20) throw new Error("GEMINI_API_KEY was not found in the shell environment or a parent .env file.");
if (typeof WebSocket !== "function") throw new Error("Node.js 24 or newer is required.");

const outputDir = resolve("public/previews/lyria");
mkdirSync(outputDir, { recursive: true });
const force = process.env.LYRIA_PREVIEW_FORCE === "true";
const requestedStyles = new Set((process.env.LYRIA_PREVIEW_STYLES ?? "").split(",").map((value) => value.trim()).filter(Boolean));
const renderStyles = requestedStyles.size > 0 ? styles.filter((style) => requestedStyles.has(style.id)) : styles;
const receipts = [];
for (const [index, style] of renderStyles.entries()) {
  const output = join(outputDir, `${style.id}.mp3`);
  if (existsSync(output) && !force) {
    console.log(`[${index + 1}/${renderStyles.length}] ${style.id}: keeping existing preview`);
    receipts.push({ id: style.id, file: `${style.id}.mp3`, bytes: statSync(output).size, model: MODEL });
    continue;
  }
  console.log(`[${index + 1}/${renderStyles.length}] ${style.id}: capturing ${CAPTURE_SECONDS}s from Lyria RealTime`);
  const pcm = await captureStyle(style, apiKey);
  const pcmPath = join(outputDir, `.${style.id}.pcm`);
  writeFileSync(pcmPath, pcm);
  if (style.oneShot) {
    const skip = style.skipSeconds ?? 1.2;
    const clip = style.clipSeconds ?? 5;
    const fadeOut = style.fadeOutSeconds ?? 0.8;
    const post = style.postFilter ? `,${style.postFilter}` : "";
    const filter = `atrim=start=${skip}:end=${skip + clip},asetpts=PTS-STARTPTS,afade=t=in:d=0.06,afade=t=out:st=${clip - fadeOut}:d=${fadeOut}${post}`;
    await run("ffmpeg", ["-y", "-f", "s16le", "-ar", String(SAMPLE_RATE_HZ), "-ac", String(CHANNELS), "-i", pcmPath, "-filter:a", filter, "-codec:a", "libmp3lame", "-b:a", "160k", output]);
  } else {
    const filter = "[0:a]atrim=start=2:end=30,asetpts=PTS-STARTPTS[body];[0:a]atrim=start=30:end=32,asetpts=PTS-STARTPTS[tail];[0:a]atrim=start=0:end=2,asetpts=PTS-STARTPTS[head];[tail][head]acrossfade=d=2:c1=tri:c2=tri[wrap];[body][wrap]concat=n=2:v=0:a=1[out]";
    await run("ffmpeg", ["-y", "-f", "s16le", "-ar", String(SAMPLE_RATE_HZ), "-ac", String(CHANNELS), "-i", pcmPath, "-filter_complex", filter, "-map", "[out]", "-t", String(OUTPUT_SECONDS), "-codec:a", "libmp3lame", "-b:a", "160k", output]);
  }
  rmSync(pcmPath, { force: true });
  receipts.push({ id: style.id, file: `${style.id}.mp3`, bytes: statSync(output).size, model: MODEL, bpm: style.config.bpm, prompt: style.prompts[0].text });
}
writeFileSync(join(outputDir, "manifest.json"), JSON.stringify({ generatedAt: new Date().toISOString(), durationSeconds: OUTPUT_SECONDS, loopCrossfadeSeconds: 2, previews: receipts }, null, 2) + "\n");
console.log(`Wrote ${receipts.length} Lyria onboarding previews to ${outputDir}`);
