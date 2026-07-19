import { createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawn } from "node:child_process";

const MODEL = "models/lyria-realtime-exp";
const WS_ENDPOINT = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic";
const SAMPLE_RATE_HZ = 48_000;
const CHANNELS = 2;
const BYTES_PER_SECOND = SAMPLE_RATE_HZ * CHANNELS * 2;
const CAPTURE_SECONDS = Number(process.env.LYRIA_REALTIME_SAMPLE_SECONDS ?? 22);
const MIN_CAPTURE_BYTES = BYTES_PER_SECOND * Math.min(8, CAPTURE_SECONDS * 0.5);

function parseEnvFile(path) {
  const values = {};
  const text = readFileSync(path, "utf8");
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = /^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(line);
    if (!match) continue;
    let value = match[2].trim();
    if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    values[match[1]] = value;
  }
  return values;
}

function findEnvFiles() {
  const found = [];
  let current = resolve(process.cwd());
  while (true) {
    const candidate = join(current, ".env");
    if (existsSync(candidate)) found.unshift(candidate);
    const parent = dirname(current);
    if (parent === current) return found;
    current = parent;
  }
}

function sendJson(socket, value) {
  socket.send(JSON.stringify(value));
}

function audioChunksFromMessage(value) {
  const chunks = value.serverContent?.audioChunks ?? value.server_content?.audio_chunks ?? [];
  return chunks.flatMap((chunk) => {
    if (typeof chunk.data !== "string") return [];
    return [Buffer.from(chunk.data, "base64")];
  });
}

async function messageDataToBuffer(data) {
  if (typeof data === "string") return Buffer.from(data, "utf8");
  if (data instanceof Buffer) return data;
  if (data instanceof ArrayBuffer) return Buffer.from(data);
  if (ArrayBuffer.isView(data)) return Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  if (typeof Blob !== "undefined" && data instanceof Blob) return Buffer.from(await data.arrayBuffer());
  throw new Error(`Unsupported WebSocket payload type: ${Object.prototype.toString.call(data)}`);
}

function encodeMp3(pcmPath, mp3Path) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn("ffmpeg", [
      "-y",
      "-f", "s16le",
      "-ar", String(SAMPLE_RATE_HZ),
      "-ac", String(CHANNELS),
      "-i", pcmPath,
      "-codec:a", "libmp3lame",
      "-b:a", "192k",
      mp3Path,
    ], { stdio: ["ignore", "ignore", "pipe"] });
    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolvePromise();
      else reject(new Error(`ffmpeg failed with exit ${code}: ${stderr.trim()}`));
    });
  });
}

const fileEnv = Object.assign({}, ...findEnvFiles().map(parseEnvFile));
const apiKey = process.env.GEMINI_API_KEY ?? fileEnv.GEMINI_API_KEY;
if (!apiKey || apiKey.trim().length < 20) {
  throw new Error("GEMINI_API_KEY was not found in the shell environment or a parent .env file.");
}
if (typeof WebSocket !== "function") {
  throw new Error("Node.js WebSocket support is required. Use Node 24 or newer.");
}

const sampleDir = resolve("samples/lyria");
mkdirSync(sampleDir, { recursive: true });
const pcmPath = join(sampleDir, "lyria-realtime-programmed-stream.pcm");
const mp3Path = join(sampleDir, "lyria-realtime-programmed-stream.mp3");
rmSync(pcmPath, { force: true });

const pcm = createWriteStream(pcmPath);
const socket = new WebSocket(`${WS_ENDPOINT}?key=${encodeURIComponent(apiKey)}`);
let setupComplete = false;
let capturedBytes = 0;
let closed = false;
let firstAudioAt;

const program = [
  {
    atMs: 0,
    name: "samba intro",
    prompts: [
      { text: "Brazilian samba house groove, nylon guitar, cavaquinho, surdo, tamborim, polished instrumental mix", weight: 1 },
      { text: "warm dance floor energy, clean bass, human percussion, no vocals", weight: 0.72 },
    ],
    config: { bpm: 104, density: 0.72, brightness: 0.62, guidance: 4.3, temperature: 1.05, topK: 40, scale: "C_MAJOR_A_MINOR", muteBass: false, muteDrums: false, onlyBassAndDrums: false, musicGenerationMode: "QUALITY" },
  },
  {
    atMs: 7_000,
    name: "rock lift",
    prompts: [
      { text: "tight live rock band, punchy drums, electric guitar hook, strong bass, instrumental festival mix", weight: 1 },
      { text: "samba percussion remains underneath, energetic but controlled, no vocals", weight: 0.38 },
    ],
    config: { bpm: 122, density: 0.66, brightness: 0.56, guidance: 4.1, temperature: 1.0, topK: 44, scale: "E_MAJOR_D_FLAT_MINOR", muteBass: false, muteDrums: false, onlyBassAndDrums: false, musicGenerationMode: "QUALITY" },
  },
  {
    atMs: 14_000,
    name: "cinematic outro",
    prompts: [
      { text: "cinematic electronic score, shimmering piano, wide synth pads, clean drums, emotional instrumental arc", weight: 1 },
      { text: "club groove gently resolves, polished master, no vocals", weight: 0.58 },
    ],
    config: { bpm: 108, density: 0.48, brightness: 0.42, guidance: 4.6, temperature: 0.95, topK: 36, scale: "C_MAJOR_A_MINOR", muteBass: false, muteDrums: false, onlyBassAndDrums: false, musicGenerationMode: "QUALITY" },
  },
];

function applyProgramStep(step) {
  console.log(`Lyria control: ${step.name}`);
  sendJson(socket, { client_content: { weightedPrompts: step.prompts } });
  sendJson(socket, { music_generation_config: step.config });
}

const timeout = setTimeout(() => {
  if (!closed) {
    closed = true;
    socket.close();
    pcm.destroy(new Error("Timed out waiting for enough Lyria RealTime audio bytes."));
  }
}, Math.max(20_000, CAPTURE_SECONDS * 1_000 + 10_000));

socket.addEventListener("open", () => {
  sendJson(socket, { setup: { model: MODEL } });
});

socket.addEventListener("message", async (event) => {
  const payload = await messageDataToBuffer(event.data);
  const text = payload.toString("utf8");
  let value;
  try {
    value = JSON.parse(text);
  } catch {
    capturedBytes += payload.length;
    pcm.write(payload);
    return;
  }
  if (value.warning) console.warn(`Lyria warning: ${value.warning}`);
  if (value.filteredPrompt ?? value.filtered_prompt) console.warn("Lyria filtered one of the prompts.");
  if (!setupComplete && (value.setupComplete ?? value.setup_complete) !== undefined) {
    setupComplete = true;
    for (const step of program) setTimeout(() => applyProgramStep(step), step.atMs);
    setTimeout(() => sendJson(socket, { playback_control: "PLAY" }), 120);
    setTimeout(() => sendJson(socket, { playback_control: "STOP" }), CAPTURE_SECONDS * 1_000);
    setTimeout(() => socket.close(), CAPTURE_SECONDS * 1_000 + 600);
    return;
  }
  for (const bytes of audioChunksFromMessage(value)) {
    if (bytes.length === 0) continue;
    if (firstAudioAt === undefined) firstAudioAt = Date.now();
    capturedBytes += bytes.length;
    pcm.write(bytes);
  }
});

await new Promise((resolvePromise, reject) => {
  socket.addEventListener("error", () => reject(new Error("Lyria RealTime WebSocket failed.")));
  socket.addEventListener("close", () => {
    closed = true;
    clearTimeout(timeout);
    pcm.end(resolvePromise);
  });
  pcm.on("error", reject);
});

if (capturedBytes < MIN_CAPTURE_BYTES) {
  rmSync(pcmPath, { force: true });
  throw new Error(`Lyria RealTime returned only ${capturedBytes} PCM bytes; expected at least ${MIN_CAPTURE_BYTES}.`);
}

await encodeMp3(pcmPath, mp3Path);
const mp3Bytes = statSync(mp3Path).size;
rmSync(pcmPath, { force: true });
console.log(`Wrote ${mp3Path}`);
console.log(`${Math.round(capturedBytes / 1024)} KB PCM captured, ${Math.round(mp3Bytes / 1024)} KB MP3 encoded.`);
