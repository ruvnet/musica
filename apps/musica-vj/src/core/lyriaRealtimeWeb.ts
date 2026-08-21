// Browser-native Lyria RealTime client — the web equivalent of
// src-tauri/src/lyria_realtime_provider.rs, talking directly to Gemini's Live
// API over a real WebSocket instead of going through Tauri IPC to Rust.
//
// Only the API-key auth path ports here (the desktop's gcloud-ADC path shells
// out to the `gcloud` CLI, which has no browser equivalent). The key comes
// from `configureLyriaRealtimeWebKey`, called by cognitumWeb.ts after
// redeeming a Cognitum sign-in for a short-lived broker key (ADR-182), or —
// in principle — a bring-your-own Gemini API key.
//
// Wire protocol (mirrors lyria_realtime_provider.rs exactly):
//   -> {"setup":{"model":"models/lyria-realtime-exp"}}
//   <- {"setupComplete":{}}                                  (or setup_complete)
//   -> {"client_content":{"weightedPrompts":[{text,weight}]}}
//   -> {"music_generation_config":{bpm,guidance,density,...}}
//   -> {"playback_control":"PLAY"}
//   <- {"serverContent":{"audioChunks":[{"data":"<base64 pcm16>"}]}}
//   -> {"playback_control":"STOP"}                            (graceful stop)
import type {
  LyriaRealtimeAudioPoll,
  LyriaRealtimeDeckId,
  LyriaRealtimeRequest,
  LyriaRealtimeSession,
  LyriaRealtimeStatus,
} from "./lyriaRealtime";

const MODEL = "models/lyria-realtime-exp";
const WS_ENDPOINT =
  "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateMusic";
const SAMPLE_RATE_HZ = 48_000;
const CHANNELS = 2;
const MAX_QUEUED_AUDIO_BYTES = 48_000 * 2 * 2 * 8; // 8s of stereo PCM16, matches MAX_QUEUED_AUDIO_BYTES in Rust
const MAX_POLL_BYTES = 48_000 * 2 * 2; // 1s per poll, matches MAX_POLL_BYTES in Rust
const RECONNECT_BASE_MS = 1_000;
const RECONNECT_MAX_MS = 15_000;
const RECONNECT_MAX_ATTEMPTS = 6;

type StreamPhase = "setup" | "streaming";

interface DeckState {
  ws: WebSocket | null;
  phase: StreamPhase;
  session: LyriaRealtimeSession | null;
  pendingRequest: LyriaRealtimeRequest | null;
  audioQueue: Uint8Array[];
  bufferedBytes: number;
  streamedBytes: number;
  warning?: string;
  closing: boolean;
  reconnectAttempt: number;
  reconnectTimer: number | null;
}

const decks = new Map<LyriaRealtimeDeckId, DeckState>();
let runtimeKey: string | null = null;

/// Injects (or clears) a runtime-brokered Gemini/Lyria API key — the browser
/// equivalent of the `lyria_realtime_configure_key` Tauri command. Pass an
/// empty string to clear it.
export function configureLyriaRealtimeWebKey(key: string): void {
  const trimmed = key.trim();
  runtimeKey = trimmed.length > 0 ? trimmed : null;
}

function freshDeckState(): DeckState {
  return {
    ws: null,
    phase: "setup",
    session: null,
    pendingRequest: null,
    audioQueue: [],
    bufferedBytes: 0,
    streamedBytes: 0,
    closing: false,
    reconnectAttempt: 0,
    reconnectTimer: null,
  };
}

function unit(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

function validateRequest(request: LyriaRealtimeRequest): void {
  const { weightedPrompts, config } = request;
  if (weightedPrompts.length === 0 || weightedPrompts.length > 4) {
    throw new Error("Lyria RealTime requires one to four weighted prompts");
  }
  for (const prompt of weightedPrompts) {
    const text = prompt.text.trim();
    if (text.length === 0 || [...text].length > 240) {
      throw new Error("Lyria RealTime prompts must be 1 to 240 characters");
    }
    if (!Number.isFinite(prompt.weight) || prompt.weight === 0 || prompt.weight < -3 || prompt.weight > 3) {
      throw new Error("Lyria RealTime prompt weights must be finite, non-zero, and between -3 and 3");
    }
  }
  if (config.bpm < 60 || config.bpm > 200) {
    throw new Error("Lyria RealTime BPM must be 60 to 200");
  }
  if (!unit(config.density) || !unit(config.brightness)) {
    throw new Error("Lyria RealTime density and brightness must be 0 to 1");
  }
  if (!Number.isFinite(config.guidance) || config.guidance < 0 || config.guidance > 6) {
    throw new Error("Lyria RealTime guidance must be 0 to 6");
  }
  if (!Number.isFinite(config.temperature) || config.temperature < 0 || config.temperature > 3) {
    throw new Error("Lyria RealTime temperature must be 0 to 3");
  }
  if (config.topK < 1 || config.topK > 1000) {
    throw new Error("Lyria RealTime topK must be 1 to 1000");
  }
  if (config.onlyBassAndDrums && (config.muteBass || config.muteDrums)) {
    throw new Error("onlyBassAndDrums cannot be combined with muted bass or drums");
  }
}

function base64ToBytes(base64: string): Uint8Array | null {
  try {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  } catch {
    return null;
  }
}

const utf8Decoder = new TextDecoder("utf-8", { fatal: true });

type Incoming = { kind: "json"; value: unknown } | { kind: "binary"; bytes: Uint8Array } | { kind: "ignore" };

function parseIncoming(event: MessageEvent): Incoming {
  if (typeof event.data === "string") {
    try {
      return { kind: "json", value: JSON.parse(event.data) };
    } catch {
      return { kind: "ignore" };
    }
  }
  if (event.data instanceof ArrayBuffer) {
    const bytes = new Uint8Array(event.data);
    try {
      const text = utf8Decoder.decode(bytes);
      return { kind: "json", value: JSON.parse(text) };
    } catch {
      return { kind: "binary", bytes };
    }
  }
  return { kind: "ignore" };
}

function pushAudio(state: DeckState, bytes: Uint8Array): void {
  if (bytes.length === 0) return;
  state.streamedBytes += bytes.length;
  state.audioQueue.push(bytes);
  state.bufferedBytes += bytes.length;
  while (state.bufferedBytes > MAX_QUEUED_AUDIO_BYTES && state.audioQueue.length > 0) {
    const dropped = state.audioQueue.shift();
    if (dropped) state.bufferedBytes -= dropped.length;
  }
}

function drainAudio(state: DeckState, maxBytes: number): Uint8Array[] {
  const drained: Uint8Array[] = [];
  let total = 0;
  while (state.audioQueue.length > 0) {
    const chunk = state.audioQueue[0]!;
    if (total > 0 && total + chunk.length > maxBytes) break;
    state.audioQueue.shift();
    state.bufferedBytes -= chunk.length;
    total += chunk.length;
    drained.push(chunk);
    if (total >= maxBytes) break;
  }
  return drained;
}

function applyServerMessage(value: unknown, state: DeckState): void {
  if (typeof value !== "object" || value === null) return;
  const record = value as Record<string, unknown>;
  if (typeof record.warning === "string") state.warning = record.warning;
  if (record.error !== undefined) state.warning = `Lyria RealTime error: ${JSON.stringify(record.error)}`;
  const filtered = record.filteredPrompt ?? record.filtered_prompt;
  if (filtered !== undefined) state.warning = `Filtered prompt: ${JSON.stringify(filtered)}`;

  const serverContent = (record.serverContent ?? record.server_content) as Record<string, unknown> | undefined;
  const chunks = serverContent?.audioChunks ?? serverContent?.audio_chunks;
  if (Array.isArray(chunks)) {
    for (const chunk of chunks) {
      const data = (chunk as Record<string, unknown> | null)?.data;
      if (typeof data === "string") {
        const bytes = base64ToBytes(data);
        if (bytes) pushAudio(state, bytes);
        else state.warning = "Lyria RealTime returned invalid Base64 audio";
      }
    }
  }
}

function sendRealtimeRequest(ws: WebSocket, request: LyriaRealtimeRequest): void {
  ws.send(JSON.stringify({ client_content: { weightedPrompts: request.weightedPrompts } }));
  ws.send(JSON.stringify({ music_generation_config: request.config }));
}

function scheduleReconnect(deck: LyriaRealtimeDeckId, state: DeckState): void {
  if (state.closing) return;
  state.reconnectAttempt += 1;
  if (state.reconnectAttempt > RECONNECT_MAX_ATTEMPTS) {
    state.warning = "Lyria RealTime lost connection and could not reconnect — press play to retry";
    return;
  }
  const delay = Math.min(RECONNECT_MAX_MS, RECONNECT_BASE_MS * 2 ** (state.reconnectAttempt - 1));
  state.reconnectTimer = window.setTimeout(() => {
    state.reconnectTimer = null;
    if (!state.closing) connect(deck, state);
  }, delay);
}

function connect(deck: LyriaRealtimeDeckId, state: DeckState): void {
  const key = runtimeKey;
  if (!key) {
    state.warning = "Lyria RealTime key is no longer available";
    return;
  }
  state.phase = "setup";
  const ws = new WebSocket(`${WS_ENDPOINT}?key=${encodeURIComponent(key)}`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.addEventListener("open", () => {
    ws.send(JSON.stringify({ setup: { model: MODEL } }));
  });

  ws.addEventListener("message", (event) => {
    const incoming = parseIncoming(event);
    if (state.phase === "setup") {
      if (incoming.kind !== "json") return;
      const value = incoming.value as Record<string, unknown>;
      if (value.setupComplete !== undefined || value.setup_complete !== undefined) {
        state.phase = "streaming";
        state.reconnectAttempt = 0;
        if (state.pendingRequest) sendRealtimeRequest(ws, state.pendingRequest);
        ws.send(JSON.stringify({ playback_control: "PLAY" }));
        return;
      }
      if (typeof value.warning === "string") state.warning = value.warning;
      if (value.error !== undefined) {
        state.warning = `Lyria RealTime setup error: ${JSON.stringify(value.error)}`;
        ws.close();
      }
      return;
    }
    if (incoming.kind === "json") applyServerMessage(incoming.value, state);
    else if (incoming.kind === "binary") pushAudio(state, incoming.bytes);
  });

  ws.addEventListener("close", () => {
    state.ws = null;
    if (state.closing) return;
    state.warning = state.phase === "setup" ? "Lyria RealTime setup did not complete" : "Lyria RealTime WebSocket closed";
    scheduleReconnect(deck, state);
  });
}

export function getLyriaRealtimeWebStatus(deck: LyriaRealtimeDeckId): LyriaRealtimeStatus {
  const state = decks.get(deck);
  const available = runtimeKey !== null;
  return {
    deck,
    available,
    provider: "lyria_realtime",
    model: MODEL,
    sampleRateHz: SAMPLE_RATE_HZ,
    channels: CHANNELS,
    audioFormat: "pcm16",
    instrumentalOnly: true,
    reason: available ? undefined : "Sign in with Cognitum One to authorize Lyria",
    activeSessionId: state?.session?.id,
    bufferedAudioBytes: state?.bufferedBytes ?? 0,
    streamedAudioBytes: state?.streamedBytes ?? 0,
    warning: state?.warning,
  };
}

export async function startLyriaRealtimeWeb(
  request: LyriaRealtimeRequest,
  deck: LyriaRealtimeDeckId,
): Promise<LyriaRealtimeSession> {
  if (!runtimeKey) throw new Error("Sign in with Cognitum One first to authorize Lyria");
  validateRequest(request);
  await stopLyriaRealtimeWeb(deck);
  const state = freshDeckState();
  decks.set(deck, state);
  state.pendingRequest = request;
  const session: LyriaRealtimeSession = {
    deck,
    id: `lrt-web-${deck}-${Date.now()}`,
    provider: "lyria_realtime",
    model: MODEL,
    state: "streaming",
    weightedPrompts: request.weightedPrompts,
    config: request.config,
    sampleRateHz: SAMPLE_RATE_HZ,
    channels: CHANNELS,
    audioFormat: "pcm16",
  };
  state.session = session;
  connect(deck, state);
  return session;
}

export async function updateLyriaRealtimeWeb(
  request: LyriaRealtimeRequest,
  deck: LyriaRealtimeDeckId,
): Promise<LyriaRealtimeSession> {
  validateRequest(request);
  const state = decks.get(deck);
  if (!state || !state.session) throw new Error("Lyria RealTime deck is not active");
  state.pendingRequest = request;
  state.session = { ...state.session, weightedPrompts: request.weightedPrompts, config: request.config };
  if (state.ws && state.ws.readyState === WebSocket.OPEN && state.phase === "streaming") {
    sendRealtimeRequest(state.ws, request);
  }
  return state.session;
}

export async function stopLyriaRealtimeWeb(deck: LyriaRealtimeDeckId): Promise<void> {
  const state = decks.get(deck);
  if (!state) return;
  state.closing = true;
  if (state.reconnectTimer !== null) {
    window.clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
  }
  if (state.ws) {
    try {
      if (state.ws.readyState === WebSocket.OPEN) {
        state.ws.send(JSON.stringify({ playback_control: "STOP" }));
      }
      state.ws.close();
    } catch {
      // already closing/closed — nothing to do
    }
  }
  decks.delete(deck);
}

export function pollLyriaRealtimeWebAudio(deck: LyriaRealtimeDeckId): LyriaRealtimeAudioPoll {
  const state = decks.get(deck);
  if (!state) {
    return {
      deck,
      sampleRateHz: SAMPLE_RATE_HZ,
      channels: CHANNELS,
      audioFormat: "pcm16",
      chunks: [],
      bufferedAudioBytes: 0,
      streamedAudioBytes: 0,
    };
  }
  const drained = drainAudio(state, MAX_POLL_BYTES);
  return {
    deck,
    sessionId: state.session?.id,
    sampleRateHz: SAMPLE_RATE_HZ,
    channels: CHANNELS,
    audioFormat: "pcm16",
    chunks: drained.map((bytes) => Array.from(bytes)),
    bufferedAudioBytes: state.bufferedBytes,
    streamedAudioBytes: state.streamedBytes,
    warning: state.warning,
  };
}
