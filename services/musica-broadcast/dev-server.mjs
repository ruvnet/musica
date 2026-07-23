import { createServer } from "node:http";
import { verifyRequest } from "./auth.mjs";
import { handle } from "./handler.mjs";

/// Local development server for the broadcast plane (ADR-182).
///
/// Two things make the feature awkward to exercise on one machine, and this
/// solves both:
///
///  1. **No Firestore.** The store is an interface, so an in-memory Map is a
///     complete implementation. Nothing to provision.
///  2. **Nobody to follow.** A broadcaster is excluded from their own listener
///     count, so a single account has no one to follow. This seeds a synthetic
///     broadcaster whose set *evolves* — tempo, energy, scene, and generative
///     direction all move — so a follow is visibly alive rather than static.
///
/// The auth gate is NOT stubbed. Requests are verified against the real
/// Cognitum JWKS exactly as in production, so signing in to the app normally is
/// what makes this work — and a bug in the gate cannot hide behind a dev flag.

const PORT = Number(process.env.PORT ?? 8080);
process.env.BROADCAST_ID_SECRET ??= "local-development-secret";

const broadcasts = new Map();
const listeners = new Map();
const key = (id, subject) => `${id}::${subject}`;

const store = {
  async getBroadcast(id) {
    return broadcasts.get(id);
  },
  async putBroadcast(id, doc) {
    broadcasts.set(id, doc);
  },
  async listBroadcasts(limit) {
    return [...broadcasts.entries()]
      .map(([id, doc]) => ({ id, doc }))
      .sort((a, b) => (b.doc.listenerCount ?? 0) - (a.doc.listenerCount ?? 0))
      .slice(0, limit);
  },
  async getListener(id, subject) {
    return listeners.get(key(id, subject));
  },
  async putListener(id, subject, doc) {
    listeners.set(key(id, subject), doc);
  },
  async deleteListener(id, subject) {
    listeners.delete(key(id, subject));
  },
  async countListeners(id, seenSinceSeconds) {
    return [...listeners.entries()]
      .filter(([entry, doc]) => entry.startsWith(`${id}::`) && doc.seenAt >= seenSinceSeconds)
      .length;
  },
};

// --- the synthetic broadcaster -------------------------------------------

const DEMO_ID = "demo-broadcaster-0001";
const SCENES = ["lasergrid", "aurora", "monolith", "pulsefield", "chromawave"];
const PALETTES = ["neon", "ice", "mono", "ember", "prism"];
const STYLES = [
  { id: "techno", label: "Warehouse Techno", prompt: "warehouse techno, metallic stabs, dry punchy drums" },
  { id: "ambient", label: "Ambient Dub", prompt: "ambient dub, tape delay chords, spacious sub pulses" },
  { id: "house", label: "Afro Cosmic House", prompt: "afro house, polyrhythmic percussion, warm analog bass" },
];

let demoRev = 0;
let demoStep = 0;

function demoSnapshot() {
  demoStep += 1;
  demoRev += 1;
  // A slow energy curve so a follower can watch tempo, brightness, and look
  // travel together rather than jumping at random.
  const phase = (Math.sin(demoStep / 6) + 1) / 2;
  const style = STYLES[Math.floor(demoStep / 8) % STYLES.length];
  return {
    v: 1,
    rev: demoRev,
    performance: {
      bpm: Math.round(100 + phase * 40),
      styleId: style.id,
      styleLabel: style.label,
      deckEnabled: { main: true, sequence: phase > 0.5, vocal: false },
      deckControls: {
        main: { volume: 0.6 + phase * 0.3, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
        sequence: { volume: 0.3 + phase * 0.3, muted: false, pitchSemitones: 0, beatNudgeMs: 0 },
        vocal: { volume: 0.4, muted: true, pitchSemitones: 0, beatNudgeMs: 0 },
      },
    },
    lyria: {
      config: {
        bpm: Math.round(100 + phase * 40),
        guidance: 4,
        density: 0.3 + phase * 0.6,
        brightness: 0.25 + phase * 0.7,
        temperature: 1.1,
        topK: 40,
        scale: "C_MAJOR_A_MINOR",
        muteBass: false,
        muteDrums: false,
        onlyBassAndDrums: false,
        musicGenerationMode: "QUALITY",
      },
      prompts: [
        { text: style.prompt, weight: 1.2 },
        { text: phase > 0.6 ? "peak energy, driving" : "restrained, spacious", weight: 0.8 },
      ],
    },
    visual: {
      scene: SCENES[Math.floor(demoStep / 5) % SCENES.length],
      // Deliberately swings the full range, so the follower's slew limit is
      // visible in action rather than only in a unit test.
      intensity: phase,
      color: {
        palette: PALETTES[Math.floor(demoStep / 7) % PALETTES.length],
        hue: phase,
        saturation: 0.5 + phase * 0.4,
        contrast: 0.5,
        diversity: 0.4,
      },
    },
    fx: {
      effects: { flanger: 0, phaser: 0, drive: 0, crush: 0, sweep: phase * 0.3, reverb: (1 - phase) * 0.4, echo: 0 },
      params: {
        flangerRate: 0.2, flangerDepth: 0.5, flangerFeedback: 0.6, phaserRate: 0.25,
        phaserDepth: 0.6, driveEdge: 0.4, crushBits: 0.45, sweepRate: 0.3, sweepReso: 0.55,
      },
    },
  };
}

function refreshDemo() {
  const nowSeconds = Math.floor(Date.now() / 1000);
  const existing = broadcasts.get(DEMO_ID) ?? {};
  const snapshot = demoSnapshot();
  broadcasts.set(DEMO_ID, {
    ...existing,
    displayName: "Demo Broadcaster",
    snapshot,
    rev: snapshot.rev,
    live: true,
    updatedAt: nowSeconds,
  });
}

refreshDemo();
setInterval(refreshDemo, 4_000).unref?.();

// --- server ---------------------------------------------------------------

function readBody(request) {
  return new Promise((resolve) => {
    let raw = "";
    request.on("data", (chunk) => { raw += chunk; });
    request.on("end", () => {
      try {
        resolve(raw ? JSON.parse(raw) : undefined);
      } catch {
        resolve(undefined);
      }
    });
  });
}

createServer(async (request, response) => {
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("Content-Type", "application/json");
  const url = new URL(request.url, `http://${request.headers.host}`);
  try {
    const { subject } = await verifyRequest(request);
    const result = await handle({
      store,
      subject,
      method: request.method,
      path: url.pathname,
      query: Object.fromEntries(url.searchParams),
      body: await readBody(request),
      nowSeconds: Math.floor(Date.now() / 1000),
    });
    console.log(`${request.method} ${url.pathname} -> ${result.status} (${subject})`);
    response.statusCode = result.status;
    response.end(JSON.stringify(result.body));
  } catch (error) {
    const status = Number.isInteger(error?.status) ? error.status : 500;
    console.log(`${request.method} ${url.pathname} -> ${status}: ${error.message}`);
    response.statusCode = status;
    response.end(JSON.stringify({ error: status === 500 ? "Internal error" : error.message }));
  }
}).listen(PORT, "127.0.0.1", () => {
  console.log(`musica-broadcast dev server on http://127.0.0.1:${PORT}`);
  console.log('Seeded "Demo Broadcaster" — its set changes every 4s, so following it is visibly live.');
  console.log("");
  console.log("Point Musica at it:");
  console.log(`  MUSICA_BROADCAST_URL=http://127.0.0.1:${PORT}`);
  console.log("");
  console.log("Auth is NOT stubbed: sign in to Cognitum One in the app as usual.");
});
