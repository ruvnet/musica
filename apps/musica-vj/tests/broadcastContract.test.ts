import { describe, expect, it } from "vitest";
// The real service handler, imported across the package boundary on purpose:
// this is a contract test, and a hand-copied fixture of the response shape
// would prove nothing about the service that actually ships.
import { broadcastIdFor, handle } from "../../../services/musica-broadcast/handler.mjs";
import {
  normalizeBroadcastDirectory,
  normalizeBroadcastFollowUpdate,
  normalizeBroadcastSnapshot,
  captureBroadcastSnapshot,
} from "../src/core/broadcast";
import { DEFAULT_LYRIA_DECK_CONTROLS } from "../src/core/lyriaDeckScenes";
import { DEFAULT_MASTER_EFFECT_PARAMS } from "../src/audio/AudioEngine";

/// Contract between `services/musica-broadcast` and `src/core/broadcast`.
///
/// Each side has its own thorough suite, and both passed while the Rust layer
/// between them emitted `display_name` to a reader expecting `displayName`.
/// Unit tests cannot catch a seam; this runs real client data through the real
/// service and back, and asserts nothing is lost on the way.

const BROADCAST_ID_SECRET = "contract-test-secret";
// Reached through globalThis so the app's tsconfig does not need @types/node
// just to run one cross-package test.
(globalThis as { process?: { env: Record<string, string | undefined> } }).process!.env
  .BROADCAST_ID_SECRET = BROADCAST_ID_SECRET;

function memoryStore() {
  const broadcasts = new Map();
  const listeners = new Map();
  const key = (id: string, subject: string) => `${id}::${subject}`;
  return {
    async getBroadcast(id: string) { return broadcasts.get(id); },
    async putBroadcast(id: string, doc: unknown) { broadcasts.set(id, doc); },
    async listBroadcasts(limit: number) {
      return [...broadcasts.entries()].map(([id, doc]) => ({ id, doc })).slice(0, limit);
    },
    async getListener(id: string, subject: string) { return listeners.get(key(id, subject)); },
    async putListener(id: string, subject: string, doc: unknown) { listeners.set(key(id, subject), doc); },
    async deleteListener(id: string, subject: string) { listeners.delete(key(id, subject)); },
    async countListeners(id: string, since: number) {
      return [...listeners.entries()].filter(([k, d]) => k.startsWith(`${id}::`) && (d as { seenAt: number }).seenAt >= since).length;
    },
  };
}

interface CallOptions {
  query?: Record<string, unknown>;
  body?: unknown;
  nowSeconds?: number;
}

const call = (store: unknown, subject: string, method: string, path: string, extra: CallOptions = {}) =>
  handle({ store, subject, method, path, query: extra.query ?? {}, body: extra.body, nowSeconds: extra.nowSeconds ?? 1_000 });

/// A snapshot built exactly the way the app builds one when going live.
function clientSnapshot(rev: number) {
  return captureBroadcastSnapshot({
    rev,
    bpm: 128,
    styleId: "techno",
    styleLabel: "Warehouse Techno",
    deckEnabled: { main: true, sequence: true, vocal: false },
    deckControls: { ...DEFAULT_LYRIA_DECK_CONTROLS },
    lyriaConfig: {
      bpm: 128, guidance: 4.25, density: 0.73, brightness: 0.61, temperature: 1.15, topK: 37,
      scale: "D_MAJOR_B_MINOR", muteBass: false, muteDrums: false, onlyBassAndDrums: false,
      musicGenerationMode: "DIVERSITY",
    },
    lyriaPrompts: [{ text: "warehouse techno, metallic stabs", weight: 1.3 }],
    visualScene: "lasergrid",
    visualIntensity: 0.82,
    visualColor: { palette: "neon", hue: 0.91, saturation: 0.72, contrast: 0.55, diversity: 0.44 },
    masterEffects: { flanger: 0.1, phaser: 0, drive: 0, crush: 0, sweep: 0.2, reverb: 0.35, echo: 0 },
    masterEffectParams: { ...DEFAULT_MASTER_EFFECT_PARAMS },
  });
}

describe("service <-> client contract", () => {
  it("round-trips a published snapshot back to the follower byte-for-byte", async () => {
    const store = memoryStore();
    const published = clientSnapshot(1);

    const publishResult = await call(store, "dj", "POST", "/broadcast", {
      body: { display_name: "DJ Nova", snapshot: published },
      nowSeconds: 1_000,
    });
    const id = publishResult.body.id;

    const followResult = await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_001 });
    const update = normalizeBroadcastFollowUpdate(followResult.body);
    expect(update).toBeDefined();
    expect(update?.snapshot).toBeDefined();

    // Everything the follower needs must survive the round trip. The service
    // owns `rev`, so compare the rest field-for-field.
    const { rev: _sent, ...sentContent } = published;
    const { rev: _got, ...gotContent } = update!.snapshot!;
    expect(gotContent).toEqual(sentContent);

    // Spot-check the generative fields specifically: these are what make a
    // follow reproduce the sound rather than just the faders (ADR-182).
    expect(update?.snapshot?.lyria.config.density).toBe(0.73);
    expect(update?.snapshot?.lyria.config.scale).toBe("D_MAJOR_B_MINOR");
    expect(update?.snapshot?.lyria.config.musicGenerationMode).toBe("DIVERSITY");
    expect(update?.snapshot?.lyria.prompts).toEqual([{ text: "warehouse techno, metallic stabs", weight: 1.3 }]);
  });

  it("assigns a server-side revision the client accepts and orders by", async () => {
    const store = memoryStore();
    const id = (await call(store, "dj", "POST", "/broadcast", {
      body: { display_name: "DJ Nova", snapshot: clientSnapshot(1) }, nowSeconds: 1_000,
    })).body.id;

    const first = normalizeBroadcastFollowUpdate((await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_001 })).body);
    await call(store, "dj", "POST", "/broadcast", { body: { snapshot: clientSnapshot(1) }, nowSeconds: 1_010 });
    const second = normalizeBroadcastFollowUpdate((await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_011 })).body);

    expect(second!.rev).toBeGreaterThan(first!.rev);
    expect(second!.snapshot!.rev).toBe(second!.rev);
  });

  it("produces a directory the client parses without dropping a listing", async () => {
    const store = memoryStore();
    await call(store, "dj", "POST", "/broadcast", {
      body: { display_name: "DJ Nova", snapshot: clientSnapshot(1) }, nowSeconds: 1_000,
    });

    const listed = await call(store, "browser", "GET", "/broadcasts", { nowSeconds: 1_012 });
    const directory = normalizeBroadcastDirectory(listed.body);

    // A field-name mismatch here would silently empty the leaderboard rather
    // than fail loudly, so assert the parsed values, not just the count.
    expect(directory.all).toHaveLength(1);
    expect(directory.top).toHaveLength(1);
    expect(directory.all[0]).toEqual({
      id: broadcastIdFor("dj", BROADCAST_ID_SECRET),
      displayName: "DJ Nova",
      live: true,
      listeners: 0,
      updatedAgoSeconds: 12,
      styleLabel: "Warehouse Techno",
      bpm: 128,
    });
  });

  it("survives the unchanged-body optimisation without the client losing state", async () => {
    const store = memoryStore();
    const id = (await call(store, "dj", "POST", "/broadcast", {
      body: { snapshot: clientSnapshot(1) }, nowSeconds: 1_000,
    })).body.id;

    const first = normalizeBroadcastFollowUpdate((await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_001 })).body);
    const repeat = normalizeBroadcastFollowUpdate((await call(store, "fan", "POST", `/listen/${id}`, {
      query: { since: String(first!.rev) }, nowSeconds: 1_003,
    })).body);

    // No body, but still a valid update the follower can act on.
    expect(repeat).toBeDefined();
    expect(repeat?.snapshot).toBeUndefined();
    expect(repeat?.rev).toBe(first!.rev);
    expect(repeat?.live).toBe(true);
  });

  it("keeps an offline broadcaster's last snapshot readable by the client", async () => {
    const store = memoryStore();
    const id = (await call(store, "dj", "POST", "/broadcast", {
      body: { snapshot: clientSnapshot(1) }, nowSeconds: 1_000,
    })).body.id;
    await call(store, "dj", "DELETE", "/broadcast", { nowSeconds: 1_100 });

    const update = normalizeBroadcastFollowUpdate((await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_200 })).body);
    expect(update?.live).toBe(false);
    expect(update?.updatedAgoSeconds).toBe(200);
    expect(normalizeBroadcastSnapshot(update?.snapshot)).toBeDefined();
  });
});
