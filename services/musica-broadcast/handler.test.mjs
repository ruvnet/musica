import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";
import {
  DIRECTORY_OFFLINE_CUTOFF_SECONDS,
  HEARTBEAT_WRITE_INTERVAL_SECONDS,
  LISTENER_COUNT_REFRESH_SECONDS,
  LISTENER_TTL_SECONDS,
  MAX_SNAPSHOT_BYTES,
  broadcastIdFor,
  handle,
  parseRoute,
} from "./handler.mjs";

const SECRET = "test-secret";
process.env.BROADCAST_ID_SECRET = SECRET;

/// In-memory store with the same interface as the Firestore one, plus counters
/// so the tests can assert on write volume — the whole point of decoupling poll
/// rate from write rate (ADR-182).
function memoryStore() {
  const broadcasts = new Map();
  const listeners = new Map();
  const writes = { broadcast: 0, listener: 0 };
  const key = (id, subject) => `${id}::${subject}`;
  return {
    writes,
    raw: { broadcasts, listeners },
    async getBroadcast(id) {
      return broadcasts.get(id);
    },
    async putBroadcast(id, doc) {
      writes.broadcast += 1;
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
      writes.listener += 1;
      listeners.set(key(id, subject), doc);
    },
    async deleteListener(id, subject) {
      listeners.delete(key(id, subject));
    },
    async countListeners(id, seenSinceSeconds) {
      return [...listeners.entries()]
        .filter(([entryKey, doc]) => entryKey.startsWith(`${id}::`) && doc.seenAt >= seenSinceSeconds)
        .length;
    },
  };
}

const SNAPSHOT = {
  v: 1,
  performance: { bpm: 128, styleLabel: "Warehouse Techno" },
  lyria: { config: { density: 0.7 }, prompts: [{ text: "techno", weight: 1 }] },
};

const call = (store, subject, method, path, extra = {}) => handle({
  store,
  subject,
  method,
  path,
  query: extra.query ?? {},
  body: extra.body,
  nowSeconds: extra.nowSeconds ?? 1_000,
});

describe("routing", () => {
  it("maps the documented surface", () => {
    assert.deepEqual(parseRoute("POST", "/broadcast"), { route: "publish" });
    assert.deepEqual(parseRoute("DELETE", "/broadcast"), { route: "stop" });
    assert.deepEqual(parseRoute("GET", "/broadcasts"), { route: "directory" });
    assert.deepEqual(parseRoute("POST", "/listen/abc123"), { route: "listen", id: "abc123" });
    assert.deepEqual(parseRoute("DELETE", "/listen/abc123"), { route: "leave", id: "abc123" });
    // Cloud Functions may present the path with or without slashes.
    assert.deepEqual(parseRoute("GET", "broadcasts/"), { route: "directory" });
  });

  it("refuses wrong methods, unknown paths, and structured ids", () => {
    assert.equal(parseRoute("GET", "/broadcast").route, "method-not-allowed");
    assert.equal(parseRoute("POST", "/broadcasts").route, "method-not-allowed");
    assert.equal(parseRoute("GET", "/anything-else").route, "not-found");
    assert.equal(parseRoute("POST", "/listen/..%2F..%2Fadmin").route, "bad-id");
    assert.equal(parseRoute("POST", "/listen/has spaces").route, "bad-id");
  });
});

describe("public identity", () => {
  it("never exposes the OAuth subject, and is stable across restarts", () => {
    const id = broadcastIdFor("auth0|user-12345", SECRET);
    assert.ok(!id.includes("user-12345"));
    assert.match(id, /^[A-Za-z0-9_-]{22}$/);
    assert.equal(id, broadcastIdFor("auth0|user-12345", SECRET));
    assert.notEqual(id, broadcastIdFor("auth0|someone-else", SECRET));
  });

  it("fails closed when the id secret is missing", () => {
    // Passing `undefined` would fall through to the env default, which is the
    // production path; an unset secret is what must refuse to mint an id.
    assert.throws(() => broadcastIdFor("someone", ""), /misconfigured/);
  });
});

describe("publishing", () => {
  let store;
  beforeEach(() => { store = memoryStore(); });

  it("assigns the revision server-side and increments it", async () => {
    const first = await call(store, "dj", "POST", "/broadcast", {
      body: { display_name: "DJ Nova", snapshot: { ...SNAPSHOT, rev: 9_999 } },
      nowSeconds: 1_000,
    });
    assert.equal(first.body.rev, 1);
    // A client-chosen revision is overwritten, so it cannot rewind a follower.
    assert.equal(store.raw.broadcasts.get(first.body.id).snapshot.rev, 1);

    const second = await call(store, "dj", "POST", "/broadcast", {
      body: { snapshot: SNAPSHOT },
      nowSeconds: 1_010,
    });
    assert.equal(second.body.rev, 2);
  });

  it("rejects a missing or oversized snapshot", async () => {
    await assert.rejects(
      call(store, "dj", "POST", "/broadcast", { body: {} }),
      (error) => error.status === 400,
    );
    const huge = { blob: "x".repeat(MAX_SNAPSHOT_BYTES + 100) };
    await assert.rejects(
      call(store, "dj", "POST", "/broadcast", { body: { snapshot: huge } }),
      (error) => error.status === 413,
    );
  });

  it("rate-limits a broadcaster hammering publish", async () => {
    await call(store, "dj", "POST", "/broadcast", { body: { snapshot: SNAPSHOT }, nowSeconds: 1_000 });
    await assert.rejects(
      call(store, "dj", "POST", "/broadcast", { body: { snapshot: SNAPSHOT }, nowSeconds: 1_000 }),
      (error) => error.status === 429,
    );
  });

  it("sanitizes the display name and keeps the previous one when blank", async () => {
    const rtl = String.fromCharCode(0x202e);
    const published = await call(store, "dj", "POST", "/broadcast", {
      body: { display_name: `DJ${rtl} Nova${String.fromCharCode(0)}`, snapshot: SNAPSHOT },
      nowSeconds: 1_000,
    });
    assert.equal(published.body.display_name, "DJ Nova");

    const kept = await call(store, "dj", "POST", "/broadcast", {
      body: { display_name: "   ", snapshot: SNAPSHOT },
      nowSeconds: 1_010,
    });
    assert.equal(kept.body.display_name, "DJ Nova");
  });
});

describe("going offline", () => {
  it("keeps the last snapshot so a late follower still lands somewhere", async () => {
    const store = memoryStore();
    const published = await call(store, "dj", "POST", "/broadcast", {
      body: { snapshot: SNAPSHOT },
      nowSeconds: 1_000,
    });
    await call(store, "dj", "DELETE", "/broadcast", { nowSeconds: 1_100 });

    const followed = await call(store, "fan", "POST", `/listen/${published.body.id}`, { nowSeconds: 1_200 });
    assert.equal(followed.body.live, false);
    assert.equal(followed.body.snapshot.performance.styleLabel, "Warehouse Techno");
    assert.equal(followed.body.updatedAgoSeconds, 200);
  });
});

describe("following", () => {
  let store;
  let id;
  beforeEach(async () => {
    store = memoryStore();
    id = (await call(store, "dj", "POST", "/broadcast", { body: { snapshot: SNAPSHOT }, nowSeconds: 1_000 })).body.id;
  });

  it("404s an unknown broadcast", async () => {
    await assert.rejects(
      call(store, "fan", "POST", "/listen/doesnotexist", { nowSeconds: 1_000 }),
      (error) => error.status === 404,
    );
  });

  it("omits the snapshot body when the follower is already current", async () => {
    const first = await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_001 });
    assert.ok(first.body.snapshot);

    const unchanged = await call(store, "fan", "POST", `/listen/${id}`, {
      query: { since: String(first.body.rev) },
      nowSeconds: 1_002,
    });
    assert.equal(unchanged.body.snapshot, undefined);
    assert.equal(unchanged.body.rev, first.body.rev);
  });

  it("sends the body again once the set moves on", async () => {
    const first = await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_001 });
    await call(store, "dj", "POST", "/broadcast", { body: { snapshot: SNAPSHOT }, nowSeconds: 1_010 });
    const next = await call(store, "fan", "POST", `/listen/${id}`, {
      query: { since: String(first.body.rev) },
      nowSeconds: 1_011,
    });
    assert.ok(next.body.snapshot);
    assert.equal(next.body.rev, first.body.rev + 1);
  });

  it("does not write presence on every poll", async () => {
    const before = store.writes.listener;
    // A 2s client poll must not mean a 2s write cadence.
    for (let second = 0; second < HEARTBEAT_WRITE_INTERVAL_SECONDS; second += 2) {
      await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_001 + second });
    }
    assert.equal(store.writes.listener - before, 1);

    await call(store, "fan", "POST", `/listen/${id}`, {
      nowSeconds: 1_001 + HEARTBEAT_WRITE_INTERVAL_SECONDS,
    });
    assert.equal(store.writes.listener - before, 2);
  });

  it("does not count a broadcaster listening to themselves", async () => {
    await call(store, "dj", "POST", `/listen/${id}`, { nowSeconds: 1_100 });
    assert.equal(store.writes.listener, 0);
    const seen = await call(store, "dj", "POST", `/listen/${id}`, { nowSeconds: 1_200 });
    assert.equal(seen.body.listeners, 0);
  });

  it("keeps one presence row per subject no matter how often they reconnect", async () => {
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_001 + attempt * 2 });
    }
    // Twenty polls, one row: presence is an upsert keyed by subject, so
    // reconnecting cannot manufacture listeners.
    const rows = [...store.raw.listeners.keys()].filter((entry) => entry.startsWith(`${id}::`));
    assert.equal(rows.length, 1);

    const counted = await call(store, "other-fan", "POST", `/listen/${id}`, { nowSeconds: 1_042 });
    assert.equal(counted.body.listeners, 2);
  });

  it("stops counting a listener who went away", async () => {
    await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_000 });
    const fresh = await call(store, "other", "POST", `/listen/${id}`, {
      nowSeconds: 1_000 + LISTENER_COUNT_REFRESH_SECONDS,
    });
    assert.equal(fresh.body.listeners, 2);

    // Past the TTL the stale presence no longer counts, whether or not a
    // sweeper has run.
    const later = await call(store, "other", "POST", `/listen/${id}`, {
      nowSeconds: 1_000 + LISTENER_TTL_SECONDS + LISTENER_COUNT_REFRESH_SECONDS + 1,
    });
    assert.equal(later.body.listeners, 1);
  });

  it("throttles the denormalized count so a popular broadcast is not a write hotspot", async () => {
    await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 2_000 });
    const before = store.writes.broadcast;
    for (let second = 1; second < LISTENER_COUNT_REFRESH_SECONDS; second += 1) {
      await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 2_000 + second });
    }
    assert.equal(store.writes.broadcast, before);
  });

  it("forgets a listener who explicitly leaves", async () => {
    await call(store, "fan", "POST", `/listen/${id}`, { nowSeconds: 1_000 });
    await call(store, "fan", "DELETE", `/listen/${id}`, { nowSeconds: 1_001 });
    const after = await call(store, "other", "POST", `/listen/${id}`, {
      nowSeconds: 1_000 + LISTENER_COUNT_REFRESH_SECONDS,
    });
    assert.equal(after.body.listeners, 1);
  });
});

describe("directory", () => {
  it("ranks by concurrent listeners and holds the leaderboard to five", async () => {
    const store = memoryStore();
    for (let index = 0; index < 8; index += 1) {
      const published = await call(store, `dj-${index}`, "POST", "/broadcast", {
        body: { display_name: `DJ ${index}`, snapshot: SNAPSHOT },
        nowSeconds: 1_000,
      });
      for (let fan = 0; fan <= index; fan += 1) {
        await call(store, `fan-${index}-${fan}`, "POST", `/listen/${published.body.id}`, {
          nowSeconds: 1_000 + fan,
        });
      }
      // Force the throttled count to refresh before ranking.
      await call(store, `fan-${index}-0`, "POST", `/listen/${published.body.id}`, {
        nowSeconds: 1_000 + LISTENER_COUNT_REFRESH_SECONDS + index,
      });
    }

    const listed = await call(store, "browser", "GET", "/broadcasts", { nowSeconds: 1_100 });
    assert.equal(listed.body.top.length, 5);
    assert.equal(listed.body.all.length, 8);
    const counts = listed.body.top.map((entry) => entry.listeners);
    assert.deepEqual(counts, [...counts].sort((a, b) => b - a));
    assert.equal(counts[0], 8);
  });

  it("surfaces the style and tempo without exposing the snapshot", async () => {
    const store = memoryStore();
    await call(store, "dj", "POST", "/broadcast", {
      body: { display_name: "DJ Nova", snapshot: SNAPSHOT },
      nowSeconds: 1_000,
    });
    const listed = await call(store, "browser", "GET", "/broadcasts", { nowSeconds: 1_030 });
    assert.deepEqual(listed.body.all[0], {
      id: broadcastIdFor("dj", SECRET),
      displayName: "DJ Nova",
      live: true,
      listeners: 0,
      updatedAgoSeconds: 30,
      styleLabel: "Warehouse Techno",
      bpm: 128,
    });
    assert.equal(listed.body.all[0].snapshot, undefined);
  });

  it("drops long-offline broadcasts so the directory does not accumulate zombies", async () => {
    const store = memoryStore();
    await call(store, "dj", "POST", "/broadcast", { body: { snapshot: SNAPSHOT }, nowSeconds: 1_000 });
    await call(store, "dj", "DELETE", "/broadcast", { nowSeconds: 1_100 });

    const soon = await call(store, "browser", "GET", "/broadcasts", { nowSeconds: 1_200 });
    assert.equal(soon.body.all.length, 1);
    assert.equal(soon.body.all[0].live, false);

    const later = await call(store, "browser", "GET", "/broadcasts", {
      nowSeconds: 1_100 + DIRECTORY_OFFLINE_CUTOFF_SECONDS + 1,
    });
    assert.equal(later.body.all.length, 0);
  });
});
