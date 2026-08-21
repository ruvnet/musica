import { createHmac } from "node:crypto";

/// Routing and policy for the settings broadcast plane (ADR-182).
///
/// Deliberately free of Firestore: the store is an interface, so this whole
/// file is testable in memory and the storage choice can change without
/// touching policy.

/// Presence older than this no longer counts toward the leaderboard. Queries
/// filter on it directly, so counts are correct immediately rather than
/// whenever a TTL sweeper happens to run.
export const LISTENER_TTL_SECONDS = 15;
/// A heartbeat write is skipped while the presence doc is still this fresh, so
/// a 2s client poll does not mean a 2s write cadence.
export const HEARTBEAT_WRITE_INTERVAL_SECONDS = 10;
/// Recompute the denormalized listener count at most this often per broadcast,
/// so a popular broadcaster does not become a write hotspot.
export const LISTENER_COUNT_REFRESH_SECONDS = 5;
/// Floor on the interval between a broadcaster's publishes.
export const PUBLISH_MIN_INTERVAL_SECONDS = 1;
/// A broadcast stops appearing in the directory once it has been offline this
/// long — the snapshot is kept for anyone following by id, but the directory
/// does not accumulate zombies.
export const DIRECTORY_OFFLINE_CUTOFF_SECONDS = 3600;

export const MAX_SNAPSHOT_BYTES = 16 * 1024;
export const MAX_DISPLAY_NAME_LENGTH = 32;
export const DIRECTORY_PAGE_SIZE = 50;
export const LEADERBOARD_SIZE = 5;

const UNSAFE_TEXT = /[\u0000-\u001f\u007f-\u009f\u200e-\u200f\u202a-\u202e\u2066-\u2069]/g;

class HttpError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function sanitizeText(value, maxLength) {
  if (typeof value !== "string") return "";
  return value.replace(UNSAFE_TEXT, "").trim().slice(0, maxLength);
}

/// The public id is an opaque HMAC of the OAuth subject, never the subject
/// itself: a `sub` in client-visible URLs and logs is a permanent identifier
/// that cannot be rotated or hidden (ADR-182). Deterministic, so a broadcaster
/// resumes their own broadcast across restarts.
export function broadcastIdFor(subject, secret = process.env.BROADCAST_ID_SECRET) {
  if (!secret) throw new HttpError(500, "Broadcast service is misconfigured");
  return createHmac("sha256", secret).update(subject).digest("base64url").slice(0, 22);
}

function listingFrom(id, doc, nowSeconds) {
  return {
    id,
    displayName: doc.displayName ?? "",
    live: doc.live === true,
    listeners: doc.listenerCount ?? 0,
    updatedAgoSeconds: Math.max(0, Math.round(nowSeconds - (doc.updatedAt ?? nowSeconds))),
    styleLabel: doc.snapshot?.performance?.styleLabel ?? "",
    bpm: doc.snapshot?.performance?.bpm ?? 120,
  };
}

async function refreshListenerCount(store, id, doc, nowSeconds) {
  if (nowSeconds - (doc.listenerCountAt ?? 0) < LISTENER_COUNT_REFRESH_SECONDS) {
    return doc.listenerCount ?? 0;
  }
  const listeners = await store.countListeners(id, nowSeconds - LISTENER_TTL_SECONDS);
  await store.putBroadcast(id, { ...doc, listenerCount: listeners, listenerCountAt: nowSeconds });
  return listeners;
}

async function publish(store, subject, body, nowSeconds) {
  const snapshot = body?.snapshot;
  if (typeof snapshot !== "object" || snapshot === null) {
    throw new HttpError(400, "Missing snapshot");
  }
  if (Buffer.byteLength(JSON.stringify(snapshot), "utf8") > MAX_SNAPSHOT_BYTES) {
    throw new HttpError(413, "Snapshot is too large");
  }

  const id = broadcastIdFor(subject);
  const existing = (await store.getBroadcast(id)) ?? {};
  if (existing.updatedAt && nowSeconds - existing.updatedAt < PUBLISH_MIN_INTERVAL_SECONDS) {
    throw new HttpError(429, "Publishing too fast");
  }

  const displayName = sanitizeText(body?.display_name ?? body?.displayName, MAX_DISPLAY_NAME_LENGTH)
    || existing.displayName
    || "Unnamed";
  // `rev` is assigned server-side and only ever increases, so a client cannot
  // replay or rewind a follower by choosing its own revision.
  const rev = (existing.rev ?? 0) + 1;
  const doc = {
    ...existing,
    displayName,
    snapshot: { ...snapshot, rev },
    rev,
    live: true,
    updatedAt: nowSeconds,
  };
  await store.putBroadcast(id, doc);
  return {
    status: 200,
    body: { id, display_name: displayName, live: true, rev, listeners: doc.listenerCount ?? 0 },
  };
}

async function stop(store, subject, nowSeconds) {
  const id = broadcastIdFor(subject);
  const existing = await store.getBroadcast(id);
  // Going offline keeps the snapshot: a follower arriving after the broadcaster
  // quits still lands on where the set ended.
  if (existing) {
    await store.putBroadcast(id, { ...existing, live: false, offlineAt: nowSeconds });
  }
  return { status: 200, body: { id, live: false } };
}

async function directory(store, nowSeconds) {
  const docs = await store.listBroadcasts(DIRECTORY_PAGE_SIZE);
  const listings = docs
    .filter(({ doc }) => doc.live === true
      || nowSeconds - (doc.offlineAt ?? doc.updatedAt ?? 0) < DIRECTORY_OFFLINE_CUTOFF_SECONDS)
    .map(({ id, doc }) => listingFrom(id, doc, nowSeconds));
  // Live broadcasters outrank offline ones at equal listener counts, so the
  // leaderboard reflects who is playing now.
  const ranked = [...listings].sort((a, b) =>
    (b.listeners - a.listeners) || (Number(b.live) - Number(a.live)) || (a.updatedAgoSeconds - b.updatedAgoSeconds));
  return { status: 200, body: { top: ranked.slice(0, LEADERBOARD_SIZE), all: listings } };
}

async function listen(store, subject, id, since, nowSeconds) {
  const doc = await store.getBroadcast(id);
  if (!doc) throw new HttpError(404, "No such broadcast");

  // Self-listening must never inflate a leaderboard.
  if (broadcastIdFor(subject) !== id) {
    const presence = await store.getListener(id, subject);
    if (!presence || nowSeconds - presence.seenAt >= HEARTBEAT_WRITE_INTERVAL_SECONDS) {
      // One document per (broadcast, listener): an upsert, never an append, so
      // reconnecting cannot manufacture listeners.
      await store.putListener(id, subject, {
        seenAt: nowSeconds,
        expiresAt: nowSeconds + LISTENER_TTL_SECONDS,
      });
    }
  }

  const listeners = await refreshListenerCount(store, id, doc, nowSeconds);
  const unchanged = typeof since === "number" && Number.isFinite(since) && since === doc.rev;
  return {
    status: 200,
    body: {
      rev: doc.rev ?? 0,
      // An unchanged set costs a heartbeat, not a payload.
      ...(unchanged ? {} : { snapshot: doc.snapshot }),
      listeners,
      live: doc.live === true,
      updatedAgoSeconds: Math.max(0, Math.round(nowSeconds - (doc.updatedAt ?? nowSeconds))),
    },
  };
}

async function leave(store, subject, id) {
  await store.deleteListener(id, subject);
  return { status: 200, body: { ok: true } };
}

const ID_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

export function parseRoute(method, path) {
  const trimmed = `/${String(path ?? "").replace(/^\/+|\/+$/g, "")}`;
  if (trimmed === "/broadcast") {
    if (method === "POST") return { route: "publish" };
    if (method === "DELETE") return { route: "stop" };
    return { route: "method-not-allowed" };
  }
  if (trimmed === "/broadcasts") {
    return method === "GET" ? { route: "directory" } : { route: "method-not-allowed" };
  }
  const listenMatch = /^\/listen\/([^/]+)$/.exec(trimmed);
  if (listenMatch) {
    const id = decodeURIComponent(listenMatch[1]);
    if (!ID_PATTERN.test(id)) return { route: "bad-id" };
    if (method === "POST") return { route: "listen", id };
    if (method === "DELETE") return { route: "leave", id };
    return { route: "method-not-allowed" };
  }
  return { route: "not-found" };
}

/// Executes a verified request. `subject` has already passed the JWT gate.
export async function handle({ store, subject, method, path, query, body, nowSeconds }) {
  const parsed = parseRoute(method, path);
  switch (parsed.route) {
    case "publish":
      return publish(store, subject, body, nowSeconds);
    case "stop":
      return stop(store, subject, nowSeconds);
    case "directory":
      return directory(store, nowSeconds);
    case "listen": {
      const raw = query?.since;
      const since = raw === undefined || raw === null || raw === "" ? undefined : Number(raw);
      return listen(store, subject, parsed.id, since, nowSeconds);
    }
    case "leave":
      return leave(store, subject, parsed.id);
    case "bad-id":
      throw new HttpError(400, "Invalid broadcast id");
    case "method-not-allowed":
      throw new HttpError(405, "Method not allowed");
    default:
      throw new HttpError(404, "Not found");
  }
}

export { HttpError };
