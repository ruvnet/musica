# musica-broadcast

The settings broadcast plane for Musica — see [ADR-182](../../docs/adr/ADR-182-settings-broadcast-follow-and-listener-leaderboard.md).

A broadcaster publishes a few hundred bytes describing *how* their set is being played; followers adopt those settings and re-synthesize locally. **No audio moves through this service.** For "let people hear exactly what I hear", that is the Restream path (ADR-174).

## Auth

Every route requires `Authorization: Bearer <cognitum-jwt>`, verified as ES256 against the live JWKS at `https://auth.cognitum.one/.well-known/jwks.json`, with issuer, expiry, and the `inference` scope all enforced. Fail-closed: no route does anything without a verified token. This mirrors the ADR-179 Lyria broker, and for the same reason — the api gateway's Firebase-based `verifyAuthToken` would reject Musica's Cognitum token, so this function verifies for itself.

The function is public-invokable because the JWT gate, not a network ACL, is the boundary.

## API

| Route | Purpose |
| --- | --- |
| `POST /broadcast` | Publish a snapshot. Body `{ display_name, snapshot }`. `rev` is assigned server-side and only increases. |
| `DELETE /broadcast` | Go offline. The snapshot is **kept**, so a follower arriving later still lands on where the set ended. |
| `GET /broadcasts` | Directory: `{ top, all }`, where `top` is the 5 broadcasters with the most concurrent listeners. |
| `POST /listen/:id?since=<rev>` | Renew presence **and** fetch state in one round-trip. The `snapshot` body is omitted when `since` already matches `rev`. |
| `DELETE /listen/:id` | Stop listening. A courtesy — correctness rests on presence expiry, not on this call. |

Public ids are opaque: `base64url(HMAC-SHA256(BROADCAST_ID_SECRET, sub))`, truncated to 22 chars. Deterministic, so a broadcaster resumes their own broadcast across restarts, but the OAuth subject never becomes a public handle.

## Why poll rate ≠ write rate

Clients poll at ~2s for responsiveness, but that is a latency target, not a storage access pattern:

- the presence write is **skipped** while the listener's document is under `HEARTBEAT_WRITE_INTERVAL_SECONDS` old;
- the response **omits the snapshot** when the client's `since` matches the current `rev`;
- `listenerCount` is denormalized onto the broadcast document and recomputed at most every `LISTENER_COUNT_REFRESH_SECONDS`, so the directory is one indexed query rather than a per-broadcaster count fan-out, and a popular broadcaster does not become a write hotspot;
- broadcasters publish only when a content hash changes (enforced client-side) and are rate-limited server-side.

`handler.test.mjs` asserts on write counts directly, so a regression here fails a test rather than a bill.

## Data model

```
musica_broadcasts/{broadcastId}
  displayName, snapshot, rev, live, updatedAt, offlineAt,
  listenerCount, listenerCountAt
  listeners/{listenerSub}          # one document per listener — upsert, never append
    seenAt, expiresAt, expiresAtTimestamp
```

Presence counting filters on `seenAt` rather than relying on the TTL sweeper, because TTL deletion can lag by hours. The TTL policy is storage hygiene, not correctness.

## Develop

```bash
npm install
npm test            # 32 tests: the JWT gate against a real ES256 keypair, and
                    # routing/policy against an in-memory store
npm start           # functions-framework on :8080
```

`handler.mjs` is deliberately free of Firestore — the store is an interface — so policy is testable in memory and the storage choice can change without touching it.

## Deploy

```bash
npm run deploy      # see deploy.sh for the secret, TTL, and index steps
```

Point Musica at a non-default deployment with `MUSICA_BROADCAST_URL`.
