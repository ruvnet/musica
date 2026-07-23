# ADR-182: Settings broadcast, live follow, and a listener leaderboard

- Status: Accepted
- Date: 2026-07-22
- Related: ADR-179 (Cognitum-gated credential broker), ADR-175 (Cognitum OAuth + Meta-LLM), ADR-170 (Lyria RealTime provider), ADR-172 (Lyria RealTime multistream decks), ADR-166 (desktop threat model), ADR-174 (Restream native live output — a *different* feature that also says "broadcast")

## Context

Musica is a single-operator instrument. Two people running it in different rooms have no way to share a set: the only export paths are a rendered capture (ADR-165) or an encoded RTMP stream (ADR-174), both of which ship *audio*, not the performance that produced it.

Three facts make a much lighter kind of sharing possible:

- Musica already mirrors its live control surface across windows. `DjControlState` (`src/core/djControls.ts`) is recomputed on every relevant state change and emitted to the secondary DJ-control windows over Tauri events (`App.tsx`). A cross-machine share is the same snapshot over a network hop instead of an IPC hop.
- Every user already authenticates through **Cognitum One**, and ADR-179 established the pattern for an org-native service that verifies those ES256 tokens against a live JWKS, fail-closed, with the bearer never leaving Rust.
- Everyone running Musica already has a synthesis engine and a Lyria session. So "hear what they're doing" does not require moving audio — it requires moving the *parameters*, a few hundred bytes, and letting the receiver re-synthesize.

The wanted behavior: a user opts into broadcasting; other signed-in users see a directory with a top-5 leaderboard of the most-listened broadcasters; clicking one adopts that broadcaster's settings and keeps following them live; a broadcaster who quits leaves their last snapshot behind so followers still land on where they ended.

### What this is not

It is **not audio streaming**, and the product language must not imply it is. Lyria RealTime is generative and session-stateful: two sessions given identical configuration produce *the same musical intent*, not the same notes, and there is no shared clock, so followers are not beat-aligned with the broadcaster. This feature reproduces a **set's direction**, not its waveform. ADR-174's Restream path remains the answer for "let people hear exactly what I hear", and keeps the word RESTREAM in the UI so the two never read as the same button.

## Decision

Add a **settings broadcast plane**: an org-native Cognitum service holding live snapshots plus listener presence, and a Musica client that can publish to it, browse it, and follow a broadcaster with a hardened apply pipeline.

### Snapshot contract (`BroadcastSnapshot`, v1)

The single most important design question is what a snapshot must contain to actually reproduce a set. `DjControlState` alone is **not sufficient** — this was the central finding of design review. The generative parameters that determine what Lyria emits (`density`, `brightness`, `guidance`, `temperature`, `topK`, `scale`, the bass/drum mute flags) and the weighted prompts themselves live in `lyriaRealtimeConfig` / `lyriaPrompts`, are directly exposed as sliders in the AUDIO · LYRIA panel, and appear nowhere in `DjControlState`. Broadcasting the control surface alone would ship a remote fader panel that does not reproduce the sound.

So a snapshot carries:

| Group | Fields | Why |
| --- | --- | --- |
| `performance` | bpm, styleId, styleLabel, deckEnabled, deckControls | The control surface — which decks, at what levels, pitch, nudge |
| `lyria` | full `LyriaRealtimeConfig` + resolved `weightedPrompts` | The generative direction. Without this the follow is cosmetic |
| `visual` | scene, intensity, color controls | The look |
| `fx` | master effect levels + their params | The processing |

Envelope fields: `v: 1` (schema version, unknown-field-tolerant) and `rev` (monotonic, broadcaster-assigned).

**Prompts are sent resolved, not by reference.** A `styleId` is a pointer into a namespace the receiver may not share — a broadcaster's custom style (`customLyriaStyles`) would resolve to nothing on a listener's machine. Sending the prompts and config that the style *resolves to* makes custom styles work across machines without shipping the broadcaster's style library, staying within a "performance + FX" scope rather than a whole-workspace sync.

**Deliberately excluded**, each for a reason:

- `masterVolume` — a follower's monitor level is theirs. A remote value here is a way to blast someone's speakers.
- `playing` as a command — see consent, below.
- Saved **deck-scene presets** — these are the broadcaster's own preset library, persisted locally by the follower too, so adopting them would overwrite the follower's saved work. Deck enablement and controls already carry whatever scene the broadcaster has active, which is what following actually needs.
- Visual plugins, set arcs, the custom-style library, capture history — whole-workspace sync is a different feature with a different risk profile (AI-generated plugin code moving between accounts). Out of scope here.

### Service (`services/musica-broadcast` → Cloud Function `musicaBroadcast`)

A 2nd-gen HTTP Cloud Function over Firestore, deployed alongside the ADR-179 broker in the Cognitum project, doing its own JWKS verification for exactly the reason ADR-179 documents: the api gateway's `verifyAuthToken` is Firebase-based and rejects Musica's Cognitum token. Every route requires a signature-verified, unexpired, `inference`-scoped bearer. Fail-closed.

| Route | Behavior |
| --- | --- |
| `POST /broadcast` | Upsert `broadcasts/{broadcastId}` with snapshot, displayName, `rev`, `live: true`, `updatedAt` |
| `DELETE /broadcast` | `live: false`, snapshot retained — "the latest they had before they closed" |
| `GET /broadcasts` | Directory: one indexed query ordered by denormalized `listenerCount`, returning top-5 plus a capped page |
| `POST /listen/{id}` | Registers listener presence **and** returns the current snapshot |
| `DELETE /listen/{id}` | Explicit leave (courtesy; correctness rests on expiry, not on this) |

`broadcastId` is **opaque and derived** — `base64url(HMAC-SHA256(secret, sub))`, truncated. Deterministic, so a user resumes their own broadcast across restarts; opaque, so the OAuth subject never becomes a public handle baked into clients, URLs, and logs.

**Poll rate is decoupled from write rate.** `POST /listen/{id}` merges heartbeat and fetch into one authenticated round-trip so the client stays simple, but internally the heartbeat write is skipped unless the listener's presence doc is older than ~10s, and the response omits the snapshot body when the client's `?since=<rev>` already matches. A 2s product latency target therefore does not imply a 2s write cadence. Symmetrically, the broadcaster publishes only when a content hash of the normalized snapshot changes, rate-limited — local IPC frequency must not become cloud write frequency.

`listenerCount` is denormalized onto the broadcast document and refreshed at most every few seconds, so the directory read is one query rather than a per-broadcaster count fan-out. Presence documents carry `expiresAt`; queries filter on it for correctness *now*, and a Firestore TTL policy reclaims storage eventually. Presence is keyed one document per `(broadcastId, listenerSub)` so a listener cannot inflate a count by reconnecting, and self-listening does not count.

### Client

- **Rust** (`broadcast_provider.rs`): commands `broadcast_status`, `broadcast_publish`, `broadcast_stop`, `broadcast_list`, `broadcast_listen`, `broadcast_leave`. All calls carry the bearer from `fresh_access_token`; the token never reaches React. Rust treats the snapshot as opaque JSON with a hard byte cap — the schema is owned by TypeScript, so Rust does not duplicate a large evolving type.
- **TypeScript** (`src/core/broadcast.ts`): the snapshot type, capture from live state, and — most importantly — `normalizeBroadcastSnapshot`, the trust boundary.
- **UI**: a BROADCAST panel with a go-live toggle and display name, the leaderboard, and follow/unfollow with a visible offline age.

### Safety posture

An incoming snapshot is **untrusted input from another user that drives the local audio and visual engines**. It is treated like any other hostile input:

- **Clamp on ingest.** bpm 60–200 (`MIN_BPM`/`MAX_BPM`), unit values 0–1, pitch ±7 semitones, beat nudge ±250ms, scene/palette/scale/mode IDs whitelisted against known values with unknown values ignored rather than defaulted, arrays and strings length-capped, whole payload byte-capped.
- **Slew, not just clamp.** Photosensitivity is a trajectory problem: `visualIntensity` flipping 0↔1 satisfies every range check and still pulses. Two bounds apply, and which does the work matters. The **poll interval** is primary — a follower's visuals cannot change faster than 0.5 Hz, well below the range photosensitivity guidance addresses. The **slew budget** bounds the magnitude of each of those steps so a full-range swing spans more than one poll rather than snapping; it is sized to actually engage at the default poll rate, which a larger budget would not (a test asserts exactly this, because a bound that never fires is worse than none — it reads as protection).
- **Consent to play.** A snapshot never starts audio. Following applies settings; the listener starts their own transport. Deck *enablement* is applied only when a Lyria session is already running — enabling a deck opens a metered session, so a follow must not do it unasked. Each follower runs their own metered session, making remote auto-play both a surprise-audio problem and a billing one; a viral broadcaster would otherwise create a thundering herd of inference.
- **Prompt text is sanitized.** Weighted prompts are free text that would reach the *listener's* Lyria credential. Count-capped, length-capped, control characters stripped, and shown in the follow UI so a follower can see what they adopted.
- **Ordering.** `rev` is monotonic; a snapshot with `rev` not greater than the applied one is dropped, so retries and reordering cannot resurrect stale state. Snapshots apply atomically, never field-by-field into live engine state.
- **No writes to the user's own saved work.** Deck-scene presets are not on the wire at all, and a locked master effect (`fxLocks`) outranks the incoming value — a follower's explicit decisions survive a follow.
- **Display names** are length-capped, control/RTL-stripped, and rate-limited on rename. Listener counts are treated as untrusted social proof, never as anything security-sensitive.

## Consequences

- Sharing a set becomes a few hundred bytes and no new streaming infrastructure, reusing the ADR-179 auth shape wholesale.
- One new operational surface (a service that stores user-authored snapshots and presence). Its safety rests on the same JWT gate as the broker, plus the client-side clamp/slew pipeline, which is where the tests concentrate.
- The follow experience is honest but bounded: same direction, not the same audio, not beat-aligned, a few seconds behind. UI language says *follow their settings*, never *listen to their set*. If true listen-along is ever wanted, that is ADR-174's Restream path or a new decision, not an extension of this one.
- Snapshot fidelity is now coupled to the Lyria config surface: a future slider added to `lyriaRealtimeConfig` that is not added to the snapshot silently degrades follow quality. Schema version + a normalization test suite make that a visible failure rather than a quiet one.
- Polling was chosen over Firestore `onSnapshot` / WebSocket push despite push being cheaper in writes, because push requires database credentials in the client and would break the "all network egress goes through Rust with the bearer" boundary this and ADR-179 depend on. The write-amplification concern is instead addressed by decoupling write cadence from poll cadence. If concurrency ever makes this the wrong trade, the fix is a push channel *terminated in Rust*, not a client-side database SDK.

## Validation

- Client normalization suite (`tests/broadcast.test.ts`, 25 tests): hostile snapshots — out-of-range bpm, non-unit levels, unknown scene/palette/scale/mode ids, oversized arrays and strings, control characters and bidi overrides in prompts and names, mutually exclusive mute flags, missing and unknown fields, stale and equal `rev` — each clamped, ignored, or rejected without throwing. Also asserts the slew bound, and that deck-scene presets and master volume never appear on the wire.
- Service (`services/musica-broadcast`, `npm test`, 32 tests): the gate accepts a valid in-scope token and rejects missing, garbage, forged-signature, expired, wrong-issuer, scope-less, and subject-less tokens against a real ES256 keypair. Policy tests assert server-assigned revisions, publish rate limiting, presence upsert-per-subject, self-listen exclusion, leaderboard ranking and truncation, offline snapshot retention, and directory zombie cutoff — and assert on **write counts** directly, so a regression in the poll-rate/write-rate decoupling fails a test rather than a bill.
- Rust (`cargo test -p musica-vj`, 53 tests incl. 4 new): broadcast id path-escape rejection, char-boundary name truncation, and https-only enforcement on the service base URL. `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` clean.
- `npm run typecheck` / `test:run` (120 tests) / `build` / `verify:no-secrets` green.

## Next steps

1. Deploy `musicaBroadcast` to the Cognitum project with `BROADCAST_ID_SECRET` in Secret Manager and a Firestore TTL policy on presence `expiresAt`; bake the URL in as the client default with a `MUSICA_BROADCAST_URL` override, exactly as ADR-179 does for the broker.
2. End-to-end interactive test across two machines: broadcast, appear in the directory, follow, observe settings adopt and track changes, quit the broadcaster and confirm the last snapshot persists with an offline age.
3. Revisit presence storage if concurrency grows — high-churn small writes and durable snapshots are different access patterns, and splitting them (presence in Memorystore/RTDB, snapshots in Firestore) is the natural next move.
4. Consider banded listener counts ("10+", "50+") if leaderboard gaming becomes visible.
