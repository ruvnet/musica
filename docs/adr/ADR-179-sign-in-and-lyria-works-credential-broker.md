# ADR-179: "Sign in and Lyria works" — a Cognitum-gated credential broker

- Status: Accepted
- Date: 2026-07-20
- Related: ADR-178 (metered Lyria via the Cognitum proxy plane), ADR-175 (Cognitum OAuth + Meta-LLM), ADR-170 (Lyria RealTime provider)

## Context

Every operator must currently supply their own `GEMINI_API_KEY` for Lyria RealTime and generation. On a packaged desktop app there is no shell to export it — this was the exact "Start Session does nothing" failure on Windows — and BYO keys leak when shared and can't be metered.

ADR-178 explored routing Lyria through Cognitum's metered proxy plane, but established a hard constraint: `meta-proxy` is a **text-only** data plane (chat/messages/sponsor), with no music, WebSocket, or Lyria surface. A full metered relay for the bidirectional audio socket is real new infrastructure and is not available today.

Two facts make a lighter path possible now:

- Musica already authenticates users through **Cognitum One** (OAuth 2.1 + PKCE), and the access tokens are **ES256 JWTs** (`iss https://auth.cognitum.one`, scope `inference`) verifiable against a live, published JWKS.
- Lyria RealTime accepts a plain Gemini API key. So "signed in" can be turned into "has a key" by a tiny broker, without any streaming relay.

## Decision

Stand up a minimal, fail-closed **credential broker** that exchanges a valid Cognitum sign-in for a short-lived Lyria/Gemini key, and have Musica redeem it automatically after OAuth. BYO key remains a permanent fallback.

### Broker (`cognitum-one/api` → `functions/lyria-broker`)

The broker is an org-native Cognitum service, deployed to the Cognitum GCP
project (`cognitum-20260110`) as a 2nd-gen Cloud Function alongside the existing
`functions/seed/*`. Crucially it verifies the **Cognitum OAuth** token Musica
holds, not a Firebase ID token — the api gateway's `verifyAuthToken` uses
Firebase and would reject Musica's token, so this function does its own JWKS
verification. It:

1. Requires `Authorization: Bearer <cognitum-jwt>`.
2. Verifies the JWT against Cognitum's live JWKS (`ES256`), enforcing issuer, expiry, and the required `inference` scope.
3. On success, returns `{ "api_key": <gemini-key> }`; anything else yields 401/403 and no key.

The Gemini key lives only in the Cognitum project's Secret Manager (`GEMINI_API_KEY`), injected at deploy via `--set-secrets` — never in the client bundle. The function is public-invokable because it does its own auth; the gate, not network ACLs, is the boundary. Responses are `no-store`. Deployed URL: `https://us-central1-cognitum-20260110.cloudfunctions.net/lyriaBroker` (baked into the client as the default, overridable via `MUSICA_LYRIA_BROKER_URL`).

### Client (Musica)

- Rust `lyria_realtime_provider` gains a `runtime_key` slot with `set_runtime_key` / `effective_auth` / `effective_enabled`: a brokered key takes precedence over env config and implicitly enables the provider, so a packaged app with no env becomes live purely from signing in. Exposed as the `lyria_realtime_configure_key` command.
- Rust `cognitum_provider` gains `cognitum_lyria_credential`, which calls the broker (URL from `MUSICA_LYRIA_BROKER_URL`) with the user's current bearer and returns the key. The bearer never leaves Rust; the key is fetched over TLS.
- The React app calls `activateCognitumLyria()` whenever sign-in completes, which fetches and injects the key and refreshes provider status. All steps are best-effort no-ops when unconfigured or signed out.

### Security posture

- Tokens stay in Rust; the key is fetched into Rust and injected into the provider — neither is ever exposed to React or persisted to disk.
- The broker is fail-closed: no key without a signature-verified, in-scope, unexpired Cognitum token.
- The vended key is a shared project key today; the natural evolution is per-user short-lived keys and audio-seconds metering once Cognitum exposes them (ADR-178's endgame).

## Consequences

- Eliminates the key-provisioning failure class for signed-in users with no new streaming infrastructure — the gap ADR-178 left open.
- Introduces one new operational surface (the broker) that vends a real key; its safety rests entirely on the JWT gate, so that gate is validated (`services/lyria-broker/verify.mjs`: live-JWKS reachability + accept-valid / reject-expired / reject-wrong-issuer / reject-scope-less).
- The shared project key means metering is coarse until Cognitum issues per-user keys; documented as the next step, not shipped here.
- BYO `GEMINI_API_KEY` via `providers.env` remains supported forever for self-hosters and offline use.

## Validation

- Broker gate (`cognitum-one/api` `functions/lyria-broker`, `npm test`): 5/5 — accepts a valid in-scope token, rejects missing / expired / wrong-issuer / scope-less, verified against a real ES256 JWKS roundtrip.
- Live endpoint: no token and garbage token → 401; a well-formed token with correct issuer+scope but forged signature → 401 "Invalid or expired token", confirming the deployed function fetches the production JWKS (kid `_jQ62WD8cCiIGkKNQB8Hg4El2TNU5rHIITV4h_ba4YM`) and verifies signatures; `Cache-Control: no-store` present.
- Client: `cargo fmt`/`clippy`/`test` (49 lib tests) and `npm run typecheck`/`test:run` (95 tests)/`verify:no-secrets` all green.

## Next steps

1. End-to-end interactive test: sign in on a desktop build → `cognitum_lyria_credential` returns the key → decks report available → live audio with no BYO key. (The 200 success path is the same verified code path the unit tests cover; it needs a real interactive Cognitum token to exercise against the live endpoint.)
2. Follow ADR-178 toward per-user short-lived keys and audio-seconds metering when the Cognitum platform exposes them.
