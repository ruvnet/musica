# ADR-178: Metered Lyria via the Cognitum proxy plane

- Status: Proposed
- Date: 2026-07-20

## Context

Today every operator must supply their own `GEMINI_API_KEY` for Lyria RealTime and generation. This is the single biggest onboarding failure: on a packaged desktop app there is no shell to export it (the v0.3.0 "Start Session does nothing" bug on Windows was exactly this), keys leak when shared, and there is no per-user metering or billing.

Musica already authenticates users through **Cognitum One** (OAuth 2.1 + PKCE, ADR-175) using the shared `meta-proxy` client, and already routes its **text** Meta-LLM features (set arcs, phrase briefs, style packs, FX/visual/vocal direction) through Cognitum's `/v1/chat/completions` inference plane with that token. `meta-proxy` is Cognitum's authenticated, fail-closed routing proxy with a metered/consent-gated Sponsored plane.

The question: route **Lyria** through the same metered plane so signing in with Cognitum One is sufficient — no bring-your-own key.

## Decision

Adopt "sign in, and Lyria just works, metered to your account" as the target model, in two tiers, and keep BYO-key as a permanent fallback.

### Protocol reality (the constraint)

- **Meta-LLM / text features** — already on the metered plane. Nothing to do; this ADR only records that they are the proof the pattern works.
- **Lyria generation (batch)** — REST request/response; a Cognitum `/v1/lyria/generate`-style metered passthrough is straightforward for the proxy to add. Musica's `creative_provider` gains a "cognitum" provider kind that sends the OAuth bearer token instead of a Gemini key.
- **Lyria RealTime (live)** — this is the hard part. It is a **bidirectional audio WebSocket** (`wss://…/BidiGenerateMusic`), not a chat completion. Metering it requires Cognitum/meta-proxy to expose a WebSocket passthrough that (a) authenticates the OAuth token, (b) opens the upstream Google socket with Cognitum's own Google credential, (c) relays PCM frames, and (d) meters by streamed audio-seconds. This is new proxy surface, not a config change.

### Client design (Musica side, once the endpoints exist)

- A new provider mode selects, per service, between `byo-key` (current `GEMINI_API_KEY`) and `cognitum` (OAuth bearer to the metered endpoint). Default to `cognitum` when signed in and the endpoint is reachable, else fall back to BYO key, else offline.
- The realtime provider's WebSocket URL and auth header become configurable so it can target either `generativelanguage.googleapis.com` (BYO key in query) or the Cognitum relay (bearer token) with no other code change.
- Metering/usage is surfaced read-only in the COGNITUM AI panel (audio-seconds this session, plan remaining) using the existing capabilities/usage endpoint shape.

### Fallback and consent

- BYO `GEMINI_API_KEY` (now discoverable via ADR-178-adjacent `providers.env`) stays supported forever for self-hosters and offline/air-gapped use.
- Metered usage is consent-gated: the Sponsored/metered plane is only used after explicit opt-in in the panel, never silently, mirroring meta-proxy's consent model.

## Consequences

- Eliminates the key-provisioning failure class entirely for signed-in users — the exact bug this release chased — and adds real per-account metering/billing.
- Requires Cognitum/meta-proxy to ship two new metered endpoints (Lyria generate REST, Lyria RealTime WebSocket relay); the RealTime relay is non-trivial infrastructure and gates the "live" tier.
- Musica's provider layer grows a mode selector but the security posture is unchanged: tokens stay in Rust, never in React.
- Keeps a hard BYO-key fallback so the app never becomes captive to the proxy.

## Next steps

1. Confirm with the Cognitum platform whether a metered Lyria REST passthrough and a RealTime WebSocket relay are on the roadmap; this ADR is blocked on that infra.
2. If yes: implement the batch `cognitum` provider mode first (low risk), then the RealTime relay client.
3. Until then, `providers.env` BYO-key remains the supported path.

## References

- ADR-175 (Cognitum OAuth + Meta-LLM), ADR-170 (Lyria RealTime provider)
- `cognitum-one/meta-proxy` — the routing/metering proxy
