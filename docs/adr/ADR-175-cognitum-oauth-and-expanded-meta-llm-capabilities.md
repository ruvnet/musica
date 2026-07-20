# ADR-175: Cognitum One OAuth and expanded Meta-LLM capabilities

- Status: Proposed
- Date: 2026-07-19

## Context

Musica already ships a governed Cognitum Meta-LLM boundary (`meta_llm_provider.rs`): the Agent Director sends a bounded goal to `api.cognitum.one` and receives a structured performance plan (template, scene, BPM, art direction, temporal controls, and a production brief). The integration is gated by `MUSICA_META_LLM_ENABLED` and authenticated with a static bearer token from `MUSICA_META_LLM_API_TOKEN`. When the provider is absent, a deterministic local planner answers instead.

Two constraints now limit what the integration can do:

1. **Static env-var tokens do not survive distribution.** They work for a developer shell but not for a packaged desktop app used by non-developers. They are long-lived, unscoped, shared across features, and cannot express per-user quotas, billing, or revocation.
2. **The Meta-LLM surface is narrow.** One command (`meta_llm_plan`) with one prompt shape. The rest of the app has since grown capabilities the planner knows nothing about: custom user styles, the 25-style preset bank, per-genre stream FX defaults, the seven-effect FX rack with parameter editors, performance pads, deck scenes, and IndexedDB-persisted workspace settings.

Separately, Restream broadcasting (ADR-174) requires the operator to hand-copy a stream key, which is the least ergonomic and most error-prone step of going live. Restream exposes an OAuth 2.0 API that can list channels and mint ingest credentials.

## Decision

Adopt OAuth 2.0 Authorization Code with PKCE as the account model for external services, and expand the Meta-LLM boundary into a set of scoped, schema-validated commands. Delivery is phased; each phase is independently shippable.

### Phase 1 — Cognitum One OAuth sign-in

- Add `musica://oauth` deep-link handling in Tauri; the authorization request opens in the system browser (never the webview), and the loopback/deep-link redirect returns only the authorization code to Rust.
- Exchange and refresh happen exclusively in Rust. Tokens are stored in the operating-system keychain (Tauri keyring), never in React state, `localStorage`, IndexedDB, or the settings export (the existing secret-canary CI checks extend to these paths).
- `MUSICA_META_LLM_API_TOKEN` remains a fallback for headless/CI use; when both exist, OAuth wins.
- The topbar gains a signed-in indicator; sign-out revokes and deletes the stored tokens.

### Phase 2 — Expanded Meta-LLM capabilities (each a separate Rust command with a strict JSON schema)

| Command | New capability |
|---|---|
| `meta_llm_style_pack` | Generate a full four-prompt Lyria style pack (identity, blend, arc, negative, config, stream-FX defaults) from a plain-language description; lands as an editable custom style, never applied silently. |
| `meta_llm_set_arc` | Plan a 30–90 minute set as a timeline of deck scenes, style switches, FX moves, and visual scene changes; the operator sees the timeline and each step still applies through existing UI actions. |
| `meta_llm_autodj_brief` | Replace the local Auto DJ phrase rotation with per-phrase briefs that know the previous phrase, the crowd direction text, and the active style's negative prompts. |
| `meta_llm_fx_direction` | Map a mood request ("make it feel underwater, then lift") to bounded stream-FX automation over the next N bars, respecting per-effect locks. |

All commands inherit the existing envelope: bounded input sizes, bounded response bytes, JSON-only parsing, numeric clamping in Rust before anything reaches React, and the deterministic local planner as the fallback for every command.

### Phase 3 — Restream OAuth

- Replace manual stream-key entry with Restream account sign-in using the same PKCE + keychain pattern.
- Rust fetches the ingest URL and stream key from the Restream API at `Go Live` time; the key never appears in the UI at all (strictly better than ADR-174's masked field, which remains as a manual fallback).
- Channel titles and destination status become visible in the AV OUTPUT panel.

### Explicitly out of scope

- OAuth for Gemini/Lyria: Google API keys remain env-provided to the Rust providers (ADR-170 governance unchanged).
- Cloud sync of workspace settings: possible later on top of Phase 1 identity, but IndexedDB persistence is local-only for now.
- Browser preview builds: no OAuth, no tokens; they keep the deterministic local planner and manual Restream handoff.

## Consequences

- A packaged, distributable Musica becomes realistic: end users sign in instead of editing shell environments, and abuse is controllable per-account server-side.
- The Meta-LLM stops being a single-purpose planner and becomes the reasoning layer for the features users actually touch (styles, FX, set arcs), while every application of its output stays inspectable and reversible in the UI.
- Two new failure domains appear: OAuth flows (browser handoff, expired refresh tokens) and keychain access. Both degrade to today's env-var behavior rather than blocking.
- Secret-handling surface grows; the no-tokens-in-React invariant now covers keychain plumbing and the settings exporter, and CI canaries must assert it.
- Restream key exposure drops to zero in the happy path, at the cost of an app registration with Restream and API-version coupling.

## Acceptance tests

1. The authorization URL opens in the system browser; the webview never navigates to the identity provider.
2. Tokens round-trip through the keychain only; grep of React state, localStorage, IndexedDB, settings exports, logs, and status payloads finds no token material.
3. With no network or no sign-in, every Meta-LLM command falls back to the local planner and the UI reports LOCAL.
4. `meta_llm_style_pack` output is rejected unless it validates against the style schema (four prompts, ≤240 chars each, exactly one negative prompt, bounded config).
5. `meta_llm_fx_direction` cannot modify a locked effect.
6. Restream Go Live with OAuth never renders the stream key in any UI element or React state snapshot.
7. Sign-out deletes stored tokens and subsequent status calls report signed-out within one poll interval.

## References

- ADR-170 (Lyria RealTime provider governance), ADR-171 (Auto DJ sequencing), ADR-174 (Restream native live output)
- `apps/musica-vj/src-tauri/src/meta_llm_provider.rs` — current Cognitum boundary
- RFC 7636 (PKCE), RFC 8252 (OAuth 2.0 for Native Apps)
