# ADR-182: Browser-hosted Lyria RealTime and a fully-functional app PWA

- Status: Accepted
- Date: 2026-07-22
- Related: ADR-163 (Three.js/WebGL2 visual engine), ADR-170 (Lyria RealTime live performance provider), ADR-172 (multistream decks), ADR-174 (Restream native output), ADR-175 (Cognitum OAuth), ADR-179 (sign-in and Lyria credential broker), ADR-181 (landing page PWA shell) — **supersedes ADR-181's stated non-goal of real in-browser Lyria generation**

## Context

ADR-181 shipped an installable PWA, but it was a marketing shell (`docs/index.html`) — a tour of the product, not the product. The actual React studio (`apps/musica-vj/src`) only ran inside the Tauri desktop app: every Lyria RealTime call in `lyriaRealtime.ts` threw `"Lyria RealTime requires the desktop app"` outside Tauri, because the real WebSocket client to Gemini's Live API lived entirely in Rust (`lyria_realtime_provider.rs`), and Cognitum sign-in (the credential path that unlocks it) was flatly disabled outside Tauri (`App.tsx:1426`, `cognitumSignInAvailable = isTauri()`).

Investigation this round found the gap was narrower than expected:
- `AudioEngine.ts` and `VisualEngine.ts` have zero Tauri dependencies — already portable.
- `OnboardingWizard.tsx` has no desktop gating at all.
- The desktop's manual/paste-a-code OAuth flow (`cognitum_auth_manual_start`/`_complete`, the RFC 8252 `urn:ietf:wg:oauth:2.0:oob` sentinel) needs no redirect_uri registration, because it doesn't redirect anywhere — Cognitum shows a code on screen that the user copies back in.
- Verified live against production via `OPTIONS` preflight before writing any code: `https://auth.cognitum.one/v1/oauth/code-exchange`, `/oauth/token`, and the `lyriaBroker` Cloud Function all already send `Access-Control-Allow-Origin: *`. **Zero changes to Cognitum's production infrastructure were needed.** (`/v1/capabilities` does *not* have CORS enabled — handled the same way the Rust side already does: an unreachable capabilities call just grants the full known capability set rather than blocking sign-in.)

## Decision

### Browser Lyria RealTime client — `apps/musica-vj/src/core/lyriaRealtimeWeb.ts`

A from-scratch port of `lyria_realtime_provider.rs`'s protocol over a real browser `WebSocket`, implementing the same five operations (`start`/`update`/`stop`/`poll`/`status`) behind `lyriaRealtime.ts`'s existing exported function signatures — every other caller in the app (`App.tsx`'s poll loop → `AudioEngine.playRealtimePcm16`) needed zero changes. Replicated faithfully: the `setup` → `setupComplete` → `client_content`/`music_generation_config` → `playback_control:"PLAY"` handshake sequence, request validation (prompt count/length/weight, bpm/density/brightness/guidance/temperature/topK ranges, the `onlyBassAndDrums` XOR mute-flags rule), the 8-second bounded audio queue with drop-oldest backpressure, base64 PCM16 chunk decoding, and the graceful `playback_control:"STOP"`-then-close shutdown. One deliberate improvement over the Rust side: reconnect-with-backoff on an unexpected socket close (up to 6 attempts, capped at 15s) — a browser tab is more failure-prone than a native process, and the desktop provider has no reconnect logic at all.

Only the API-key auth path ports (`?key=` query param on the WS handshake). The desktop's `gcloud`-ADC path shells out to the `gcloud` CLI — no browser equivalent, and out of scope here.

### Browser Cognitum auth — `apps/musica-vj/src/core/cognitumWeb.ts`

Implements the manual/OOB flow as real `fetch()` calls: PKCE verifier/challenge via `crypto.subtle.digest("SHA-256", ...)` (base64url, mirroring `random_url_safe`/`URL_SAFE_NO_PAD` in `cognitum_provider.rs`), `window.open()` to the authorize URL, and a paste-back code exchange. Token refresh (`/oauth/token`, `grant_type=refresh_token`) is also ported, since without it a 15-minute Gemini session would silently die mid-set. Tokens persist in `localStorage`, not the OS keychain the desktop uses (ADR-179's `keyring`-backed persistence) — a real, stated tradeoff: a browser tab cannot reach the OS keychain, and `localStorage` is the standard web equivalent, but it is not keychain-grade protection. `App.tsx`'s primary sign-in button is now always available (`cognitumSignInAvailable = true`) and routes to this manual flow when `!isTauri()`, since the desktop's loopback-listener flow genuinely cannot work in a browser (no way to bind a TCP listener).

### Out of scope for browser sign-in (unchanged, still Tauri-only)

`generateCognitumStylePack`, `generateCognitumVisualPlugin`, and `generateCognitumAutoDjBrief` remain desktop-only — optional AI-assist extras, not on the core "sign in → hear music" path. `generateCognitumFxDirection`/`VisualDirection`/`SetArc`/`VocalGuidance` already have local deterministic fallbacks in `cognitum.ts` used when Cognitum is unavailable, so browser users aren't blocked, just get the local version.

### Build & hosting

`vite.config.ts` gained a conditional `base: isWebBuild ? "./" : "/"` (via `MUSICA_WEB_BUILD=true`, wired to a new `npm run build:web` script) — Tauri's webview needs the root-absolute base it already had; a GitHub Pages subpath needs relative asset URLs instead. Built and committed to `docs/app/`, the same "no separate deploy pipeline, merging to `main` is the deploy" pattern as the rest of `docs/`. Live at `https://ruvnet.github.io/musica/app/`.

### PWA installability for the real app

`apps/musica-vj/public/manifest.json` + `public/sw.js` + `public/icons/` (reusing the same icons generated for ADR-181, sourced from the real desktop app icon) — copied into `docs/app/` on every build via Vite's `public/` convention. **Verified, not assumed**: Chrome DevTools Protocol's `Page.getInstallabilityErrors` against the built, served app returns `[]` — zero installability errors, the actual signal Chrome uses to decide whether to offer the install prompt. An explicit install-toast (vanilla JS in `index.html`, `beforeinstallprompt` → show, `appinstalled`/dismiss → hide, `localStorage`-remembered) sits alongside Chrome's own native install icon, and — like the marketing page's install button — never renders on browsers without `beforeinstallprompt` support.

### SEO / link previews

`index.html` gained a real `<meta name="description">`, Open Graph and Twitter Card tags, and a dedicated 1200×630 JPEG preview image (`docs/assets/screenshots/musica-vj-og-preview.jpg`, cropped/compressed from the existing studio screenshot — the original 3MB PNG was too large for reliable social-crawler fetching).

## Verification

- `npm run typecheck` and `npm run test:run` (95 tests) pass unchanged.
- Headless-Chromium (Playwright) walkthrough of the built, served `docs/app/`: sign-in button enabled and generates the correct manual/OOB authorize URL with a real PKCE challenge; the paste-a-code UI appears; full onboarding → welcome screen → studio flow completes via "Enter without audio"; zero console errors at every stage.
- **UI parity directly confirmed against a live desktop screenshot the user supplied**: the browser-rendered studio (Visual Bank, VJ Presets, Animation, Lyria RealTime deck panel, beat sequencer, top/footer bars) matches structurally and visually — expected, since it's the same React app and CSS, not a separate lite build, but verified rather than assumed.
- `Page.getInstallabilityErrors` via CDP: `[]`. Service worker confirmed `active`, scope correctly resolved.

## Non-Goals

- Restream/RTMP broadcast, the persistent native capture library, the Logitech MX Console bridge, and auto-updates remain desktop-only — genuinely native capabilities, unchanged from ADR-181.
- A live, human-completed Cognitum sign-in end-to-end (this session has no test Cognitum account) — automated verification covers everything up to that boundary; a real sign-in and live Lyria audio check needs a human or real test credentials.
- Push notifications, background sync, or any service-worker capability beyond app-shell caching.

## Consequences

- The "browser version" claim in the README's Web + WASM Direction section is now true, not aspirational — updated accordingly.
- `localStorage`-based token storage is a real, if standard, downgrade from the desktop's OS-keychain persistence — worth revisiting if browser session hijacking becomes a concern (e.g., a future Trusted Types / stricter CSP pass).
- Two PWAs now exist side by side under `docs/` — the marketing shell (`docs/`) and the real app (`docs/app/`) — each with its own manifest/service worker/icons, deliberately not merged, since they have different scopes and different install identities.
