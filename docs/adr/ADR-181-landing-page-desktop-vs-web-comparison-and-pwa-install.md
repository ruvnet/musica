# ADR-181: Landing page desktop-vs-browser comparison and installable PWA shell

- Status: Accepted
- Date: 2026-07-22
- Related: ADR-163 (Three.js/WebGL2 visual engine), ADR-170 (Lyria RealTime live performance provider), ADR-174 (Restream native live output), ADR-175 (Cognitum OAuth), ADR-179 (sign-in and Lyria credential broker)

## Context

`docs/index.html` is a musician/DJ-facing marketing page, served live by GitHub Pages directly from `docs/` on the `main` branch (no separate build step — merging to `main` is the deploy). It shipped in commit `1ed7c72` ("Add musician/DJ-focused landing page with three.js hero") plus a follow-up legibility fix in `99dff3a`, both without an ADR ("ADR n/a" in the commit message). This ADR retroactively covers that decision and records the additions made alongside it.

The page sells the product but, until now, didn't say what a visitor gets by staying in the browser versus installing the desktop app — and wasn't installable itself. Two things needed fixing:

1. **Clarity on what's desktop-only.** Musica's most load-bearing recent features are native and cannot run in a browser tab: live AI music generation (Lyria RealTime, gated behind `isTauri()` in `apps/musica-vj/src/core/lyriaRealtime.ts` — every entry point throws `"Lyria RealTime requires the desktop app"` outside Tauri), Restream/RTMP broadcast via a bundled FFmpeg subprocess, a persistent on-device capture library, and OS-keychain-persisted Cognitum sign-in. None of these are portable to a static page without a substantial separate architecture (a browser-side Lyria client, a relay for RTMP, etc.) — that is explicitly out of scope here (see Non-Goals).
2. **No install path.** The page had no manifest, no service worker, and no way to add it to a home screen or app launcher, despite being a natural top-of-funnel artifact that benefits from one-tap return visits.

## Decision

### Landing page (retroactive record of `1ed7c72` + `99dff3a`)

Keep the page as shipped: a dark, Space-Grotesk-set, cyan/mint/magenta-accented single-scroll page (`--bg:#0a0a10`, `--accent:#24c8db`, `--accent-2:#75f4c5`, `--accent-3:#ff31d2`) with a decorative, self-animating Three.js hero (loaded via `https://unpkg.com/three@0.185.1/build/three.module.js` — a plain CDN import is fine here, this is a normal site with no CSP, unlike a sandboxed Claude Artifact), a feature card grid, a performance-template chip list, visuals/capture split sections with real screenshots, and a download section linking each platform to `https://github.com/ruvnet/musica/releases/latest` (durable — doesn't hardcode a version that goes stale). No functional changes made to these sections in this ADR.

### New: `#capabilities` — "what runs where"

A two-column comparison, inserted between `#features` and `#templates`, framed as **desktop app vs. browser (this page)** — not "free vs. paid," since both are free:

- **Desktop-only:** live AI music generation (Lyria RealTime), Restream/RTMP broadcast, persistent on-device capture library, sign-in that stays signed in, MIDI/DJ controller + Logitech MX Creative Console support, in-app auto-updates.
- **Browser (this page):** the full tour and screenshots, install as a launcher icon, works offline as a shell back to the downloads. Explicitly states no AI music generation happens here.

A silent preview toggle sits below the comparison: it does **not** add a second rendering pipeline — it exaggerates the existing hero's fake beat-envelope pulse (`window.__musicaPreviewBoost`, read by the existing `tick()` loop) for 8 seconds, clearly labeled "silent, no AI generation." This was chosen over porting a real Web-Audio-driven demo into the page: browsers block audio autoplay (so it couldn't "just play" the way the current hero does), and a real-sounding demo risks implying AI music generation works in-browser, which it doesn't.

The CTA — "Install this page for one-tap access — then download the desktop app for the full DJ+VJ rig." — appears both next to the install button in `#capabilities` and again in `#download`, so whichever section a visitor lands on states the same next step.

### New: installable PWA shell

- `docs/manifest.json` — name/short_name "Musica VJ", `start_url: "/musica/?source=pwa"`, `scope`/`id: "/musica/"`, `display: "standalone"`, theme/background color `#0a0a10`, icons at 192/512 (`any`) and a padded 512 `maskable` variant.
- Icons generated once from the real desktop app icon (`apps/musica-vj/src-tauri/icons/icon.png`, 512×512) via Python/PIL — not hand-drawn, so the web install icon matches the actual installed desktop app icon. The maskable variant pads the mark to ~80% of the canvas on the brand background so Android/Chrome's circular or squircle crop never clips it.
- `docs/sw.js` — a minimal app-shell service worker. Same-origin shell assets (the page, the manifest, local icons, local screenshots) are cache-first under a versioned cache name. Anything cross-origin — specifically the `unpkg.com` three.js import — is deliberately left to the network every time and never cached, so the worker can never pin a stale or broken copy of a dependency it doesn't own.
- Install UX: a single `#pwa-install-btn` in `#capabilities`, hidden until the browser fires `beforeinstallprompt` (Chrome/Edge only; the button and its section presence stay inert on Safari/Firefox, which don't support it — never show a dead control). Visiting the page with `?install=1` surfaces and highlights the button immediately if the prompt is already available, or arms it to fire the moment it becomes available — this is the "browser install option in the URL" entry point, usable from social links, the app's own onboarding, or docs. `appinstalled` persists a `localStorage` flag so a returning, already-installed visitor doesn't see the button again.
- `docs/.nojekyll` — added as a standing safety no-op; GitHub Pages defaults to Jekyll processing on `/docs`, and nothing in this change currently trips it, but it removes the risk once JSON/JS assets live alongside the markdown-heavy ADR/spec trees.

## Non-Goals

- **Real in-browser Lyria RealTime generation.** Out of scope. The desktop-only guard in `lyriaRealtime.ts` is treated as a hard boundary here, not a gap to paper over.
- **A functional lite web app.** The installed PWA is an app-shell/shortcut to the marketing site — a fast way back to the tour and the download links — not a reduced-feature version of Musica itself.
- **Push notifications, background sync, or any other service-worker capability beyond offline app-shell caching.**

## Consequences

- Small, one-time bundle growth: four generated icon PNGs, `manifest.json`, `sw.js`.
- The CDN three.js import is now relied on by both the decorative hero and the silent preview toggle — still not cached by the service worker, by design (see above), so a network hiccup degrades to "no preview," never to a broken/stale render.
- Copy on the page must keep tracking the real feature boundary as the desktop app evolves — if a capability ever becomes available in-browser, `#capabilities` needs a corresponding edit, not just the app.
- No change to the desktop app's release process — this page deploys by merging to `main` and is versioned independently of `musica-vj-v*` tags.

## References

- `apps/musica-vj/src/core/lyriaRealtime.ts` (the `isTauri()` guard this ADR treats as authoritative)
- `apps/musica-vj/src-tauri/icons/icon.png` (source art for the generated PWA icons)
- `docs/index.html`, `docs/manifest.json`, `docs/sw.js`
