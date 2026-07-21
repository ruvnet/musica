# ADR-180: Extension system for integrations, sensors, camera, and custom plugins

- Status: Proposed
- Date: 2026-07-20
- Related: ADR-170 (Lyria RealTime provider), ADR-173 (first-run profile + previews), visual plugin system (`src/core/visualPlugins.ts`), controller bridge (`src-tauri/src/controller_bridge.rs`)

## Context

Musica already integrates external systems, but each one is bespoke and hard-wired:

- **Controllers** — a native Logitech MX Console bridge (`controller_bridge.rs`) and Web MIDI (`ControlRouter`), each with hand-written routing.
- **Visual plugins** — `visualPlugins.ts` already loads AI-generated visual scene specs (`VisualPluginSpec`, normalized and validated), the closest thing to an extension point today.
- **Outputs** — Restream (native FFmpeg/RTMPS), social capture.

There is no general way for the community — or the operator — to add **new input sources (sensors, cameras, OSC, network, biometrics), new transforms (audio DSP, generative visuals), or new outputs (DMX/lighting, external show-control, webhooks)** without patching the app. Every new integration is a code change and a release.

The project already commits to two relevant directions (see memory / prior ADRs): **WASM for DSP and advanced capabilities**, and **AI-generated visual plugins**. An extension system is the unifying architecture those both want.

## Decision

Adopt a **manifest-based, capability-scoped extension system** with a typed **Host API**, where untrusted compute runs in a **WASM sandbox** and all I/O flows through declared, permissioned channels. Extensions are first-class, hot-loadable units — not forks.

### Extension kinds

| Kind | Examples | Runs as |
|------|----------|---------|
| **Input source** | Camera (frame → features/motion), sensors (Arduino/serial, BLE, OSC, MQTT), network/webhook, biometrics, MIDI/OSC clock | Native provider (device access) + optional WASM transform |
| **Transform** | Audio DSP (filters, analysis), generative/visual reactivity, ML feature extraction | WASM (sandboxed, deterministic) |
| **Output / integration** | DMX/Art-Net lighting, show-control (OSC/MIDI out), webhooks, external systems, second-screen | Native provider behind a capability |
| **Visual plugin** | Scene renderers (the existing `VisualPluginSpec`, generalized) | JS/WASM against the visual Host API |

### Manifest + capabilities (the trust boundary)

Every extension ships a manifest declaring identity, kind, entry point, and the **exact capabilities it requests** — nothing is ambient:

```jsonc
{
  "id": "com.example.motion-cam",
  "name": "Motion Camera",
  "version": "1.0.0",
  "kind": "input-source",
  "entry": "plugin.wasm",
  "capabilities": {
    "camera": { "reason": "Drive visuals from body motion" },
    "publish": ["visual.intensity", "visual.hue", "control.macro"],
    "subscribe": ["audio.metrics", "transport.beat"]
  }
}
```

The host resolves the manifest, shows the operator a **consent prompt** listing the requested capabilities in plain language, and grants a scoped token. An extension can only touch host state it was granted — mirroring the app's existing Tauri capability model and the Cognitum consent posture (ADR-178/179).

### Host API (the typed surface)

A single, versioned `HostApi` is the only way an extension reaches Musica. Channels are typed and rate-limited:

- **Subscribe (read):** `audio.metrics` (spectrum/waveform/level from `AudioEngine.getMetrics()`), `transport` (playing, bpm, bar/beat), `visual.state`, `deck.state`.
- **Publish (write):** `visual.*` (intensity, hue, palette, morph, camera), `control.macro`, `deck.*` (guidance, arm), `fx.*`, `stream.marker`.
- **Frames:** input sources push typed frames (camera `ImageBitmap`/features, sensor scalars, OSC bundles); outputs receive typed frames (DMX universes, OSC/MIDI messages).
- **Storage:** a per-extension key/value store (IndexedDB-backed, like the capture library) — no shared global state.

The Host API is versioned; a manifest declares `hostApi: ">=1.0"`, and the host refuses incompatible extensions rather than crashing.

### Sandboxing

- **Transforms and visual plugins run in WASM** (the committed DSP direction) or a locked-down JS worker — no DOM, no network, no filesystem, only the granted Host API. This makes third-party and AI-generated code safe to run.
- **Native providers** (camera, serial/BLE, DMX, OSC) are **first-party Rust** behind Tauri capabilities; a WASM extension *requests* them via the manifest but never gets raw device handles — the native side brokers frames. This keeps device access in audited Rust while letting untrusted logic drive it.

### Lifecycle & distribution

- **Discover:** an `extensions/` directory in the app data dir (drop-in), plus a signed registry later.
- **Load → grant → run → hot-reload → unload**, all without restarting a set. A failing extension is isolated and disabled with a toast, never taking down the show.
- **Distribution:** local install now; a signed extension registry (Ed25519, reusing the updater's signing posture from ADR-179's `latest.json` model) later, so third-party extensions are integrity-verified.

## Consequences

- New integrations (a camera, an OSC sensor rig, a DMX lighting fixture, an AI visual) become **extensions, not releases** — the community and the operator extend Musica without touching core.
- Generalizes today's one-offs: the visual plugin loader, the controller bridge, and MIDI become the first consumers of the Host API rather than parallel code paths.
- The WASM sandbox makes third-party and AI-generated code **safe to run**, unblocking the AI-visual-plugin direction and a real ecosystem.
- Cost: a versioned Host API is a compatibility contract to maintain, and native providers (camera/serial/BLE/DMX) each need a Tauri capability + platform work. This is deliberately phased.

## Phased delivery

1. **Host API v1 + WASM/JS visual plugins** — generalize `visualPlugins.ts` behind the typed Host API and a sandboxed runtime; manifest + consent prompt. (No new device access.)
2. **Input sources — camera + OSC/MIDI + generic sensor (serial/BLE)** — native providers brokering typed frames to sandboxed transforms.
3. **Outputs — DMX/Art-Net + OSC/MIDI out + webhooks.**
4. **Signed extension registry** (Ed25519), reusing the updater signing infrastructure.

## References

- `apps/musica-vj/src/core/visualPlugins.ts` (existing visual plugin precedent)
- `apps/musica-vj/src-tauri/src/controller_bridge.rs` (native input precedent)
- ADR-178 / ADR-179 (consent + signing posture reused for capabilities and the registry)
