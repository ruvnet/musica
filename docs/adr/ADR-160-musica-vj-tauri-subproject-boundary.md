# ADR-160: Musica VJ as an Isolated Tauri 2 Desktop Subproject

## Status
Accepted

## Date
2026-07-18

## Context

Musica is a portable Rust library for graph-based audio analysis and source separation. The proposed product is different: a macOS performance application combining a six-track synthesizer, imported audio, reactive visuals, physical controllers, cloud generation, and social-video capture.

Putting desktop concerns into the root crate would force Tauri, WebKit, UI, code-signing, and vendor SDK dependencies onto library users. Putting privileged operations in browser code would expose provider credentials and broad filesystem access. A separate product boundary is therefore required.

The first release targets macOS 13 or later. The UI must still run in an ordinary browser during development, with native-only features reported as unavailable rather than emulated insecurely.

## Decision

Create the desktop product at `apps/musica-vj` as a private Tauri 2 application.

```text
musica (root Rust crate)
    ^
    | optional, one-way dependency
apps/musica-vj/src-tauri (Rust trust broker)
    ^
    | typed Tauri commands and events
apps/musica-vj/src (React, TypeScript, Web Audio, Three.js)
```

The dependency rule is an invariant:

1. The root `musica` crate MUST NOT depend on Tauri, JavaScript packages, Three.js, Logitech SDK assemblies, provider SDKs, or macOS application frameworks.
2. The Tauri crate MAY depend on `musica` for offline jobs.
3. The frontend MAY depend only on versioned data-transfer types and Tauri commands. It MUST NOT receive provider secrets or unrestricted filesystem handles.
4. Vendor-specific integrations MUST stay under the application boundary and communicate through normalized control or provider contracts.

The root Cargo workspace may include `apps/musica-vj/src-tauri` so a repository-level build can validate the native shell. The root package remains the default workspace member, preserving the existing `cargo test` and library release workflow.

### Runtime ownership

| Concern | Owner | Boundary rule |
|---|---|---|
| Music separation and reusable DSP | Root `musica` crate | Platform-neutral Rust API |
| Synth graph, transport, track mix | Web Audio frontend | Uses the audio sample clock; see ADR-161 |
| Reactive rendering | Three.js frontend | No privileged commands in a frame loop; see ADR-163 |
| Native dialogs, cache, secrets, network providers | Tauri Rust backend | Least-privilege commands only |
| Logitech Actions SDK | Separate official SDK companion | Authenticated local protocol; see ADR-162 |
| Recording UI and live capture | Frontend | Browser capability-probed path; see ADR-165 |
| Signing, notarization, release | Tauri bundle workflow | Protected release environment; see ADR-167 |

Tauri commands use explicit request and response structs. Breaking command changes increment a `schemaVersion`; unknown fields are rejected for security-sensitive requests. Command handlers validate and delegate to modules rather than containing DSP, provider, or rendering policy.

The browser development build provides the local synthesizer, visuals, and in-memory recording when supported. It returns a typed `Unavailable` result for Keychain access, native file operations, provider calls, global shortcuts, and the Logitech bridge.

### Product and build targets

The application identifier is `one.cognitum.musica.vj`. The JavaScript package is private and is never published as part of the core crate. Dependencies are pinned by lockfiles and production assets are bundled locally. No runtime JavaScript is loaded from a CDN.

Initial budgets on the reference Apple M1, 16 GB machine are:

| Metric | Budget |
|---|---|
| Cold launch to interactive controls | less than 2.0 seconds, p95 over 20 launches |
| Compressed DMG | less than 35 MB before optional sample packs |
| Idle resident memory after warm-up | less than 250 MB |
| Webview-to-Rust non-streaming command overhead | less than 10 ms p95 locally |

Audio, visual, and capture budgets are defined in ADR-161, ADR-163, and ADR-167. These numbers are release gates, not claims about every Mac.

## Alternatives Considered

### Electron application

Electron provides a uniform Chromium runtime and stronger WebCodecs predictability. It also bundles Chromium and Node, increasing download size, memory, update surface, and native attack surface. A realistic minimum compressed application would be roughly 80 to 150 MB rather than the 35 MB target. Rejected for v1.

### Native Swift and Metal application

Swift, AVAudioEngine, and Metal can deliver lower audio latency and deterministic media export. They would require rebuilding the Three.js experience and would make the initial product macOS-only at every layer. Rejected as the primary UI; native media components remain possible behind Tauri commands.

### Browser or PWA only

This is the cheapest distribution path but cannot safely own long-lived provider secrets, global shortcuts, notarized helpers, or the official Logitech plugin lifecycle. Retained only as the reduced development preview.

### Add UI features to the root crate examples

This would blur the reusable-library contract and make release, licensing, and security policy inseparable from DSP. Rejected.

## Consequences

### Positive

- Core Musica users do not inherit desktop dependencies.
- The web stack can iterate on visual performance without compromising Rust portability.
- Privileged capabilities have one auditable Rust boundary.
- Removing the product subproject cannot break the root library API.

### Negative

- There are two build ecosystems and two lockfiles.
- WebKit behavior can differ by macOS version even when the JavaScript bundle is unchanged.
- Typed IPC adds serialization and is unsuitable for real-time audio or per-frame pixel transfer.

## Risks and Mitigations

The largest uncertainty is WKWebView media capability drift across supported macOS releases. Every optional feature performs a runtime capability check, the supported macOS matrix is tested before release, and native fallback work is isolated behind the Tauri boundary.

Dependency duplication is controlled by exact versions, automated update PRs, and bundle-size checks. IPC misuse is controlled by architectural tests that reject privileged imports in audio and render-loop modules.

## Rollback

The application can be removed from the Cargo workspace and `apps/musica-vj` without changing the root crate's public API or data formats. If Tauri proves unsuitable, the frontend can be hosted in another shell while retaining the normalized control, provider, project, and provenance contracts.

## Acceptance Tests

1. `cargo test --workspace` and the root default `cargo test` both pass; the latter does not build Tauri.
2. A dependency scan confirms the root package graph contains no Tauri, webview, Logitech, or creative-provider dependency.
3. `npm run build` produces a self-contained frontend with no CDN script or style reference.
4. Browser preview starts, plays the six local tracks, and renders visuals while native features report `Unavailable` without throwing.
5. The current `macos-15` CI job builds and uploads an unsigned application bundle; Apple Silicon and Intel builds are required before claiming both architectures in a public release.
6. Twenty cold starts on the reference M1 meet the 2.0 second p95 budget and the compressed bundle remains below 35 MB.

## References

- [Tauri 2 architecture](https://v2.tauri.app/concept/architecture/)
- [Tauri 2 capabilities](https://v2.tauri.app/security/capabilities/)
- [Tauri 2 Content Security Policy](https://v2.tauri.app/security/csp/)
- [Tauri 2 macOS distribution](https://v2.tauri.app/distribute/sign/macos/)
