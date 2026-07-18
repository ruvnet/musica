# ADR-166: Desktop Threat Model, CSP, Secrets, and Filesystem Boundaries

## Status
Accepted

## Date
2026-07-18

## Lyria 3 Pro amendment

ADRs 168 and 169 add a provider-specific Rust path without widening webview network authority. `GEMINI_API_KEY` is consumed only by Rust application code. The adapter is pinned to `generativelanguage.googleapis.com:443`, disables redirects, system proxies, and `reqwest` retries, uses a five-second connect timeout and a configurable request deadline that defaults to 600 seconds and is restricted to 60 through 900 seconds, limits concurrency to two, sends `store: false`, and returns only typed task state and validated local audio bytes to the webview. Provider failures become allowlisted error codes rather than raw bodies or headers. Successful raw responses are staged; upstream error bodies are inspected locally for classification but are neither exposed nor preserved as exact failure artifacts.

Successful Lyria audio, the exact JSON response, and a receipt are written under the private application-data `generated` directory with per-task directory mode `0700`, file mode `0600`, and create-new semantics. A failure stores a typed immutable failure receipt, not the raw provider error response. The success receipt can omit plaintext prompts through `MUSICA_CREATIVE_RETAIN_PROMPTS=false`; it always retains a prompt hash. That setting does not redact the exact provider response, which can contain generated lyrics, structural or safety text, and potentially echoed sensitive content. The entire task directory is therefore privacy-sensitive. C2PA and SynthID are marked expected and unverified. The production bundle and credentialed runtime logs must be scanned for a test Gemini key before promotion.

The Lyria path closes the generic provider's hashed-staging gap but not the hostile-decoder gap. MP3/WAV size, MIME, signature, encoded duration, sample rate, and stereo channel count are inspected in Rust, then the validated bytes are decoded by WKWebView/Web Audio. A constrained native media subprocess, checked decoded-allocation ceiling before codec execution, C2PA parser/validator, Keychain credential setup, durable workspace isolation, and restart-safe job ledger remain deferred security gates. Image and PDF conditioning are rejected in V1 before network dispatch.

## Context

A Tauri desktop application combines code from different trust domains: bundled JavaScript in a webview, privileged Rust commands, user-selected media, remote provider responses, a local Logitech process, and paths chosen for export. It also operates speakers at live volume and can spend money through creative APIs.

Desktop locality is not a security boundary. A cross-site scripting defect, malicious media file, compromised provider response, or same-user local process must not obtain API keys, arbitrary file access, shell execution, or an unauthenticated control channel.

## Decision

Adopt a deny-by-default desktop threat model. The Tauri Rust backend is a narrow trust broker, not a general native API. The webview is treated as potentially compromised for capability design even though all production code is bundled.

### Trust boundaries

| Component | Trust level | Permitted authority |
|---|---|---|
| Bundled webview | Constrained | UI, Web Audio, WebGL, typed commands |
| Tauri Rust backend | Trusted broker | Validation, scoped I/O, approved network, secret-source abstraction |
| Root Musica algorithms | Trusted computation | Bytes and parameters supplied by validated backend |
| Imported/generated media | Hostile data | Decode only under size, duration, and resource limits |
| Creative provider | Untrusted remote | Exact-host HTTPS through provider adapter |
| Logitech companion | Same-user untrusted process | Authenticated enumerated control actions only |
| Project/export path | User controlled | One selected operation, canonicalized and scoped |

### Tauri capabilities and commands

Only the `main` window receives capabilities. V1 enables `core:default`, save dialog, file write, and global-shortcut registration/unregistration. There is no shell plugin, Tauri HTTP plugin, process launch, clipboard read, arbitrary filesystem read, or broad home-directory read scope. The save dialog dynamically scopes the selected destination for the filesystem plugin.

Implemented provider and bridge messages use typed or strict structs with maximum lengths, closed enums, unknown-field rejection, finite numeric bounds, and exact-host URL validation. Command handlers never interpolate input into a shell.

The target native import and project API replaces plugin-managed paths with opaque, short-lived Rust path tokens bound to operation, canonical path, window, and expiry. Atomic replacement, explicit overwrite, and token replay tests are required before that target API ships. V1 does not claim this path-token layer.

### Content Security Policy

V1 blocks remote scripts, `eval`, frames, objects, and all external webview network origins. `connect-src` permits only the application and Tauri IPC. `img-src` permits local application, asset protocol, blob, and data images; `media-src` permits local application, asset protocol, and blob media but deliberately omits `data:`. Provider metadata and generated audio travel through exact-host Rust commands, so a partner build does not widen webview CSP.

The implemented base production CSP is:

```text
default-src 'self';
script-src 'self';
style-src 'self' 'unsafe-inline';
connect-src 'self' ipc: http://ipc.localhost;
img-src 'self' asset: http://asset.localhost blob: data:;
media-src 'self' asset: http://asset.localhost blob:;
worker-src 'self' blob:;
object-src 'none';
frame-src 'none';
base-uri 'none';
form-action 'none';
```

In that target policy, `unsafe-eval`, remote scripts, CDN assets, arbitrary `https:`, remote frames, and objects are prohibited. Once Rust owns provider staging, the webview no longer needs any provider origin. The Rust adapter retains its exact compile-time API-host allowlist. Development-server exceptions live in a development-only configuration.

Inline style is the only initial CSP concession. It is tracked for removal when the component and Three.js UI no longer require it. User prompts, filenames, provider messages, and project metadata are rendered as text, never injected HTML.

### Secrets

The standard release ships with creative providers disabled. The generic operator and partner path reads its bearer token from the Tauri process environment, while Lyria reads `GEMINI_API_KEY`; application code keeps both in Rust memory. Neither is sent to JavaScript, `localStorage`, project files, crash reports, URLs, provider responses, or logs. Runtime environment configuration cannot expand either provider's host policy.

Environment-based development has a narrower guarantee than a consumer credential store. When an operator launches `npm run tauri dev` from a shell containing a provider key, child processes used by the development stack inherit that environment even though Vite does not expose a non-`VITE_` key to browser code or bundle it. Development must use a dedicated low-quota credential and must not execute untrusted npm or build tooling in the same environment. Keychain-backed secret injection directly into the native process remains the promotion path.

A user-facing credential setup flow is not accepted until it stores provider tokens as generic-password items in macOS Keychain under service `one.cognitum.musica.vj`, returns only a non-secret key identifier to the webview, and fails closed when Keychain is unavailable. Plaintext application configuration is not a fallback.

The Logitech installation token is intentionally separate. It is 256 random bits encoded as 64 lowercase hexadecimal characters in `controller.token`, a regular current-user-owned file with mode `0600` and one hard link. The companion needs same-user filesystem access because it is hosted by Logi Plugin Service. Both sides reject a symlink or unsafe metadata and never log the token. Sequence and timestamp checks limit replay; this bearer token is not represented as an HMAC protocol.

Logs redact authorization headers, query strings, prompt bodies by default, local paths beyond basename, controller tokens, task response bodies, and values matching registered secret fingerprints. Telemetry is off by default and never contains content.

### Media and filesystem controls

V1 direct import uses a browser file input, rejects files above 250 MB before `arrayBuffer()`, decodes with `decodeAudioData`, and then enforces 10 minutes, eight channels, and 192 kHz. It does not yet provide Rust magic-byte validation, pre-decode duration/channel metadata, or a checked decoded-sample allocation. Explicit file selection and the lack of ambient read permission reduce authority, but hostile decoder input remains a known gap.

The target native import path checks a magic signature and enforces 250 MB, 10 minutes, eight channels, 192 kHz, and a checked maximum sample count before allocation. Unsupported, encrypted, archive, and malformed inputs fail before Musica.

Generic-partner JSON is capped at 512 KiB, HTTPS-only, exact-host allowlisted, proxy-disabled, and redirect-disabled. Rust repolls a completed task, downloads its audio with the same client, enforces a 64 MB streaming cap regardless of `Content-Length`, requires an `audio/*` content type plus a recognized MP3, WAV, FLAC, Ogg, or MP4 signature, and returns raw IPC bytes. `application/octet-stream` and other generic MIME types are rejected. The signed URL is not persisted. Lyria instead bounds successful JSON to 96 MiB and decoded audio to 72 MiB, then uses the private hashed staging described in ADR-169. Stronger decoder isolation and private-address defense for any future configurable host remain target work. No redirect support is required; if added, every hop must pass the same exact-host policy.

Content-addressed application cache, separate project storage, asset-protocol scopes, and path-token exports are target work. V1 live exports use a save dialog and Tauri's dynamically scoped file-write permission.

### Local bridge and updates

The Logitech bridge follows ADR-162: a mode `0600` per-user socket and token file, constant-time token equality, sequence and timestamp replay protection, 4 KiB messages, and a closed action enum. It never listens on TCP and never runs as root.

Public release bundles must be signed and notarized under ADR-167; the current CI artifact is unsigned. Any future updater must verify a Tauri update signature in addition to TLS before installation. The app does not self-update unsigned code or load executable plugins from a project.

### Abuse controls

Master output starts below unity, is limited, and clamps all controller values. Replayed or malformed adjustment events cannot drive NaN or unbounded gain. Provider calls are bounded by the implemented limits and roadmap gates distinguished in ADR-164. Recording always displays a visible indicator and records only the application's canvas and master mix, never the screen, microphone, or camera without a new permission decision.

## Alternatives Considered

### Trust bundled frontend code fully

Bundled code can still contain dependency or injection flaws. Granting broad Tauri permissions would convert one webview defect into native compromise. Rejected.

### Store API keys in localStorage or a dotfile

Both are easy for frontend code, crash tools, backup systems, or other processes to read. Rejected.

### Use unrestricted Tauri filesystem and HTTP plugins

This reduces Rust code but gives a compromised webview ambient authority over user files and remote hosts. Rejected.

### Send remote media URLs directly to Web Audio

This widens CSP, leaks user IP and request metadata, and weakens size controls. Rejected. V1 downloads remote media through bounded Rust IPC; persistent Rust staging and content hashing remain required before general-user provider promotion.

### Depend on the macOS App Sandbox alone

The sandbox can reduce system access but does not define application-level provider spend, socket authentication, path-token semantics, or project separation. It is defense in depth, not the architecture.

## Consequences

### Positive

- A webview compromise has no shell, provider token, arbitrary filesystem read, process capability, external network origin, or direct generated-media URL download.
- Provider and local-device risks are explicit and testable.
- Future projects remain portable without carrying credentials.
- Offline mode is a complete security fallback.

### Negative

- Target opaque path tokens remain roadmap work. Lyria has provider-specific private hashed staging, while the generic partner and manual import do not.
- A future user-facing provider setup can be interrupted by Keychain prompts or lock state; the operator-only environment path is intentionally not a consumer credential store.
- The generic partner's raw IPC download must be replaced by private hashed staging before its promotion to general users. Lyria already stages bytes but still needs isolated media decoding and durable project integration.

## Risks and Mitigations

The largest residual uncertainty is WKWebView decoder attack surface when parsing a user-selected file or bounded remote preview. The local byte cap prevents unbounded source reads, but post-decode duration, channel, and sample-rate checks cannot prevent every decoder allocation. Provider-off-by-default operation, Rust audio signature checks, the generic 64 MB cap, and the Lyria 72 MiB cap limit source exposure; they do not constitute a constrained decoder. Native isolated decoding with checked output allocation remains required before broad provider distribution.

Exact provider hosts and terms can change. Adapter configuration changes require review, tests, and a release; arbitrary redirects are not a compatibility mechanism.

## Rollback

Leaving the provider enable environment variable unset disables provider calls while preserving local synthesis, direct file import, visuals, and live export. The Logitech bridge can be removed independently while F13 through F24, keyboard, MIDI, and UI remain. Individual native capabilities can be removed without changing the root Musica crate.

## Acceptance Tests

1. V1 CSP tests prove remote scripts, `eval`, frames, objects, and every external network origin are blocked from the webview; `connect-src` permits only the application and Tauri IPC.
2. Capability tests prove the frontend cannot invoke shell, process, Tauri HTTP, arbitrary read, or a write path not granted by the save dialog.
3. Before the target path-token API ships, traversal, symlink substitution, expiry, replay, and overwrite-without-confirmation fixtures must fail closed.
4. Before native import ships, fuzz fixtures must exercise malformed headers, oversized sample counts, 250 MB cap, duration cap, channel cap, and checked allocation without crash or unbounded memory.
5. V1 rejects every provider redirect, non-HTTPS origin, non-443 port, credentialed URL, non-allowlisted host, oversized audio response, unsupported content type, and unsupported audio signature. The future configurable staged downloader also rejects private and metadata addresses.
6. Repository, bundle, logs, project, receipts, localStorage, and crash fixtures contain no test provider token after a full provider workflow; a future user-facing provider release also passes locked-Keychain tests.
7. Wrong installation token, replayed sequence, stale timestamp, oversized Logitech message, and unknown action never emit a frontend control event.
8. Provider-disabled mode makes zero external connection during a 30-minute performance and capture test.
9. Audio fuzzing confirms every gain remains finite and bounded despite malformed controller values.

## References

- [Tauri 2 security overview](https://v2.tauri.app/security/)
- [Tauri 2 capabilities](https://v2.tauri.app/security/capabilities/)
- [Tauri 2 Content Security Policy](https://v2.tauri.app/security/csp/)
- [Tauri 2 asset protocol scope](https://v2.tauri.app/security/asset-protocol/)
- [Tauri 2 filesystem security](https://v2.tauri.app/plugin/file-system/)
- [Apple Keychain Services](https://developer.apple.com/documentation/security/keychain-services)
- [OWASP File Upload Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html)
- ADR-162: Logitech controller bridge
- ADR-164: Creative AI provider governance
- ADR-167: Release signing and notarization
- ADR-168: Lyria 3 Pro provider, routing, and prompt contract
- ADR-169: Lyria paid job lifecycle, assets, analysis, and provenance
