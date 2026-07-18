# ADR-167: Quality, Performance, macOS CI, Signing, and Notarization Gates

## Status
Accepted

## Date
2026-07-18

## Lyria 3 Pro amendment

Secret-free pull-request CI validates structured prompt compilation, routing, cost math, request validation, one-attempt state transitions, MP3/WAV fixture inspection, decoded PCM analysis, frontend bundle secret scanning, and the provider-disabled path. The analysis suite includes a generated 180-second, 48 kHz stereo asset and requires the complete local analysis to finish in under five seconds on the CI runner. This is an automated regression budget, not proof of decode or analysis time on a supported Mac. CI cannot prove current preview availability, a paid 120 to 180 second result, Google account billing, SynthID presence, C2PA validity, or external generation latency.

Before the Lyria capability is promoted beyond preview, a protected manual integration run must generate at least one MP3 and one WAV, retain account-side request and billing evidence, decode both, compare actual encoded and decoded metadata, and preserve the private receipts. Two requests in one process require a generation ceiling of at least USD 0.16; a USD 0.08 ceiling requires a restart between separately approved candidates because recorded charges accumulate for the process lifetime. A physical reference-Mac run must confirm post-receipt analysis under five seconds, waveform availability under two seconds, visualization setup under three seconds, playback readiness under 500 ms after decode, and 60 FPS Three.js playback. The packaged recorder must refuse non-MP4 output and reject a completed MP4 unless its ISO `stsd` sample entries prove H.264 and AAC. A separate physical decode must still prove actual 1080 by 1920 dimensions, delivered frame rate, audio/video tracks, and synchronized timestamps.

None of those provider or hardware checks supplies Apple distribution evidence. Physical Logitech qualification, Developer ID signing, hardened runtime, notarization, stapling, Gatekeeper assessment, and clean-Mac validation remain separate manual or protected-release gates. The integration specification must not label the unsigned CI `.app` production ready.

## Context

A music and visual-performance application can pass ordinary unit tests while still failing on stage through audio starvation, frame drops, controller disconnects, corrupt recordings, or macOS security rejection. Tauri also combines Rust, npm, WebKit, native signing, and optional external plugin artifacts, so a single green build is not sufficient evidence of release readiness.

Physical Logitech hardware, audio latency, GPU performance, and notarization cannot all be validated on an ordinary pull-request runner. The release process needs clear separation between deterministic CI gates, reference-hardware performance gates, and manual device gates.

## Decision

Adopt three release layers: pull-request CI, scheduled reference-Mac qualification, and protected signed release. The repository implements an initial functional CI layer now; the expanded matrix, reference-hardware gates, and signed-release automation below are required before the corresponding public release claim is made.

### Implemented initial pull-request CI

The checked-in `musica-vj.yml` workflow is configured with locked npm dependencies and no production secret. Its Linux frontend job is intended to execute:

```text
npm ci
npm run typecheck
npm run test:run
npm run verify:assets
npm run build
npm run verify:no-secrets
npm run verify:samples
npm run verify:samples:reproducible
```

The sample steps install CI `ffmpeg`; it is not bundled with the application. The checked-in files are verified before regeneration, and regenerated SHA-256 values must match. The configured `macos-15` bootstrap job installs and builds the frontend, runs .NET 8 companion unit tests, formats Rust, runs targeted Musica library tests plus `musica-vj` Clippy and tests, and invokes `tauri build --bundles app`. It is configured to upload the generated workspace `Cargo.lock`, formatted Rust sources, and the resulting unsigned `.app` as bootstrap/functional artifacts. Native platform icon bundles are generated from the tracked source PNG, so derived ICNS and other platform icon files remain outside version control. None of this is evidence until a referenced run completes successfully, and even a green run is not evidence of Intel compatibility, reference-hardware performance, signing, or notarization.

### Required pull-request CI expansion

Before a public release, the functional gate expands to the following locked command set or exact workspace-equivalent commands:

```text
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
npm ci
npm run typecheck
npm run test:run
npm run build
npm run tauri build -- --no-bundle
```

The target macOS build matrix covers Apple Silicon and Intel whenever GitHub-hosted runner availability permits. At minimum, the oldest supported deployment target is compiled and one current macOS runner executes tests. The application declares macOS 13.0 as its minimum system version.

As each feature ships, CI must add:

- TypeScript units for deterministic music helpers, track state, controller mappings, MIME negotiation, presets, receipts, and provider state machines;
- Rust units for request validation, path tokens, provider secret-source isolation, redirect policy, receipts, controller-token metadata, replay defense, and command DTOs;
- contract fixtures between TypeScript request types and Rust command structs;
- integration tests for six-track transport, live-recorder state, and security denials, plus offline-job and save/reopen tests only when those roadmap features ship;
- stable visual reference images on a software-controlled same-backend fixture;
- generated Tauri capability schemas and production CSP;
- exact package and crate lockfiles, license inventory, and vulnerability audit;
- compressed frontend and unsigned bundle size budgets.

No pull-request workflow may receive Apple signing certificates, provider keys, controller tokens, or notarization credentials. Forked code therefore cannot exfiltrate release secrets from the release environment.

### Reference-Mac qualification

Before public performance claims, a scheduled or manually dispatched workflow must run on a dedicated Apple M1, 16 GB reference Mac with a wired audio path. Results are retained as machine-readable artifacts and compared to the previous release. This qualification workflow is a release target, not part of the current initial CI.

| Budget | Release gate |
|---|---:|
| Cold start to interactive, 20 runs | less than 2.0 seconds p95 |
| Idle resident memory | less than 250 MB after warm-up |
| 10-minute active resident memory | less than 500 MB with no upward leak over final 5 minutes |
| Controller action to sound, wired | less than 35 ms p95 and 60 ms p99 |
| Scheduler late events | fewer than 1 per 10,000 under stress |
| 60-minute audio soak | zero audible underruns |
| Interactive visual frame time | less than 20 ms p95; fewer than 1% over 33 ms |
| 1080 by 1920 live capture | at least 29.7 average FPS and no more than 1% dropped frames |
| 30-second live capture A/V drift | no more than 33 ms |
| Deterministic 60-second export A/V drift | no more than 10 ms when that path ships |
| Compressed DMG without sample packs | less than 35 MB |

The initial stress profile plays all six tracks, cycles scenes, records, and receives a fixed MIDI/controller trace. When the offline Musica background-job roadmap in ADR-161 ships, the profile adds one such job. A result outside budget blocks release unless this ADR is amended with evidence and a new target; it is not waived as “machine noise.” Three repeated runs distinguish regression from transient load.

### Manual hardware and media qualification

Before a public release candidate is promoted:

1. Test current Options+ and the pinned Logi Actions SDK on a physical MX Creative Keypad and Dialpad.
2. Test the F13 through F24 fallback with the plugin removed.
3. Verify wired speakers or headphones and one Bluetooth device, recording the external Bluetooth latency separately.
4. Capture and decode every social preset on the oldest and current supported macOS versions.
5. Open both six-second sample fixtures in QuickTime and upload private test posts to currently supported social test accounts when available. Add 15- and 30-second preset captures once those app-driven export fixtures exist.
6. Test provider credential unavailable, provider offline, disk full, controller disconnect, display sleep, and WebGL context loss. Add app-restart-during-job and locked-Keychain tests when those roadmap features ship.

The hardware checklist records device firmware, Options+ version, Mac model, macOS build, audio device, detected recorder MIME, and result hashes.

### Required signed release

No signed or notarized release workflow exists in the initial implementation. Before public desktop distribution, releases must be produced only from a protected `vj-v*` tag in a release environment requiring approval. Dependencies are rebuilt from lockfiles; CI artifacts from an untrusted job are not simply re-signed.

The release build:

1. creates a universal Apple binary when supported by dependencies, otherwise separate signed `aarch64` and `x86_64` artifacts;
2. uses a Developer ID Application identity and hardened runtime;
3. signs every nested executable and framework with a secure timestamp;
4. creates the application and DMG through Tauri;
5. submits the final artifact to Apple's notary service using an App Store Connect API key held by the protected environment;
6. waits for acceptance and fails on any notarization issue;
7. staples the ticket to distributable artifacts;
8. verifies with `codesign --verify --deep --strict`, `spctl --assess`, and `stapler validate` on a clean Mac;
9. emits SHA-256 checksums, CycloneDX SBOMs for Rust and npm, license reports, test summaries, performance artifacts, and sample-video receipts;
10. creates release notes with known capability differences by macOS and detected media codec.

The Logitech `.lplug4` is packaged and verified separately with `LogiPluginTool`. It is not hidden inside the application signature as an auto-installed executable. Marketplace submission follows Logitech's review process and version compatibility is recorded independently.

Any future application updater uses a separate Tauri update signing key. Apple code signing and notarization do not replace the update signature, and the update private key is not stored in the repository or general CI.

### Reproducibility and versioning

When project persistence ships, projects must record app version, project schema, audio engine version, visual engine version, and asset hashes. When deterministic export receipts ship, they additionally record encoder and OS versions. Migration tests then open fixtures from every released project schema and verify that saving is idempotent.

JavaScript tests use seeded randomness and fake time except for explicit real-time tests. Performance numbers always identify hardware and OS. Codec output is allowed to vary across systems; sample count, logical frame count, dimensions, duration, and receipt hashes remain testable invariants.

## Alternatives Considered

### Linux-only CI for speed

This misses WKWebView, macOS deployment target, Tauri bundle, macOS secret storage, and signing configuration errors. Rejected.

### Run performance gates on shared GitHub runners

Shared hosts have variable contention and cannot provide stable audio hardware or GPU evidence. Rejected as a release gate; they remain useful for functional builds.

### Sign every pull request build

This exposes high-value credentials to untrusted code and produces artifacts users may mistake for releases. Rejected.

### Notarize after publishing

Users could receive a Gatekeeper-rejected artifact and release assets could change after announcement. Rejected. Acceptance and stapling precede publication.

### Test only Apple Silicon

This reduces cost but contradicts an Intel-supported product claim. If Intel qualification becomes uneconomic, support must be explicitly removed in a future ADR rather than silently untested.

### Treat sample videos as visual demos only

A beautiful clip can still have broken audio, wrong codecs, or drift. Rejected. The current synthetic fixtures are decoded and measured with `ffprobe`; receipt hashing becomes mandatory when the governed receipt format in ADR-165 ships.

## Consequences

### Positive

- Release claims are tied to measurements and artifacts.
- Signing secrets never enter ordinary pull-request jobs.
- Audio, visual, controller, and capture regressions have independent gates.
- The release target gives users notarized artifacts that Gatekeeper can verify offline after ticket stapling.

### Negative

- A complete qualification run takes at least 60 to 90 minutes plus notarization time.
- Intel and oldest-macOS testing require maintained hardware or paid runners.
- Physical Logitech checks remain partly manual until a simulator is supplied by the vendor.

## Risks and Mitigations

The largest process risk is a green functional build being mistaken for release readiness. The planned protected environment requires reference-Mac and physical-device artifact IDs as inputs, and the release checklist is stored with the tag. A human may approve an explained platform outage only after this ADR is amended; signature, notarization, checksum, and security tests are not bypassable public-release gates.

Apple, GitHub, and Logitech can change tooling independently. The target release system pins tool versions, runs weekly smoke tests, and exercises credentials with a non-publishing dry run before expiration.

## Rollback

A failed candidate is never published over the previous release. If a published version has a critical regression, revoke or remove it from the download channel, disable affected provider adapters, and republish the last good source as a new signed patch version. When project persistence ships, migrations are forward-only and must preserve an untouched backup before first save, allowing users to reopen with the previous version when the schema did not require an irreversible change.

## Acceptance Tests

1. A clean checkout with no production secrets must pass both jobs in `musica-vj.yml`: the Linux frontend/typecheck/test/build/sample-fixture path and the `macos-15` frontend plus .NET companion tests, targeted Rust check/test, and unsigned-app path. Frontend tests include the under-five-second 180-second stereo analysis fixture and MP4 fixtures that require H.264/AAC `stsd` sample entries. Until a workflow URL and successful run are recorded, this criterion is pending rather than “green today.”
2. Before a public performance release, a deliberately over-budget bundle, frame test, audio-latency test, drift fixture, or memory leak blocks its corresponding reference-Mac gate.
3. Pull-request jobs cannot enumerate, decrypt, or request protected signing and provider secrets.
4. Before any public `vj-v*` tag, a candidate cannot reach publication without reference-Mac artifacts and the physical hardware checklist.
5. Before public distribution, the final app and DMG pass `codesign`, Gatekeeper assessment, notarization log inspection, and staple validation on a clean Mac.
6. Before public distribution, SHA-256 checksums match downloaded artifacts, and SBOM and license reports identify every shipped Rust crate, npm package, and native companion component.
7. When project persistence ships, every released project fixture opens and saves twice without a second byte change to normalized project metadata.
8. The two current six-second, 360 by 640, 30 FPS H.264/AAC synthetic sample fixtures decode with video and audio, contain exactly 180 delivered frames, stay within 20 ms stream-duration drift and 1 MB each, and regenerate to identical SHA-256 values; they are CI previews, not app-capture or deterministic-export evidence.
9. Removing the Logitech plugin still leaves all critical actions reachable through the documented fallback.

## References

- [Tauri 2 macOS code signing and notarization](https://v2.tauri.app/distribute/sign/macos/)
- [Tauri 2 macOS application bundle distribution](https://v2.tauri.app/distribute/dmg/)
- [Apple: Notarizing macOS software before distribution](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
- [Apple: Hardened Runtime](https://developer.apple.com/documentation/security/hardened-runtime)
- [Apple: Resolving common notarization issues](https://developer.apple.com/documentation/security/resolving-common-notarization-issues)
- [GitHub Actions runner images](https://github.com/actions/runner-images)
- [GitHub Actions encrypted secrets](https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions)
- [CycloneDX specification](https://cyclonedx.org/specification/overview/)
- [Logitech Marketplace approval guidelines](https://logitech.github.io/actions-sdk-docs/marketplace-approval-guidelines/)
- ADR-161: Web Audio sample-clock scheduling
- ADR-163: Three.js visual performance
- ADR-165: Live capture and deterministic export
- ADR-166: Desktop threat model
- ADR-168: Lyria 3 Pro provider, routing, and prompt contract
- ADR-169: Lyria paid job lifecycle, assets, analysis, and provenance
