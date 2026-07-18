# ADR-165: Live Social Capture and Deterministic Export Roadmap

## Status
Accepted

## Date
2026-07-18

## Lyria 3 Pro amendment

A completed Lyria asset enters the same Web Audio track and master capture destination as local synthesis and imported audio. The provider does not receive the visualization or rendered video, and the export path does not depend on Lyria being available after the asset is staged.

The Lyria integration does not promote the current recorder to deterministic export. The packaged Tauri app fails closed unless the physical macOS WKWebView reports an MP4 recorder, then parses the ISO box hierarchy and requires H.264 (`avc1` or `avc3`) plus AAC (`mp4a`) as entries in an `stsd` sample-description box before reporting success. This is stronger than scanning arbitrary bytes for codec text but remains shallow structural inspection. WebM remains a browser-development fallback only. Actual encoded dimensions, delivered frame rate, timestamps, decoder integrity, and A/V drift are not inspected, so synchronized 1080 by 1920 MP4 acceptance still requires a manual physical-Mac decode and drift gate. The checked-in 360 by 640 sample videos remain synthetic CI fixtures and are not evidence of a real Lyria-to-Three.js export. A deterministic fixed-timestep 1080 by 1920 MP4 pipeline is still deferred.

## Context

The product loop is performance, capture, post. TikTok and Instagram favor fullscreen vertical media, fast hooks, and clips that work immediately without a separate editor. Live capture must therefore be one action from the Logitech console or UI.

WebKit's `MediaRecorder` output is capability-dependent. Container names do not guarantee particular codecs, frame delivery is driven by real time, and encoder availability can change with macOS. A live recording can be synchronized enough to post without being deterministic or bit-reproducible. That distinction must be explicit.

TikTok's official in-feed guidance recommends 9:16 vertical media and 9 to 15 second creative. Meta recommends 9:16 for fullscreen placements and reserves edge regions for interface overlays. Platform requirements can change independently of the application.

## Decision

Ship a capability-probed live recorder in v1 and define a separate fixed-timestep deterministic export path for the next release stage.

### Live capture

The live path combines:

1. `HTMLCanvasElement.captureStream(preset.fps)` from the Three.js canvas;
2. the master mix from `MediaStreamAudioDestinationNode`;
3. one `MediaStream` containing those video and audio tracks;
4. `MediaRecorder` with 500 ms chunks;
5. an explicit stop/finalize/save state machine.

The recording indicator, elapsed progress, preset, and stop control remain visible throughout capture. `start()` moves synchronously from `idle` to `starting` before it awaits the first animation frame, so repeated button presses cannot create overlapping recorders. Stop/finalization has a 15-second deadline and transitions back to `idle` on encoder, inspection, synchronous-stop, or timeout failure.

The application probes `MediaRecorder.isTypeSupported()` at runtime in this order:

```text
video/mp4;codecs=h264,aac
video/mp4
video/webm;codecs=vp9,opus
video/webm;codecs=vp8,opus
video/webm
```

The recorder chooses the first supported value. If none is reported, capture fails closed instead of guessing a container. The packaged Tauri app additionally refuses to start when the selected type is not MP4; WebM fallback is limited to browser development. Setup requires at least one canvas video track and one master audio track. The result must be non-empty and match its declared container. MP4 inspection requires a top-level ISO `ftyp`, bounded box nesting, and `avc1`/`avc3` plus `mp4a` sample entries inside `stsd`; codec-looking strings elsewhere in the file do not count. WebM inspection requires an EBML signature and reports VP8/VP9 and Opus markers when present. The result records container, detected codecs, and target dimensions and FPS. Every setup, encoder, synchronous stop, inspection, and 15-second finalization timeout failure unlocks export resolution, stops the capture video track, clears timers, and notifies the UI. V1 still does not parse actual encoded dimensions, delivered frames, or stream timestamps and does not run a complete post-record decode probe.

Audio is requested at 256 kbps. Initial social presets are:

| Preset | Dimensions | FPS | Duration | Video bitrate | Approximate payload |
|---|---:|---:|---:|---:|---:|
| 6-second loop | 1080 by 1920 | 30 | 6 s | 10 Mbps | 7.7 MB |
| 9-second hook | 1080 by 1920 | 30 | 9 s | 10 Mbps | 11.5 MB |
| 15-second reel | 1080 by 1920 | 30 | 15 s | 10 Mbps | 19.2 MB |
| 30-second reel | 1080 by 1920 | 30 | 30 s | 10 Mbps | 38.5 MB |
| 15-second square | 1080 by 1080 | 30 | 15 s | 8 Mbps | 15.5 MB |

Payload estimates include requested audio bitrate but exclude container overhead. The six and nine second presets optimize for seamless visual and musical loops; the 15-second preset maps to TikTok's stated 9 to 15 second recommendation. A 60 FPS vertical preset remains hidden until the reference hardware meets the capture gate.

V1 does not auto-publish or store social credentials. During capture, a Three.js overlay scene draws the Musica wordmark, active scene, and live-set label inside the recorded canvas. Stored Meta/TikTok edge-mask guides and automatic title-safe checks remain roadmap UI.

### Implemented deterministic sample fixtures

Repository sample videos are generated by `scripts/render-samples.mjs` with an external CI FFmpeg installation. Fixed filter seeds, one encoder thread, bit-exact flags, and stripped metadata create two bit-reproducible six-second, 360 by 640, 30 FPS vertical previews with synthetic 48 kHz stereo audio. `verify-samples.mjs` uses `ffprobe` to require H.264 video, AAC audio, exact preview dimensions, exactly 180 delivered frames, duration from 5.95 to 6.05 seconds, no more than 20 ms stream-duration drift, and less than 1 MB per file. A second script regenerates the fixtures and requires identical SHA-256 values.

These previews validate repository media generation and provide shareable examples. They are not recordings of the Three.js canvas, do not use the application's `MediaRecorder`, and do not mean deterministic desktop export is implemented. FFmpeg is installed in CI for sample generation and is not bundled in the Tauri app.

### Deterministic export roadmap

The implemented live `RecordingResult` contains Blob, MIME, filename, measured duration, and byte size. It has no export receipt. A future deterministic export is a different mode with these invariants:

- freeze the project, asset hashes, engine version, visual seed, scene parameters, automation, BPM map, sample rate, dimensions, FPS, color space, and requested codec into an export manifest;
- render audio offline from frame zero to an exact integer sample count;
- render visual frame `i` at logical time `i / fps`, never from wall-clock time or `requestAnimationFrame()`;
- render exactly `ceil(durationSeconds * fps)` frames at the preset dimensions;
- mux the fixed audio and frame streams with timestamps derived from the same rational timeline;
- record content hashes and encoder settings in the export receipt.

The macOS roadmap should use AVFoundation `AVAssetWriter` and VideoToolbox through a narrow native module, avoiding a bundled FFmpeg dependency by default. Hardware H.264 output is not promised to be byte-identical across GPU or OS versions. Here, deterministic means the same frame count, logical visual state, audio samples, timestamps, and manifest under the same engine build. A lossless frame-hash mode is a required test before promotion.

The future release gate for deterministic 1080 by 1920, 30 FPS export is no more than 1.5 times clip duration on an M1, A/V drift no greater than 10 ms over 60 seconds, and no missing or repeated logical frame. Pause and cancellation are also roadmap requirements.

### Codec and legal constraints

MP4 is a container; H.264 and AAC are codecs. Support is tested independently. TikTok and Meta documentation can guide presets but does not guarantee that a particular organic upload will be accepted. The saved result displays container, detected codecs, target dimensions and FPS, wall-clock duration, and size. Actual encoded dimensions, delivered frame rate, and timestamps remain properties of the physical decode gate.

The project does not bundle FFmpeg until an inventory identifies every enabled component's LGPL/GPL obligations, redistribution terms, and codec-patent implications. Using Apple's system encoder avoids distributing that encoder but does not transfer music rights to the user.

Future governed exports include the target asset receipts from ADR-164. V1 recordings do not claim that provenance. The user remains responsible for music, sample, voice, likeness, trademark, and platform rights. The app does not label generated or imported audio royalty-free.

## Alternatives Considered

### Native deterministic export only

This produces the strongest output but delays the viral performance loop and requires a larger native media implementation. Rejected as the only v1 path.

### MediaRecorder only

This is sufficient for live posting but cannot guarantee exact frame count, portable H.264/AAC, reproducible timing, or stable encoder behavior. Accepted for live capture, rejected as the final export architecture.

### Bundle FFmpeg immediately

FFmpeg supports broad codecs and containers, but build flags determine LGPL or GPL obligations and codec licensing exposure. It would also materially increase the app and release matrix. Deferred pending a documented component and license review.

### Record the display with macOS ScreenCaptureKit

This can capture overlays, other windows, notifications, and color transformations outside the scene. It also requires screen-recording permission. Rejected for canvas-only social output; useful later for tutorials.

### Upload directly to social networks

This adds OAuth secrets, changing platform scopes, account risk, moderation, and accidental publication. Rejected for v1. Save and hand off through the user's normal posting workflow.

## Consequences

### Positive

- A physical record button creates a postable clip with no DAW or editor.
- Six, nine, and 15-second vertical formats support loop and hook experimentation.
- Runtime codec probing avoids false MP4 claims.
- The deterministic roadmap has checkable timing invariants and a macOS-native encoder path.

### Negative

- A packaged Mac whose WKWebView cannot produce the required MP4 fails closed; only browser development can return WebM.
- The implemented live recorder has no deterministic or post-decode guarantee.
- 1080 by 1920 rendering and encoding can force a lower visual quality than the interactive canvas.

## Risks and Mitigations

The largest v1 risk is a clip whose shallow MP4 inspection proves H.264 and AAC `stsd` entries but misses encoder corruption, wrong actual dimensions, dropped frames, or timestamp drift. V1 requires audio and video tracks, locks startup synchronously, requests chunks every 500 ms, bounds finalization to 15 seconds, rejects an empty or mismatched container, and records byte count, wall-clock duration, detected codecs, and target dimensions. It does not run a one-second preflight or a complete native decode and timestamp validation. Adding `ffprobe`-equivalent native inspection before presenting a production-ready success state remains the fix path.

Resource use is bounded by fixed presets, a maximum 30-second live preset, one active recorder, and 500 ms chunks. V1 has no 500 MB dynamic hard stop or capability-performance preflight. The renderer is resolution-locked under ADR-163; size enforcement and a lower-preset recommendation are hardening roadmap items.

## Rollback

If `MediaRecorder` regresses on a supported macOS release, disable live capture for that capability signature and retain performance mode. Users can route audio to an external recorder. Deterministic native export can ship independently behind the same `SocialPreset` and export-manifest contracts.

## Acceptance Tests

1. Unit fixtures require valid MP4 `ftyp` plus H.264 and AAC `stsd` sample entries, reject codec text outside `stsd`, and reject a declared MP4 that cannot prove both codecs. A physical packaged-Mac run must independently decode every preset and verify actual dimensions, frame rate, audio track, duration, and extension against the target metadata.
2. A 30-second reference capture has A/V drift no greater than one 30 FPS frame and no more than 1% dropped video frames on the M1 reference machine.
3. Unit tests prove unsupported MIME candidates are skipped, no platform default is guessed, MP4 is preferred, and an actual WebM MIME is never named MP4. The packaged Tauri code path refuses a WebM-only recorder; a packaged-runtime fixture remains required to exercise that platform branch.
4. Rapid record commands encounter the synchronous `starting` lock and create exactly one recorder; encoder, stop, inspection, and 15-second finalization-timeout errors release tracks and unlock visual resolution. Window-close and disk-failure behavior remain physical/integration gates rather than unit-test evidence.
5. V1 tests enforce the fixed 30-second maximum preset and one-recorder state. A byte hard stop is required before custom-duration recording ships.
6. CI first verifies at least two checked-in six-second 360 by 640 H.264/AAC sample previews under 1 MB, including exactly 180 frames and at most 20 ms stream-duration drift, then regenerates them and requires identical SHA-256 values; documentation states that they are FFmpeg fixtures, not app captures.
7. Before the safe-zone UI ships, its reference title and focal-object fixtures must remain unobscured by stored TikTok and Meta masks.
8. Before deterministic desktop export ships, fixtures must produce exact audio sample and logical frame counts, matching lossless frame hashes, and no more than 10 ms A/V drift over 60 seconds.

## References

- [W3C MediaStream Recording specification](https://www.w3.org/TR/mediastream-recording/)
- [W3C Media Capture from DOM Elements](https://www.w3.org/TR/mediacapture-fromelement/)
- [W3C Media Capture and Streams synchronization model](https://www.w3.org/TR/mediacapture-streams/)
- [TikTok Auction In-Feed Ads specifications](https://ads.tiktok.com/help/article/tiktok-auction-in-feed-ads)
- [TikTok Reservation In-Feed Ads specifications](https://ads.tiktok.com/help/article/tiktok-reservation-in-feed-ads-reach-frequency)
- [Meta aspect-ratio best practices](https://www.facebook.com/business/help/103816146375741)
- [Meta Reels and Stories safe-zone guidance](https://www.facebook.com/business/help/980593475366490)
- [Apple AVAssetWriter](https://developer.apple.com/documentation/avfoundation/avassetwriter)
- [Apple VideoToolbox](https://developer.apple.com/documentation/videotoolbox)
- ADR-163: Three.js visual engine
- ADR-164: Creative asset provenance
- ADR-168: Lyria 3 Pro provider and routing
- ADR-169: Lyria paid job lifecycle and assets
