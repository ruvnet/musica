# ADR-163: Three.js WebGL2-First Visual Engine with Adaptive Quality

## Status
Accepted

## Date
2026-07-18

## Context

Musica VJ needs high-impact, audio-reactive visuals that remain smooth during live synthesis and 9:16 capture. The supported macOS range spans several Apple and Intel GPU generations. WebGPU is strategically important but its WKWebView availability and Three.js post-processing parity cannot be assumed across macOS 13 and later.

Visual overload must degrade image quality before it causes audio starvation or dropped controller input. Export resolution, meanwhile, must not silently change because an adaptive controller reacts to encoder load.

## Decision

Use Three.js `WebGLRenderer`, which targets WebGL 2, as the production renderer for v1. Pin Three.js exactly in the application lockfile and use local shaders and assets only. The implemented v1 renderer is a concrete `VisualEngine`; the backend abstraction and WebGPU experiment described below are migration work, not current features.

The visual engine is independent of React rendering. It owns one canvas, renderer, scene, camera, `EffectComposer`, and animation loop. React sends scene, intensity, and bounded artist-macro commands only when the performer changes a control. On each animation frame, the engine reads one immutable audio-metrics snapshot and updates GPU uniforms or preallocated geometry buffers. It never calls Tauri, a provider, or the filesystem in the frame loop.

### Initial scenes and budgets

| Scene | Primitive | Initial ceiling |
|---|---|---|
| Neon Fold | Reused torus geometry | 30 rings |
| Signal Bloom | Shader particles plus preallocated spectral trails and reflections | 7,000 particles, 18 trails, 128 points per trail |
| Spectral Field | Reused plane position buffer | 64 by 64 segments |

The shared scene adds a central mesh, ambient light, two point lights, exponential fog, a 1,600-point deterministic atmosphere, a dynamic 256-point waveform ribbon, a low-opacity floor grid, ACES tone mapping, and a bloom pass. Signal Bloom adds 18 preallocated spectral trails and 18 low-opacity floor reflections; these trail draws and their CPU geometry updates are disabled in the other scenes. Initial budgets remain fewer than 50 visible draw calls, fewer than 150,000 visible vertices, and less than 128 MB estimated GPU resources per scene. Scene switching toggles existing groups; it does not compile a shader or fetch an asset during a performance.

Frequency energy is reduced into low, mid, and high bands plus master level, waveform samples, and beat phase. Bass controls camera displacement and the spectral sculpture envelope, beat proximity controls the waveform excursion and radial pulse, and high-frequency energy controls particles and haze. Visual smoothing may interpolate those values, but it may not write back to transport or audio state. The audio path therefore cannot wait for a frame.

Signal Bloom also exposes four normalized artist macros so the renderer functions as a playable instrument instead of a fixed visualization. `sculpture` changes fan width and spectral relief, `motion` changes drift and camera movement, `atmosphere` changes haze and particle presence, and `ribbon` changes waveform excursion, glow, and reflection. Values are finite-clamped from zero to one before entering the render loop. Macro changes reuse existing uniforms and position buffers; they do not add geometry, draw calls, network work, or per-frame object allocation. Audio analysis and artist intent remain independent inputs, with the global reactivity control acting as the master audio-response amount.

### Adaptive quality

Interactive rendering starts at a pixel ratio of `min(devicePixelRatio, 1.75)`. The implemented frame-time exponential moving average controls pixel ratio with hysteresis:

1. If average frame time exceeds 20 ms for 90 consecutive frames, reduce pixel ratio by 0.18, down to 1.0.
2. If average frame time stays below 13.8 ms for 300 consecutive frames, increase it by 0.12, up to the initial ceiling.
Pixel ratio increase takes at least 300 frames, approximately five seconds at 60 FPS. A future second quality tier must, when pixel ratio is already 1.0 and p95 frame time remains above 25 ms for five seconds, reduce bloom resolution, then particle count, then terrain resolution. That deeper tier is not implemented in v1.

The UI reports FPS, smoothed frame time, pixel ratio, and whether quality is adaptive or export-locked. It does not report a synthetic 60 FPS when frames are being dropped.

When recording, both `WebGLRenderer` and `EffectComposer` lock to the preset's exact pixel dimensions at pixel ratio 1. Adaptive changes are disabled for the clip and it does not resize mid-clip. Keeping the composer ratio synchronized avoids allocating post-processing targets at the display DPR. A Three.js overlay scene renders the Musica wordmark, full scene name, and live-set label into the captured canvas while remaining hidden in the ordinary stage view. The interactive stage adds non-captured editorial telemetry, scanlines, edge color keys, and BPM/48 kHz readouts in DOM overlays; export branding remains canvas-native. V1 has no capability-performance preflight, so an overloaded machine may still drop frames. Failing before capture or offering a lower preset is a release-hardening requirement shared with ADR-165.

### WebGPU migration

The planned `VisualBackend` boundary keeps renderer-specific creation and post-processing separate while scene identifiers, deterministic seeds, audio metrics, palette, camera, and quality policy remain backend-neutral. Extracting the current `VisualEngine` behind this boundary is the first migration step.

Three.js `WebGPURenderer` will be developed behind an explicit experimental flag. No WebGPU flag or backend ships in v1, and it must not become an automatic production choice until all gates pass:

- supported on every currently supported macOS and WKWebView combination, or cleanly limited to an advertised subset;
- all three scenes have feature parity, including bloom and capture;
- reference-image structural similarity is at least 0.98 for fixed seeds;
- frame time is at least 15% better, or energy use at least 20% lower, on two representative Apple GPUs;
- no regression in 30-second capture drift or dropped-frame rate;
- a forced `webgl2` setting remains available for one full major release.

Three.js can use a WebGL 2 backend as part of its WebGPU renderer stack, but Musica VJ will not rely on that implicit fallback as its rollback mechanism. The production choice remains explicit and observable.

### Performance targets

The reference viewport is 1440 by 900 on an Apple M1, with the six-track engine active.

| Metric | Target |
|---|---|
| Interactive frame rate | at least 55 FPS over a 10-minute run |
| Interactive frame time | less than 20 ms p95; fewer than 1% above 33 ms |
| Scene-switch stall after warm-up | less than 20 ms p95 |
| Per-frame JavaScript allocation | less than 2 KB average after warm-up |
| GPU/context loss | recover scene within 2 seconds without stopping audio |
| 1080 by 1920, 30 FPS capture | no more than 1% dropped frames on reference M1 |

## Alternatives Considered

### WebGPU first

WebGPU offers better compute and modern resource management, but WKWebView support, shader tooling, and post-processing behavior vary across the supported OS range. Rejected for the production default until the migration gates pass.

### Raw WebGL 2

Raw WebGL reduces abstraction overhead but requires building scene graphs, resource disposal, materials, render targets, and post-processing infrastructure. The expected engineering cost is several person-months without a differentiating product benefit. Rejected.

### PixiJS

PixiJS is strong for 2D sprites and compositing but does not fit the 3D tunnel, terrain, lighting, and shader roadmap as directly. Rejected as the primary engine; it can remain an asset-generation tool.

### Canvas 2D or CSS animation

These maximize compatibility but cannot meet the scene density or shader aesthetic at the target frame rate. Rejected.

### Native Metal

Metal would provide the best macOS control but would duplicate the full scene system and remove browser preview. Deferred as an optional future export backend rather than the live renderer.

## Consequences

### Positive

- A mature scene and post-processing ecosystem supports rapid visual iteration.
- WebGL 2 covers the broadest declared macOS range.
- Explicit budgets and adaptive quality protect the audio product.
- Backend-neutral state creates a measurable WebGPU path rather than a rewrite promise.

### Negative

- Three.js and post-processing contribute materially to bundle size.
- WebGL context loss and driver behavior remain platform-dependent.
- Export locking can expose a performance failure that adaptive interactive mode hides.

## Risks and Mitigations

The largest failure mode is a visually impressive scene that saturates the GPU and makes the encoder or webview miss frames. New scenes must declare geometry, draw-call, render-target, and shader-compilation budgets. They are warmed before use and enter the adaptive stress suite before release.

V1 does not yet implement explicit WebGL context-loss reconstruction. Before public release, CPU-side scene parameters and deterministic seeds must be retained so resources can be reconstructed. If reconstruction fails twice in a session, the application must fall back to a low-cost diagnostic scene while audio continues.

## Rollback

V1 is fixed to WebGL 2. Any release that introduces WebGPU must retain a `webgl2` backend setting for at least one full major release. A problematic scene can be disabled independently through the local scene registry, and quality can be capped at pixel ratio 1.0 without changing projects. WebGPU code remains feature-gated until its acceptance gates pass.

## Acceptance Tests

1. Fixed seed, audio metrics, viewport, and time produce stable reference images for every scene within a structural-similarity threshold of 0.98 on the same backend.
2. A 10-minute reference run meets the frame-time, allocation, draw-call, and vertex budgets while all six tracks play.
3. V1 artificial GPU pressure reduces pixel ratio by the specified increments and never below 1.0; recovery requires at least 300 stable frames. Before the deeper adaptive tier ships, tests must verify bloom, particles, and terrain degrade in that order.
4. Export lock renders both the renderer and post-processing targets at exactly 1080 by 1920 and pixel ratio 1 for the full clip, includes the canvas branding layer, and never changes quality mid-recording.
5. Before public release, context-loss injection reconstructs the active scene within two seconds while audio transport continues.
6. Scene switching after warm-up causes no network request, shader compilation over 20 ms p95, or undisposed growth across 1,000 switches.
7. Experimental WebGPU cannot become the release default unless all migration gates are recorded in benchmark artifacts.

## References

- [Three.js WebGLRenderer](https://threejs.org/docs/#api/en/renderers/WebGLRenderer)
- [Three.js WebGPURenderer](https://threejs.org/docs/#api/en/renderers/webgpu/WebGPURenderer)
- [W3C WebGL 2 specification](https://www.w3.org/TR/webgl2/)
- [W3C WebGPU specification](https://www.w3.org/TR/webgpu/)
- [Three.js post-processing manual](https://threejs.org/manual/en/post-processing.html)
- ADR-161: Web Audio sample-clock scheduling
- ADR-165: Live capture and deterministic export
