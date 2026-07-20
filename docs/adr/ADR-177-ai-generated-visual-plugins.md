# ADR-177: AI-generated visual plugins

- Status: Accepted
- Date: 2026-07-20

## Context

Musica ships nine built-in scenes rendered by hand-written Three.js code. The operator has asked for AI-generated visual plugins/extensions: describe a look, get a *new* scene — not just new parameters for an existing one. The standing architecture directive says advanced or generated runtime components must be WASM or declarative data, never eval'd JavaScript: the app is a distributable desktop binary whose webview must stay sandbox-safe.

## Decision

Introduce a **parametric scene specification** as the plugin format, with three tiers of ambition; tier 1 ships now.

### Tier 1 (implemented): declarative parametric scenes

A `VisualPluginSpec` is pure data interpreted by a new generic renderer inside `VisualEngine` — no code crosses the boundary:

```
{ name, base: "particles" | "ribbons" | "rings",
  count (50-4000), size, spread,
  motion: { orbit, pulse, drift, twist },      // 0-1 modulators mapped to audio bands
  audio: { bassTo: "scale"|"speed"|"none", highTo: "brightness"|"jitter"|"none" },
  colors: { primary, accent, background },      // validated hex
  fog, exposure }
```

`cognitum_visual_plugin` asks the Meta-LLM for a spec from a described look; Rust validates every field against hard ranges (counts, hex colors, enum values) before it reaches React. Specs persist in the workspace settings store, appear as user scenes ("P1", "P2"…) in the scene grid, and are removable. A curated local fallback generates specs from keywords when Cognitum is unavailable. The generic renderer draws each spec with additive-blended primitive groups driven by the same band-envelope/beat-pulse inputs as built-in scenes, so plugins inherit the audio reactivity and the feedback pass for free.

### Tier 2 (planned): WASM update kernels

A plugin gains an optional sandboxed WASM module exporting a pure `update(positions, time, bands) -> positions` kernel, instantiated with no imports (no WASI, no memory sharing beyond the positions buffer). This unlocks novel motion behavior beyond the parametric modulators while keeping the no-generated-JS invariant.

### Tier 3 (exploratory): shader fragments

AI-authored GLSL is the most expressive tier but the hardest to validate; it requires a shader sanitizer/timeout harness and is explicitly out of scope until tier 2 proves the pipeline.

## Consequences

- New scenes without new code paths: the spec interpreter is the only renderer addition, and validation bounds worst-case cost (≤4000 particles).
- Generated specs are inspectable, editable, exportable data — they ride the existing settings export/import and IndexedDB persistence.
- The plugin grid grows the scene vocabulary that set arcs and AI LOOK can reference.
- Tier 2's WASM sandbox contract (pure kernel, no imports) is the template for future DSP plugins as well.

## Acceptance tests

1. A spec with out-of-range count, malformed color, or unknown enum is rejected in Rust and never reaches the renderer.
2. Plugin scenes render with audio reactivity and respect the feedback pass and palette/hue controls.
3. Deleting a plugin scene while it is active falls back to a built-in scene without a crash.
4. Specs survive export → import round-trips unchanged.
