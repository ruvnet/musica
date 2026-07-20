# ADR-176: Performance memory (LEARNING) and ruv ecosystem review

- Status: Accepted
- Date: 2026-07-20

## Context

ADR-175 shipped four Cognitum-backed capabilities (style packs, set arcs, Auto DJ phrase briefs, FX/visual mood direction) with local fallbacks. The LEARNING capability chip had no substance yet, and three ecosystem components were flagged for integration review: `npx metaharness`, `npx ruvector`, and `ruvnet/musicai`.

## Decision

### Performance memory (implemented)

`src/core/performanceMemory.ts` records every AI direction the operator actually uses (`ai-look`, `fx-mood`, plus room for `ai-style`/`set-arc`) into IndexedDB with a bounded 400-entry LRU, and recalls the closest past direction by token-overlap similarity with a 0.45 threshold and recency tiebreak. When Cognitum is unavailable, AI LOOK and AI MOOD consult memory before the keyword fallback — so the app genuinely learns the operator's vocabulary over time, entirely offline. The LEARNING chip shows the remembered-entry count.

The interface is deliberately vector-shaped (record → recall-by-similarity) so the scorer can be replaced without touching call sites.

### Ecosystem review

| Component | Verdict | Rationale |
|---|---|---|
| `ruvector` (npm 0.2.35) | **Adopt later, interface ready** | Self-learning vector DB (HNSW, hybrid search). The natural upgrade for `performanceMemory`'s scorer once embeddings are available in the webview; native Node bindings do not run in WKWebView, so integration waits for its wasm build to stabilize or for a Rust-side ruvector behind a Tauri command. |
| `metaharness` (npm 0.4.1) | **Not a runtime dependency** | It mints agent harnesses (dev tooling). Useful for repo automation, not for the performance app itself. |
| `ruvnet/musicai` | **Concepts only** | Raspberry Pi real-time auto-tune/enhancement in Rust using ruvector. Its pitch/timing-correction DSP is interesting for a future vocal-deck enhancement path, but it targets Pi hardware and duplicates the mastering chain Musica already has. Revisit if live input processing becomes a goal. |

## Consequences

- LEARNING is real and offline-first; Cognitum remains optional.
- Memory never blocks a performance: all failures degrade to the keyword fallbacks.
- A ruvector swap-in is a contained change to one module.
