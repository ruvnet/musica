# ADR-145: ClipCannon → Musica — Rust Exploration for Realtime Avatar & Analysis

## Status
Accepted

## Date
2026-04-12

## Context

[ClipCannon](https://github.com/mrjcleaver/clipcannon) is a Python/CUDA MCP server
that ships an impressive 23-stage video-analysis pipeline, a LatentSync-based
lip-sync avatar engine ("Phoenix"), a voice-clone engine with SECS verification,
and a "Jarvis" realtime voice agent that joins Google Meet. It exposes 54 MCP
tools and targets NVIDIA GPUs with 8–32 GB VRAM.

The **analysis+avatar** slice of ClipCannon is compelling for musica because it
demands exactly the properties musica already provides:

| ClipCannon requirement              | Musica asset                                                                 |
|-------------------------------------|------------------------------------------------------------------------------|
| Realtime audio understanding (ASR↔TTS gap must be bridged in <50 ms)         | Musica `hearing_aid` streams at 0.20 ms/frame (sub-millisecond separation)   |
| Per-frame lip-sync coefficients from audio                                    | Musica STFT + graph features give per-frame spectral envelope for free       |
| Speaker embedding / diarization                                               | `crowd.rs` tracks thousands of speakers distributively                       |
| Emotion / prosody                                                             | STFT magnitudes + Fiedler partition already expose voicing/energy/pitch      |
| Edge / embedded / WASM deployment                                             | Zero-runtime-dep core; WASM bridge exists                                    |
| Interpretability / regulatory surface                                         | Graph partition is fully explainable                                         |

ClipCannon's Python+CUDA stack is unreachable on the edge: it needs ≥8 GB VRAM,
CUDA 12.1, an Ollama host, FluidSynth, and a 31-table SQLite-vec index per
project. A full port would be a multi-year effort and duplicate upstream work.

### The question

> *Is a full Rust rewrite of ClipCannon worth doing?*

**No.** Most of ClipCannon's value sits in diffusion models (LatentSync 1.6,
ACE-Step, Wav2Vec2, SigLIP, Qwen3-TTS) that we cannot reproduce and should not
try to. A faithful clone would be a thin wrapper around the same Python
libraries, adding no value.

### The better question

> *Which ClipCannon stages have a latency profile that Python+CUDA cannot hit,
> where a zero-dep Rust implementation unlocks a capability upstream cannot?*

Exactly three:

1. **Realtime prosody & voicing analysis** — 2 ms frame rate, <0.5 ms budget.
   Wav2Vec2-large cannot run this fast; musica STFT already does.
2. **Audio-driven viseme / jaw-open coefficients** — feeding a lip-sync avatar
   at 60 fps from a streaming mic. LatentSync needs video input; we generate
   the driving signal from audio features alone at sub-ms latency.
3. **Low-footprint speaker embedding** — ECAPA-TDNN is 2048-dim and a 6 MB
   model; musica's graph-based speaker fingerprint is ~64-dim and 0 bytes.

These three form a self-contained *bounded context* we can implement cleanly in
Rust inside this crate. The rest of ClipCannon (editing, rendering, billing,
dashboard, MCP server) is out of scope.

## Decision

Create a new `src/clipcannon/` module that **is not a port of ClipCannon** but
rather a *realtime analysis & avatar-driving subsystem inspired by the analysis
layer of ClipCannon*, designed to interoperate with a ClipCannon deployment via
a narrow well-typed interface.

The module:

- Follows the same `AudioProcessor` trait, `AudioBlock` in-place contract, and
  inline `#[cfg(test)]` test style as `hearmusica/`.
- Adds **zero runtime dependencies** (honours the musica zero-dep ethos).
- Targets <1 ms per 128-sample block at 16 kHz.
- Exposes features as plain-old-data structs that can be serialised into
  ClipCannon's SQLite/sqlite-vec tables over an FFI or MCP boundary *if* a user
  chooses to wire it up — but that wiring lives outside this crate.

### Scope — IN

| Capability                       | Upstream analogue                   | Musica implementation                                   |
|----------------------------------|-------------------------------------|---------------------------------------------------------|
| Streaming prosody features       | Wav2Vec2 prosody stage 12           | `clipcannon/prosody.rs` — F0/energy/voicing/ZCR/centroid|
| Viseme / jaw-open coefficients   | Phoenix avatar driving signal       | `clipcannon/viseme.rs` — 8-class viseme + jaw_open      |
| Lightweight speaker fingerprint  | ECAPA-TDNN 2048-dim                 | `clipcannon/speaker_embed.rs` — 64-dim spectral MFCC-ish|
| Streaming realtime pipeline      | "Jarvis" ASR→LLM→TTS front-end      | `clipcannon/pipeline.rs` — ring-buffered block runner   |
| Analysis DAG aggregation         | 23-stage DAG stages 4,5,9,10,12,15  | `clipcannon/analysis.rs` — struct-of-arrays frame record|

### Scope — OUT

- Transcription (musica has `transcriber.rs` behind a feature flag already).
- Scene detection / OCR / motion vectors (video, not audio).
- Diffusion (LatentSync, ACE-Step) — remains Python+CUDA upstream.
- MCP server — upstream hosts the tools; musica is a library.
- Billing, dashboard, Stripe, HMAC credits.
- Voice cloning (SECS verification) — upstream model-specific.

### Why Not Fork ClipCannon Python Directly?

| Concern                  | ClipCannon (Python/CUDA)              | Musica Rust realtime subsystem            |
|--------------------------|----------------------------------------|-------------------------------------------|
| Frame latency            | 30–80 ms (Wav2Vec2 + PyTorch overhead) | <1 ms (pure arithmetic on f32)            |
| Memory per session       | 2–6 GB VRAM                            | <1 MB heap                                |
| Deployment               | NVIDIA GPU + CUDA 12.1                 | Any CPU, WASM, MCU                        |
| License                  | BSL 1.1 (commercial-hostile until '30) | Dual MIT/Apache-2.0                       |
| Dependency               | ~40 pip packages                       | 1 crate (`ruvector-mincut`)               |
| Explainability           | Opaque embeddings                      | Graph partition + explicit features       |

## Integration Boundary

A ClipCannon deployment that wants to use this subsystem sees a single
`RealtimeAvatarAnalyzer` type that accepts `AudioBlock`s and emits
`AvatarFrame`s. An optional `--feature clipcannon-bridge` can later be added to
serialise those frames into ClipCannon's per-project SQLite schema, but the
bridge lives in a separate crate and is **not** part of this decision.

```text
┌──────────── ClipCannon (Python/CUDA) ─────────────┐
│                                                    │
│  ┌──────────────┐   ┌─────────────┐   ┌────────┐  │
│  │ LatentSync   │◀──│ viseme +    │   │ SQLite │  │
│  │ diffusion    │   │ jaw_open    │   │ + vec  │  │
│  └──────────────┘   └──────▲──────┘   └───▲────┘  │
│                            │              │       │
└────────────────────────────┼──────────────┼───────┘
                             │              │
                       AvatarFrame     AnalysisFrame
                             │              │
┌────────────────────────────┼──────────────┼───────┐
│        Musica Rust realtime subsystem             │
│  ┌──────────────────────────────────────────────┐ │
│  │ clipcannon::RealtimeAvatarAnalyzer           │ │
│  │   STFT → prosody → viseme → speaker_embed    │ │
│  └──────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

## Consequences

### Positive

- Unlocks sub-ms realtime analysis for ClipCannon's avatar & voice-agent tracks.
- Zero additional runtime dependencies; preserves musica's edge story.
- Module is standalone — anyone running a hearing aid, meeting bot, or lip-sync
  avatar gets the same `AudioProcessor`-based API.
- All outputs are explainable (graph partition + named features), which is a
  differentiator against Wav2Vec2 black-boxes.

### Negative

- Emotion classification will be coarser than a 1024-dim Wav2Vec2 embedding.
  We ship **categorical prosody buckets**, not a continuous emotion vector.
- Speaker fingerprint is not interchangeable with ECAPA-TDNN 2048-dim; a bridge
  adapter would be needed for existing ClipCannon voice profiles.

### Risks

- The driving signal for a LatentSync avatar is *trained* on specific prosody
  features; ours may need calibration before it lip-syncs cleanly.
- Viseme classification from audio alone (no phoneme recogniser) will miss
  some bilabials — we accept this as a v1 trade-off.

## References

- ClipCannon: https://github.com/mrjcleaver/clipcannon (BSL 1.1)
- ADR-143: HEARmusica — Tympan Rust port (establishes `AudioProcessor` trait)
- ADR-144: Candle-Whisper transcription integration
- ADR-146: Realtime Avatar Pipeline — DDD Architecture (sibling)
- ADR-147: Streaming Prosody, Emotion & Speaker Analysis (sibling)
- ADR-148: Benchmark Methodology & Performance Targets (sibling)
