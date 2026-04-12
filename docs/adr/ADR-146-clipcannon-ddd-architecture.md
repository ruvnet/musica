# ADR-146: ClipCannon Realtime Subsystem — DDD Architecture

## Status
Accepted

## Date
2026-04-12

## Context

ADR-145 defined *what* to build inside `src/clipcannon/`: a realtime
analysis & avatar-driving subsystem inspired by ClipCannon's analysis layer.
This ADR defines *how* to model it using Domain-Driven Design.

The problem domain mixes three very different vocabularies:

1. **DSP** — sample rate, window, hop, STFT frame, magnitude, phase.
2. **Phonetics / visual** — viseme class, jaw-open coefficient, emotion bucket.
3. **Identity** — speaker fingerprint, cosine similarity, session.

Mashing these into one "AvatarEngine" god-struct is how the Python stack ended
up with 31 SQLite tables cross-cutting every concern. Rust gives us the tools
to do better: disjoint types per bounded context, with explicit anti-corruption
layers between them.

## Decision

Model the subsystem as **four bounded contexts** connected by
**context-map relationships**, all inside the `clipcannon` module.

### Ubiquitous Language

| Term                  | Meaning                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| **Frame**             | One STFT analysis window (typically 16 ms @ 16 kHz)                     |
| **Block**             | One realtime audio buffer handed to `process_block` (typically 8 ms)    |
| **ProsodySnapshot**   | Per-frame prosody features (F0, voicing, energy, centroid, ZCR, roll-off)|
| **Viseme**            | Discrete mouth-shape class driving a lip-sync avatar (AA/EE/IH/OH/UW/FV/MBP/REST) |
| **AvatarFrame**       | The full packet emitted to a downstream lip-sync renderer               |
| **SpeakerFingerprint**| 64-dim spectral signature uniquely identifying a speaker                |
| **AnalysisFrame**     | The persistence-friendly record for ClipCannon's SQLite analysis DAG    |
| **Session**           | A contiguous conversation or recording belonging to one pipeline run    |

### Bounded Contexts

```text
┌────────────────────────────────────────────────────────────────────┐
│                       clipcannon module                            │
│                                                                    │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │ Signal Analysis  │───▶│ Avatar Driving   │                      │
│  │ (prosody.rs)     │    │ (viseme.rs)      │                      │
│  │                  │    │                  │                      │
│  │ ProsodySnapshot  │    │ AvatarFrame      │                      │
│  │ value objects    │    │ value objects    │                      │
│  └────────┬─────────┘    └──────────────────┘                      │
│           │                                                        │
│           │                                                        │
│           ▼                                                        │
│  ┌──────────────────┐    ┌──────────────────┐                      │
│  │ Speaker Identity │    │ Analysis DAG     │                      │
│  │ (speaker_embed)  │───▶│ (analysis.rs)    │                      │
│  │                  │    │                  │                      │
│  │ SpeakerFingerprint│   │ AnalysisFrame    │                      │
│  │ aggregate root   │    │ aggregate root   │                      │
│  └──────────────────┘    └──────────────────┘                      │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### Context 1: Signal Analysis (`prosody.rs`)

**Purpose**: convert raw samples into per-frame prosody.

**Aggregate root**: none. This context is purely functional; it produces
immutable `ProsodySnapshot` value objects from stateless function calls.

**Value objects**:

```rust
pub struct ProsodySnapshot {
    pub f0_hz:        f32,   // fundamental frequency, 0 if unvoiced
    pub voicing:      f32,   // [0,1] voicing probability
    pub energy_db:    f32,   // RMS in dB
    pub centroid_hz:  f32,   // spectral centroid
    pub rolloff_hz:   f32,   // 85% rolloff
    pub zcr:          f32,   // zero-crossing rate
    pub flatness:     f32,   // spectral flatness [0,1]
}
```

**Invariants**: all fields are finite; `voicing ∈ [0,1]`; `f0_hz ∈ [0, nyquist]`.

**Services**: `ProsodyExtractor::extract(frame: &[f32], sample_rate: f32) -> ProsodySnapshot`.

### Context 2: Avatar Driving (`viseme.rs`)

**Purpose**: translate prosody + spectrum into mouth-shape coefficients suitable
for driving a lip-sync renderer (the downstream Phoenix/LatentSync equivalent).

**Aggregate root**: `AvatarFrame` — the packet emitted per block.

**Entities**: none (single-frame output is immutable).

**Value objects**:

```rust
pub enum Viseme { Rest, Aa, Ee, Ih, Oh, Uw, Fv, Mbp }

pub struct VisemeCoeffs {
    pub viseme:      Viseme,
    pub jaw_open:    f32,   // [0,1]
    pub lip_round:   f32,   // [0,1]
    pub lip_spread:  f32,   // [0,1]
    pub confidence:  f32,   // [0,1]
}

pub struct AvatarFrame {
    pub timestamp_us: u64,
    pub prosody:      ProsodySnapshot,
    pub viseme:       VisemeCoeffs,
    pub speaker_id:   Option<u32>,   // set by Speaker Identity context
    pub emotion:      EmotionBucket,
}
```

**Invariants**: all coefficients ∈ [0,1]; exactly one viseme per frame; jaw_open
monotonic in energy_db when voiced.

**Services**: `VisemeMapper::from_prosody_and_spectrum(snap, mags) -> VisemeCoeffs`.

### Context 3: Speaker Identity (`speaker_embed.rs`)

**Purpose**: produce and match compact speaker fingerprints.

**Aggregate root**: `SpeakerTracker` — owns the set of known speakers in a
session.

**Entities**:

```rust
pub struct SpeakerFingerprint {
    pub id:         u32,
    pub embedding:  [f32; 64],
    pub frames:     u32,       // how many frames accumulated
}

pub struct SpeakerTracker {
    // invariants: at most `max_speakers`; cosine_threshold strictly enforced
    speakers:        Vec<SpeakerFingerprint>,
    next_id:         u32,
    max_speakers:    usize,
    cosine_threshold: f32,
}
```

**Domain events**: `SpeakerEnrolled { id }`, `SpeakerMatched { id, similarity }`.

**Services**:
- `SpeakerTracker::observe(&mut self, snap, mags) -> u32` — identify or enrol.
- `SpeakerTracker::cosine(a, b) -> f32` — pure value function.

### Context 4: Analysis DAG (`analysis.rs`)

**Purpose**: aggregate frames into a persistence-friendly stream that mirrors
(a subset of) ClipCannon's 23-stage DAG output.

**Aggregate root**: `AnalysisFrame` — one record per `process_block`.

**Value objects**:

```rust
pub struct AnalysisFrame {
    pub session_id:    u64,
    pub frame_index:   u64,
    pub timestamp_us:  u64,
    pub prosody:       ProsodySnapshot,
    pub viseme:        VisemeCoeffs,
    pub emotion:       EmotionBucket,
    pub speaker_id:    Option<u32>,
    pub highlight:     f32,   // [0,1] highlight salience
    pub safe_cut:      bool,  // silence-padded safe boundary
}
```

**Services**:
- `HighlightScorer::score(&AnalysisFrame) -> f32`
- `SafeCutDetector::is_safe_cut(frames_window: &[AnalysisFrame]) -> bool`

### Context Map

| Upstream          | Downstream        | Relationship                            |
|-------------------|-------------------|-----------------------------------------|
| Signal Analysis   | Avatar Driving    | **Customer/Supplier** — viseme consumes prosody by value |
| Signal Analysis   | Speaker Identity  | **Shared Kernel** — both read STFT mags |
| Avatar Driving    | Analysis DAG      | **Conformist** — DAG stores AvatarFrame subset as-is |
| Speaker Identity  | Analysis DAG      | **Customer/Supplier** — DAG attaches speaker_id |

Note there is **no** dependency from Signal Analysis on any other context: the
purest, most reusable context sits at the bottom of the dependency graph.

### Anti-Corruption Layer

A single struct `RealtimeAvatarAnalyzer` in `clipcannon/pipeline.rs` implements
`AudioProcessor` and is the **only** place where the four contexts are
composed. Downstream code never touches individual contexts directly.

```rust
pub struct RealtimeAvatarAnalyzer {
    extractor:      ProsodyExtractor,
    viseme_mapper:  VisemeMapper,
    speaker:        SpeakerTracker,
    last_frame:     Option<AnalysisFrame>,
    session_id:     u64,
    frame_index:    u64,
}

impl AudioProcessor for RealtimeAvatarAnalyzer {
    fn prepare(&mut self, sample_rate: f32, block_size: usize) { /* ... */ }
    fn process(&mut self, block: &mut AudioBlock)               { /* ... */ }
    fn name(&self) -> &str { "RealtimeAvatarAnalyzer" }
    fn latency_samples(&self) -> usize { /* window/2 */ }
}

impl RealtimeAvatarAnalyzer {
    pub fn last_frame(&self) -> Option<&AnalysisFrame> { self.last_frame.as_ref() }
}
```

This preserves the `hearmusica` convention — all side-effects flow through
`AudioBlock::metadata` and the analyser's own borrowed state — and keeps the
realtime contract (no allocation in `process`).

## Consequences

### Positive
- Four small contexts are individually testable and independently reusable.
- Zero cross-context mutable state; only the ACL (`RealtimeAvatarAnalyzer`)
  composes them.
- `AnalysisFrame` is a plain POD that can be serialised to any downstream
  store (SQLite, WASM postMessage, JSON) without leaking domain types.

### Negative
- Four modules instead of one; higher file count.
- Emotion is coarse (bucket) not a continuous embedding — see ADR-147.

### Risks
- Context boundaries drift over time; mitigated by making cross-context calls
  value-only (never mut refs across contexts).

## References
- Evans, *Domain-Driven Design*, Chapters 4–5 & 14 (bounded contexts, context map).
- ADR-145: ClipCannon Rust Exploration — Scope.
- ADR-147: Streaming prosody/emotion/speaker analysis.
- ADR-148: Benchmark methodology & performance targets.
