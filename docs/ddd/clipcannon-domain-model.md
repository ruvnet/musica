# ClipCannon Realtime Subsystem — Domain Model

> Companion to ADRs 145–148. This document is the **living domain reference**
> for `src/clipcannon/`. Update it whenever the ubiquitous language or context
> boundaries change.

## 1. Ubiquitous Language

| Term                   | Formal definition                                                                                   |
|------------------------|------------------------------------------------------------------------------------------------------|
| **Sample**             | A single `f32` amplitude in the interleaved PCM stream.                                              |
| **Frame**              | One STFT analysis window. Default: 256 samples @ 16 kHz (16 ms).                                     |
| **Block**              | One realtime audio buffer passed to `AudioProcessor::process`. Default: 128 samples @ 16 kHz (8 ms). |
| **Hop**                | Distance between successive frame starts. Default: 128 samples (= block size).                      |
| **ProsodySnapshot**    | Value object. Immutable per-frame prosody features. See §3.1.                                       |
| **Viseme**             | Enum value. Discrete mouth-shape class from the set {REST, AA, EE, IH, OH, UW, FV, MBP}.            |
| **VisemeCoeffs**       | Value object. Per-block viseme class + continuous coefficients (jaw_open, lip_round, lip_spread).   |
| **EmotionBucket**      | Enum value. Coarse valence/arousal bucket from {Neutral, Happy, Sad, Angry, Calm}.                  |
| **AvatarFrame**        | Aggregate root of the Avatar Driving context. See §3.2.                                             |
| **SpeakerFingerprint** | Entity. 64-dim L2-normalised spectral signature + stable integer id.                               |
| **SpeakerTracker**     | Aggregate root of the Speaker Identity context. Owns a bounded set of `SpeakerFingerprint`.         |
| **AnalysisFrame**      | Aggregate root of the Analysis DAG context. One record per `process_block` call.                    |
| **Session**            | Logical grouping of `AnalysisFrame`s with a shared `session_id`. No explicit type in v1.            |
| **Highlight**          | Scalar in [0,1]. Salience score used by ClipCannon to surface moments.                              |
| **Safe Cut**           | Boolean. Whether the current frame sits inside a silence-padded safe boundary.                     |

## 2. Bounded Contexts

```text
                 ┌──────────────────┐
                 │ Signal Analysis  │   pure, stateless, value-only
                 │   (prosody)      │
                 └────────┬─────────┘
                          │ ProsodySnapshot
              ┌───────────┴────────────┐
              ▼                        ▼
 ┌──────────────────┐       ┌──────────────────┐
 │ Avatar Driving   │       │ Speaker Identity │   stateful (tracker)
 │   (viseme)       │       │ (speaker_embed)  │
 └────────┬─────────┘       └────────┬─────────┘
          │ VisemeCoeffs             │ speaker_id
          └────────┬─────────────────┘
                   ▼
         ┌──────────────────┐
         │ Analysis DAG     │   write-only aggregate root
         │   (analysis)     │
         └──────────────────┘
```

### 2.1 Signal Analysis
- **Role**: factory for `ProsodySnapshot` values.
- **State**: none.
- **Public API**: `ProsodyExtractor::new(sample_rate, window)`, `extract(frame: &[f32], mags: &[f32]) -> ProsodySnapshot`.
- **Invariants**: see §3.1.

### 2.2 Avatar Driving
- **Role**: convert prosody + spectrum to `VisemeCoeffs` for downstream lip-sync.
- **State**: 3-frame median history for viseme flicker suppression.
- **Public API**: `VisemeMapper::new()`, `map(&mut self, snap, mags) -> VisemeCoeffs`.
- **Invariants**: see §3.3.

### 2.3 Speaker Identity
- **Role**: on-line diarisation via compact fingerprint matching.
- **State**: bounded vector of enrolled `SpeakerFingerprint` entities (≤ max_speakers).
- **Public API**: `SpeakerTracker::new(max_speakers, cosine_threshold)`, `observe(&mut self, mags) -> u32`.
- **Invariants**: see §3.4.

### 2.4 Analysis DAG
- **Role**: compose upstream contexts into persistable `AnalysisFrame`s.
- **State**: `frame_index`, `session_id`, rolling window for safe-cut detection.
- **Public API**: `Analyzer::analyse(&mut self, snap, viseme, speaker_id) -> AnalysisFrame`.
- **Invariants**: `frame_index` strictly monotonic; `session_id` stable.

## 3. Types — Invariants Table

### 3.1 `ProsodySnapshot`
| Field         | Type | Range / invariant              | Unit     |
|---------------|------|--------------------------------|----------|
| `f0_hz`       | f32  | [0, sample_rate/2]             | Hz       |
| `voicing`     | f32  | [0, 1]                         | ratio    |
| `energy_db`   | f32  | (-∞, ~+6]                      | dB FS    |
| `centroid_hz` | f32  | [0, sample_rate/2]             | Hz       |
| `rolloff_hz`  | f32  | [0, sample_rate/2]             | Hz       |
| `zcr`         | f32  | [0, 1]                         | ratio    |
| `flatness`    | f32  | [0, 1]                         | ratio    |

All fields must be finite (no NaN / inf). `f0_hz = 0` indicates "unvoiced".

### 3.2 `AvatarFrame`
| Field         | Type              | Invariant                             |
|---------------|-------------------|---------------------------------------|
| `timestamp_us`| u64               | monotonic per session                 |
| `prosody`     | ProsodySnapshot   | §3.1                                  |
| `viseme`      | VisemeCoeffs      | §3.3                                  |
| `speaker_id`  | Option<u32>       | `Some` if tracker enrolled ≥1 speaker |
| `emotion`     | EmotionBucket     | valid variant                         |

### 3.3 `VisemeCoeffs`
| Field       | Type  | Invariant               |
|-------------|-------|-------------------------|
| `viseme`    | Viseme| one of 8 variants       |
| `jaw_open`  | f32   | [0, 1]                  |
| `lip_round` | f32   | [0, 1]                  |
| `lip_spread`| f32   | [0, 1]                  |
| `confidence`| f32   | [0, 1]                  |

Additional cross-field rule: `lip_round + lip_spread ≈ 1` (tolerance 1e-3).

### 3.4 `SpeakerTracker`
| Field             | Invariant                                     |
|-------------------|-----------------------------------------------|
| `speakers.len()`  | ≤ `max_speakers`                              |
| `cosine_threshold`| (0, 1)                                        |
| `next_id`         | strictly monotonic                            |
| Each embedding    | L2 norm within 1e-3 of 1.0                    |

### 3.5 `AnalysisFrame`
All sub-values above, plus:

| Field         | Type    | Invariant                                    |
|---------------|---------|----------------------------------------------|
| `session_id`  | u64     | stable across the life of the analyser       |
| `frame_index` | u64     | strictly monotonic                            |
| `highlight`   | f32     | [0, 1]                                        |
| `safe_cut`    | bool    | derived from rolling window, no free params |

## 4. Domain Services

### 4.1 `RealtimeAvatarAnalyzer`

The **anti-corruption layer** between `hearmusica::AudioProcessor` and the
four domain contexts. Implements `AudioProcessor::process` by:

1. Pull the L channel of the current `AudioBlock`.
2. STFT-ish windowed magnitude pass (reusing `stft::TwiddleCache` is *not*
   required — we do it internally to keep allocation-free).
3. `ProsodyExtractor::extract(frame, mags)`.
4. `VisemeMapper::map(snap, mags)`.
5. `SpeakerTracker::observe(mags)`.
6. `Analyzer::analyse(snap, viseme, id)` → `AnalysisFrame`.
7. Store latest frame on `self.last_frame` and on `block.metadata`.

The `AudioBlock` buffers are **not** modified (this is a read-only analyser),
so downstream blocks in the pipeline see untouched audio.

### 4.2 `HighlightScorer`

```text
highlight = 0.4 · arousal + 0.3 · voicing + 0.2 · energy_norm + 0.1 · pitch_dynamics
```

Output clamped to [0,1]. All inputs are already available in `ProsodySnapshot`.

### 4.3 `SafeCutDetector`

Given a rolling window of 8 `AnalysisFrame`s, a frame is a safe cut iff:

- current `voicing < 0.2`,
- current `energy_db < −38 dB`,
- at least 4 of the 8 frames in the window satisfy the above,
- previous frame did **not** satisfy the above (edge, not plateau).

This mirrors ClipCannon's "silence + silence padding" heuristic.

## 5. Domain Events

| Event              | Fields                  | Triggered by                         |
|--------------------|-------------------------|--------------------------------------|
| `SpeakerEnrolled`  | `id: u32`               | `SpeakerTracker::observe` new speaker |
| `SpeakerMatched`   | `id: u32`, `sim: f32`   | `SpeakerTracker::observe` match       |
| `HighlightDetected`| `frame_index: u64`      | highlight crosses 0.8 threshold      |
| `SafeCutDetected`  | `frame_index: u64`      | `SafeCutDetector::is_safe_cut` true   |

In v1 events are **not** emitted via a channel; the host polls
`analyzer.last_frame()` and inspects it. A later version can add a
`EventSink` trait without breaking the analyser API.

## 6. Persistence Boundary

No persistence in v1. `AnalysisFrame` is a plain POD, serialisable by any
external code into (e.g.) ClipCannon's SQLite `frames` table via an
adapter crate.

## 7. Open Questions

- Should `VisemeMapper` be parameterised by language (bias table)? — **Not in v1.**
- Should the emotion bucket become a continuous vector? — **Not in v1, see ADR-147.**
- Should we expose the STFT result for reuse by other analyser blocks? — **Yes**,
  via `AudioBlock::metadata` once a suitable field is added upstream.

## 8. Change Log

- 2026-04-12: Initial draft, aligned with ADRs 145–148.
