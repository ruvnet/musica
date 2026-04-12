# ADR-147: Streaming Prosody, Emotion & Speaker Analysis — Algorithms

## Status
Accepted

## Date
2026-04-12

## Context

ADR-146 named the bounded contexts. This ADR fixes the *algorithms* used inside
`clipcannon/prosody.rs`, `clipcannon/viseme.rs`, and `clipcannon/speaker_embed.rs`.

All three share three hard constraints:

1. **No external dependencies** beyond what musica already imports.
2. **No allocation in the hot path** (steady-state `process_block`).
3. **<0.5 ms / 128-sample block** on a laptop CPU at 16 kHz.

ClipCannon uses Wav2Vec2 (prosody), Wav2Vec2-large (emotion), WavLM (speaker)
and ECAPA-TDNN (voice identity). All four are impossible under those
constraints. We need classical DSP-first approximations that are "good enough
to drive a lip-sync avatar and diarise a meeting" while staying interpretable.

## Decision

### 1. Prosody Extraction (`prosody.rs`)

**Features per STFT frame (window=256, hop=128, sr=16000):**

| Feature       | Algorithm                                          | Complexity  |
|---------------|----------------------------------------------------|-------------|
| `f0_hz`       | ACF peak in 50–400 Hz on time-domain frame         | O(N·L)      |
| `voicing`     | ACF peak / ACF[0], clipped to [0,1]                | O(1)        |
| `energy_db`   | 10·log10(Σ x²/N + ε)                               | O(N)        |
| `centroid_hz` | Σ f·|X[f]| / Σ |X[f]|                              | O(B)        |
| `rolloff_hz`  | smallest f where Σ|X[0..f]|² ≥ 0.85·Σ|X|²          | O(B)        |
| `zcr`         | sign-change count / N                              | O(N)        |
| `flatness`    | geometric_mean / arithmetic_mean of |X|            | O(B)        |

where N = window size, B = N/2+1 bins, L = lag range (≈N/2).

**Why ACF for F0?** YIN and PYIN are more accurate but more expensive and
introduce state. Autocorrelation with quadratic interpolation around the peak
gives F0 to ±2 Hz which is sufficient for mouth-shape driving.

**State**: none. Extractor is pure. Makes it trivially thread-safe and
WASM-compatible.

### 2. Viseme Mapping (`viseme.rs`)

Driving a diffusion lip-sync model (upstream LatentSync) from audio requires at
minimum:

- a discrete viseme class (mouth shape),
- a continuous jaw-open coefficient (mouth opening),
- a lip-round coefficient (vs. lip-spread — distinguishes UW from EE).

We derive all three from the first three formants approximated by **band-energy
ratios** over the magnitude spectrum:

```text
F1 band = 300-1000 Hz   → jaw openness (AA, OH)
F2 band = 1000-2500 Hz  → lip roundness vs spreading (EE vs UW)
F3 band = 2500-5000 Hz  → fricatives (FV), sibilants (implicit S→REST)
```

**Classification** uses a hand-coded decision table on (voicing, energy_db,
f1_ratio, f2_ratio, f3_ratio, flatness). Calibrated on typical English speech:

| Viseme | Condition                                                        |
|--------|------------------------------------------------------------------|
| REST   | energy < −40 dB or voicing < 0.25                                |
| MBP    | voicing > 0.5 AND f1_ratio < 0.10 (closed lips burst)            |
| FV     | voicing < 0.5 AND f3_ratio > 0.35 AND flatness > 0.35            |
| AA     | voicing > 0.5 AND f1_ratio > 0.55                                |
| OH     | voicing > 0.5 AND f1_ratio > 0.40 AND f2_ratio < 0.30            |
| UW     | voicing > 0.5 AND f2_ratio < 0.25 AND f1_ratio < 0.35            |
| IH     | voicing > 0.5 AND f2_ratio > 0.35 AND f1_ratio < 0.30            |
| EE     | voicing > 0.5 AND f2_ratio > 0.50                                |

`jaw_open = clamp((energy_db + 40) / 40 · f1_ratio_norm, 0, 1)`
`lip_round = smoothstep(0.15, 0.35, 1 − f2_ratio)`
`lip_spread = 1 − lip_round`
`confidence = voicing · (1 − flatness)`

A 3-frame median filter on `Viseme` suppresses single-frame flicker at <1 ms
cost per block.

### 3. Emotion Bucket

Continuous emotion embeddings require Wav2Vec2-large; we ship a **coarse
valence-arousal bucket** computed from prosody alone:

```text
arousal = normalise(energy_db + α · f0_std_3frames)
valence = β · f0_slope_3frames − γ · flatness
```

```rust
pub enum EmotionBucket { Neutral, Happy, Sad, Angry, Calm }
```

mapped from the (valence, arousal) quadrant. This is deliberately coarse — it
is not meant to replace Wav2Vec2, only to give the avatar a gesture bias.

### 4. Speaker Fingerprint (`speaker_embed.rs`)

Instead of ECAPA-TDNN 2048-dim (6 MB, ~30 ms / frame on CPU), we build a
**64-dim spectral fingerprint** derived from a log-mel-ish band energy vector
and a long-term running average:

1. Split magnitude spectrum into 32 log-spaced bands over [0, 8000] Hz.
2. Log-compress: `b_i = log(1 + energy_i)`.
3. Concatenate with 32 first-order temporal differences → 64-dim vector.
4. L2-normalise.
5. Identity match via cosine similarity, threshold = 0.85.

A `SpeakerTracker` maintains a running *mean* fingerprint per enrolled speaker,
updated with exponential moving average `α = 0.05` — giving ~20-frame
convergence and immunity to single noisy frames.

**Diarisation**: per block, we call `observe()` which either matches an
existing speaker (returning its id) or enrols a new one (up to
`max_speakers`, default 16). This gives cheap diarisation without any neural
network.

**Known limitation**: two speakers with similar timbre will merge. For the
target use case (meeting bot with ≤8 participants) this is acceptable.

## Consequences

### Positive

- All three stages run in <0.5 ms on a laptop CPU.
- Zero heap allocation in steady state — all buffers preallocated.
- All features are interpretable and can be inspected live.
- Code is auditable for regulatory contexts (no opaque embeddings).

### Negative

- Emotion is coarse; a user wanting Ekman-grade emotion classification must
  still go to Wav2Vec2.
- Speaker fingerprint is not interoperable with ECAPA-TDNN; ClipCannon voice
  profiles would need a separate adapter to match up.
- Viseme mapping is English-calibrated; tonal languages may need a different
  F1/F2/F3 band table.

### Risks

- ACF-based F0 fails on whispered speech. We mitigate by flagging
  `voicing < 0.3` and falling back to REST viseme (mouth closed) rather than
  emitting garbage.

## References
- Rabiner, *Fundamentals of Speech Recognition*, Ch. 4 (formants, F0 by ACF).
- Eyben et al., *The Geneva Minimalistic Acoustic Parameter Set* (prosody defs).
- ADR-145, ADR-146.
