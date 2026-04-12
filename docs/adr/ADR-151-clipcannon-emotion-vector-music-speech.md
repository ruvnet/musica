# ADR-151: Continuous Emotion Vector & Music/Speech Discriminator

## Status
Accepted

## Date
2026-04-12

## Context

ADR-147 ships a coarse `EmotionBucket` (Neutral/Happy/Sad/Angry/Calm)
because the v1 contract had no continuous emotion field. Two follow-up
needs surfaced:

1. **Smoother avatar gestures.** Discrete buckets cause snap transitions.
   Animators want a continuous (valence, arousal) pair to interpolate.
2. **Music vs speech.** A meeting bot wants to mute itself for music; a
   karaoke app wants the opposite. The VAD in ADR-150 fires for *both*
   speech and singing — we need to discriminate.

Both signals are computable from features we already extract — no new
algorithms or data, just better synthesis.

## Decision

### 1. Continuous valence-arousal vector

Add `EmotionVector` and keep `EmotionBucket` as a *derived* projection.

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmotionVector {
    pub valence:  f32,   // [-1, +1] negative → positive
    pub arousal:  f32,   // [-1, +1] calm    → excited
    pub confidence: f32, // [0, 1]
}

impl EmotionVector {
    pub fn neutral() -> Self;
    pub fn to_bucket(self) -> EmotionBucket;   // backward-compat projection
    pub fn lerp(self, other: Self, t: f32) -> Self;   // for animator smoothing
}
```

**Computation** (per `AnalysisFrame`, with a 5-frame F0 history window):

```text
energy_norm = clamp((energy_db + 50) / 50, 0, 1)
f0_var      = std(f0_history) / 50            // 50 Hz norm
arousal_raw = 0.6·energy_norm + 0.4·clamp(f0_var, 0, 1)
arousal     = arousal_raw·2 - 1               // remap to [-1,+1]

mean_f0     = mean(f0_history)
slope_f0    = (f0_history[last] - f0_history[0]) / max(1, len-1)
valence_raw = 0.5·tanh(slope_f0 / 25)         // rising pitch → positive
            + 0.3·(1 - flatness)              // tonal → positive
            - 0.2·clamp(zcr - 0.2, 0, 0.5)*4  // hissy → negative
valence     = clamp(valence_raw, -1, 1)

confidence  = voicing · (1 - flatness)
```

A 4-frame EMA smoothes the output (`α = 0.35`). The smoother lives inside
the analyser, not the value object — the value object is immutable.

### 2. Music vs speech discriminator

Add `clipcannon/music_speech.rs` exposing:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SignalKind { Silence, Speech, Music, Mixed }

pub struct MusicSpeechDetector {
    history: [f32; 16],   // 16-frame rolling window of "musicness" score
    head:    usize,
    filled:  usize,
    threshold: f32,
}

impl MusicSpeechDetector {
    pub fn new() -> Self;
    pub fn observe(&mut self, snap: &ProsodySnapshot, mags: &[f32]) -> SignalKind;
    pub fn reset(&mut self);
}
```

**Discriminator features:**

| Feature                  | Speech (typical) | Music (typical) |
|--------------------------|------------------|-----------------|
| ZCR variance over 16 fr. | high             | low             |
| F0 stability             | unstable         | stable (tones)  |
| Spectral flatness        | 0.1 – 0.4        | 0.05 – 0.3      |
| Long-term spectral peak  | < 4 kHz          | up to nyquist   |
| 4 Hz modulation energy   | high (syllables) | low             |

The 4 Hz syllabic-modulation cue is the classic Scheirer-Slaney
discriminator. We approximate it with the variance of the energy envelope
over the 16-frame window (≈128 ms at 8 ms blocks).

**Decision:**

```text
musicness = 0.4·(1 - zcr_variance_norm)
          + 0.3·f0_stability
          + 0.3·(1 - syllabic_mod_norm)
```

A 16-frame rolling mean of `musicness`, hysteresis-thresholded against
`threshold = 0.55` and an offset `0.40`, gives the `SignalKind`. Mixed is
returned when both VAD and music score are high simultaneously.

### Realtime contract
- Both extensions are O(1) per call.
- Allocation-free.
- Target: <2 µs combined per block.

## Consequences

### Positive
- Animators get a continuous emotion signal with cheap LERP.
- Meeting bots can suppress themselves during background music.
- Karaoke apps can isolate "is the user singing".
- Backward compatible: `EmotionBucket` stays as a derived projection.

### Negative
- Music/speech discriminator can be fooled by a cappella vocals (no
  instruments, ZCR/F0 look like speech). We accept this in v1.

### Risks
- Continuous emotion is still derived from acoustic prosody only — it
  cannot match a multimodal model. Documented honestly.

## References
- Scheirer & Slaney, *Construction and Evaluation of a Robust Multifeature
  Speech/Music Discriminator*, ICASSP 1997.
- Russell, *A Circumplex Model of Affect*, 1980 (valence/arousal).
- ADR-147 (original bucket model).
