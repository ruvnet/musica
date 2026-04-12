# ADR-154: Singing Analysis Subsystem — Pitch, Vibrato, Style, Genre

## Status
Accepted

## Date
2026-04-12

## Context

A natural extension of the ClipCannon realtime subsystem is **singing**.
The same audio features that drive an avatar (F0, energy, formants,
spectral envelope) are exactly what a singing system needs:
- **Karaoke scoring** — does the singer match a reference melody?
- **Vibrato / portamento detection** — performance qualities.
- **Auto-tune** target generation — sub-cent pitch correction.
- **Singing voice → animated singer mouth shapes** — ADR-147 visemes
  with pitch-aware jaw bias.
- **Style / genre classification** — pop vs. classical vs. metal vs. R&B.
- **Sing-along avatars** — drive a virtual idol from a live mic.

The user explicitly asked: *"Can we implement high fidelity singing
capabilities based on sampling of existing styles and popularity and
genres?"* This ADR answers that with an honest scope split.

### What is **out of scope** for this ADR

A neural singing voice **synthesiser** (DiffSinger, NNSVS, ACE-Step,
SingTalk, Suno-style) requires gigabytes of trained weights, GPUs, and
copyright-encumbered training corpora. That violates every musica
constraint (zero-dep, sub-ms, edge-deployable, reproducible). It would
also overlap exactly with ClipCannon's upstream ACE-Step pipeline, which
we explicitly chose **not** to duplicate (ADR-145).

### What **is** in scope

A **singing analysis** subsystem that:

1. Tracks pitch with cent-level accuracy in real time.
2. Detects musical performance traits (vibrato rate/depth, portamento,
   note onsets).
3. Classifies the singing style/genre by sampling a small library of
   pre-baked **acoustic prototypes** at runtime — no neural training.
4. Drives the existing avatar pipeline so animated singers lip-sync
   correctly with pitch-aware jaw motion.
5. Optionally aligns to a reference melody for karaoke scoring.

What this ADR is **not** doing: rendering audio. Synthesis stays an
out-of-tree concern (a host can wire our pitch/style outputs into a
local TTS, an external Suno-style API, or a sample-based synth).

## Decision

Add `clipcannon/singing.rs` with three coordinated services:

### 1. `PitchTracker` — cent-accurate F0

A higher-resolution F0 estimator than the prosody-class one (which is
optimised for speech). Uses the same Wiener-Khinchin ACF from ADR-153
but with:

- **Octave error correction** via subharmonic summation: check the
  argmax of `r[τ] + 0.5·r[2τ] + 0.25·r[4τ]` to avoid octave doubling
  (the bane of every classical pitch tracker).
- **Cent-resolution interpolation**: parabolic interpolation already in
  ADR-147; we expose the result in cents as `1200·log2(f0/ref_hz)`
  rather than Hz.
- **Voiced/unvoiced gate**: voicing > 0.65 (singing demands more
  certainty than speech).

```rust
pub struct PitchTracker {
    sample_rate:   f32,
    ref_hz:        f32,    // typically 440 Hz (A4)
    history:       [f32; 16],   // F0 history for stability
    head:          usize,
    filled:        usize,
}

#[derive(Debug, Clone, Copy)]
pub struct PitchSnapshot {
    pub f0_hz:      f32,    // 0 if unvoiced
    pub cents:      f32,    // semitone-cents from ref_hz, NaN if unvoiced
    pub midi_note:  f32,    // float MIDI note number
    pub stability:  f32,    // [0, 1] — std-dev of last 8 frames
    pub voiced:     bool,
}
```

### 2. `VibratoDetector` — performance quality

Vibrato is a quasi-periodic 4–8 Hz modulation of F0 with depth typically
20–100 cents. Detect via FFT of a 32-frame F0 history (≈256 ms at 8 ms
hop), peak in the 4–8 Hz band:

```rust
pub struct VibratoDetector {
    history:   [f32; 32],   // F0 in cents over the last 32 blocks
    head:      usize,
    filled:    usize,
}

#[derive(Debug, Clone, Copy)]
pub struct VibratoSnapshot {
    pub rate_hz:    f32,    // 0 if no vibrato
    pub depth_cents:f32,    // 0 if no vibrato
    pub presence:   f32,    // [0,1] — peak/total ratio in 4-8 Hz band
}
```

### 3. `StyleClassifier` — genre / style by acoustic prototype matching

The "high fidelity singing capabilities based on sampling of existing
styles and popularity and genres" request is, structurally, a
*classification* problem. We solve it without training data:

- A small **prototype library** ships compiled into the binary as
  `const` arrays (zero on-disk weights). Each prototype is a fixed-size
  acoustic fingerprint summarising one style:

  ```rust
  pub struct StylePrototype {
      pub name:               &'static str,   // "pop_belt", "classical_legit", "metal_scream", ...
      pub mean_centroid_hz:   f32,
      pub mean_rolloff_hz:    f32,
      pub mean_flatness:      f32,
      pub mean_zcr:           f32,
      pub mean_vibrato_rate:  f32,
      pub mean_vibrato_depth: f32,
      pub formant_bias:       [f32; 4],       // F1..F4 normalised
      pub popularity_weight:  f32,            // [0,1] — Bayesian prior
  }
  ```

- The classifier maintains a 64-frame rolling fingerprint of the live
  audio in the same shape, then scores each prototype with a weighted
  L2 distance. Top-K matches are returned with confidences.

- **Popularity / genre weighting** is a Bayesian prior: a host can pass
  `popularity_weights[name]` (e.g. derived from the top-N current
  Billboard, a Spotify chart, or a corpus frequency table) to bias the
  classifier toward currently popular styles. The weights are *runtime
  data*, not training data — the host owns them.

```rust
pub struct StyleClassifier {
    fingerprint:    StyleFingerprint,
    prototypes:     &'static [StylePrototype],
    rolling_count:  u32,
}

#[derive(Debug, Clone, Copy)]
pub struct StyleMatch {
    pub name:       &'static str,
    pub similarity: f32,    // [0,1]
}

impl StyleClassifier {
    pub fn new(prototypes: &'static [StylePrototype]) -> Self;
    pub fn observe(
        &mut self,
        prosody: &ProsodySnapshot,
        pitch:   &PitchSnapshot,
        vibrato: &VibratoSnapshot,
    );
    pub fn top_k(&self, k: usize, out: &mut [StyleMatch]) -> usize;
    pub fn reset(&mut self);
}
```

### Default prototype library

Ship **eight** built-in style prototypes covering common popular and
classical singing styles:

| Name              | Centroid (Hz) | Vibrato | Notes                        |
|-------------------|--------------:|---------|------------------------------|
| `pop_belt`        | 2200          | 5.5 Hz / 50 c | Modern pop chest belt   |
| `pop_breathy`     | 1400          | weak           | Indie / acoustic pop    |
| `classical_legit` | 2800          | 6.0 Hz / 80 c | Operatic, full vibrato   |
| `musical_theatre` | 2500          | 5.0 Hz / 60 c | Belt+legit hybrid        |
| `rock_grit`       | 3200          | 4.5 Hz / 40 c | Rock / hard rock         |
| `metal_scream`    | 4500          | none           | Distorted, wide-band     |
| `rnb_melisma`     | 1900          | 6.5 Hz / 70 c | High pitch agility       |
| `folk_plain`      | 1800          | minimal        | Folk / singer-songwriter |

Eight is chosen so the in-memory prototype table is < 1 KB.

The numeric values are **acoustic centroids derived from public-domain
academic literature** on singing voice analysis (not from sampling
copyrighted recordings). Hosts that want their own genre table swap in
their own `&'static [StylePrototype]` — the library is just a starting
point.

### Pitch-aware viseme bias

Singing demands wider mouth openings on sustained vowels. Extend
`viseme.rs` (additively) so that when a `PitchSnapshot` is supplied with
`stability > 0.7`, the `jaw_open` coefficient gets a `+0.2` bias on
voiced visemes. Optional, controlled by a flag on the analyser.

### Karaoke scoring (optional convenience)

```rust
pub struct KaraokeScorer {
    reference: Vec<f32>,    // reference melody in MIDI floats per block
    cursor:    usize,
}

impl KaraokeScorer {
    pub fn new(reference_midi: Vec<f32>) -> Self;
    pub fn score(&mut self, pitch: &PitchSnapshot) -> f32;   // [0,1] per block
    pub fn reset(&mut self);
}
```

A 1-cent error gives ~0.99 score, 50 cents (≈half-semitone) gives ~0.5,
100 cents gives ~0.0. Score formula: `exp(-(cents_err/30)²)`.

## Realtime contract

- All four services are allocation-free in steady state.
- All consume the existing `SharedSpectrum` from ADR-153 plus a
  precomputed `ProsodySnapshot`. No new FFTs.
- Combined target: **<8 µs / block** for the full singing pipeline.

## Consequences

### Positive
- Adds a complete singing analysis surface without violating zero-dep.
- Genre/popularity weighting is host-supplied runtime data, sidestepping
  copyright concerns entirely.
- Karaoke, autotune front-end, virtual idol lip-sync all become trivial.
- Composes naturally with the existing avatar pipeline.

### Negative
- We **do not synthesise** singing. Hosts that need synthesis still go
  to ClipCannon upstream / ACE-Step / DiffSinger.
- Style classification is acoustic-only: a low-effort cover of a metal
  song will classify as folk, not metal.

### Risks
- Cent accuracy depends on the F0 tracker handling octave errors
  cleanly. We add a dedicated subharmonic-summation step in
  `PitchTracker` to mitigate.
- The 8-prototype default library is a starting point, not a complete
  taxonomy. Documented as such.

## References
- Sundberg, *The Science of the Singing Voice*, 1987.
- de Cheveigné & Kawahara, *YIN, a fundamental frequency estimator for
  speech and music*, JASA 2002.
- Goto et al., *Singing information processing*, 2014 survey.
- ADR-145 (no diffusion synthesis), ADR-147 (visemes), ADR-153 (shared FFT).
