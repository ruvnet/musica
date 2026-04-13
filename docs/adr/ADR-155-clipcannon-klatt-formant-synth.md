# ADR-155: Klatt-Style Formant Voice Synthesiser

## Status
Accepted

## Date
2026-04-12

## Context

ADR-145 explicitly excluded diffusion synthesis (LatentSync, ACE-Step,
Qwen3-TTS) from the musica scope. ADRs 149–154 built the *analysis* side
of a complete voice agent. The asymmetry leaves musica unable to talk
back: it can hear, classify, diarise, and drive an avatar from audio,
but it cannot emit audio of its own.

To close the loop without violating the zero-dep ethos we need a
synthesiser that:

1. Runs realtime on a microcontroller (no GPU, no big models).
2. Has no on-disk weights.
3. Is fully interpretable and auditable.
4. Produces *intelligible* (not necessarily natural) speech.

The classical answer is **Klatt's formant synthesiser** (1980,
*Software for a Cascade/Parallel Formant Synthesizer*, JASA 67(3)). It
powered DECtalk, Stephen Hawking's voice, MITalk, and the `klatt`
backend of eSpeak. The acoustic model is:

```
glottal pulse  ──►  cascade resonators (F1..F5)  ──►  spectral envelope
white noise    ──►  parallel resonators (frication branch)  ──►  fricatives
                          │
                          ▼
                       output
```

State per voice: a few hundred floats. Compute per sample: ~50 multiplies.
Sub-microsecond per block on any CPU.

## Decision

Add `clipcannon/klatt.rs` implementing a self-contained Klatt-class
formant synthesiser driven by a phoneme stream + the existing
`ProsodySnapshot` for prosody.

### Public types

```rust
pub struct KlattSynthesiser {
    sample_rate:    f32,
    /// Glottal pulse generator state (LF model approximation).
    glottal:        GlottalSource,
    /// Five cascade formant resonators (F1..F5).
    cascade:        [Resonator; 5],
    /// Parallel resonators for the frication branch (4 stages).
    parallel:       [Resonator; 4],
    /// Currently-targeted formant frequencies and bandwidths.
    target:         FormantTarget,
    /// Interpolation state for smooth phoneme transitions.
    current:        FormantTarget,
    interp_alpha:   f32,
    /// Voicing on/off (off for unvoiced fricatives).
    voicing:        bool,
}

pub struct FormantTarget {
    pub f0_hz:  f32,
    pub formants: [(f32 /* freq */, f32 /* bw */); 5],
    pub voicing_amp:    f32,   // [0, 1]
    pub frication_amp:  f32,   // [0, 1]
}

pub enum Phoneme { /* 39 ARPAbet symbols, see ADR-156 */ }

impl KlattSynthesiser {
    pub fn new(sample_rate: f32) -> Self;
    pub fn set_phoneme(&mut self, p: Phoneme, f0_hz: f32);
    /// Render `n` samples in-place into `out` (mono f32).
    pub fn render(&mut self, out: &mut [f32]);
    pub fn reset(&mut self);
}
```

### Algorithm

**Glottal source** — LF (Liljencrants-Fant) approximation. A single sine
oscillator at F0 modulated by a non-symmetric envelope:

```text
phase += 2π·F0/sr
g(t) = sin(phase) · open_quotient(phase)   // simplified LF
```

**Cascade resonators** — biquad form, two parameters per formant
(centre frequency, bandwidth):

```text
y[n] = c·g[n] + b·y[n-1] - a·y[n-2]
where
  c =  1 - b - a
  b =  2·exp(-π·BW/sr)·cos(2π·F/sr)
  a = -exp(-2π·BW/sr)
```

The output of stage `i` feeds the input of stage `i+1`. After all five
stages, the signal carries the vowel envelope.

**Parallel branch** — same biquad form fed by white noise, used for
fricatives `/s/`, `/sh/`, `/f/`, `/h/`. Each parallel resonator has its
own gain. Output is summed with the cascade output.

**Phoneme-to-formant table** — 39 ARPAbet phonemes mapped to
`FormantTarget` presets, derived from public-domain values in Klatt 1980
(reproduced widely; not copyrighted). Vowels get F1–F3, consonants get
appropriate frication bands.

**Phoneme transitions** — linear interpolation of formant targets over
~30 ms (the `interp_alpha` field) gives smooth transitions and avoids
clicks.

### Realtime contract
- No allocation in the hot path.
- Renders at any sample rate via parameter recomputation in `set_phoneme`.
- Target: <0.5 µs per 128-sample block at 16 kHz (≈4 ns/sample, well
  within budget for 50 multiplies/sample).

### Quality

Intelligible but unmistakably synthetic — "1980s computer voice".
Comparable to eSpeak's `klatt` backend. Good for accessibility tools,
embedded notifications, edge voice agents, hearing-aid feedback prompts.
Not a replacement for ElevenLabs or LPCNet.

## Consequences

### Positive
- Closes the loop: musica can now talk back without a GPU or model file.
- Fits on any MCU; runs in WASM at 1000× realtime.
- Fully interpretable: every formant is a named, tunable parameter.
- Foundation for ADR-158 singing synthesis (drive the same engine with
  a target pitch curve).

### Negative
- Robotic timbre. Quality is below any neural vocoder.
- Vowel naturalness depends on accurate formant tables; we ship
  English-tuned defaults only.

### Risks
- Glottal pulse aliasing at high F0. Mitigated by an oversampled
  generator (4×) for F0 > 300 Hz.

## References
- Klatt, *Software for a Cascade/Parallel Formant Synthesizer*, JASA 67(3),
  1980.
- Klatt & Klatt, *Analysis, synthesis, and perception of voice quality
  variations among female and male talkers*, JASA 87(2), 1990.
- eSpeak NG `klatt` backend (MIT licensed reference implementation).
- ADR-156, ADR-158, ADR-159.
