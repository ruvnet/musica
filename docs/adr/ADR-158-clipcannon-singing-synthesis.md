# ADR-158: Singing Synthesiser — Pitch-Driven Klatt + PSOLA Stretching

## Status
Accepted

## Date
2026-04-12

## Context

ADR-154 added singing **analysis** (pitch tracking, vibrato detection,
style classification, karaoke scoring). The natural follow-up is
**synthesis** — making musica actually sing.

The user explicitly asked: *"Can we implement high fidelity singing
capabilities… implement realtime voice and singing?"* The honest
answer (Q&A above) is that diffusion synthesis is out of scope, but a
**classical-DSP singing synthesiser** is well within budget and gives
real value: realtime virtual idols, MIDI-to-singing, lyric-driven
karaoke avatars, music education, accessibility (sung notifications).

The two classical paths are:

1. **Klatt-driven** — feed our existing Klatt formant synthesiser
   (ADR-155) with a melody-driven F0 envelope. Adds vibrato by
   modulating F0. Vowels duck Klatt's `interp_alpha` so notes hold
   without slurring. Quality: 1980s-vocaloid robotic.
2. **PSOLA pitch shift** — store short vowel/consonant samples,
   pitch-shift them in the time domain via Pitch-Synchronous Overlap-Add
   to hit each note. Quality: 1990s Vocaloid 1 — recognisably synthetic
   but musical.

We implement **both** as alternative backends behind a `SingerVoice`
enum. Klatt is the default (zero data); PSOLA is available when the
host supplies a `VoiceBank` of recorded samples.

## Decision

Add `clipcannon/sing_synth.rs` with two backends and a unified
front-end.

### Public types

```rust
#[derive(Debug, Clone, Copy)]
pub enum SingerVoice {
    /// Klatt formant backend, no data needed.
    Klatt,
    /// PSOLA backend backed by a host-supplied voice bank.
    Psola,
}

pub struct VoiceBank {
    pub sample_rate: f32,
    /// Short recorded samples per ARPAbet vowel (or selected consonants).
    pub samples: Vec<(Phoneme, Vec<f32>)>,
    /// Reference pitch each sample was recorded at, in Hz.
    pub ref_f0_hz: f32,
}

pub struct SingingSynthesiser {
    sample_rate:    f32,
    backend:        SingerVoice,
    klatt:          KlattSynthesiser,
    voice_bank:     Option<VoiceBank>,
    /// Current target note in MIDI floats.
    target_midi:    f32,
    /// Current vibrato config.
    vibrato_rate_hz: f32,
    vibrato_depth_cents: f32,
    /// Phase accumulator for vibrato LFO.
    vibrato_phase:  f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SungNote {
    pub phoneme:    Phoneme,
    pub midi_note:  f32,
    pub duration_ms: u32,
    pub velocity:   f32,    // [0, 1]
}

impl SingingSynthesiser {
    pub fn new_klatt(sample_rate: f32) -> Self;
    pub fn new_psola(sample_rate: f32, bank: VoiceBank) -> Self;

    pub fn set_vibrato(&mut self, rate_hz: f32, depth_cents: f32);

    /// Set the current sung note. Held until `set_note` is called again.
    pub fn set_note(&mut self, note: SungNote);

    /// Render `n` samples in-place.
    pub fn render(&mut self, out: &mut [f32]);

    pub fn reset(&mut self);
}
```

### Klatt backend

The simplest path. On `set_note`:

1. Convert MIDI to F0: `f0 = 440 · 2^((midi-69)/12)`.
2. Push the phoneme into the underlying Klatt synthesiser.
3. Per-sample `render` loop applies an LFO-modulated F0:
   `f0_now = f0_target · 2^(vibrato_phase / 1200)`
   where `vibrato_phase = depth_cents · sin(2π·rate·t)`.
4. Velocity scales the cascade output gain.

### PSOLA backend

Time-Domain PSOLA (Moulines & Charpentier, 1990) is the workhorse for
small-footprint pitch-shifting. The musica implementation:

1. **Pitch mark** the source sample once at load time: find peaks at
   the source's reference pitch period.
2. **Extract grains** centred on each pitch mark, two periods long,
   Hann-windowed.
3. **Re-space grains** on the output timeline at the *target* period
   (`sr / target_f0`).
4. **Overlap-add** the grains into the output buffer.
5. For sustain notes longer than the source, **loop** the grain stream
   from the middle of the source.

Vibrato is implemented by modulating `target_f0` per output frame, not
per grain — same LFO as the Klatt path.

PSOLA works well for ratios 0.5–2.0 (one octave each side); beyond that
formants distort. We document the recommended range and clamp.

### Realtime contract

- Klatt backend: ~50 multiplies/sample, allocation-free.
- PSOLA backend: O(grain_size) per output sample, allocation-free once
  the bank is loaded.
- Target: <2 µs per 128-sample block (Klatt), <10 µs per block (PSOLA),
  both at 16 kHz.

## Consequences

### Positive
- Closes the singing loop entirely on CPU/edge.
- Two backends cover the quality/data trade-off cleanly:
  - Klatt: zero data, robotic but always available.
  - PSOLA: sample data, recognisable timbre, no neural net.
- Drives downstream things naturally: animator pipelines, karaoke
  apps, virtual idols, accessibility.
- Composes with ADR-154 analysis: a singer can sing along with itself
  in a closed loop.

### Negative
- Klatt backend sounds like 1980s robot. Acceptable as a baseline.
- PSOLA needs a sample bank. Hosts must record/license one.

### Risks
- Pitch-shift artefacts beyond ±octave. Documented range; clamped.

## References
- Moulines & Charpentier, *Pitch-synchronous waveform processing
  techniques for text-to-speech synthesis using diphones*, Speech
  Communication, 1990.
- Vocaloid 1 (Yamaha, 2003) — original commercial PSOLA singing engine.
- ADR-154 (analysis), ADR-155 (Klatt), ADR-159 (pipeline).
