# ADR-149: Binaural Localization — ITD + ILD → Azimuth

## Status
Accepted

## Date
2026-04-12

## Context

The v1 `RealtimeAvatarAnalyzer` (ADRs 145–148) reads only the L channel of
the `AudioBlock`. The R channel is right there, free, and contains the
single most useful spatial cue for any avatar / meeting bot / hearing aid:
**direction of arrival**.

Use cases:
- Avatar head-turn toward whoever is speaking.
- Camera auto-framing for meeting bots.
- Beam-forming front-end for the hearing aid pipeline.
- Per-speaker spatial labelling in `AnalysisFrame`.

ClipCannon's upstream Phoenix engine doesn't compute this — it gets it
implicitly from face landmarks on video. We have **no video**, just two
microphone channels. Classical psychoacoustic localization gives us the
azimuth from those two channels alone.

## Decision

Add `clipcannon/localize.rs` implementing **GCC-PHAT** (Generalised Cross-
Correlation with Phase Transform) for the interaural time difference (ITD)
and a band-limited log-magnitude difference for the interaural level
difference (ILD). Both fuse into an azimuth estimate via a
known-microphone-spacing model.

### Public types

```rust
pub struct LocalizationSnapshot {
    pub itd_us:        f32,   // microseconds, signed (positive = right side)
    pub ild_db:        f32,   // dB, signed
    pub azimuth_deg:   f32,   // -90..+90, 0 = front
    pub confidence:    f32,   // [0, 1]
}

pub struct Localizer {
    sample_rate:    f32,
    window:         usize,
    mic_spacing_m:  f32,      // default 0.18 (typical head width)
    max_lag:        usize,    // bounded by spacing / speed_of_sound
    // preallocated FFT scratch
}

impl Localizer {
    pub fn new(sample_rate: f32, window: usize, mic_spacing_m: f32) -> Self;
    pub fn locate(
        &mut self,
        left_frame:  &[f32],
        right_frame: &[f32],
        l_mags:      &[f32],
        r_mags:      &[f32],
    ) -> LocalizationSnapshot;
}
```

### Algorithm

**ITD via GCC-PHAT** (the SOTA classical approach, robust to reverberation):

```text
X_L = FFT(left)
X_R = FFT(right)
G   = X_L · conj(X_R) / |X_L · conj(X_R)|     (PHAT weighting)
r   = IFFT(G)
itd = argmax(|r|) over lags in [-max_lag, +max_lag]
```

The PHAT weighting whitens the cross-spectrum, so the inverse-FFT peak is
sharp even with broadband background noise. We get sub-sample resolution
via parabolic interpolation around the peak (same trick as ADR-147 §1).

`max_lag = ceil(mic_spacing_m * sample_rate / 343.0)` — roughly 8 samples
at 16 kHz with an 18 cm head, so the search is tiny.

**ILD via band-limited log-magnitude:**

```text
ild_db = 10 · log10( Σ|L|² in [500, 4000 Hz]  /  Σ|R|² in [500, 4000 Hz] )
```

500–4 kHz is the band where head-shadow ILD is psychoacoustically dominant.

**Azimuth fusion:**

ITD-only model (Woodworth approximation):
```text
itd_seconds = (mic_spacing_m / c) · sin(θ)
θ_itd = asin( clamp(itd_seconds * c / mic_spacing_m, -1, 1) )
```

ILD-only model (linear from -10..+10 dB → -60..+60°):
```text
θ_ild = clamp(ild_db / 10.0, -1.0, 1.0) · 60°
```

Final azimuth is a confidence-weighted blend: ITD dominates below 1.5 kHz
(precise but ambiguous in reverb), ILD dominates above. We weight ITD by
peak-to-mean ratio of the GCC-PHAT correlation (sharper peak → higher
weight).

### Realtime contract

- Reuses the FFT scratch from `RealtimeAvatarAnalyzer` (see ADR-153).
- Allocation-free in steady state.
- Target: <30 µs / block at 16 kHz, 256-sample window.

## Consequences

### Positive
- Adds a spatial dimension to `AnalysisFrame` with no extra hardware.
- All-classical, interpretable, no model file.
- Plays nicely with the existing hearing-aid binaural pipeline.

### Negative
- Two-mic localization gives azimuth only — no elevation, no front/back.
- Reverberant rooms reduce ITD precision; we mitigate with PHAT weighting.

### Risks
- The 18 cm mic-spacing default is a *head* model, not a mic-array model.
  Hosts that know their actual array geometry must override
  `mic_spacing_m`.

## References
- Knapp & Carter, *The Generalised Correlation Method for Estimation of
  Time Delay*, IEEE TASSP 1976 (the GCC-PHAT paper).
- Woodworth, *Experimental Psychology*, 1938 (ITD-azimuth model).
- ADR-145, ADR-153.
