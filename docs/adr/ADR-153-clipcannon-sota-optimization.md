# ADR-153: SOTA Optimisation — Shared FFT, Wiener-Khinchin ACF, Wide ILP

## Status
Accepted

## Date
2026-04-12

## Context

ADR-148 baseline measured the v1 analyser at:

```
prosody_frame   18.24 µs   (target <80 µs)
analyzer_block  26.04 µs   (target <250 µs)
```

The dominant cost (≈70%) is the prosody extractor's autocorrelation
inner loop, which is **independent** of the FFT the pipeline already runs
to compute magnitudes. Two FFT-sized passes touch the same data twice.
That's silly.

Worse: ADRs 149, 150, 151 add four new analyses (`Localizer`,
`VadDetector`, `EmotionVector` smoother, `MusicSpeechDetector`). Naïvely,
each new analysis would add another FFT or another scan over `mags`.
Without consolidation we'd grow from one 256-pt FFT/block to four.

## Decision

Restructure the hot path around a single shared **`SharedSpectrum`**
context, computed once per block, that all downstream analyses borrow.
Combined with a Wiener-Khinchin trick this **eliminates** the dedicated
ACF pass entirely.

### 1. Shared spectrum context

```rust
pub(crate) struct SharedSpectrum {
    pub window:        usize,
    pub sample_rate:   f32,
    pub frame_l:       Vec<f32>,    // raw L channel, length window
    pub frame_r:       Vec<f32>,    // raw R channel, length window
    pub mags_l:        Vec<f32>,    // length window/2 + 1
    pub mags_r:        Vec<f32>,    // length window/2 + 1
    pub power_l:       Vec<f32>,    // |X|² (precomputed, used by ACF & spectral feats)
    pub acf_l:         Vec<f32>,    // length window, computed via Wiener-Khinchin
    pub cross_real:    Vec<f32>,    // Re(X_L · conj(X_R)) for GCC-PHAT
    pub cross_imag:    Vec<f32>,    // Im(X_L · conj(X_R))
    pub r0_l:          f32,         // ACF[0] = energy
}
```

The pipeline computes `SharedSpectrum` once per block via a single
`compute(&mut self, block)` call. All downstream extractors take
`&SharedSpectrum` as input.

### 2. Wiener-Khinchin ACF

> ACF(τ) = IFFT( |FFT(x)|² )

We already compute `|X|²` for spectral features. One additional in-place
IFFT (the same radix-2 we built for ADR-145) on the power spectrum gives
us the full ACF for free. Cost ≈ one FFT, replacing an O(N·L) explicit
ACF that was O(256·240) ≈ 60 k multiplies.

For N=256: FFT is 256·8 ≈ 2 k operations. **Direct ACF was ~30× more
expensive than the Wiener-Khinchin path** — and the optimiser was already
beating it because the explicit ACF didn't share work with the FFT.

### 3. Cross-spectrum reuse for GCC-PHAT

`Localizer` consumes `cross_real` / `cross_imag` directly — they're the
real and imaginary parts of `X_L · conj(X_R)`, computed once per block.
GCC-PHAT then divides by magnitude and IFFTs in place into a small
`gcc` scratch buffer.

### 4. Wide ILP (no SIMD intrinsics)

Even without `std::simd` we can squeeze ~1.6× out of the inner loops by
**8-way ILP** unrolling instead of the current 4-way. LLVM auto-vectorises
the unrolled scalar loops to AVX2 on x86 and NEON on ARM, since musica
forbids `unsafe` SIMD intrinsics.

Pattern:

```rust
let mut a0 = 0.0; let mut a1 = 0.0; let mut a2 = 0.0; let mut a3 = 0.0;
let mut a4 = 0.0; let mut a5 = 0.0; let mut a6 = 0.0; let mut a7 = 0.0;
let mut i = 0;
while i + 8 <= n {
    a0 += x[i]   * y[i];
    a1 += x[i+1] * y[i+1];
    a2 += x[i+2] * y[i+2];
    a3 += x[i+3] * y[i+3];
    a4 += x[i+4] * y[i+4];
    a5 += x[i+5] * y[i+5];
    a6 += x[i+6] * y[i+6];
    a7 += x[i+7] * y[i+7];
    i += 8;
}
let sum = (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
```

Apply to: spectral feature accumulators (`spectral_features`), the
log-mel projection in `speaker_embed`, the cosine similarity, and the
band ratios in `viseme`.

### 5. Inverse-Hann elimination of windowing constants

The FFT path already multiplies by Hann. The cross-spectrum is computed
on **windowed** signals so the GCC-PHAT peak inherits the same gain
profile in both channels, cancelling exactly. No correction needed.

### 6. Branch elimination in viseme classifier

Replace the if-cascade with a small lookup table indexed by quantised
(f1_bucket, f2_bucket, voiced_bit). 24 entries, fits in L1.

## Targets (release build, 16 kHz, 128-sample blocks)

| Bench                | Pre  (v1) | Target (v2) | Notes                              |
|----------------------|----------:|------------:|------------------------------------|
| `prosody_frame`      |  18.2 µs  |  **<10 µs** | Wiener-Khinchin ACF + 8-wide ILP   |
| `viseme_map`         |   0.88 µs |  **<0.6 µs**| LUT classifier                     |
| `speaker_observe`    |   1.64 µs |  **<1.1 µs**| 8-wide cosine + log_mel            |
| `localize_block`     |   (new)   |  **<25 µs** | shared cross-spectrum              |
| `vad_observe`        |   (new)   |  **<0.3 µs**| pure state machine                 |
| `analyzer_block`     |  26.0 µs  |  **<22 µs** | one FFT total (was effectively 1 + ACF) |
| `analyzer_composite` |  3.20 ms  |  **<2.7 ms**| 1 s of audio                       |

The composite target is *more* than just the sum of the per-stage targets
because it includes four new analyses on top.

## Stopping criterion

Same as ADR-148: means meet target **and** p99 within 3× of mean **and**
all oracle tests pass.

## Consequences

### Positive
- Single FFT per block regardless of how many downstream analyses run.
- Adding a 5th, 6th, 7th analysis is now ~free.
- ACF accuracy actually *improves* (Wiener-Khinchin is exact, bias-free).

### Negative
- Slightly more memory: ~2 KB of additional scratch per analyser instance.
- Wiener-Khinchin ACF gives a *circular* autocorrelation; we zero-pad to
  2N to avoid wrap-around (one extra FFT stage).

### Risks
- The Hann-window cancellation in GCC-PHAT relies on identical L/R
  windowing. Tested explicitly.

## References
- Wiener-Khinchin theorem.
- Knapp & Carter 1976 (GCC-PHAT).
- ADR-148 (baseline & methodology).
