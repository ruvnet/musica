//! Signal Analysis bounded context — per-frame prosody features.
//!
//! See [ADR-147](../../../docs/adr/ADR-147-clipcannon-prosody-emotion-speaker.md)
//! for the algorithm rationale, [ADR-153](../../../docs/adr/ADR-153-clipcannon-sota-optimization.md)
//! for the SOTA-optimised path that consumes a [`SharedSpectrum`], and
//! [the domain model](../../../docs/ddd/clipcannon-domain-model.md#31-prosodysnapshot)
//! for invariants.

use core::f32::consts::PI;

use super::spectrum::SharedSpectrum;

/// Per-frame prosody features. Immutable value object.
///
/// All fields are guaranteed finite (no `NaN` / `inf`). `f0_hz == 0.0`
/// indicates "unvoiced".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProsodySnapshot {
    /// Fundamental frequency in Hz, or `0.0` if unvoiced.
    pub f0_hz: f32,
    /// Voicing probability in `[0, 1]`.
    pub voicing: f32,
    /// RMS energy in dB FS (full-scale).
    pub energy_db: f32,
    /// Spectral centroid in Hz.
    pub centroid_hz: f32,
    /// 85% spectral roll-off in Hz.
    pub rolloff_hz: f32,
    /// Zero-crossing rate in `[0, 1]`.
    pub zcr: f32,
    /// Spectral flatness (Wiener entropy) in `[0, 1]`.
    pub flatness: f32,
}

impl ProsodySnapshot {
    /// All-zero snapshot, suitable for "no signal".
    pub const fn silent() -> Self {
        Self {
            f0_hz: 0.0,
            voicing: 0.0,
            energy_db: -120.0,
            centroid_hz: 0.0,
            rolloff_hz: 0.0,
            zcr: 0.0,
            flatness: 1.0,
        }
    }
}

/// Stateless prosody extractor. All buffers are preallocated in [`new`] so
/// [`extract`] is allocation-free.
///
/// The extractor caches a Hann window and an autocorrelation scratch buffer
/// sized to the configured window length.
pub struct ProsodyExtractor {
    sample_rate: f32,
    window: usize,
    /// Hann window of length `window`.
    hann: Vec<f32>,
    /// Scratch windowed time-domain frame.
    scratch: Vec<f32>,
    /// Preallocated ACF buffer indexed by `lag - min_lag`.
    acf: Vec<f32>,
    /// Min lag (samples) corresponding to the F0 search ceiling.
    min_lag: usize,
    /// Max lag (samples) corresponding to the F0 search floor.
    max_lag: usize,
}

impl ProsodyExtractor {
    /// Create a new extractor sized to `window` samples at `sample_rate` Hz.
    ///
    /// F0 search range is fixed to 50–400 Hz which covers adult speech.
    pub fn new(sample_rate: f32, window: usize) -> Self {
        assert!(window >= 64, "prosody window must be ≥64 samples");
        assert!(sample_rate > 0.0, "sample_rate must be positive");

        let mut hann = vec![0.0_f32; window];
        let n = window as f32;
        for (i, h) in hann.iter_mut().enumerate() {
            *h = 0.5 * (1.0 - (2.0 * PI * i as f32 / n).cos());
        }
        let scratch = vec![0.0_f32; window];

        // F0 floor 50 Hz → max_lag, ceiling 400 Hz → min_lag.
        let min_lag = (sample_rate / 400.0).round() as usize;
        let max_lag = ((sample_rate / 50.0).round() as usize).min(window / 2);
        let acf = vec![0.0_f32; max_lag - min_lag + 1];

        Self {
            sample_rate,
            window,
            hann,
            scratch,
            acf,
            min_lag,
            max_lag,
        }
    }

    /// Window length in samples.
    #[inline]
    pub fn window(&self) -> usize {
        self.window
    }

    /// Configured sample rate.
    #[inline]
    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Extract prosody from a precomputed [`SharedSpectrum`] (ADR-153).
    ///
    /// This is the SOTA path: zero re-computation of FFT, ACF, or magnitudes.
    /// It is the path used by `RealtimeAvatarAnalyzer`.
    pub fn extract_from_spectrum(&mut self, spec: &SharedSpectrum) -> ProsodySnapshot {
        debug_assert_eq!(spec.window, self.window);

        // Energy from spectrum's precomputed value.
        let energy_db = spec.energy_db_l;

        // Zero-crossing rate (cheap, time-domain only).
        let mut zc = 0_u32;
        let mut prev = spec.frame_l[0];
        for &x in &spec.frame_l[1..] {
            if (prev >= 0.0) ^ (x >= 0.0) {
                zc += 1;
            }
            prev = x;
        }
        let zcr = zc as f32 / (self.window as f32 - 1.0).max(1.0);

        // F0 from the shared Wiener-Khinchin ACF.
        let r0 = spec.r0_l();
        let mut best_idx = 0_usize;
        let mut best_val = spec.acf_l[self.min_lag];
        let acf = &spec.acf_l;
        for lag in (self.min_lag + 1)..=self.max_lag {
            let v = acf[lag];
            if v > best_val {
                best_val = v;
                best_idx = lag - self.min_lag;
            }
        }
        let best_lag = self.min_lag + best_idx;

        // Wiener-Khinchin ACF is unbiased — no length normalisation needed,
        // because zero-padding to 2N gives a true linear (not circular) ACF.
        let voicing = (best_val / r0).clamp(0.0, 1.0);

        // Sub-sample parabolic interpolation around the peak.
        let f0_hz = if voicing > 0.3 && best_lag > self.min_lag && best_lag + 1 <= self.max_lag {
            let l = acf[best_lag - 1];
            let c = acf[best_lag];
            let r = acf[best_lag + 1];
            let denom = l - 2.0 * c + r;
            let delta = if denom.abs() > 1e-9 {
                0.5 * (l - r) / denom
            } else {
                0.0
            };
            let refined = best_lag as f32 + delta;
            if refined > 0.0 {
                self.sample_rate / refined
            } else {
                0.0
            }
        } else if voicing > 0.3 && best_lag > 0 {
            self.sample_rate / best_lag as f32
        } else {
            0.0
        };

        // Spectral features from the shared mags.
        let (centroid_hz, rolloff_hz, flatness) = spectral_features(&spec.mags_l, self.sample_rate);

        ProsodySnapshot {
            f0_hz: sanitize(f0_hz),
            voicing: sanitize(voicing),
            energy_db: sanitize(energy_db),
            centroid_hz: sanitize(centroid_hz),
            rolloff_hz: sanitize(rolloff_hz),
            zcr: sanitize(zcr),
            flatness: sanitize(flatness),
        }
    }

    /// Extract prosody from one frame and its precomputed magnitude spectrum.
    ///
    /// `frame` must contain exactly `window` time-domain samples. `mags`
    /// must contain at least `window/2 + 1` non-negative magnitudes (the
    /// caller is expected to have already done an STFT pass — typically the
    /// host shares one across multiple analysers).
    ///
    /// If `mags` is empty, spectral fields fall back to time-domain
    /// approximations and `flatness` is set to `1.0`.
    pub fn extract(&mut self, frame: &[f32], mags: &[f32]) -> ProsodySnapshot {
        debug_assert_eq!(frame.len(), self.window, "frame length mismatch");

        // -------- Energy (dB FS) --------
        let mut sum_sq = 0.0_f32;
        for &x in frame {
            sum_sq += x * x;
        }
        let n = self.window as f32;
        let rms = (sum_sq / n).sqrt();
        let energy_db = 20.0 * (rms.max(1e-7)).log10();

        // -------- Zero crossing rate --------
        let mut zc = 0_u32;
        let mut prev = frame[0];
        for &x in &frame[1..] {
            if (prev >= 0.0) ^ (x >= 0.0) {
                zc += 1;
            }
            prev = x;
        }
        let zcr = zc as f32 / (n - 1.0).max(1.0);

        // -------- F0 via autocorrelation --------
        // Use the raw (unwindowed) frame: a Hann taper biases the ACF for
        // short windows that contain only a few periods. Edge artifacts are
        // negligible because we normalise by `r0`.
        for i in 0..self.window {
            self.scratch[i] = frame[i];
        }
        // r0 = ACF at lag 0 (== frame energy).
        let mut r0 = 0.0_f32;
        for &x in &self.scratch {
            r0 += x * x;
        }
        let r0 = r0.max(1e-12);

        // Compute ACF for all lags in [min_lag, max_lag] into scratch buffer.
        for (out_idx, lag) in (self.min_lag..=self.max_lag).enumerate() {
            let end = self.window - lag;
            let mut a0 = 0.0_f32;
            let mut a1 = 0.0_f32;
            let mut a2 = 0.0_f32;
            let mut a3 = 0.0_f32;
            let mut i = 0;
            while i + 4 <= end {
                a0 += self.scratch[i] * self.scratch[i + lag];
                a1 += self.scratch[i + 1] * self.scratch[i + lag + 1];
                a2 += self.scratch[i + 2] * self.scratch[i + lag + 2];
                a3 += self.scratch[i + 3] * self.scratch[i + lag + 3];
                i += 4;
            }
            let mut sum = a0 + a1 + a2 + a3;
            while i < end {
                sum += self.scratch[i] * self.scratch[i + lag];
                i += 1;
            }
            self.acf[out_idx] = sum;
        }

        // Find global maximum across the ACF.
        let mut best_idx = 0_usize;
        let mut best_val = self.acf[0];
        for (idx, &v) in self.acf.iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best_idx = idx;
            }
        }
        let best_lag = self.min_lag + best_idx;

        // Voicing = peak / r0, *unbiased*: each ACF[lag] has only (N-lag)
        // pair contributions vs N for r0, so we rescale so a pure tone gives
        // close to 1.0.
        let n_eff = (self.window - best_lag).max(1) as f32;
        let voicing = (best_val * (self.window as f32) / (n_eff * r0)).clamp(0.0, 1.0);

        // Parabolic interpolation around the global max for sub-sample accuracy.
        let f0_hz = if voicing > 0.3 && best_idx > 0 && best_idx + 1 < self.acf.len() {
            let l = self.acf[best_idx - 1];
            let c = self.acf[best_idx];
            let r = self.acf[best_idx + 1];
            let denom = l - 2.0 * c + r;
            let delta = if denom.abs() > 1e-9 {
                0.5 * (l - r) / denom
            } else {
                0.0
            };
            let refined_lag = best_lag as f32 + delta;
            if refined_lag > 0.0 {
                self.sample_rate / refined_lag
            } else {
                0.0
            }
        } else if voicing > 0.3 && best_lag > 0 {
            self.sample_rate / best_lag as f32
        } else {
            0.0
        };

        // -------- Spectral features --------
        let (centroid_hz, rolloff_hz, flatness) = if !mags.is_empty() {
            spectral_features(mags, self.sample_rate)
        } else {
            (0.0, 0.0, 1.0)
        };

        ProsodySnapshot {
            f0_hz: sanitize(f0_hz),
            voicing: sanitize(voicing),
            energy_db: sanitize(energy_db),
            centroid_hz: sanitize(centroid_hz),
            rolloff_hz: sanitize(rolloff_hz),
            zcr: sanitize(zcr),
            flatness: sanitize(flatness),
        }
    }
}

#[inline]
fn sanitize(x: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        0.0
    }
}

/// Compute spectral centroid, 85% roll-off and Wiener-entropy flatness.
///
/// `mags` is `[|X[0]|, |X[1]|, ..., |X[B-1]|]` where `B = window/2 + 1`.
fn spectral_features(mags: &[f32], sample_rate: f32) -> (f32, f32, f32) {
    let b = mags.len();
    if b == 0 {
        return (0.0, 0.0, 1.0);
    }
    let nyq = sample_rate * 0.5;
    let bin_hz = nyq / (b as f32 - 1.0).max(1.0);

    let mut total_mag = 0.0_f32;
    let mut weighted = 0.0_f32;
    let mut total_pow = 0.0_f32;
    let mut log_sum = 0.0_f32;
    let mut nz = 0_u32;
    for (i, &m) in mags.iter().enumerate() {
        let m = m.max(0.0);
        total_mag += m;
        weighted += m * (i as f32 * bin_hz);
        let p = m * m;
        total_pow += p;
        if m > 1e-9 {
            log_sum += m.ln();
            nz += 1;
        }
    }
    let centroid_hz = if total_mag > 1e-9 {
        weighted / total_mag
    } else {
        0.0
    };

    // 85% rolloff
    let target = 0.85 * total_pow;
    let mut acc = 0.0_f32;
    let mut rolloff_bin = b - 1;
    for (i, &m) in mags.iter().enumerate() {
        acc += m * m;
        if acc >= target {
            rolloff_bin = i;
            break;
        }
    }
    let rolloff_hz = rolloff_bin as f32 * bin_hz;

    // Spectral flatness = geo_mean / arith_mean.
    let flatness = if nz > 4 && total_mag > 1e-9 {
        let geo = (log_sum / nz as f32).exp();
        let arith = total_mag / b as f32;
        (geo / arith).clamp(0.0, 1.0)
    } else {
        1.0
    };

    (centroid_hz, rolloff_hz, flatness)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sr: f32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin() * amp)
            .collect()
    }

    fn naive_mags(frame: &[f32]) -> Vec<f32> {
        // tiny DFT only used in tests (window is 256 → 33 bins typical)
        let n = frame.len();
        let b = n / 2 + 1;
        let mut mags = vec![0.0_f32; b];
        for k in 0..b {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            for i in 0..n {
                let a = -2.0 * PI * (k * i) as f32 / n as f32;
                re += frame[i] * a.cos();
                im += frame[i] * a.sin();
            }
            mags[k] = (re * re + im * im).sqrt();
        }
        mags
    }

    #[test]
    fn silent_frame_is_unvoiced() {
        let mut ext = ProsodyExtractor::new(16_000.0, 256);
        let frame = vec![0.0_f32; 256];
        let mags = vec![0.0_f32; 129];
        let snap = ext.extract(&frame, &mags);
        assert_eq!(snap.f0_hz, 0.0);
        assert!(snap.voicing < 0.10, "voicing = {}", snap.voicing);
        assert!(snap.energy_db < -100.0);
    }

    #[test]
    fn pure_tone_recovers_f0() {
        let mut ext = ProsodyExtractor::new(16_000.0, 256);
        let frame = sine(220.0, 16_000.0, 256, 0.6);
        let mags = naive_mags(&frame);
        let snap = ext.extract(&frame, &mags);
        assert!(
            (snap.f0_hz - 220.0).abs() < 4.0,
            "expected ~220 Hz, got {}",
            snap.f0_hz
        );
        assert!(snap.voicing > 0.85, "voicing = {}", snap.voicing);
    }

    #[test]
    fn full_scale_tone_has_zero_db_ish_energy() {
        let mut ext = ProsodyExtractor::new(16_000.0, 256);
        let frame = sine(1000.0, 16_000.0, 256, 1.0);
        let mags = naive_mags(&frame);
        let snap = ext.extract(&frame, &mags);
        // RMS of unit sine is 1/√2 → −3 dB FS
        assert!(
            (snap.energy_db + 3.0).abs() < 1.0,
            "expected ~−3 dB, got {}",
            snap.energy_db
        );
    }

    #[test]
    fn centroid_is_near_tone_freq() {
        let mut ext = ProsodyExtractor::new(16_000.0, 256);
        let frame = sine(2000.0, 16_000.0, 256, 0.5);
        let mags = naive_mags(&frame);
        let snap = ext.extract(&frame, &mags);
        assert!(
            (snap.centroid_hz - 2000.0).abs() < 250.0,
            "centroid = {}",
            snap.centroid_hz
        );
    }

    #[test]
    fn fields_are_finite() {
        let mut ext = ProsodyExtractor::new(16_000.0, 256);
        // White-noise-ish input.
        let mut x = 0.123_f32;
        let frame: Vec<f32> = (0..256)
            .map(|_| {
                x = (x * 1664525.0 + 1013904223.0) % 1.0;
                x - 0.5
            })
            .collect();
        let mags = naive_mags(&frame);
        let s = ext.extract(&frame, &mags);
        assert!(s.f0_hz.is_finite());
        assert!(s.voicing.is_finite());
        assert!(s.energy_db.is_finite());
        assert!(s.centroid_hz.is_finite());
        assert!(s.rolloff_hz.is_finite());
        assert!(s.zcr.is_finite());
        assert!(s.flatness.is_finite());
        assert!((0.0..=1.0).contains(&s.voicing));
        assert!((0.0..=1.0).contains(&s.zcr));
        assert!((0.0..=1.0).contains(&s.flatness));
    }
}
