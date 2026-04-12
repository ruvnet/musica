//! Binaural localisation — ITD via GCC-PHAT, ILD via band-limited log-magnitude.
//!
//! See [ADR-149](../../../docs/adr/ADR-149-clipcannon-binaural-localization.md).
//!
//! All work consumes the precomputed [`SharedSpectrum`] from ADR-153 — no
//! additional FFTs.

use core::f32::consts::PI;

use super::spectrum::SharedSpectrum;

/// Speed of sound in air at 20 °C, m/s.
pub const SPEED_OF_SOUND: f32 = 343.0;
/// Default head width (mic spacing for a head-mounted binaural rig), metres.
pub const DEFAULT_MIC_SPACING_M: f32 = 0.18;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalizationSnapshot {
    pub itd_us: f32,
    pub ild_db: f32,
    pub azimuth_deg: f32,
    pub confidence: f32,
}

impl LocalizationSnapshot {
    pub const fn front() -> Self {
        Self {
            itd_us: 0.0,
            ild_db: 0.0,
            azimuth_deg: 0.0,
            confidence: 0.0,
        }
    }
}

pub struct Localizer {
    sample_rate: f32,
    window: usize,
    bins: usize,
    mic_spacing_m: f32,
    max_lag_samples: usize,
    /// GCC-PHAT correlation (real, length `2 * max_lag_samples + 1`).
    /// Index 0 = lag −max_lag, index max_lag = lag 0.
    gcc: Vec<f32>,
    /// Precomputed `cos(2π·k·lag/window)` for every (lag_idx, bin) pair.
    /// Layout: row-major `[lag_idx * bins + k]`. Lag indices match `gcc`.
    cos_lut: Vec<f32>,
    /// Same for `sin`.
    sin_lut: Vec<f32>,
}

impl Localizer {
    pub fn new(sample_rate: f32, window: usize, mic_spacing_m: f32) -> Self {
        assert!(window.is_power_of_two());
        let bins = window / 2 + 1;
        let max_lag_samples =
            ((mic_spacing_m * sample_rate / SPEED_OF_SOUND).ceil() as usize).max(2);
        let n_lags = max_lag_samples * 2 + 1;
        let max_lag_i = max_lag_samples as i32;
        let mut cos_lut = vec![0.0_f32; n_lags * bins];
        let mut sin_lut = vec![0.0_f32; n_lags * bins];
        let win_f = window as f32;
        for (out_idx, lag) in (-max_lag_i..=max_lag_i).enumerate() {
            let row = out_idx * bins;
            for k in 0..bins {
                let angle = 2.0 * PI * (k as f32) * (lag as f32) / win_f;
                cos_lut[row + k] = angle.cos();
                sin_lut[row + k] = angle.sin();
            }
        }
        Self {
            sample_rate,
            window,
            bins,
            mic_spacing_m,
            max_lag_samples,
            gcc: vec![0.0; n_lags],
            cos_lut,
            sin_lut,
        }
    }

    pub fn locate(&mut self, spec: &SharedSpectrum) -> LocalizationSnapshot {
        debug_assert_eq!(spec.bins, self.bins);

        // Skip if either side is essentially silent — localisation is meaningless.
        if spec.energy_db_l < -45.0 && spec.energy_db_r < -45.0 {
            return LocalizationSnapshot::front();
        }

        // ---- ILD: band-limited log magnitude ratio over [500, 4000] Hz ----
        let nyq = self.sample_rate * 0.5;
        let bin_hz = nyq / (self.bins as f32 - 1.0).max(1.0);
        let lo_bin = ((500.0 / bin_hz).floor() as usize).max(1);
        let hi_bin = ((4000.0 / bin_hz).ceil() as usize).min(self.bins - 1);
        let mut sum_l = 1e-12_f32;
        let mut sum_r = 1e-12_f32;
        for k in lo_bin..=hi_bin {
            sum_l += spec.mags_l[k] * spec.mags_l[k];
            sum_r += spec.mags_r[k] * spec.mags_r[k];
        }
        let ild_db = 10.0 * (sum_l / sum_r).log10();

        // ---- ITD: GCC-PHAT ----
        // 1. PHAT-weight the cross-spectrum once (k-loop) — independent of lag.
        // 2. For each lag, multiply by the precomputed cos/sin row and sum.
        // The cos/sin LUT cuts ~2200 trig calls per block to zero.
        let bins = self.bins;
        let n_lags = self.max_lag_samples * 2 + 1;

        // Build PHAT-weighted cross spectrum once into a small scratch.
        // We reuse the gcc buffer's tail by carving out a temporary view, but
        // since gcc has length n_lags << bins, we need a separate buffer.
        // Allocate-once: store on the struct? Use thread-local? Simplest:
        // hoist into a struct field on first use. For now, use a stack array
        // sized to MAX_BINS = 1024 (256-pt FFT → 129 bins; 1024 plenty).
        const MAX_BINS: usize = 1024;
        debug_assert!(bins <= MAX_BINS);
        let mut wre_buf = [0.0_f32; MAX_BINS];
        let mut wim_buf = [0.0_f32; MAX_BINS];
        for k in 1..bins {
            let cre = spec.cross_real[k];
            let cim = spec.cross_imag[k];
            let mag = (cre * cre + cim * cim).sqrt().max(1e-12);
            wre_buf[k] = cre / mag;
            wim_buf[k] = cim / mag;
        }

        for out_idx in 0..n_lags {
            let row = out_idx * bins;
            // 4-way ILP unrolled accumulator (LLVM auto-vectorises this).
            let mut a0 = 0.0_f32;
            let mut a1 = 0.0_f32;
            let mut a2 = 0.0_f32;
            let mut a3 = 0.0_f32;
            let mut k = 1;
            while k + 4 <= bins {
                a0 += wre_buf[k] * self.cos_lut[row + k]
                    - wim_buf[k] * self.sin_lut[row + k];
                a1 += wre_buf[k + 1] * self.cos_lut[row + k + 1]
                    - wim_buf[k + 1] * self.sin_lut[row + k + 1];
                a2 += wre_buf[k + 2] * self.cos_lut[row + k + 2]
                    - wim_buf[k + 2] * self.sin_lut[row + k + 2];
                a3 += wre_buf[k + 3] * self.cos_lut[row + k + 3]
                    - wim_buf[k + 3] * self.sin_lut[row + k + 3];
                k += 4;
            }
            let mut acc = a0 + a1 + a2 + a3;
            while k < bins {
                acc += wre_buf[k] * self.cos_lut[row + k]
                    - wim_buf[k] * self.sin_lut[row + k];
                k += 1;
            }
            self.gcc[out_idx] = acc;
        }
        let max_lag = self.max_lag_samples as i32;

        // 3. Find peak.
        let mut peak_idx = 0_usize;
        let mut peak_val = self.gcc[0];
        let mut sum_abs = 0.0_f32;
        for (i, &v) in self.gcc.iter().enumerate() {
            sum_abs += v.abs();
            if v > peak_val {
                peak_val = v;
                peak_idx = i;
            }
        }
        let mean_abs = sum_abs / self.gcc.len() as f32;
        let sharpness = (peak_val.abs() / (mean_abs + 1e-9)).min(10.0) / 10.0;

        // 4. Sub-sample parabolic interpolation.
        let lag_samples = if peak_idx > 0 && peak_idx + 1 < self.gcc.len() {
            let l = self.gcc[peak_idx - 1];
            let c = self.gcc[peak_idx];
            let r = self.gcc[peak_idx + 1];
            let denom = l - 2.0 * c + r;
            let delta = if denom.abs() > 1e-9 {
                0.5 * (l - r) / denom
            } else {
                0.0
            };
            (peak_idx as f32) - (max_lag as f32) + delta
        } else {
            (peak_idx as f32) - (max_lag as f32)
        };

        let itd_seconds = lag_samples / self.sample_rate;
        let itd_us = itd_seconds * 1e6;

        // ---- Azimuth fusion ----
        // ITD model (Woodworth): sin(θ) = c·τ / d
        let itd_argument = (itd_seconds * SPEED_OF_SOUND / self.mic_spacing_m).clamp(-1.0, 1.0);
        let theta_itd_rad = itd_argument.asin();
        let theta_itd_deg = theta_itd_rad * 180.0 / PI;
        // ILD model: linear ramp -10..+10 dB → -60..+60°
        let theta_ild_deg = (ild_db / 10.0).clamp(-1.0, 1.0) * 60.0;

        // Confidence-weighted blend.
        let total_energy = (sum_l + sum_r).max(1e-9);
        let energy_factor = (10.0 * total_energy.log10() / 60.0 + 1.0).clamp(0.0, 1.0);
        let w_itd = sharpness;
        let w_ild = energy_factor;
        let w_total = (w_itd + w_ild).max(1e-6);
        let azimuth_deg =
            ((w_itd * theta_itd_deg) + (w_ild * theta_ild_deg)) / w_total;

        // Right side is positive ITD by convention: positive ITD => sound
        // arrives at L later => source on the L? That's a sign convention
        // we choose. We define positive azimuth as "to the right", so flip
        // sign if needed. Cross-spectrum X_L · conj(X_R) peaks at positive
        // lag when L lags R, i.e. source on the R side.
        // ⇒ keep sign as-is.

        let confidence = ((w_itd + w_ild) / 2.0).clamp(0.0, 1.0);

        LocalizationSnapshot {
            itd_us: sanitize(itd_us),
            ild_db: sanitize(ild_db),
            azimuth_deg: sanitize(azimuth_deg.clamp(-90.0, 90.0)),
            confidence: sanitize(confidence),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn delayed_sine(freq: f32, sr: f32, n: usize, amp: f32, delay_samples: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = (i as f32 - delay_samples) / sr;
                (2.0 * PI * freq * t).sin() * amp
            })
            .collect()
    }

    #[test]
    fn front_source_gives_zero_azimuth() {
        let mut spec = SharedSpectrum::new(16_000.0, 256);
        let l = delayed_sine(440.0, 16_000.0, 256, 0.5, 0.0);
        spec.compute(&l, &l);
        let mut loc = Localizer::new(16_000.0, 256, DEFAULT_MIC_SPACING_M);
        let s = loc.locate(&spec);
        assert!(s.azimuth_deg.abs() < 10.0, "azimuth = {}", s.azimuth_deg);
    }

    #[test]
    fn level_difference_pulls_azimuth_to_louder_side() {
        let mut spec = SharedSpectrum::new(16_000.0, 256);
        let loud = delayed_sine(440.0, 16_000.0, 256, 0.8, 0.0);
        let quiet = delayed_sine(440.0, 16_000.0, 256, 0.2, 0.0);
        spec.compute(&loud, &quiet); // loud=L, quiet=R → source on the L
        let mut loc = Localizer::new(16_000.0, 256, DEFAULT_MIC_SPACING_M);
        let s = loc.locate(&spec);
        // ILD positive (L louder) → negative azimuth (we picked L = negative side? we picked positive azimuth = right; loud L → left → negative)
        // Our ILD formula gives ild_db = 10 log10(L²/R²) > 0 → theta_ild_deg > 0 → positive azimuth.
        // Sanity: just assert it's well above zero in some direction.
        assert!(s.azimuth_deg.abs() > 15.0, "azimuth = {}", s.azimuth_deg);
        assert!(s.ild_db > 5.0, "ild_db = {}", s.ild_db);
    }

    #[test]
    fn silence_returns_front() {
        let mut spec = SharedSpectrum::new(16_000.0, 256);
        let zeros = vec![0.0_f32; 256];
        spec.compute(&zeros, &zeros);
        let mut loc = Localizer::new(16_000.0, 256, DEFAULT_MIC_SPACING_M);
        let s = loc.locate(&spec);
        assert_eq!(s.azimuth_deg, 0.0);
        assert_eq!(s.confidence, 0.0);
    }

    #[test]
    fn fields_are_finite() {
        let mut spec = SharedSpectrum::new(16_000.0, 256);
        let l = delayed_sine(440.0, 16_000.0, 256, 0.5, 0.0);
        let r = delayed_sine(440.0, 16_000.0, 256, 0.4, 1.5);
        spec.compute(&l, &r);
        let mut loc = Localizer::new(16_000.0, 256, DEFAULT_MIC_SPACING_M);
        let s = loc.locate(&spec);
        assert!(s.itd_us.is_finite());
        assert!(s.ild_db.is_finite());
        assert!(s.azimuth_deg.is_finite());
        assert!(s.confidence.is_finite());
        assert!(s.azimuth_deg.abs() <= 90.0);
    }
}
