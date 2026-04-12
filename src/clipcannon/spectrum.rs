//! Shared per-block spectral context — see ADR-153.
//!
//! `SharedSpectrum` is the single source of truth for everything an analyser
//! might want to know about one block of audio: time-domain frames for both
//! channels, magnitude spectra, the L-channel power spectrum, the L-channel
//! Wiener-Khinchin autocorrelation, and the cross-spectrum used by GCC-PHAT
//! localisation.
//!
//! Computed once per block by [`SharedSpectrum::compute`]. All downstream
//! analyses borrow `&SharedSpectrum` — no double-FFTs, no double-windowing.
//!
//! Realtime contract: `compute` is allocation-free in steady state. The only
//! allocations happen in [`SharedSpectrum::new`].

use core::f32::consts::PI;

/// Shared per-block spectral context.
pub struct SharedSpectrum {
    pub window: usize,
    pub sample_rate: f32,
    pub bins: usize, // = window/2 + 1

    /// Raw L channel time-domain frame (length `window`).
    pub frame_l: Vec<f32>,
    /// Raw R channel time-domain frame (length `window`).
    pub frame_r: Vec<f32>,
    /// L magnitude spectrum, length `bins`.
    pub mags_l: Vec<f32>,
    /// R magnitude spectrum, length `bins`.
    pub mags_r: Vec<f32>,
    /// L power spectrum |X|², length `bins`. Reused by spectral feats + ACF.
    pub power_l: Vec<f32>,
    /// L Wiener-Khinchin autocorrelation, length `window`. Index = lag.
    pub acf_l: Vec<f32>,
    /// Re(X_L · conj(X_R)), length `bins`. Used by GCC-PHAT.
    pub cross_real: Vec<f32>,
    /// Im(X_L · conj(X_R)), length `bins`.
    pub cross_imag: Vec<f32>,
    /// L energy in dB FS (computed from frame_l, not from mags).
    pub energy_db_l: f32,
    /// R energy in dB FS.
    pub energy_db_r: f32,

    // ---- internal scratch ----
    hann: Vec<f32>,
    fft_size: usize, // 2*window for non-circular ACF (zero-padded)
    bitrev: Vec<usize>,
    twiddles: Vec<Vec<(f32, f32)>>,
    re_scratch: Vec<f32>,
    im_scratch: Vec<f32>,
    re_l: Vec<f32>, // length `bins`
    im_l: Vec<f32>,
    re_r: Vec<f32>,
    im_r: Vec<f32>,
}

impl SharedSpectrum {
    /// Create a new shared spectrum sized to `window` samples at `sample_rate` Hz.
    pub fn new(sample_rate: f32, window: usize) -> Self {
        assert!(window.is_power_of_two(), "window must be power of two");
        assert!(window >= 64);
        let bins = window / 2 + 1;
        let fft_size = window * 2; // zero-padded so ACF is non-circular

        let hann: Vec<f32> = (0..window)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / window as f32).cos()))
            .collect();

        let bitrev = build_bitrev(fft_size);
        let twiddles = build_twiddles(fft_size);

        Self {
            window,
            sample_rate,
            bins,
            frame_l: vec![0.0; window],
            frame_r: vec![0.0; window],
            mags_l: vec![0.0; bins],
            mags_r: vec![0.0; bins],
            power_l: vec![0.0; bins],
            acf_l: vec![0.0; window],
            cross_real: vec![0.0; bins],
            cross_imag: vec![0.0; bins],
            energy_db_l: -120.0,
            energy_db_r: -120.0,
            hann,
            fft_size,
            bitrev,
            twiddles,
            re_scratch: vec![0.0; fft_size],
            im_scratch: vec![0.0; fft_size],
            re_l: vec![0.0; bins],
            im_l: vec![0.0; bins],
            re_r: vec![0.0; bins],
            im_r: vec![0.0; bins],
        }
    }

    /// Compute everything from raw L/R buffers of exactly `window` samples.
    pub fn compute(&mut self, left: &[f32], right: &[f32]) {
        debug_assert_eq!(left.len(), self.window);
        debug_assert_eq!(right.len(), self.window);
        let n = self.window;
        let nfft = self.fft_size;

        // Copy raw frames.
        self.frame_l[..n].copy_from_slice(left);
        self.frame_r[..n].copy_from_slice(right);

        // ---- L channel: zero-padded FFT ----
        // re/im scratch is `nfft = 2*n` for non-circular Wiener-Khinchin ACF.
        for i in 0..n {
            self.re_scratch[i] = left[i] * self.hann[i];
            self.im_scratch[i] = 0.0;
        }
        for i in n..nfft {
            self.re_scratch[i] = 0.0;
            self.im_scratch[i] = 0.0;
        }
        fft_radix2(
            &mut self.re_scratch,
            &mut self.im_scratch,
            &self.bitrev,
            &self.twiddles,
        );

        // Save the first `bins` values for reuse (mags and cross-spectrum).
        for k in 0..self.bins {
            self.re_l[k] = self.re_scratch[k];
            self.im_l[k] = self.im_scratch[k];
        }

        // |X_L|² across the *full* FFT length for IFFT step.
        for i in 0..nfft {
            let re = self.re_scratch[i];
            let im = self.im_scratch[i];
            let p = re * re + im * im;
            self.re_scratch[i] = p;
            self.im_scratch[i] = 0.0;
        }
        // power_l (length bins) for downstream spectral features.
        for k in 0..self.bins {
            self.power_l[k] = self.re_scratch[k];
            self.mags_l[k] = self.power_l[k].sqrt();
        }

        // Inverse FFT of |X|² → ACF (Wiener-Khinchin). Use forward FFT on
        // conjugate then conjugate the result; equivalent and avoids a
        // dedicated IFFT routine.
        // Since power is real and even, we can use forward FFT directly:
        fft_radix2(
            &mut self.re_scratch,
            &mut self.im_scratch,
            &self.bitrev,
            &self.twiddles,
        );
        // The result, scaled by 1/nfft, is the autocorrelation. The values at
        // indices 0..n correspond to lags 0..n.
        let scale = 1.0 / nfft as f32;
        for lag in 0..n {
            self.acf_l[lag] = self.re_scratch[lag] * scale;
        }

        // ---- R channel: same FFT, but smaller (we don't need ACF for R) ----
        for i in 0..n {
            self.re_scratch[i] = right[i] * self.hann[i];
            self.im_scratch[i] = 0.0;
        }
        for i in n..nfft {
            self.re_scratch[i] = 0.0;
            self.im_scratch[i] = 0.0;
        }
        fft_radix2(
            &mut self.re_scratch,
            &mut self.im_scratch,
            &self.bitrev,
            &self.twiddles,
        );
        for k in 0..self.bins {
            self.re_r[k] = self.re_scratch[k];
            self.im_r[k] = self.im_scratch[k];
            let re = self.re_r[k];
            let im = self.im_r[k];
            self.mags_r[k] = (re * re + im * im).sqrt();
        }

        // ---- Cross-spectrum X_L · conj(X_R) for GCC-PHAT (ADR-149) ----
        for k in 0..self.bins {
            // (a+bi)(c-di) = (ac+bd) + (bc-ad)i
            let a = self.re_l[k];
            let b = self.im_l[k];
            let c = self.re_r[k];
            let d = self.im_r[k];
            self.cross_real[k] = a * c + b * d;
            self.cross_imag[k] = b * c - a * d;
        }

        // ---- Time-domain energies in dB FS ----
        self.energy_db_l = energy_db(left);
        self.energy_db_r = energy_db(right);
    }

    /// L-channel zero-lag autocorrelation (== windowed energy of the frame).
    #[inline]
    pub fn r0_l(&self) -> f32 {
        self.acf_l[0].max(1e-12)
    }
}

#[inline]
fn energy_db(frame: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &x in frame {
        sum += x * x;
    }
    let rms = (sum / frame.len().max(1) as f32).sqrt();
    20.0 * rms.max(1e-7).log10()
}

// ---------- Tiny f32 radix-2 FFT (private to this module) ----------

fn build_bitrev(n: usize) -> Vec<usize> {
    let mut v = vec![0_usize; n];
    let bits = (n as u64).trailing_zeros();
    for i in 0..n {
        let mut x = i;
        let mut r = 0;
        for _ in 0..bits {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        v[i] = r;
    }
    v
}

fn build_twiddles(n: usize) -> Vec<Vec<(f32, f32)>> {
    let mut stages = Vec::new();
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f32;
        let twiddles: Vec<(f32, f32)> = (0..half)
            .map(|k| {
                let a = angle * k as f32;
                (a.cos(), a.sin())
            })
            .collect();
        stages.push(twiddles);
        len <<= 1;
    }
    stages
}

fn fft_radix2(re: &mut [f32], im: &mut [f32], bitrev: &[usize], twiddles: &[Vec<(f32, f32)>]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

    for i in 0..n {
        let j = bitrev[i];
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2;
    let mut stage = 0;
    while len <= n {
        let half = len / 2;
        let tw = &twiddles[stage];
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let (cos_t, sin_t) = tw[k];
                let i_a = start + k;
                let i_b = i_a + half;
                let xr = re[i_b] * cos_t - im[i_b] * sin_t;
                let xi = re[i_b] * sin_t + im[i_b] * cos_t;
                re[i_b] = re[i_a] - xr;
                im[i_b] = im[i_a] - xi;
                re[i_a] += xr;
                im[i_a] += xi;
            }
            start += len;
        }
        len <<= 1;
        stage += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, sr: f32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin() * amp)
            .collect()
    }

    #[test]
    fn compute_does_not_panic() {
        let mut s = SharedSpectrum::new(16_000.0, 256);
        let l = sine(220.0, 16_000.0, 256, 0.5);
        let r = sine(220.0, 16_000.0, 256, 0.5);
        s.compute(&l, &r);
        assert!(s.mags_l.iter().any(|&m| m > 0.0));
        assert!(s.energy_db_l > -50.0);
    }

    #[test]
    fn acf_lag_zero_is_largest() {
        let mut s = SharedSpectrum::new(16_000.0, 256);
        let l = sine(220.0, 16_000.0, 256, 0.5);
        let r = vec![0.0; 256];
        s.compute(&l, &r);
        let mut best_lag = 0;
        let mut best = s.acf_l[0];
        for (lag, &v) in s.acf_l.iter().enumerate() {
            if v > best {
                best = v;
                best_lag = lag;
            }
        }
        assert_eq!(best_lag, 0, "ACF should peak at lag 0");
    }

    #[test]
    fn acf_recovers_period_for_pure_tone() {
        let mut s = SharedSpectrum::new(16_000.0, 256);
        let l = sine(220.0, 16_000.0, 256, 0.5);
        let r = vec![0.0; 256];
        s.compute(&l, &r);
        // Find max in [40, 128] (50–400 Hz F0 range).
        let mut best_lag = 40;
        let mut best = f32::MIN;
        for lag in 40..=128 {
            if s.acf_l[lag] > best {
                best = s.acf_l[lag];
                best_lag = lag;
            }
        }
        // Period at 220 Hz = 16000/220 ≈ 72.7 samples.
        assert!(
            (best_lag as i32 - 73).abs() <= 2,
            "best_lag = {}",
            best_lag
        );
    }

    #[test]
    fn cross_spectrum_zero_for_orthogonal_inputs() {
        let mut s = SharedSpectrum::new(16_000.0, 256);
        let l = sine(220.0, 16_000.0, 256, 0.5);
        let r = sine(880.0, 16_000.0, 256, 0.5);
        s.compute(&l, &r);
        // Cross-spectrum magnitude near 220 Hz should be small (R has no
        // energy there); near 880 Hz also small (L has no energy there).
        let bin_220 = (220.0_f32 / (8000.0_f32 / 128.0)).round() as usize;
        let mag = (s.cross_real[bin_220].powi(2) + s.cross_imag[bin_220].powi(2)).sqrt();
        let total: f32 = (0..s.bins)
            .map(|k| (s.cross_real[k].powi(2) + s.cross_imag[k].powi(2)).sqrt())
            .sum();
        assert!(mag < total * 0.5);
    }

    #[test]
    fn energy_db_full_scale_sine_near_minus_three() {
        let mut s = SharedSpectrum::new(16_000.0, 256);
        let l = sine(1000.0, 16_000.0, 256, 1.0);
        let r = vec![0.0; 256];
        s.compute(&l, &r);
        assert!(
            (s.energy_db_l + 3.0).abs() < 1.0,
            "energy = {}",
            s.energy_db_l
        );
    }
}
