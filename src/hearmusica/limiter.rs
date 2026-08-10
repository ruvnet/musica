//! Brick-wall output limiter, plus a downward-expansion noise gate.
//!
//! Both processors here share the same envelope-follower + attack/release
//! technique, just aimed in opposite directions: [`Limiter`] reins in
//! peaks above a ceiling, [`NoiseGate`] reins in everything below a floor.
//!
//! `Limiter` prevents the output signal from exceeding a configurable
//! ceiling (default -1 dB FS). `NoiseGate` attenuates the signal whenever
//! its envelope drops below a threshold, suppressing residual hiss, fan
//! noise, and low-level room tone that survives upstream separation and
//! compression.

use super::block::{AudioBlock, AudioProcessor};

// ---------------------------------------------------------------------------
// Limiter
// ---------------------------------------------------------------------------

/// Brick-wall output limiter with envelope-based gain reduction.
///
/// Tracks peak levels with a fast attack and slow release. When the
/// envelope exceeds the threshold, gain is reduced proportionally so
/// the output never clips.
pub struct Limiter {
    /// Threshold in dB FS. Output will never exceed this level.
    pub threshold_db: f32,
    /// Attack time in milliseconds (default: 0.1 ms — very fast).
    pub attack_ms: f32,
    /// Release time in milliseconds (default: 50 ms).
    pub release_ms: f32,
    // Internal state
    envelope_l: f32,
    envelope_r: f32,
    attack_coeff: f32,
    release_coeff: f32,
    threshold_linear: f32,
    sample_rate: f32,
}

impl Limiter {
    /// Create a new limiter with the given ceiling in dB FS.
    pub fn new(threshold_db: f32) -> Self {
        Self {
            threshold_db,
            attack_ms: 0.1,
            release_ms: 50.0,
            envelope_l: 0.0,
            envelope_r: 0.0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            threshold_linear: 10.0f32.powf(threshold_db / 20.0),
            sample_rate: 0.0,
        }
    }

    /// Default limiter at -1 dB FS (standard headroom for hearing aids).
    pub fn default_ceiling() -> Self {
        Self::new(-1.0)
    }

    fn update_coefficients(&mut self) {
        if self.sample_rate > 0.0 {
            self.attack_coeff = (-1.0 / (self.attack_ms * 0.001 * self.sample_rate)).exp();
            self.release_coeff = (-1.0 / (self.release_ms * 0.001 * self.sample_rate)).exp();
            self.threshold_linear = 10.0f32.powf(self.threshold_db / 20.0);
        }
    }

    /// Process a single sample with envelope tracking and gain reduction.
    #[inline]
    fn limit_sample(
        x: f32,
        envelope: &mut f32,
        threshold: f32,
        attack_coeff: f32,
        release_coeff: f32,
    ) -> f32 {
        let abs_x = x.abs();

        // Envelope follower: fast attack, slow release
        if abs_x > *envelope {
            *envelope = attack_coeff * *envelope + (1.0 - attack_coeff) * abs_x;
        } else {
            *envelope = release_coeff * *envelope + (1.0 - release_coeff) * abs_x;
        }

        // Compute gain reduction
        if *envelope > threshold {
            let gain = threshold / *envelope;
            x * gain
        } else {
            x
        }
    }
}

impl AudioProcessor for Limiter {
    fn prepare(&mut self, sample_rate: f32, _block_size: usize) {
        self.sample_rate = sample_rate;
        self.update_coefficients();
        self.envelope_l = 0.0;
        self.envelope_r = 0.0;
    }

    fn process(&mut self, block: &mut AudioBlock) {
        let threshold = self.threshold_linear;
        let attack = self.attack_coeff;
        let release = self.release_coeff;

        for s in block.left.iter_mut() {
            *s = Self::limit_sample(*s, &mut self.envelope_l, threshold, attack, release);
        }
        for s in block.right.iter_mut() {
            *s = Self::limit_sample(*s, &mut self.envelope_r, threshold, attack, release);
        }
    }

    fn name(&self) -> &str {
        "Limiter"
    }
}

// ---------------------------------------------------------------------------
// NoiseGate — smooth downward-expansion gate
// ---------------------------------------------------------------------------

/// Downward-expansion noise gate with attack/release/hold.
///
/// Attenuates the signal whenever its envelope drops below a threshold,
/// suppressing residual hiss, fan noise, and low-level room tone that
/// survives upstream separation/compression. Unlike a hard on/off gate,
/// this pulls the signal toward `range_db` of attenuation smoothly rather
/// than muting outright, and a `hold_ms` window prevents rapid open/close
/// "chattering" on syllabic or percussive material.
pub struct NoiseGate {
    pub threshold_db: f32,
    pub range_db: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub hold_ms: f32,
    // Internal state (per channel)
    envelope_l: f32,
    envelope_r: f32,
    gain_l: f32,
    gain_r: f32,
    hold_remaining_l: usize,
    hold_remaining_r: usize,
    // Derived coefficients
    attack_coeff: f32,
    release_coeff: f32,
    envelope_coeff: f32,
    threshold_linear: f32,
    floor_linear: f32,
    hold_samples: usize,
    sample_rate: f32,
}

impl NoiseGate {
    /// Create a new gate with the given threshold (dB FS) and attenuation
    /// range (dB, negative). Attack/release/hold use sensible defaults.
    pub fn new(threshold_db: f32, range_db: f32) -> Self {
        Self {
            threshold_db,
            range_db,
            attack_ms: 2.0,
            release_ms: 150.0,
            hold_ms: 50.0,
            envelope_l: 0.0,
            envelope_r: 0.0,
            gain_l: 1.0,
            gain_r: 1.0,
            hold_remaining_l: 0,
            hold_remaining_r: 0,
            attack_coeff: 0.0,
            release_coeff: 0.0,
            envelope_coeff: 0.0,
            threshold_linear: 10.0f32.powf(threshold_db / 20.0),
            floor_linear: 10.0f32.powf(range_db / 20.0),
            hold_samples: 0,
            sample_rate: 0.0,
        }
    }

    /// Gentle default gate tuned for low-level room-noise cleanup
    /// (-45 dB threshold, 30 dB max attenuation).
    pub fn default_gate() -> Self {
        Self::new(-45.0, -30.0)
    }

    fn update_coefficients(&mut self) {
        if self.sample_rate > 0.0 {
            self.attack_coeff = (-1.0 / (self.attack_ms * 0.001 * self.sample_rate)).exp();
            self.release_coeff = (-1.0 / (self.release_ms * 0.001 * self.sample_rate)).exp();
            // Envelope follower: fixed 5ms smoothing for level detection.
            self.envelope_coeff = (-1.0 / (0.005 * self.sample_rate)).exp();
            self.threshold_linear = 10.0f32.powf(self.threshold_db / 20.0);
            self.floor_linear = 10.0f32.powf(self.range_db / 20.0);
            self.hold_samples = (self.hold_ms * 0.001 * self.sample_rate) as usize;
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn gate_sample(
        x: f32,
        envelope: &mut f32,
        gain: &mut f32,
        hold_remaining: &mut usize,
        envelope_coeff: f32,
        attack_coeff: f32,
        release_coeff: f32,
        threshold: f32,
        floor: f32,
        hold_samples: usize,
    ) -> f32 {
        let abs_x = x.abs();
        *envelope = envelope_coeff * *envelope + (1.0 - envelope_coeff) * abs_x;

        let target_gain = if *envelope >= threshold {
            *hold_remaining = hold_samples;
            1.0
        } else if *hold_remaining > 0 {
            *hold_remaining -= 1;
            1.0
        } else {
            floor
        };

        // Attack when opening (gain increasing), release when closing.
        if target_gain > *gain {
            *gain = attack_coeff * *gain + (1.0 - attack_coeff) * target_gain;
        } else {
            *gain = release_coeff * *gain + (1.0 - release_coeff) * target_gain;
        }

        x * *gain
    }
}

impl AudioProcessor for NoiseGate {
    fn prepare(&mut self, sample_rate: f32, _block_size: usize) {
        self.sample_rate = sample_rate;
        self.update_coefficients();
        self.envelope_l = 0.0;
        self.envelope_r = 0.0;
        self.gain_l = 1.0;
        self.gain_r = 1.0;
        self.hold_remaining_l = 0;
        self.hold_remaining_r = 0;
    }

    fn process(&mut self, block: &mut AudioBlock) {
        let threshold = self.threshold_linear;
        let floor = self.floor_linear;
        let hold_samples = self.hold_samples;
        let envelope_coeff = self.envelope_coeff;
        let attack = self.attack_coeff;
        let release = self.release_coeff;

        for s in block.left.iter_mut() {
            *s = Self::gate_sample(
                *s,
                &mut self.envelope_l,
                &mut self.gain_l,
                &mut self.hold_remaining_l,
                envelope_coeff,
                attack,
                release,
                threshold,
                floor,
                hold_samples,
            );
        }
        for s in block.right.iter_mut() {
            *s = Self::gate_sample(
                *s,
                &mut self.envelope_r,
                &mut self.gain_r,
                &mut self.hold_remaining_r,
                envelope_coeff,
                attack,
                release,
                threshold,
                floor,
                hold_samples,
            );
        }
    }

    fn name(&self) -> &str {
        "NoiseGate"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn make_block(samples: &[f32], sr: f32) -> AudioBlock {
        AudioBlock {
            left: samples.to_vec(),
            right: samples.to_vec(),
            sample_rate: sr,
            block_size: samples.len(),
            metadata: super::super::block::BlockMetadata::default(),
        }
    }

    #[test]
    fn signal_below_threshold_passes_unchanged() {
        let sr = 48000.0;
        let mut limiter = Limiter::new(-1.0); // threshold at ~0.891
        limiter.prepare(sr, 256);

        // Signal at -6 dB ≈ 0.5 amplitude — well below -1 dB threshold
        let amplitude = 0.5;
        let samples: Vec<f32> = (0..1024)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / sr).sin())
            .collect();

        let mut block = make_block(&samples, sr);
        limiter.process(&mut block);

        // Check that output matches input closely
        let max_diff: f32 = block
            .left
            .iter()
            .zip(samples.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.01,
            "Signal below threshold should pass unchanged, max diff={:.4}",
            max_diff
        );
    }

    #[test]
    fn signal_above_threshold_is_limited() {
        let sr = 48000.0;
        let mut limiter = Limiter::new(-6.0); // threshold at ~0.501
        limiter.prepare(sr, 256);

        // Signal at 0 dB = 1.0 amplitude — well above -6 dB threshold
        let amplitude = 1.0;
        let samples: Vec<f32> = (0..2048)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / sr).sin())
            .collect();

        let mut block = make_block(&samples, sr);
        limiter.process(&mut block);

        // After the limiter settles, peaks should be near the threshold
        let threshold_linear = 10.0f32.powf(-6.0 / 20.0);
        // Check the last 1024 samples (skip transient at start)
        let max_output: f32 = block.left[1024..]
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_output < threshold_linear * 1.15, // 15% tolerance for envelope lag
            "Limiter output peak={:.4} should be near threshold={:.4}",
            max_output,
            threshold_linear
        );
    }

    #[test]
    fn limiter_prevents_clipping() {
        let sr = 48000.0;
        let mut limiter = Limiter::new(-1.0);
        limiter.prepare(sr, 256);

        // Very loud signal: amplitude = 2.0 (well above 0 dBFS)
        let samples: Vec<f32> = (0..4096)
            .map(|i| 2.0 * (2.0 * PI * 440.0 * i as f32 / sr).sin())
            .collect();

        let mut block = make_block(&samples, sr);
        limiter.process(&mut block);

        let threshold_linear = 10.0f32.powf(-1.0 / 20.0);
        // After settling, output should be limited
        let max_output: f32 = block.left[2048..]
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_output < threshold_linear * 1.2,
            "Limiter should prevent clipping: peak={:.4}, threshold={:.4}",
            max_output,
            threshold_linear
        );
    }

    #[test]
    fn gate_signal_above_threshold_passes_near_unchanged() {
        let sr = 48000.0;
        let mut gate = NoiseGate::new(-45.0, -30.0);
        gate.prepare(sr, 256);

        // -6 dB tone, well above the -45 dB threshold.
        let amplitude = 0.5;
        let samples: Vec<f32> = (0..4096)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / sr).sin())
            .collect();

        let mut block = make_block(&samples, sr);
        gate.process(&mut block);

        // After the gate opens (skip attack transient), amplitude should be ~unchanged.
        let peak_out: f32 = block.left[2048..].iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(
            peak_out > amplitude * 0.95,
            "Open gate should pass signal near-unchanged, peak={:.4}",
            peak_out
        );
    }

    #[test]
    fn gate_signal_below_threshold_is_attenuated_after_hold() {
        let sr = 48000.0;
        let mut gate = NoiseGate::new(-20.0, -40.0);
        gate.hold_ms = 5.0;
        gate.release_ms = 20.0;
        gate.prepare(sr, 256);

        // -40 dB tone, well below the -20 dB threshold -> gate should close.
        let amplitude = 0.01;
        let samples: Vec<f32> = (0..8192)
            .map(|i| amplitude * (2.0 * PI * 1000.0 * i as f32 / sr).sin())
            .collect();

        let mut block = make_block(&samples, sr);
        gate.process(&mut block);

        // After hold+release settle, the tail should be attenuated well
        // below the input amplitude (floor is -40 dB relative to input).
        let tail_peak: f32 = block.left[6000..].iter().map(|s| s.abs()).fold(0.0, f32::max);
        assert!(
            tail_peak < amplitude * 0.5,
            "Closed gate should attenuate low-level signal, tail_peak={:.5}, input={:.5}",
            tail_peak,
            amplitude
        );
    }

    #[test]
    fn gate_silence_stays_silent() {
        let sr = 16000.0;
        let mut gate = NoiseGate::default_gate();
        gate.prepare(sr, 128);

        let samples = vec![0.0f32; 512];
        let mut block = make_block(&samples, sr);
        gate.process(&mut block);

        assert!(block.energy() == 0.0);
    }

    #[test]
    fn gate_latency_is_zero() {
        let gate = NoiseGate::default_gate();
        assert_eq!(gate.latency_samples(), 0);
    }
}
