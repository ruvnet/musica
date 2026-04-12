//! Continuous valence-arousal emotion vector — see ADR-151.
//!
//! Replaces the discrete `EmotionBucket` with a continuous (valence, arousal)
//! pair, with a `to_bucket()` projection for backward compatibility.

use super::analysis::EmotionBucket;
use super::prosody::ProsodySnapshot;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmotionVector {
    pub valence: f32,
    pub arousal: f32,
    pub confidence: f32,
}

impl EmotionVector {
    pub const fn neutral() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.0,
            confidence: 0.0,
        }
    }

    /// Linear interpolation between two emotion vectors. `t` ∈ [0,1].
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            valence: self.valence + (other.valence - self.valence) * t,
            arousal: self.arousal + (other.arousal - self.arousal) * t,
            confidence: self.confidence + (other.confidence - self.confidence) * t,
        }
    }

    /// Project to the discrete bucket from ADR-147 (backward compat).
    pub fn to_bucket(self) -> EmotionBucket {
        if self.confidence < 0.10 {
            return EmotionBucket::Neutral;
        }
        let high_arousal = self.arousal > 0.10;
        let positive = self.valence > 0.0;
        match (high_arousal, positive) {
            (true, true) => EmotionBucket::Happy,
            (true, false) => EmotionBucket::Angry,
            (false, true) => EmotionBucket::Calm,
            (false, false) => {
                if self.arousal < -0.30 {
                    EmotionBucket::Sad
                } else {
                    EmotionBucket::Neutral
                }
            }
        }
    }
}

/// Stateful emotion vector estimator with EMA smoothing.
pub struct EmotionEstimator {
    alpha: f32,
    smoothed: EmotionVector,
    f0_history: [f32; 5],
    f0_head: usize,
    f0_filled: usize,
}

impl Default for EmotionEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl EmotionEstimator {
    pub fn new() -> Self {
        Self {
            alpha: 0.35,
            smoothed: EmotionVector::neutral(),
            f0_history: [0.0; 5],
            f0_head: 0,
            f0_filled: 0,
        }
    }

    pub fn reset(&mut self) {
        self.smoothed = EmotionVector::neutral();
        self.f0_head = 0;
        self.f0_filled = 0;
    }

    pub fn observe(&mut self, snap: &ProsodySnapshot) -> EmotionVector {
        if snap.f0_hz > 0.0 {
            self.f0_history[self.f0_head] = snap.f0_hz;
            self.f0_head = (self.f0_head + 1) % 5;
            self.f0_filled = (self.f0_filled + 1).min(5);
        }

        let energy_norm = ((snap.energy_db + 50.0) / 50.0).clamp(0.0, 1.0);
        let f0_var = if self.f0_filled >= 2 {
            let len = self.f0_filled as f32;
            let mean = self.f0_history[..self.f0_filled].iter().sum::<f32>() / len;
            let var = self.f0_history[..self.f0_filled]
                .iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f32>()
                / len;
            (var.sqrt() / 50.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let arousal_raw = 0.6 * energy_norm + 0.4 * f0_var;
        let arousal = (arousal_raw * 2.0 - 1.0).clamp(-1.0, 1.0);

        let slope_f0 = if self.f0_filled >= 2 {
            let last_idx = (self.f0_head + 5 - 1) % 5;
            let first_idx = (self.f0_head + 5 - self.f0_filled) % 5;
            (self.f0_history[last_idx] - self.f0_history[first_idx])
                / (self.f0_filled as f32 - 1.0).max(1.0)
        } else {
            0.0
        };
        let valence_raw = 0.5 * (slope_f0 / 25.0).tanh()
            + 0.3 * (1.0 - snap.flatness)
            - 0.2 * (snap.zcr - 0.2).clamp(0.0, 0.5) * 4.0;
        let valence = valence_raw.clamp(-1.0, 1.0);

        let confidence = (snap.voicing * (1.0 - snap.flatness)).clamp(0.0, 1.0);

        let raw = EmotionVector {
            valence: sanitize(valence),
            arousal: sanitize(arousal),
            confidence: sanitize(confidence),
        };
        self.smoothed = self.smoothed.lerp(raw, self.alpha);
        self.smoothed
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

    fn voiced(energy_db: f32, f0: f32) -> ProsodySnapshot {
        ProsodySnapshot {
            f0_hz: f0,
            voicing: 0.9,
            energy_db,
            centroid_hz: 1500.0,
            rolloff_hz: 4000.0,
            zcr: 0.05,
            flatness: 0.1,
        }
    }

    #[test]
    fn neutral_constructor_is_zero() {
        let n = EmotionVector::neutral();
        assert_eq!(n.valence, 0.0);
        assert_eq!(n.arousal, 0.0);
    }

    #[test]
    fn loud_voiced_speech_increases_arousal() {
        let mut e = EmotionEstimator::new();
        for _ in 0..4 {
            let _ = e.observe(&voiced(-30.0, 220.0));
        }
        let calm = e.smoothed.arousal;
        for _ in 0..6 {
            let _ = e.observe(&voiced(-5.0, 260.0));
        }
        let excited = e.smoothed.arousal;
        assert!(excited > calm, "calm={}, excited={}", calm, excited);
    }

    #[test]
    fn lerp_midpoint() {
        let a = EmotionVector {
            valence: -1.0,
            arousal: 0.0,
            confidence: 0.0,
        };
        let b = EmotionVector {
            valence: 1.0,
            arousal: 1.0,
            confidence: 1.0,
        };
        let m = a.lerp(b, 0.5);
        assert!((m.valence - 0.0).abs() < 1e-6);
        assert!((m.arousal - 0.5).abs() < 1e-6);
    }

    #[test]
    fn silent_audio_yields_neutral_bucket() {
        let mut e = EmotionEstimator::new();
        let v = e.observe(&ProsodySnapshot::silent());
        assert_eq!(v.to_bucket(), EmotionBucket::Neutral);
    }

    #[test]
    fn fields_finite() {
        let mut e = EmotionEstimator::new();
        let v = e.observe(&voiced(-10.0, 220.0));
        assert!(v.valence.is_finite());
        assert!(v.arousal.is_finite());
        assert!(v.confidence.is_finite());
    }
}
