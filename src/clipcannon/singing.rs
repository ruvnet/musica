//! Singing analysis subsystem — see ADR-154.
//!
//! Provides:
//! - [`PitchTracker`]: cent-accurate F0 with octave-error correction.
//! - [`VibratoDetector`]: 4–8 Hz vibrato rate & depth.
//! - [`StyleClassifier`]: prototype-matched style/genre classification.
//! - [`KaraokeScorer`]: per-block reference-melody scoring.
//!
//! All four consume the existing `SharedSpectrum` + `ProsodySnapshot`.

use core::f32::consts::PI;

use super::prosody::ProsodySnapshot;
use super::spectrum::SharedSpectrum;

// ─────────────────────────────────────────────────────────────────────────
// Pitch tracking
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchSnapshot {
    pub f0_hz: f32,
    /// Cents from `ref_hz`. NaN if unvoiced.
    pub cents: f32,
    /// Float MIDI note number (e.g. 69.5 = halfway between A4 and A#4).
    pub midi_note: f32,
    /// Stability over the recent F0 history, [0, 1].
    pub stability: f32,
    pub voiced: bool,
}

impl PitchSnapshot {
    pub const fn unvoiced() -> Self {
        Self {
            f0_hz: 0.0,
            cents: f32::NAN,
            midi_note: f32::NAN,
            stability: 0.0,
            voiced: false,
        }
    }
}

pub struct PitchTracker {
    sample_rate: f32,
    ref_hz: f32,
    history: [f32; 16],
    head: usize,
    filled: usize,
    min_lag: usize,
    max_lag: usize,
}

impl PitchTracker {
    pub fn new(sample_rate: f32) -> Self {
        Self::with_reference(sample_rate, 440.0)
    }

    pub fn with_reference(sample_rate: f32, ref_hz: f32) -> Self {
        Self {
            sample_rate,
            ref_hz,
            history: [0.0; 16],
            head: 0,
            filled: 0,
            // Singing range: ~60 Hz (low bass) to ~1100 Hz (soprano whistle).
            min_lag: (sample_rate / 1100.0).round() as usize,
            max_lag: (sample_rate / 60.0).round() as usize,
        }
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
        for h in &mut self.history {
            *h = 0.0;
        }
    }

    /// Track pitch from a precomputed shared spectrum and prosody snapshot.
    pub fn track(&mut self, spec: &SharedSpectrum, prosody: &ProsodySnapshot) -> PitchSnapshot {
        if prosody.voicing < 0.45 || prosody.energy_db < -45.0 {
            return PitchSnapshot::unvoiced();
        }

        // Search the shared ACF, but use *subharmonic summation* to suppress
        // octave doubling errors common in classical pitch tracking.
        let acf = &spec.acf_l;
        let max_lag_capped = self.max_lag.min(acf.len() - 1);

        let mut best_lag = self.min_lag;
        let mut best_score = f32::MIN;
        for lag in self.min_lag..=max_lag_capped {
            // ssh = r[τ] + 0.5·r[2τ] + 0.25·r[4τ]
            let mut score = acf[lag];
            let l2 = lag * 2;
            if l2 < acf.len() {
                score += 0.5 * acf[l2];
            }
            let l4 = lag * 4;
            if l4 < acf.len() {
                score += 0.25 * acf[l4];
            }
            if score > best_score {
                best_score = score;
                best_lag = lag;
            }
        }

        // Parabolic interpolation around the chosen lag.
        let refined_lag = if best_lag > self.min_lag && best_lag + 1 <= max_lag_capped {
            let l = acf[best_lag - 1];
            let c = acf[best_lag];
            let r = acf[best_lag + 1];
            let denom = l - 2.0 * c + r;
            let delta = if denom.abs() > 1e-9 {
                0.5 * (l - r) / denom
            } else {
                0.0
            };
            best_lag as f32 + delta
        } else {
            best_lag as f32
        };

        let f0_hz = if refined_lag > 0.0 {
            self.sample_rate / refined_lag
        } else {
            return PitchSnapshot::unvoiced();
        };

        // Update history (cents).
        let cents = 1200.0 * (f0_hz / self.ref_hz).log2();
        self.history[self.head] = cents;
        self.head = (self.head + 1) % 16;
        self.filled = (self.filled + 1).min(16);

        // Stability = 1 / (1 + std-dev cents over last 8 frames).
        let take = self.filled.min(8);
        let mut sum = 0.0_f32;
        let start = (self.head + 16 - take) % 16;
        for i in 0..take {
            sum += self.history[(start + i) % 16];
        }
        let mean = sum / take as f32;
        let mut var = 0.0_f32;
        for i in 0..take {
            let d = self.history[(start + i) % 16] - mean;
            var += d * d;
        }
        let std = (var / take.max(1) as f32).sqrt();
        let stability = 1.0 / (1.0 + std / 30.0);

        let midi_note = 69.0 + cents / 100.0;

        PitchSnapshot {
            f0_hz,
            cents,
            midi_note,
            stability,
            voiced: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Vibrato detection
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VibratoSnapshot {
    pub rate_hz: f32,
    pub depth_cents: f32,
    pub presence: f32,
}

impl VibratoSnapshot {
    pub const fn none() -> Self {
        Self {
            rate_hz: 0.0,
            depth_cents: 0.0,
            presence: 0.0,
        }
    }
}

pub struct VibratoDetector {
    history: [f32; 32],
    head: usize,
    filled: usize,
    /// Hop length per block, ms (block_size / sample_rate * 1000).
    hop_ms: f32,
}

impl VibratoDetector {
    pub fn new(hop_ms: f32) -> Self {
        Self {
            history: [0.0; 32],
            head: 0,
            filled: 0,
            hop_ms,
        }
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
    }

    pub fn observe(&mut self, pitch: &PitchSnapshot) -> VibratoSnapshot {
        if !pitch.voiced || !pitch.cents.is_finite() {
            // Push 0 to keep window aligned.
            self.history[self.head] = 0.0;
            self.head = (self.head + 1) % 32;
            self.filled = (self.filled + 1).min(32);
            return VibratoSnapshot::none();
        }
        self.history[self.head] = pitch.cents;
        self.head = (self.head + 1) % 32;
        self.filled = (self.filled + 1).min(32);

        if self.filled < 16 {
            return VibratoSnapshot::none();
        }

        // Detrend & DFT (small enough to do directly, ~32×32 = 1k mults).
        let n = 32;
        let mean = self.history.iter().sum::<f32>() / n as f32;

        // We probe explicit frequencies in [3, 9] Hz at 0.5 Hz steps.
        let block_hz = 1000.0 / self.hop_ms; // sample rate of the F0 stream
        let mut peak_val = 0.0_f32;
        let mut peak_freq = 0.0_f32;
        let mut total = 0.0_f32;
        let mut probe = 3.0_f32;
        while probe <= 9.0 {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            let omega = 2.0 * PI * probe / block_hz;
            for i in 0..n {
                let v = self.history[(self.head + i) % 32] - mean;
                let phase = omega * i as f32;
                re += v * phase.cos();
                im += v * phase.sin();
            }
            let mag = (re * re + im * im).sqrt() / n as f32;
            total += mag;
            if mag > peak_val {
                peak_val = mag;
                peak_freq = probe;
            }
            probe += 0.5;
        }

        let presence = if total > 1e-6 {
            (peak_val * 13.0 / total).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let rate_hz = if presence > 0.30 { peak_freq } else { 0.0 };
        // Depth = peak amplitude × 2 (sine peak-to-peak).
        let depth_cents = if presence > 0.30 { peak_val * 2.0 } else { 0.0 };

        VibratoSnapshot {
            rate_hz,
            depth_cents,
            presence,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Style / genre classification
// ─────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct StylePrototype {
    pub name: &'static str,
    pub mean_centroid_hz: f32,
    pub mean_rolloff_hz: f32,
    pub mean_flatness: f32,
    pub mean_zcr: f32,
    pub mean_vibrato_rate: f32,
    pub mean_vibrato_depth: f32,
    pub popularity_weight: f32,
}

/// Eight built-in styles. Acoustic centroids derived from public-domain
/// singing voice analysis literature; not from sampling copyrighted audio.
pub const DEFAULT_STYLE_PROTOTYPES: &[StylePrototype] = &[
    StylePrototype {
        name: "pop_belt",
        mean_centroid_hz: 2200.0,
        mean_rolloff_hz: 5500.0,
        mean_flatness: 0.18,
        mean_zcr: 0.10,
        mean_vibrato_rate: 5.5,
        mean_vibrato_depth: 50.0,
        popularity_weight: 1.0,
    },
    StylePrototype {
        name: "pop_breathy",
        mean_centroid_hz: 1400.0,
        mean_rolloff_hz: 4500.0,
        mean_flatness: 0.32,
        mean_zcr: 0.14,
        mean_vibrato_rate: 4.0,
        mean_vibrato_depth: 20.0,
        popularity_weight: 1.0,
    },
    StylePrototype {
        name: "classical_legit",
        mean_centroid_hz: 2800.0,
        mean_rolloff_hz: 6500.0,
        mean_flatness: 0.12,
        mean_zcr: 0.07,
        mean_vibrato_rate: 6.0,
        mean_vibrato_depth: 80.0,
        popularity_weight: 1.0,
    },
    StylePrototype {
        name: "musical_theatre",
        mean_centroid_hz: 2500.0,
        mean_rolloff_hz: 6000.0,
        mean_flatness: 0.16,
        mean_zcr: 0.09,
        mean_vibrato_rate: 5.0,
        mean_vibrato_depth: 60.0,
        popularity_weight: 1.0,
    },
    StylePrototype {
        name: "rock_grit",
        mean_centroid_hz: 3200.0,
        mean_rolloff_hz: 7000.0,
        mean_flatness: 0.28,
        mean_zcr: 0.13,
        mean_vibrato_rate: 4.5,
        mean_vibrato_depth: 40.0,
        popularity_weight: 1.0,
    },
    StylePrototype {
        name: "metal_scream",
        mean_centroid_hz: 4500.0,
        mean_rolloff_hz: 8000.0,
        mean_flatness: 0.55,
        mean_zcr: 0.22,
        mean_vibrato_rate: 0.0,
        mean_vibrato_depth: 0.0,
        popularity_weight: 1.0,
    },
    StylePrototype {
        name: "rnb_melisma",
        mean_centroid_hz: 1900.0,
        mean_rolloff_hz: 5000.0,
        mean_flatness: 0.20,
        mean_zcr: 0.10,
        mean_vibrato_rate: 6.5,
        mean_vibrato_depth: 70.0,
        popularity_weight: 1.0,
    },
    StylePrototype {
        name: "folk_plain",
        mean_centroid_hz: 1800.0,
        mean_rolloff_hz: 4800.0,
        mean_flatness: 0.22,
        mean_zcr: 0.11,
        mean_vibrato_rate: 3.5,
        mean_vibrato_depth: 25.0,
        popularity_weight: 1.0,
    },
];

#[derive(Debug, Clone, Copy)]
pub struct StyleMatch {
    pub name: &'static str,
    pub similarity: f32,
}

#[derive(Default, Clone, Copy)]
struct StyleFingerprint {
    centroid_hz: f32,
    rolloff_hz: f32,
    flatness: f32,
    zcr: f32,
    vibrato_rate: f32,
    vibrato_depth: f32,
    samples: u32,
}

pub struct StyleClassifier {
    fingerprint: StyleFingerprint,
    prototypes: &'static [StylePrototype],
}

impl StyleClassifier {
    pub fn new(prototypes: &'static [StylePrototype]) -> Self {
        Self {
            fingerprint: StyleFingerprint::default(),
            prototypes,
        }
    }

    pub fn default_library() -> Self {
        Self::new(DEFAULT_STYLE_PROTOTYPES)
    }

    pub fn reset(&mut self) {
        self.fingerprint = StyleFingerprint::default();
    }

    /// Number of samples accumulated into the rolling fingerprint.
    pub fn samples(&self) -> u32 {
        self.fingerprint.samples
    }

    pub fn observe(
        &mut self,
        prosody: &ProsodySnapshot,
        pitch: &PitchSnapshot,
        vibrato: &VibratoSnapshot,
    ) {
        if !pitch.voiced {
            return;
        }
        // Online mean update with cap = 64 (prevents drift).
        let n = (self.fingerprint.samples + 1).min(64);
        let inv = 1.0 / n as f32;
        let f = &mut self.fingerprint;
        f.centroid_hz += (prosody.centroid_hz - f.centroid_hz) * inv;
        f.rolloff_hz += (prosody.rolloff_hz - f.rolloff_hz) * inv;
        f.flatness += (prosody.flatness - f.flatness) * inv;
        f.zcr += (prosody.zcr - f.zcr) * inv;
        f.vibrato_rate += (vibrato.rate_hz - f.vibrato_rate) * inv;
        f.vibrato_depth += (vibrato.depth_cents - f.vibrato_depth) * inv;
        f.samples = n;
    }

    /// Fill `out` with the top `k` matches sorted by descending similarity.
    /// Returns the number of matches written.
    pub fn top_k(&self, k: usize, out: &mut [StyleMatch]) -> usize {
        let f = &self.fingerprint;
        if f.samples == 0 || self.prototypes.is_empty() {
            return 0;
        }
        // Score every prototype.
        let mut scratch: [StyleMatch; 16] = [StyleMatch {
            name: "",
            similarity: 0.0,
        }; 16];
        let n_proto = self.prototypes.len().min(16);
        for (i, p) in self.prototypes.iter().take(16).enumerate() {
            let dc = norm_diff(f.centroid_hz, p.mean_centroid_hz, 5000.0);
            let dr = norm_diff(f.rolloff_hz, p.mean_rolloff_hz, 8000.0);
            let df = (f.flatness - p.mean_flatness).abs();
            let dz = (f.zcr - p.mean_zcr).abs();
            let dv = norm_diff(f.vibrato_rate, p.mean_vibrato_rate, 10.0);
            let dd = norm_diff(f.vibrato_depth, p.mean_vibrato_depth, 100.0);
            let dist = 0.30 * dc
                + 0.20 * dr
                + 0.15 * df
                + 0.10 * dz
                + 0.15 * dv
                + 0.10 * dd;
            let raw = (1.0 - dist).clamp(0.0, 1.0);
            let weighted = (raw * p.popularity_weight).clamp(0.0, 1.0);
            scratch[i] = StyleMatch {
                name: p.name,
                similarity: weighted,
            };
        }
        // Insertion-sort the top n_proto.
        for i in 1..n_proto {
            let mut j = i;
            while j > 0 && scratch[j - 1].similarity < scratch[j].similarity {
                scratch.swap(j - 1, j);
                j -= 1;
            }
        }
        let to_write = k.min(n_proto).min(out.len());
        out[..to_write].copy_from_slice(&scratch[..to_write]);
        to_write
    }
}

#[inline]
fn norm_diff(a: f32, b: f32, scale: f32) -> f32 {
    ((a - b).abs() / scale).clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────
// Karaoke scoring
// ─────────────────────────────────────────────────────────────────────────

pub struct KaraokeScorer {
    reference_midi: Vec<f32>,
    cursor: usize,
}

impl KaraokeScorer {
    pub fn new(reference_midi: Vec<f32>) -> Self {
        Self {
            reference_midi,
            cursor: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Returns a per-block score in `[0, 1]` and advances the cursor.
    pub fn score(&mut self, pitch: &PitchSnapshot) -> f32 {
        if self.cursor >= self.reference_midi.len() || !pitch.voiced {
            self.cursor = self.cursor.saturating_add(1);
            return 0.0;
        }
        let target = self.reference_midi[self.cursor];
        self.cursor += 1;
        let cents_err = (pitch.midi_note - target) * 100.0;
        (-((cents_err / 30.0).powi(2))).exp().clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voiced_snap() -> ProsodySnapshot {
        ProsodySnapshot {
            f0_hz: 220.0,
            voicing: 0.9,
            energy_db: -10.0,
            centroid_hz: 2200.0,
            rolloff_hz: 5500.0,
            zcr: 0.10,
            flatness: 0.18,
        }
    }

    fn sine(freq: f32, sr: f32, n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sr).sin() * amp)
            .collect()
    }

    #[test]
    fn pitch_tracker_unvoiced_on_silence() {
        let mut spec = SharedSpectrum::new(16_000.0, 256);
        let zeros = vec![0.0; 256];
        spec.compute(&zeros, &zeros);
        let mut pt = PitchTracker::new(16_000.0);
        let p = pt.track(&spec, &ProsodySnapshot::silent());
        assert!(!p.voiced);
    }

    #[test]
    fn pitch_tracker_recovers_a4_at_440hz() {
        let mut spec = SharedSpectrum::new(16_000.0, 256);
        let l = sine(440.0, 16_000.0, 256, 0.6);
        let r = vec![0.0; 256];
        spec.compute(&l, &r);
        let mut pt = PitchTracker::new(16_000.0);
        let p = pt.track(&spec, &voiced_snap());
        assert!(p.voiced);
        assert!((p.f0_hz - 440.0).abs() < 6.0, "f0 = {}", p.f0_hz);
        // A4 = MIDI 69
        assert!((p.midi_note - 69.0).abs() < 0.4, "midi = {}", p.midi_note);
    }

    #[test]
    fn vibrato_detector_silent_returns_none() {
        let mut vd = VibratoDetector::new(8.0);
        for _ in 0..32 {
            let v = vd.observe(&PitchSnapshot::unvoiced());
            assert_eq!(v, VibratoSnapshot::none());
        }
    }

    #[test]
    fn vibrato_detector_finds_5_5hz_modulation() {
        let mut vd = VibratoDetector::new(8.0);
        // Synthesise 5.5 Hz vibrato over 32 frames @ 8 ms hop = 256 ms.
        // Block hz = 125. Period = 125/5.5 ≈ 22.7 frames. We have ~1.4 cycles.
        // Increase to deeper depth to be detectable.
        for i in 0..32 {
            let cents = 60.0 * (2.0 * PI * 5.5 * i as f32 / 125.0).sin();
            let p = PitchSnapshot {
                f0_hz: 440.0,
                cents: 0.0 + cents,
                midi_note: 69.0,
                stability: 0.9,
                voiced: true,
            };
            vd.observe(&p);
        }
        let v = vd.observe(&PitchSnapshot {
            f0_hz: 440.0,
            cents: 0.0,
            midi_note: 69.0,
            stability: 0.9,
            voiced: true,
        });
        assert!(v.presence > 0.20, "presence = {}", v.presence);
        assert!((v.rate_hz - 5.5).abs() < 1.5, "rate = {}", v.rate_hz);
    }

    #[test]
    fn style_classifier_matches_pop_belt() {
        let mut sc = StyleClassifier::default_library();
        let snap = voiced_snap();
        let pitch = PitchSnapshot {
            f0_hz: 220.0,
            cents: 0.0,
            midi_note: 69.0,
            stability: 0.9,
            voiced: true,
        };
        let vib = VibratoSnapshot {
            rate_hz: 5.5,
            depth_cents: 50.0,
            presence: 0.6,
        };
        for _ in 0..16 {
            sc.observe(&snap, &pitch, &vib);
        }
        let mut out = [StyleMatch {
            name: "",
            similarity: 0.0,
        }; 3];
        let n = sc.top_k(3, &mut out);
        assert_eq!(n, 3);
        assert_eq!(out[0].name, "pop_belt");
        assert!(out[0].similarity > 0.85, "sim = {}", out[0].similarity);
    }

    #[test]
    fn karaoke_scorer_perfect_pitch_scores_one() {
        let mut k = KaraokeScorer::new(vec![69.0, 71.0, 72.0]);
        let p = PitchSnapshot {
            f0_hz: 440.0,
            cents: 0.0,
            midi_note: 69.0,
            stability: 1.0,
            voiced: true,
        };
        let s = k.score(&p);
        assert!(s > 0.99, "score = {}", s);
    }

    #[test]
    fn karaoke_scorer_off_by_semitone_low_score() {
        let mut k = KaraokeScorer::new(vec![70.0]);
        let p = PitchSnapshot {
            f0_hz: 440.0,
            cents: 0.0,
            midi_note: 69.0,
            stability: 1.0,
            voiced: true,
        };
        let s = k.score(&p);
        assert!(s < 0.10, "score = {}", s);
    }
}
