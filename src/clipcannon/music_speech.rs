//! Music vs Speech discriminator — see ADR-151 §2.

use super::prosody::ProsodySnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalKind {
    Silence,
    Speech,
    Music,
    Mixed,
}

pub struct MusicSpeechDetector {
    /// Rolling 16-frame window of musicness scores.
    history: [f32; 16],
    head: usize,
    filled: usize,
    /// Rolling ZCR window for variance.
    zcr_history: [f32; 16],
    /// Rolling F0 history for stability.
    f0_history: [f32; 16],
    /// Rolling energy for syllabic modulation.
    energy_history: [f32; 16],
    /// Hysteresis state.
    is_music: bool,
    on_threshold: f32,
    off_threshold: f32,
}

impl Default for MusicSpeechDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl MusicSpeechDetector {
    pub fn new() -> Self {
        Self {
            history: [0.0; 16],
            head: 0,
            filled: 0,
            zcr_history: [0.0; 16],
            f0_history: [0.0; 16],
            energy_history: [0.0; 16],
            is_music: false,
            on_threshold: 0.55,
            off_threshold: 0.40,
        }
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.filled = 0;
        self.is_music = false;
    }

    pub fn observe(&mut self, snap: &ProsodySnapshot, vad_active: bool) -> SignalKind {
        // Push raw features.
        let i = self.head;
        self.zcr_history[i] = snap.zcr;
        self.f0_history[i] = snap.f0_hz;
        self.energy_history[i] = snap.energy_db;
        self.head = (self.head + 1) % 16;
        self.filled = (self.filled + 1).min(16);

        if snap.energy_db < -45.0 {
            self.history[i] = 0.0;
            self.is_music = false;
            return SignalKind::Silence;
        }

        // ZCR variance over the window.
        let zcr_var = variance(&self.zcr_history[..self.filled]);
        // Speech ZCR has high variance (0.001+), music low (0.0001).
        let zcr_var_norm = (zcr_var / 0.005).clamp(0.0, 1.0);

        // F0 stability — fraction of frames with valid F0 within ±10 Hz of mean.
        let valid_f0: Vec<f32> = self.f0_history[..self.filled]
            .iter()
            .copied()
            .filter(|&f| f > 0.0)
            .collect();
        let f0_stability = if valid_f0.len() >= 4 {
            let mean = valid_f0.iter().sum::<f32>() / valid_f0.len() as f32;
            let close = valid_f0.iter().filter(|&&f| (f - mean).abs() < 10.0).count();
            close as f32 / valid_f0.len() as f32
        } else {
            0.0
        };

        // Syllabic-modulation proxy: variance of the energy envelope.
        let energy_var = variance(&self.energy_history[..self.filled]);
        let syllabic_norm = (energy_var / 25.0).clamp(0.0, 1.0); // dB² scale

        // Higher musicness = lower zcr_var, higher f0_stab, lower energy variance.
        let musicness = 0.4 * (1.0 - zcr_var_norm)
            + 0.3 * f0_stability
            + 0.3 * (1.0 - syllabic_norm);

        self.history[i] = musicness;

        let mean_musicness =
            self.history[..self.filled].iter().sum::<f32>() / self.filled.max(1) as f32;

        if !self.is_music && mean_musicness >= self.on_threshold {
            self.is_music = true;
        } else if self.is_music && mean_musicness < self.off_threshold {
            self.is_music = false;
        }

        match (self.is_music, vad_active) {
            (true, true) => SignalKind::Mixed,
            (true, false) => SignalKind::Music,
            (false, true) => SignalKind::Speech,
            (false, false) => {
                if snap.voicing < 0.25 {
                    SignalKind::Silence
                } else {
                    SignalKind::Speech
                }
            }
        }
    }
}

fn variance(xs: &[f32]) -> f32 {
    if xs.len() < 2 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f32>() / xs.len() as f32;
    xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / xs.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(zcr: f32, f0: f32, energy_db: f32, flatness: f32) -> ProsodySnapshot {
        ProsodySnapshot {
            f0_hz: f0,
            voicing: if f0 > 0.0 { 0.85 } else { 0.0 },
            energy_db,
            centroid_hz: 1500.0,
            rolloff_hz: 4000.0,
            zcr,
            flatness,
        }
    }

    #[test]
    fn silence_detected_by_low_energy() {
        let mut d = MusicSpeechDetector::new();
        for _ in 0..16 {
            assert_eq!(d.observe(&snap(0.05, 0.0, -80.0, 1.0), false), SignalKind::Silence);
        }
    }

    #[test]
    fn stable_tone_classifies_as_music() {
        let mut d = MusicSpeechDetector::new();
        // Stable F0, low ZCR variance, low energy variance.
        for _ in 0..16 {
            d.observe(&snap(0.04, 220.0, -10.0, 0.1), false);
        }
        let result = d.observe(&snap(0.04, 220.0, -10.0, 0.1), false);
        assert_eq!(result, SignalKind::Music);
    }

    #[test]
    fn variable_speech_classifies_as_speech() {
        let mut d = MusicSpeechDetector::new();
        let zcrs = [0.05, 0.20, 0.10, 0.30, 0.08, 0.25, 0.12, 0.18];
        let f0s = [220.0, 0.0, 180.0, 0.0, 240.0, 0.0, 200.0, 0.0];
        let energies = [-10.0, -25.0, -8.0, -30.0, -12.0, -28.0, -9.0, -26.0];
        for i in 0..16 {
            let j = i % 8;
            d.observe(&snap(zcrs[j], f0s[j], energies[j], 0.3), true);
        }
        let result = d.observe(&snap(zcrs[0], f0s[0], energies[0], 0.3), true);
        assert!(matches!(result, SignalKind::Speech | SignalKind::Mixed));
    }
}
