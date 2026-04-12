//! Analysis DAG bounded context — aggregator emitting `AnalysisFrame`s.
//!
//! See [the domain
//! model](../../../docs/ddd/clipcannon-domain-model.md#24-analysis-dag) for
//! invariants and [ADR-145](../../../docs/adr/ADR-145-clipcannon-rust-exploration.md)
//! for the relationship to ClipCannon's 23-stage DAG.

use super::prosody::ProsodySnapshot;
use super::viseme::VisemeCoeffs;

/// Coarse valence/arousal emotion bucket. See ADR-147 §3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmotionBucket {
    Neutral,
    Happy,
    Sad,
    Angry,
    Calm,
}

/// Aggregate root of the Analysis DAG context. One record per `process_block`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalysisFrame {
    pub session_id: u64,
    pub frame_index: u64,
    pub timestamp_us: u64,
    pub prosody: ProsodySnapshot,
    pub viseme: VisemeCoeffs,
    pub emotion: EmotionBucket,
    pub speaker_id: Option<u32>,
    pub highlight: f32,
    pub safe_cut: bool,
}

/// Compose contexts into `AnalysisFrame`s. Stateful — keeps a small rolling
/// window for safe-cut detection.
pub struct Analyzer {
    session_id: u64,
    frame_index: u64,
    sample_rate: f32,
    block_size: usize,
    /// Rolling window for safe-cut edge detection.
    history: [SafeCutSample; 8],
    history_filled: usize,
    history_pos: usize,
    /// Last 3 voiced energies for valence/arousal pitch dynamics.
    last_f0: [f32; 3],
    last_f0_filled: usize,
    last_f0_pos: usize,
    last_safe_cut: bool,
}

#[derive(Default, Clone, Copy)]
struct SafeCutSample {
    voicing: f32,
    energy_db: f32,
}

impl Analyzer {
    pub fn new(session_id: u64) -> Self {
        Self {
            session_id,
            frame_index: 0,
            sample_rate: 16_000.0,
            block_size: 128,
            history: [SafeCutSample::default(); 8],
            history_filled: 0,
            history_pos: 0,
            last_f0: [0.0; 3],
            last_f0_filled: 0,
            last_f0_pos: 0,
            last_safe_cut: false,
        }
    }

    pub fn prepare(&mut self, sample_rate: f32, block_size: usize) {
        self.sample_rate = sample_rate;
        self.block_size = block_size;
        self.frame_index = 0;
        self.history_filled = 0;
        self.history_pos = 0;
        self.last_f0_filled = 0;
        self.last_f0_pos = 0;
        self.last_safe_cut = false;
    }

    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    /// Combine prosody, viseme, and speaker id into one `AnalysisFrame`.
    pub fn analyse(
        &mut self,
        prosody: ProsodySnapshot,
        viseme: VisemeCoeffs,
        speaker_id: Option<u32>,
    ) -> AnalysisFrame {
        // Update F0 history.
        if prosody.f0_hz > 0.0 {
            self.last_f0[self.last_f0_pos] = prosody.f0_hz;
            self.last_f0_pos = (self.last_f0_pos + 1) % 3;
            self.last_f0_filled = (self.last_f0_filled + 1).min(3);
        }

        // Update safe-cut history with the new sample.
        let prev_was_safe = self.last_safe_cut;
        let new_sample = SafeCutSample {
            voicing: prosody.voicing,
            energy_db: prosody.energy_db,
        };
        self.history[self.history_pos] = new_sample;
        self.history_pos = (self.history_pos + 1) % 8;
        self.history_filled = (self.history_filled + 1).min(8);

        let safe_now = SafeCutDetector::evaluate(&self.history[..self.history_filled])
            && new_sample.voicing < 0.20
            && new_sample.energy_db < -38.0;
        let safe_cut = safe_now && !prev_was_safe;
        self.last_safe_cut = safe_now;

        let emotion = classify_emotion(&prosody, &self.last_f0[..self.last_f0_filled]);
        let highlight = HighlightScorer::score_inner(&prosody, &self.last_f0[..self.last_f0_filled]);

        let timestamp_us = self.frame_index_to_us(self.frame_index);
        let frame = AnalysisFrame {
            session_id: self.session_id,
            frame_index: self.frame_index,
            timestamp_us,
            prosody,
            viseme,
            emotion,
            speaker_id,
            highlight,
            safe_cut,
        };
        self.frame_index = self.frame_index.saturating_add(1);
        frame
    }

    fn frame_index_to_us(&self, idx: u64) -> u64 {
        if self.sample_rate <= 0.0 {
            return 0;
        }
        let secs = (idx as f64) * (self.block_size as f64) / (self.sample_rate as f64);
        (secs * 1_000_000.0) as u64
    }
}

/// Stateless highlight scoring. Combines arousal, voicing, energy and pitch
/// dynamics into a `[0,1]` salience.
pub struct HighlightScorer;

impl HighlightScorer {
    pub fn score(prosody: &ProsodySnapshot) -> f32 {
        Self::score_inner(prosody, &[])
    }

    fn score_inner(p: &ProsodySnapshot, f0_history: &[f32]) -> f32 {
        let arousal = arousal_from(p, f0_history);
        let energy_norm = ((p.energy_db + 50.0) / 50.0).clamp(0.0, 1.0);
        let pitch_dynamics = pitch_dynamics(f0_history);
        (0.4 * arousal + 0.3 * p.voicing + 0.2 * energy_norm + 0.1 * pitch_dynamics).clamp(0.0, 1.0)
    }
}

/// Stateless safe-cut detector. See ADR-146 §4.3.
pub struct SafeCutDetector;

impl SafeCutDetector {
    /// Returns true iff at least 4 of the supplied frames look like silence.
    pub fn evaluate(frames: &[SafeCutSample]) -> bool {
        if frames.len() < 4 {
            return false;
        }
        let silent = frames
            .iter()
            .filter(|s| s.voicing < 0.20 && s.energy_db < -38.0)
            .count();
        silent >= 4
    }
}

fn arousal_from(p: &ProsodySnapshot, f0_history: &[f32]) -> f32 {
    let energy_norm = ((p.energy_db + 50.0) / 50.0).clamp(0.0, 1.0);
    let f0_var = if f0_history.len() >= 2 {
        let mean = f0_history.iter().copied().sum::<f32>() / f0_history.len() as f32;
        let var = f0_history.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / f0_history.len() as f32;
        (var.sqrt() / 50.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0.7 * energy_norm + 0.3 * f0_var).clamp(0.0, 1.0)
}

fn pitch_dynamics(f0_history: &[f32]) -> f32 {
    if f0_history.len() < 2 {
        return 0.0;
    }
    let max = f0_history.iter().cloned().fold(f32::MIN, f32::max);
    let min = f0_history.iter().cloned().fold(f32::MAX, f32::min);
    ((max - min) / 100.0).clamp(0.0, 1.0)
}

fn classify_emotion(p: &ProsodySnapshot, f0_history: &[f32]) -> EmotionBucket {
    if p.voicing < 0.25 || p.energy_db < -45.0 {
        return EmotionBucket::Neutral;
    }
    let arousal = arousal_from(p, f0_history);
    let valence = if f0_history.len() >= 2 {
        let last = *f0_history.last().unwrap();
        let first = f0_history[0];
        ((last - first) / 50.0).clamp(-1.0, 1.0) - 0.4 * p.flatness
    } else {
        -0.2 * p.flatness
    };

    match (arousal > 0.55, valence > 0.0) {
        (true, true) => EmotionBucket::Happy,
        (true, false) => EmotionBucket::Angry,
        (false, true) => EmotionBucket::Calm,
        (false, false) => {
            if p.energy_db < -25.0 {
                EmotionBucket::Sad
            } else {
                EmotionBucket::Neutral
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::viseme::{Viseme, VisemeCoeffs};
    use super::*;

    fn make_snap(voicing: f32, energy_db: f32, f0: f32) -> ProsodySnapshot {
        ProsodySnapshot {
            f0_hz: f0,
            voicing,
            energy_db,
            centroid_hz: 1500.0,
            rolloff_hz: 4000.0,
            zcr: 0.05,
            flatness: 0.1,
        }
    }

    fn rest_viseme() -> VisemeCoeffs {
        VisemeCoeffs::rest()
    }

    fn aa_viseme() -> VisemeCoeffs {
        VisemeCoeffs {
            viseme: Viseme::Aa,
            jaw_open: 0.7,
            lip_round: 0.4,
            lip_spread: 0.6,
            confidence: 0.8,
        }
    }

    #[test]
    fn frame_index_is_monotonic() {
        let mut a = Analyzer::new(42);
        a.prepare(16_000.0, 128);
        for i in 0..10 {
            let f = a.analyse(make_snap(0.0, -80.0, 0.0), rest_viseme(), None);
            assert_eq!(f.frame_index, i);
            assert_eq!(f.session_id, 42);
        }
    }

    #[test]
    fn silence_yields_neutral_emotion() {
        let mut a = Analyzer::new(1);
        a.prepare(16_000.0, 128);
        let f = a.analyse(make_snap(0.0, -80.0, 0.0), rest_viseme(), None);
        assert_eq!(f.emotion, EmotionBucket::Neutral);
        assert!(f.highlight < 0.2);
    }

    #[test]
    fn loud_voiced_speech_has_high_highlight() {
        let mut a = Analyzer::new(1);
        a.prepare(16_000.0, 128);
        // prime f0 history
        for _ in 0..3 {
            let _ = a.analyse(make_snap(0.9, -8.0, 220.0), aa_viseme(), Some(1));
        }
        let f = a.analyse(make_snap(0.95, -6.0, 260.0), aa_viseme(), Some(1));
        assert!(f.highlight > 0.5, "highlight = {}", f.highlight);
    }

    #[test]
    fn safe_cut_fires_on_silence_edge() {
        let mut a = Analyzer::new(1);
        a.prepare(16_000.0, 128);
        // 5 voiced frames first
        for _ in 0..5 {
            let _ = a.analyse(make_snap(0.9, -10.0, 220.0), aa_viseme(), Some(1));
        }
        // then 5 silent frames; safe_cut should fire on the first silent frame
        // that brings the rolling window to ≥4 silent samples.
        let mut fired = 0;
        for _ in 0..5 {
            let f = a.analyse(make_snap(0.05, -60.0, 0.0), rest_viseme(), None);
            if f.safe_cut {
                fired += 1;
            }
        }
        assert_eq!(fired, 1, "safe_cut should be an edge, not a plateau");
    }

    #[test]
    fn timestamp_grows_with_frames() {
        let mut a = Analyzer::new(1);
        a.prepare(16_000.0, 128);
        let f0 = a.analyse(make_snap(0.0, -80.0, 0.0), rest_viseme(), None);
        let f1 = a.analyse(make_snap(0.0, -80.0, 0.0), rest_viseme(), None);
        let f2 = a.analyse(make_snap(0.0, -80.0, 0.0), rest_viseme(), None);
        assert_eq!(f0.timestamp_us, 0);
        assert!(f1.timestamp_us > f0.timestamp_us);
        assert!(f2.timestamp_us > f1.timestamp_us);
    }
}
