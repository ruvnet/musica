//! Voice Activity Detection + End-of-Turn detection.
//!
//! See [ADR-150](../../../docs/adr/ADR-150-clipcannon-vad-end-of-turn.md).
//!
//! Pure state machine over a [`ProsodySnapshot`]. Allocation-free.

use super::prosody::ProsodySnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    Inactive,
    Active,
}

#[derive(Debug, Clone, Copy)]
pub struct VadDecision {
    pub state: VadState,
    pub speech_score: f32,
    pub onset_edge: bool,
    pub offset_edge: bool,
    pub end_of_turn: bool,
}

pub struct VadDetector {
    state: VadState,
    pub onset_thresh: f32,
    pub offset_thresh: f32,
    pub onset_frames: u32,
    pub offset_frames: u32,
    above_count: u32,
    below_count: u32,
    pub eot_silence_ms: u32,
    silence_ms_acc: u32,
    eot_armed: bool,
}

impl Default for VadDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl VadDetector {
    pub fn new() -> Self {
        Self {
            state: VadState::Inactive,
            onset_thresh: 0.55,
            offset_thresh: 0.35,
            onset_frames: 3,
            offset_frames: 6,
            above_count: 0,
            below_count: 0,
            eot_silence_ms: 700,
            silence_ms_acc: 0,
            eot_armed: false,
        }
    }

    pub fn reset(&mut self) {
        self.state = VadState::Inactive;
        self.above_count = 0;
        self.below_count = 0;
        self.silence_ms_acc = 0;
        self.eot_armed = false;
    }

    pub fn state(&self) -> VadState {
        self.state
    }

    pub fn observe(&mut self, snap: &ProsodySnapshot, block_ms: f32) -> VadDecision {
        let speech_score = compute_speech_score(snap);
        let mut onset_edge = false;
        let mut offset_edge = false;
        let mut end_of_turn = false;

        match self.state {
            VadState::Inactive => {
                if speech_score >= self.onset_thresh {
                    self.above_count += 1;
                    self.below_count = 0;
                    if self.above_count >= self.onset_frames {
                        self.state = VadState::Active;
                        self.above_count = 0;
                        self.silence_ms_acc = 0;
                        self.eot_armed = true;
                        onset_edge = true;
                    }
                } else {
                    self.above_count = 0;
                    if self.eot_armed {
                        self.silence_ms_acc =
                            self.silence_ms_acc.saturating_add(block_ms as u32);
                        if self.silence_ms_acc >= self.eot_silence_ms {
                            end_of_turn = true;
                            self.eot_armed = false;
                            self.silence_ms_acc = 0;
                        }
                    }
                }
            }
            VadState::Active => {
                if speech_score < self.offset_thresh {
                    self.below_count += 1;
                    self.above_count = 0;
                    if self.below_count >= self.offset_frames {
                        self.state = VadState::Inactive;
                        self.below_count = 0;
                        offset_edge = true;
                    }
                } else {
                    self.below_count = 0;
                }
                self.silence_ms_acc = 0;
            }
        }

        VadDecision {
            state: self.state,
            speech_score,
            onset_edge,
            offset_edge,
            end_of_turn,
        }
    }
}

fn compute_speech_score(s: &ProsodySnapshot) -> f32 {
    let voicing_score = s.voicing.clamp(0.0, 1.0);
    let energy_score = sigmoid((s.energy_db + 38.0) / 6.0);
    let flatness_score = (1.0 - s.flatness).clamp(0.0, 1.0);
    let zcr_score = 1.0 - smoothstep(0.20, 0.40, s.zcr);
    (0.45 * voicing_score + 0.30 * energy_score + 0.15 * flatness_score + 0.10 * zcr_score)
        .clamp(0.0, 1.0)
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn voiced(energy_db: f32) -> ProsodySnapshot {
        ProsodySnapshot {
            f0_hz: 220.0,
            voicing: 0.9,
            energy_db,
            centroid_hz: 1500.0,
            rolloff_hz: 4000.0,
            zcr: 0.05,
            flatness: 0.1,
        }
    }

    fn silent() -> ProsodySnapshot {
        ProsodySnapshot::silent()
    }

    #[test]
    fn starts_inactive() {
        let v = VadDetector::new();
        assert_eq!(v.state(), VadState::Inactive);
    }

    #[test]
    fn voiced_frames_trigger_onset_after_hysteresis() {
        let mut v = VadDetector::new();
        let s = voiced(-10.0);
        // 2 frames: not yet
        let d = v.observe(&s, 8.0);
        assert!(!d.onset_edge);
        let _ = v.observe(&s, 8.0);
        // 3rd frame triggers
        let d = v.observe(&s, 8.0);
        assert!(d.onset_edge);
        assert_eq!(d.state, VadState::Active);
    }

    #[test]
    fn silent_frames_trigger_offset_after_hysteresis() {
        let mut v = VadDetector::new();
        let s = voiced(-10.0);
        for _ in 0..3 {
            v.observe(&s, 8.0);
        }
        assert_eq!(v.state(), VadState::Active);
        let q = silent();
        for _ in 0..5 {
            let d = v.observe(&q, 8.0);
            assert!(!d.offset_edge);
        }
        let d = v.observe(&q, 8.0);
        assert!(d.offset_edge);
        assert_eq!(d.state, VadState::Inactive);
    }

    #[test]
    fn end_of_turn_fires_after_silence() {
        let mut v = VadDetector::new();
        v.eot_silence_ms = 80;
        let s = voiced(-10.0);
        for _ in 0..3 {
            v.observe(&s, 8.0);
        }
        let q = silent();
        // Drop out
        for _ in 0..6 {
            v.observe(&q, 8.0);
        }
        // Now keep going silent — eot should fire after 80 ms = 10 blocks
        let mut fired = 0;
        for _ in 0..20 {
            let d = v.observe(&q, 8.0);
            if d.end_of_turn {
                fired += 1;
            }
        }
        assert_eq!(fired, 1);
    }

    #[test]
    fn single_glitch_does_not_offset() {
        let mut v = VadDetector::new();
        let s = voiced(-10.0);
        for _ in 0..5 {
            v.observe(&s, 8.0);
        }
        // Single silent frame
        let d = v.observe(&silent(), 8.0);
        assert!(!d.offset_edge);
        // Continue voiced
        let d = v.observe(&s, 8.0);
        assert_eq!(d.state, VadState::Active);
    }
}
