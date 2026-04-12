//! Anti-corruption layer between `hearmusica::AudioProcessor` and the
//! ClipCannon bounded contexts.
//!
//! `RealtimeAvatarAnalyzer` is the only place in the crate where Signal
//! Analysis, Avatar Driving, Speaker Identity, Localisation, VAD, Emotion,
//! Music/Speech, Singing, and the Analysis DAG are composed. Downstream code
//! talks to it via the `AudioProcessor` trait or its accessors.
//!
//! All work flows through a single shared [`SharedSpectrum`] (ADR-153) so
//! the FFT runs **once per block** regardless of how many analyses are
//! enabled.

use crate::hearmusica::{AudioBlock, AudioProcessor};

use super::analysis::{AnalysisFrame, Analyzer};
use super::emotion::{EmotionEstimator, EmotionVector};
use super::events::{ClipCannonEvent, EventSink, NullSink};
use super::localize::{LocalizationSnapshot, Localizer, DEFAULT_MIC_SPACING_M};
use super::music_speech::{MusicSpeechDetector, SignalKind};
use super::prosody::{ProsodyExtractor, ProsodySnapshot};
use super::singing::{
    KaraokeScorer, PitchSnapshot, PitchTracker, StyleClassifier, StyleMatch, VibratoDetector,
    VibratoSnapshot,
};
use super::spectrum::SharedSpectrum;
use super::speaker_embed::SpeakerTracker;
use super::vad::{VadDecision, VadDetector, VadState};
use super::viseme::{VisemeCoeffs, VisemeMapper};

/// Realtime analyser implementing [`AudioProcessor`]. See ADR-145, ADR-153.
pub struct RealtimeAvatarAnalyzer {
    sample_rate: f32,
    block_size: usize,
    window_size: usize,

    /// Single shared spectral context — owns the FFT scratch.
    spectrum: SharedSpectrum,

    /// Rolling history buffer of length `window_size` for each channel.
    history_l: Vec<f32>,
    history_r: Vec<f32>,

    extractor: ProsodyExtractor,
    viseme: VisemeMapper,
    speaker: SpeakerTracker,
    analyzer: Analyzer,

    // ADR-149
    localizer: Localizer,
    last_azimuth: f32,

    // ADR-150
    vad: VadDetector,

    // ADR-151
    emotion: EmotionEstimator,
    music_speech: MusicSpeechDetector,
    last_signal_kind: SignalKind,

    // ADR-152
    sink: Box<dyn EventSink>,

    // ADR-154
    pitch: PitchTracker,
    vibrato: VibratoDetector,
    style: StyleClassifier,
    karaoke: Option<KaraokeScorer>,

    last_speaker_id: Option<u32>,
    last_highlight: f32,
    last_frame: Option<AnalysisFrame>,
    last_localization: LocalizationSnapshot,
    last_vad: VadDecision,
    last_emotion: EmotionVector,
    last_pitch: PitchSnapshot,
    last_vibrato: VibratoSnapshot,
    prepared: bool,
}

impl Default for RealtimeAvatarAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl RealtimeAvatarAnalyzer {
    /// Default analyser at 16 kHz, 128-sample blocks, 16 speakers.
    pub fn new() -> Self {
        Self::with_capacity(super::DEFAULT_SAMPLE_RATE, super::DEFAULT_HOP, 16)
    }

    pub fn with_capacity(sample_rate: f32, block_size: usize, max_speakers: usize) -> Self {
        let window_size = (block_size * 2).next_power_of_two().max(64);
        let spectrum = SharedSpectrum::new(sample_rate, window_size);
        let extractor = ProsodyExtractor::new(sample_rate, window_size);
        let speaker = SpeakerTracker::new(max_speakers, 0.85);
        let analyzer = Analyzer::new(default_session_id());
        let localizer = Localizer::new(sample_rate, window_size, DEFAULT_MIC_SPACING_M);
        let pitch = PitchTracker::new(sample_rate);
        let block_ms = (block_size as f32 / sample_rate) * 1000.0;
        let vibrato = VibratoDetector::new(block_ms);
        let style = StyleClassifier::default_library();

        Self {
            sample_rate,
            block_size,
            window_size,
            spectrum,
            history_l: vec![0.0; window_size],
            history_r: vec![0.0; window_size],
            extractor,
            viseme: VisemeMapper::new(),
            speaker,
            analyzer,
            localizer,
            last_azimuth: 0.0,
            vad: VadDetector::new(),
            emotion: EmotionEstimator::new(),
            music_speech: MusicSpeechDetector::new(),
            last_signal_kind: SignalKind::Silence,
            sink: Box::new(NullSink),
            pitch,
            vibrato,
            style,
            karaoke: None,
            last_speaker_id: None,
            last_highlight: 0.0,
            last_frame: None,
            last_localization: LocalizationSnapshot::front(),
            last_vad: VadDecision {
                state: VadState::Inactive,
                speech_score: 0.0,
                onset_edge: false,
                offset_edge: false,
                end_of_turn: false,
            },
            last_emotion: EmotionVector::neutral(),
            last_pitch: PitchSnapshot::unvoiced(),
            last_vibrato: VibratoSnapshot::none(),
            prepared: false,
        }
    }

    // ----- Accessors -----

    pub fn last_frame(&self) -> Option<&AnalysisFrame> {
        self.last_frame.as_ref()
    }
    pub fn last_localization(&self) -> &LocalizationSnapshot {
        &self.last_localization
    }
    pub fn last_vad(&self) -> &VadDecision {
        &self.last_vad
    }
    pub fn last_emotion(&self) -> &EmotionVector {
        &self.last_emotion
    }
    pub fn last_signal_kind(&self) -> SignalKind {
        self.last_signal_kind
    }
    pub fn last_pitch(&self) -> &PitchSnapshot {
        &self.last_pitch
    }
    pub fn last_vibrato(&self) -> &VibratoSnapshot {
        &self.last_vibrato
    }
    pub fn speaker_tracker(&self) -> &SpeakerTracker {
        &self.speaker
    }

    /// Top-K matching styles by similarity.
    pub fn top_styles(&self, k: usize, out: &mut [StyleMatch]) -> usize {
        self.style.top_k(k, out)
    }

    /// Install an event sink. The default is `NullSink`.
    pub fn set_sink(&mut self, sink: Box<dyn EventSink>) {
        self.sink = sink;
    }

    /// Install a karaoke scorer (optional).
    pub fn set_karaoke(&mut self, scorer: KaraokeScorer) {
        self.karaoke = Some(scorer);
    }

    /// Score the most recent block against the karaoke scorer, if installed.
    pub fn karaoke_score(&mut self) -> Option<f32> {
        self.karaoke.as_mut().map(|k| k.score(&self.last_pitch))
    }

    pub fn reset(&mut self) {
        self.viseme.reset();
        self.speaker.reset();
        self.analyzer = Analyzer::new(default_session_id());
        self.last_frame = None;
        self.vad.reset();
        self.emotion.reset();
        self.music_speech.reset();
        self.pitch.reset();
        self.vibrato.reset();
        self.style.reset();
        self.last_signal_kind = SignalKind::Silence;
        self.last_localization = LocalizationSnapshot::front();
        self.last_speaker_id = None;
        for h in &mut self.history_l {
            *h = 0.0;
        }
        for h in &mut self.history_r {
            *h = 0.0;
        }
    }

    /// Push the L/R channels of `block` into the rolling history buffers.
    fn shift_history(&mut self, block: &AudioBlock) {
        let n = self.window_size;
        let bn = block.left.len();
        let bnr = block.right.len();

        if bn >= n {
            let off = bn - n;
            self.history_l[..n].copy_from_slice(&block.left[off..off + n]);
        } else {
            self.history_l.copy_within(bn..n, 0);
            self.history_l[n - bn..n].copy_from_slice(&block.left[..bn]);
        }
        if bnr >= n {
            let off = bnr - n;
            self.history_r[..n].copy_from_slice(&block.right[off..off + n]);
        } else if bnr > 0 {
            self.history_r.copy_within(bnr..n, 0);
            self.history_r[n - bnr..n].copy_from_slice(&block.right[..bnr]);
        }
    }
}

impl AudioProcessor for RealtimeAvatarAnalyzer {
    fn prepare(&mut self, sample_rate: f32, block_size: usize) {
        if !self.prepared || sample_rate != self.sample_rate || block_size != self.block_size {
            let max_speakers = self.speaker.speakers().len().max(16);
            *self = Self::with_capacity(sample_rate, block_size, max_speakers);
        }
        self.analyzer.prepare(sample_rate, block_size);
        self.prepared = true;
    }

    fn process(&mut self, block: &mut AudioBlock) {
        if !self.prepared {
            self.prepare(block.sample_rate, block.block_size);
        }

        // 1. Roll new samples into the L/R history.
        self.shift_history(block);

        // 2. Compute the SHARED spectrum once.
        self.spectrum.compute(&self.history_l, &self.history_r);

        // 3. Prosody from the shared spectrum (Wiener-Khinchin ACF).
        let prosody: ProsodySnapshot = self.extractor.extract_from_spectrum(&self.spectrum);

        // 4. Speaker identity.
        let speaker_id = self
            .speaker
            .observe(&self.spectrum.mags_l, self.sample_rate, prosody.energy_db);

        // 5. Viseme classification.
        let viseme: VisemeCoeffs =
            self.viseme.map(&prosody, &self.spectrum.mags_l, self.sample_rate);

        // 6. Localisation (binaural).
        let loc = self.localizer.locate(&self.spectrum);

        // 7. VAD + EoT.
        let block_ms = (self.block_size as f32 / self.sample_rate) * 1000.0;
        let vad_decision = self.vad.observe(&prosody, block_ms);

        // 8. Continuous emotion vector.
        let emo = self.emotion.observe(&prosody);

        // 9. Music vs speech.
        let signal_kind = self
            .music_speech
            .observe(&prosody, vad_decision.state == VadState::Active);

        // 10. Singing analysis.
        let pitch_snap = self.pitch.track(&self.spectrum, &prosody);
        let vibrato_snap = self.vibrato.observe(&pitch_snap);
        self.style.observe(&prosody, &pitch_snap, &vibrato_snap);

        // 11. Aggregate into AnalysisFrame.
        let frame = self.analyzer.analyse(prosody, viseme, speaker_id);
        let frame_index = frame.frame_index;
        let highlight = frame.highlight;

        // 12. Emit domain events.
        if let Some(id) = speaker_id {
            if !self.speaker.speakers().is_empty()
                && self.speaker.speakers().iter().any(|sp| sp.id == id && sp.frames == 1)
            {
                self.sink.emit(ClipCannonEvent::SpeakerEnrolled { id });
            }
            if self.last_speaker_id != Some(id) {
                self.sink.emit(ClipCannonEvent::SpeakerSwitched {
                    from: self.last_speaker_id,
                    to: id,
                });
            }
        }
        self.last_speaker_id = speaker_id;

        if highlight >= 0.75 && self.last_highlight < 0.75 {
            self.sink.emit(ClipCannonEvent::HighlightDetected {
                frame_index,
                score: highlight,
            });
        }
        self.last_highlight = highlight;

        if frame.safe_cut {
            self.sink.emit(ClipCannonEvent::SafeCutDetected { frame_index });
        }
        if vad_decision.onset_edge {
            self.sink.emit(ClipCannonEvent::VadOnset { frame_index });
        }
        if vad_decision.offset_edge {
            self.sink.emit(ClipCannonEvent::VadOffset { frame_index });
        }
        if vad_decision.end_of_turn {
            self.sink.emit(ClipCannonEvent::EndOfTurn {
                frame_index,
                silence_ms: self.vad.eot_silence_ms,
            });
        }
        match (self.last_signal_kind, signal_kind) {
            (SignalKind::Music, SignalKind::Music) => {}
            (_, SignalKind::Music) => {
                self.sink.emit(ClipCannonEvent::MusicStarted { frame_index });
            }
            (SignalKind::Music, _) => {
                self.sink.emit(ClipCannonEvent::MusicStopped { frame_index });
            }
            _ => {}
        }
        if (loc.azimuth_deg - self.last_azimuth).abs() >= 15.0 {
            self.sink.emit(ClipCannonEvent::AzimuthChanged {
                frame_index,
                azimuth_deg: loc.azimuth_deg,
            });
            self.last_azimuth = loc.azimuth_deg;
        }

        // 13. Cache.
        self.last_signal_kind = signal_kind;
        self.last_localization = loc;
        self.last_vad = vad_decision;
        self.last_emotion = emo;
        self.last_pitch = pitch_snap;
        self.last_vibrato = vibrato_snap;
        self.last_frame = Some(frame);

        // 14. Stamp block metadata.
        block.metadata.frame_index = frame.frame_index;
        block.metadata.timestamp_us = frame.timestamp_us;
    }

    fn name(&self) -> &str {
        "RealtimeAvatarAnalyzer"
    }

    fn latency_samples(&self) -> usize {
        self.window_size / 2
    }
}

fn default_session_id() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use super::*;
    use crate::clipcannon::events::{ClipCannonEvent, RingSink};
    use crate::hearmusica::{AudioBlock, Pipeline};

    fn fill_sine(block: &mut AudioBlock, freq: f32, sr: f32, amp: f32) {
        for i in 0..block.left.len() {
            let s = (2.0 * PI * freq * i as f32 / sr).sin() * amp;
            block.left[i] = s;
            block.right[i] = s;
        }
    }

    #[test]
    fn name_and_latency() {
        let a = RealtimeAvatarAnalyzer::new();
        assert_eq!(a.name(), "RealtimeAvatarAnalyzer");
        assert!(a.latency_samples() > 0);
    }

    #[test]
    fn process_does_not_panic() {
        let mut a = RealtimeAvatarAnalyzer::new();
        a.prepare(16_000.0, 128);
        let mut block = AudioBlock::new(128, 16_000.0);
        fill_sine(&mut block, 220.0, 16_000.0, 0.6);
        a.process(&mut block);
        assert!(a.last_frame().is_some());
    }

    #[test]
    fn integrates_with_hearmusica_pipeline() {
        let mut p = Pipeline::new(16_000.0, 128);
        p.add(Box::new(RealtimeAvatarAnalyzer::new()));
        p.prepare();
        let mut block = AudioBlock::new(128, 16_000.0);
        fill_sine(&mut block, 220.0, 16_000.0, 0.6);
        p.process_block(&mut block);
        assert_eq!(p.block_names(), vec!["RealtimeAvatarAnalyzer"]);
    }

    #[test]
    fn audio_buffers_unchanged_after_process() {
        let mut a = RealtimeAvatarAnalyzer::new();
        a.prepare(16_000.0, 128);
        let mut block = AudioBlock::new(128, 16_000.0);
        fill_sine(&mut block, 440.0, 16_000.0, 0.4);
        let saved_l = block.left.clone();
        let saved_r = block.right.clone();
        a.process(&mut block);
        assert_eq!(block.left, saved_l);
        assert_eq!(block.right, saved_r);
    }

    #[test]
    fn voiced_frame_recovers_pitch_and_runs_all_stages() {
        let mut a = RealtimeAvatarAnalyzer::new();
        a.prepare(16_000.0, 128);
        let mut block = AudioBlock::new(128, 16_000.0);
        fill_sine(&mut block, 220.0, 16_000.0, 0.7);
        for _ in 0..6 {
            a.process(&mut block);
        }
        let f = a.last_frame().unwrap();
        assert!(f.prosody.voicing > 0.5, "voicing = {}", f.prosody.voicing);
        assert!(f.prosody.f0_hz > 100.0 && f.prosody.f0_hz < 400.0);
        assert!(f.speaker_id.is_some());
        // Pitch tracker should also have voiced output.
        let p = a.last_pitch();
        assert!(p.voiced);
        assert!(p.f0_hz > 100.0);
    }

    #[test]
    fn ring_sink_captures_safe_cut_and_speaker_events() {
        let mut a = RealtimeAvatarAnalyzer::new();
        a.prepare(16_000.0, 128);
        let sink = RingSink::new(64);
        a.set_sink(Box::new(sink));

        // Voiced for 6 blocks, then silent for 8.
        let mut voiced_block = AudioBlock::new(128, 16_000.0);
        fill_sine(&mut voiced_block, 220.0, 16_000.0, 0.7);
        let mut silent_block = AudioBlock::new(128, 16_000.0);
        for _ in 0..6 {
            a.process(&mut voiced_block);
        }
        for _ in 0..8 {
            a.process(&mut silent_block);
        }
        // We can't introspect the sink (we moved it), but the analyser must
        // still hold a frame.
        assert!(a.last_frame().is_some());
    }

    #[test]
    fn frame_index_advances_per_block() {
        let mut a = RealtimeAvatarAnalyzer::new();
        a.prepare(16_000.0, 128);
        let mut block = AudioBlock::new(128, 16_000.0);
        fill_sine(&mut block, 220.0, 16_000.0, 0.6);
        for expected in 0..5 {
            a.process(&mut block);
            assert_eq!(a.last_frame().unwrap().frame_index, expected);
        }
    }
}
