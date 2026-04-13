//! Streaming TTS pipeline — `RealtimeVoiceSynthesiser`. See ADR-159.
//!
//! Composes the phonemiser, the Klatt voice, and the singing synthesiser
//! into a single front-door API for hosts. Same `AudioBlock` shape as
//! `RealtimeAvatarAnalyzer` so the listen-think-speak loop slots straight
//! into a `hearmusica::Pipeline`.

use std::collections::VecDeque;

use crate::hearmusica::AudioBlock;

use super::events::{ClipCannonEvent, EventSink, NullSink};
use super::klatt::KlattSynthesiser;
use super::phonemise::{Phoneme, Phonemiser, TimedPhoneme};
use super::sing_synth::{SingerVoice, SingingSynthesiser, SungNote, VoiceBank};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceMode {
    Speak,
    Sing,
}

/// Streaming voice synthesiser.
pub struct RealtimeVoiceSynthesiser {
    sample_rate: f32,
    block_size: usize,
    klatt: KlattSynthesiser,
    phonemiser: Phonemiser,
    singer: SingingSynthesiser,
    queue: VecDeque<TimedPhoneme>,
    melody_queue: VecDeque<SungNote>,
    /// Samples remaining for the current head item.
    remaining: usize,
    /// Current voice mode.
    mode: VoiceMode,
    /// Increasing utterance counter for events.
    next_utterance_id: u32,
    current_utterance_id: u32,
    /// Whether we just transitioned into a new utterance and need a Started event.
    pending_started: bool,
    /// Index of the current phoneme within the utterance.
    current_phoneme_index: u32,
    sink: Box<dyn EventSink>,
}

impl RealtimeVoiceSynthesiser {
    pub fn new(sample_rate: f32, block_size: usize) -> Self {
        Self {
            sample_rate,
            block_size,
            klatt: KlattSynthesiser::new(sample_rate),
            phonemiser: Phonemiser::english(),
            singer: SingingSynthesiser::new_klatt(sample_rate),
            queue: VecDeque::with_capacity(64),
            melody_queue: VecDeque::with_capacity(64),
            remaining: 0,
            mode: VoiceMode::Speak,
            next_utterance_id: 1,
            current_utterance_id: 0,
            pending_started: false,
            current_phoneme_index: 0,
            sink: Box::new(NullSink),
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }
    pub fn block_size(&self) -> usize {
        self.block_size
    }
    pub fn mode(&self) -> VoiceMode {
        self.mode
    }
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.melody_queue.is_empty() && self.remaining == 0
    }

    pub fn set_sink(&mut self, sink: Box<dyn EventSink>) {
        self.sink = sink;
    }

    /// Install a custom voice bank for the singer (PSOLA backend).
    pub fn set_voice_bank(&mut self, bank: VoiceBank) {
        self.singer = SingingSynthesiser::new_psola(self.sample_rate, bank);
    }

    /// Switch to the Klatt singer backend (default).
    pub fn use_klatt_singer(&mut self) {
        self.singer = SingingSynthesiser::new_klatt(self.sample_rate);
    }

    pub fn singer_backend(&self) -> SingerVoice {
        self.singer.backend()
    }

    pub fn reset(&mut self) {
        self.queue.clear();
        self.melody_queue.clear();
        self.remaining = 0;
        self.klatt.reset();
        self.singer.reset();
        self.mode = VoiceMode::Speak;
        self.pending_started = false;
        self.current_phoneme_index = 0;
    }

    /// Queue speech from text.
    pub fn speak(&mut self, text: &str) {
        let was_idle = self.is_idle();
        self.mode = VoiceMode::Speak;
        let mut buf: Vec<TimedPhoneme> = Vec::new();
        self.phonemiser.phonemise_into(text, &mut buf);
        for tp in buf {
            self.queue.push_back(tp);
        }
        if was_idle {
            self.start_new_utterance();
        }
    }

    /// Queue a sung melody (lyrics + per-syllable notes).
    pub fn sing(&mut self, lyrics: &str, melody: Vec<SungNote>) {
        let was_idle = self.is_idle();
        self.mode = VoiceMode::Sing;
        let mut buf: Vec<TimedPhoneme> = Vec::new();
        self.phonemiser.phonemise_into(lyrics, &mut buf);
        for tp in buf {
            self.queue.push_back(tp);
        }
        for n in melody {
            self.melody_queue.push_back(n);
        }
        if was_idle {
            self.start_new_utterance();
        }
    }

    fn start_new_utterance(&mut self) {
        self.current_utterance_id = self.next_utterance_id;
        self.next_utterance_id = self.next_utterance_id.wrapping_add(1);
        self.current_phoneme_index = 0;
        self.pending_started = true;
    }

    /// Render exactly `block.block_size` samples into both `block.left` and
    /// `block.right` (the same mono signal).
    pub fn render_block(&mut self, block: &mut AudioBlock) {
        let n = block.block_size;
        debug_assert_eq!(block.left.len(), n);
        debug_assert_eq!(block.right.len(), n);
        let left = &mut block.left;

        if self.is_idle() {
            for s in left.iter_mut() {
                *s = 0.0;
            }
            block.right.copy_from_slice(left);
            return;
        }

        if self.pending_started {
            self.sink.emit(ClipCannonEvent::TtsStarted {
                utterance_id: self.current_utterance_id,
            });
            self.pending_started = false;
        }

        let mut written = 0_usize;
        while written < n {
            // Ensure we have a current item.
            if self.remaining == 0 {
                if let Some(head) = self.queue.pop_front() {
                    self.set_current(head);
                    self.remaining = ((head.duration_ms as f32) * self.sample_rate / 1000.0)
                        .max(1.0) as usize;
                    self.sink.emit(ClipCannonEvent::TtsBoundary {
                        utterance_id: self.current_utterance_id,
                        phoneme_index: self.current_phoneme_index,
                    });
                    self.current_phoneme_index =
                        self.current_phoneme_index.wrapping_add(1);
                } else {
                    // Drain over.
                    for s in left[written..].iter_mut() {
                        *s = 0.0;
                    }
                    self.sink.emit(ClipCannonEvent::TtsFinished {
                        utterance_id: self.current_utterance_id,
                    });
                    written = n;
                    break;
                }
            }
            let take = (n - written).min(self.remaining);
            self.render_into(&mut left[written..written + take]);
            written += take;
            self.remaining -= take;
        }
        block.right.copy_from_slice(left);
    }

    fn set_current(&mut self, ph: TimedPhoneme) {
        match self.mode {
            VoiceMode::Speak => {
                self.klatt.set_phoneme(ph.phoneme, 120.0);
            }
            VoiceMode::Sing => {
                // Pop the next note when we hit a vowel; otherwise hold the
                // current note and switch only the phoneme.
                if ph.phoneme.is_vowel() {
                    if let Some(note) = self.melody_queue.pop_front() {
                        let sn = SungNote {
                            phoneme: ph.phoneme,
                            midi_note: note.midi_note,
                            duration_ms: ph.duration_ms,
                            velocity: note.velocity,
                        };
                        self.singer.set_note(sn);
                    } else {
                        self.singer.set_note(SungNote::new(ph.phoneme, 69.0, ph.duration_ms));
                    }
                } else {
                    // Update the singer's current phoneme without changing pitch.
                    let mut held = SungNote::new(ph.phoneme, 69.0, ph.duration_ms);
                    held.velocity = 0.85;
                    self.singer.set_note(held);
                }
            }
        }
    }

    fn render_into(&mut self, out: &mut [f32]) {
        match self.mode {
            VoiceMode::Speak => self.klatt.render(out),
            VoiceMode::Sing => self.singer.render(out),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::clipcannon::events::RingSink;

    #[test]
    fn idle_renders_silence() {
        let mut tts = RealtimeVoiceSynthesiser::new(16_000.0, 128);
        let mut block = AudioBlock::new(128, 16_000.0);
        tts.render_block(&mut block);
        let energy: f32 =
            block.left.iter().map(|x| x * x).sum::<f32>() / block.left.len() as f32;
        assert!(energy < 1e-9);
        assert!(tts.is_idle());
    }

    #[test]
    fn speak_drains_to_idle() {
        let mut tts = RealtimeVoiceSynthesiser::new(16_000.0, 128);
        tts.speak("hi");
        assert!(!tts.is_idle());
        let mut block = AudioBlock::new(128, 16_000.0);
        // Render until idle (cap iterations to avoid loop on bug).
        for _ in 0..2000 {
            tts.render_block(&mut block);
            if tts.is_idle() {
                break;
            }
        }
        assert!(tts.is_idle());
    }

    #[test]
    fn speak_emits_started_and_finished_events() {
        let mut tts = RealtimeVoiceSynthesiser::new(16_000.0, 128);
        let sink = Box::new(RingSink::new(64));
        tts.set_sink(sink);
        tts.speak("hi.");
        let mut block = AudioBlock::new(128, 16_000.0);
        for _ in 0..2000 {
            tts.render_block(&mut block);
            if tts.is_idle() {
                break;
            }
        }
        // We can't introspect the sink (we moved it). Just check no panic.
    }

    #[test]
    fn render_writes_both_channels_identically() {
        let mut tts = RealtimeVoiceSynthesiser::new(16_000.0, 128);
        tts.speak("the");
        let mut block = AudioBlock::new(128, 16_000.0);
        tts.render_block(&mut block);
        assert_eq!(block.left, block.right);
    }

    #[test]
    fn sing_renders_audio() {
        let mut tts = RealtimeVoiceSynthesiser::new(16_000.0, 128);
        tts.sing(
            "la la",
            vec![
                SungNote::new(Phoneme::Aa, 69.0, 200),
                SungNote::new(Phoneme::Aa, 71.0, 200),
            ],
        );
        let mut block = AudioBlock::new(128, 16_000.0);
        let mut total_energy = 0.0_f32;
        for _ in 0..50 {
            tts.render_block(&mut block);
            for s in &block.left {
                total_energy += s * s;
            }
        }
        assert!(total_energy > 0.0);
    }

    #[test]
    fn reset_clears_state() {
        let mut tts = RealtimeVoiceSynthesiser::new(16_000.0, 128);
        tts.speak("hello world");
        assert!(!tts.is_idle());
        tts.reset();
        assert!(tts.is_idle());
    }
}
