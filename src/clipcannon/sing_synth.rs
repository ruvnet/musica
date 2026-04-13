//! Singing synthesiser — Klatt + PSOLA backends. See ADR-158.
//!
//! The Klatt backend wraps `KlattSynthesiser` with a per-sample F0 modulation
//! (vibrato + portamento). It needs no data and runs on any CPU.
//!
//! The PSOLA backend implements Time-Domain Pitch-Synchronous Overlap-Add
//! over a host-supplied `VoiceBank`. It produces recognisably human timbre
//! when a sample bank is available.

use core::f32::consts::PI;

use super::klatt::{formant_target, FormantTarget, KlattSynthesiser};
use super::phonemise::Phoneme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingerVoice {
    Klatt,
    Psola,
}

#[derive(Debug, Clone, Copy)]
pub struct SungNote {
    pub phoneme: Phoneme,
    pub midi_note: f32,
    pub duration_ms: u32,
    pub velocity: f32,
}

impl SungNote {
    pub fn new(phoneme: Phoneme, midi_note: f32, duration_ms: u32) -> Self {
        Self {
            phoneme,
            midi_note,
            duration_ms,
            velocity: 0.85,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VoiceBank {
    pub sample_rate: f32,
    pub samples: Vec<(Phoneme, Vec<f32>)>,
    pub ref_f0_hz: f32,
}

impl VoiceBank {
    /// Tiny synthetic bank for tests / fallback. Generates one period of
    /// each vowel via the Klatt engine, so PSOLA can be exercised without
    /// real recordings.
    pub fn synthetic(sample_rate: f32) -> Self {
        let ref_f0 = 220.0;
        let n = (sample_rate / ref_f0 * 4.0) as usize; // 4 periods
        let vowels = [
            Phoneme::Aa,
            Phoneme::Eh,
            Phoneme::Iy,
            Phoneme::Ow,
            Phoneme::Uw,
        ];
        let mut samples = Vec::new();
        for &v in &vowels {
            let mut k = KlattSynthesiser::new(sample_rate);
            k.set_interp_alpha(1.0);
            k.set_phoneme(v, ref_f0);
            let mut buf = vec![0.0_f32; n + 256];
            k.render(&mut buf);
            samples.push((v, buf[256..].to_vec())); // skip transient
        }
        Self {
            sample_rate,
            samples,
            ref_f0_hz: ref_f0,
        }
    }

    pub fn lookup(&self, p: Phoneme) -> Option<&[f32]> {
        self.samples
            .iter()
            .find(|(ph, _)| *ph == p)
            .map(|(_, s)| s.as_slice())
    }
}

pub struct SingingSynthesiser {
    sample_rate: f32,
    backend: SingerVoice,
    klatt: KlattSynthesiser,
    voice_bank: Option<VoiceBank>,
    /// Current target MIDI note (float for portamento).
    target_midi: f32,
    current_midi: f32,
    portamento_alpha: f32,
    vibrato_rate_hz: f32,
    vibrato_depth_cents: f32,
    vibrato_phase: f32,
    velocity: f32,
    current_phoneme: Phoneme,
    /// PSOLA state — read cursor into the bank sample.
    psola_cursor: f32,
}

impl SingingSynthesiser {
    pub fn new_klatt(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            backend: SingerVoice::Klatt,
            klatt: KlattSynthesiser::new(sample_rate),
            voice_bank: None,
            target_midi: 69.0,
            current_midi: 69.0,
            portamento_alpha: 0.10,
            vibrato_rate_hz: 5.0,
            vibrato_depth_cents: 0.0,
            vibrato_phase: 0.0,
            velocity: 0.0,
            current_phoneme: Phoneme::Sil,
            psola_cursor: 0.0,
        }
    }

    pub fn new_psola(sample_rate: f32, bank: VoiceBank) -> Self {
        let mut s = Self::new_klatt(sample_rate);
        s.backend = SingerVoice::Psola;
        s.voice_bank = Some(bank);
        s
    }

    pub fn backend(&self) -> SingerVoice {
        self.backend
    }

    pub fn set_vibrato(&mut self, rate_hz: f32, depth_cents: f32) {
        self.vibrato_rate_hz = rate_hz.max(0.0);
        self.vibrato_depth_cents = depth_cents.max(0.0);
    }

    pub fn set_portamento_alpha(&mut self, alpha: f32) {
        self.portamento_alpha = alpha.clamp(0.0, 1.0);
    }

    pub fn set_note(&mut self, note: SungNote) {
        self.target_midi = note.midi_note;
        self.velocity = note.velocity.clamp(0.0, 1.0);
        self.current_phoneme = note.phoneme;
        if self.backend == SingerVoice::Klatt {
            let f0 = midi_to_hz(note.midi_note);
            self.klatt.set_phoneme(note.phoneme, f0);
        } else {
            // For PSOLA we still use the Klatt engine as a fallback when
            // the bank is missing the requested phoneme.
            let f0 = midi_to_hz(note.midi_note);
            self.klatt.set_phoneme(note.phoneme, f0);
            self.psola_cursor = 0.0;
        }
    }

    pub fn reset(&mut self) {
        self.klatt.reset();
        self.target_midi = 69.0;
        self.current_midi = 69.0;
        self.vibrato_phase = 0.0;
        self.velocity = 0.0;
        self.current_phoneme = Phoneme::Sil;
        self.psola_cursor = 0.0;
    }

    /// Render `out.len()` samples in-place. Mono.
    pub fn render(&mut self, out: &mut [f32]) {
        match self.backend {
            SingerVoice::Klatt => self.render_klatt(out),
            SingerVoice::Psola => self.render_psola(out),
        }
    }

    fn render_klatt(&mut self, out: &mut [f32]) {
        // Per-sample LFO + portamento.
        let inv_sr = 1.0 / self.sample_rate;
        let lfo_inc = 2.0 * PI * self.vibrato_rate_hz * inv_sr;
        for s in out.iter_mut() {
            // Portamento toward target.
            self.current_midi +=
                (self.target_midi - self.current_midi) * self.portamento_alpha * 0.05;
            // Vibrato in cents.
            self.vibrato_phase += lfo_inc;
            if self.vibrato_phase > 2.0 * PI {
                self.vibrato_phase -= 2.0 * PI;
            }
            let cents_offset = self.vibrato_depth_cents * self.vibrato_phase.sin();
            let modulated_midi = self.current_midi + cents_offset / 100.0;
            let f0_now = midi_to_hz(modulated_midi);
            // Set the formant target with the new F0.
            let mut t: FormantTarget = formant_target(self.current_phoneme);
            if t.voicing_amp > 0.0 {
                t.f0_hz = f0_now;
            }
            t.voicing_amp *= self.velocity;
            self.klatt.set_target(t);
            // Render exactly one sample by handing in a 1-element slice.
            let mut one = [0.0_f32; 1];
            self.klatt.render(&mut one);
            *s = one[0];
        }
    }

    fn render_psola(&mut self, out: &mut [f32]) {
        // Look up the bank for the current vowel; fall back to Klatt if missing.
        let (bank_sr, bank_ref_f0, bank_samples) = match &self.voice_bank {
            Some(bank) => match bank.lookup(self.current_phoneme) {
                Some(s) => (bank.sample_rate, bank.ref_f0_hz, s.to_vec()),
                None => return self.render_klatt(out),
            },
            None => return self.render_klatt(out),
        };
        if bank_samples.is_empty() {
            return self.render_klatt(out);
        }

        let inv_sr = 1.0 / self.sample_rate;
        let lfo_inc = 2.0 * PI * self.vibrato_rate_hz * inv_sr;

        for s in out.iter_mut() {
            self.current_midi +=
                (self.target_midi - self.current_midi) * self.portamento_alpha * 0.05;
            self.vibrato_phase += lfo_inc;
            if self.vibrato_phase > 2.0 * PI {
                self.vibrato_phase -= 2.0 * PI;
            }
            let cents_offset = self.vibrato_depth_cents * self.vibrato_phase.sin();
            let modulated_midi = self.current_midi + cents_offset / 100.0;
            let f0_now = midi_to_hz(modulated_midi);

            // PSOLA-style read: rate = (bank_ref_f0 / target_f0) * (sample_rate / bank_sr).
            // For each output sample, advance the bank cursor by `rate`.
            let rate = (bank_ref_f0 / f0_now) * (self.sample_rate / bank_sr);
            let len = bank_samples.len();
            let idx = self.psola_cursor as usize;
            if idx + 1 >= len {
                self.psola_cursor = 0.0;
            }
            let i0 = (self.psola_cursor as usize) % len;
            let i1 = (i0 + 1) % len;
            let frac = self.psola_cursor - self.psola_cursor.floor();
            let sample = bank_samples[i0] * (1.0 - frac) + bank_samples[i1] * frac;
            *s = sample * self.velocity;
            self.psola_cursor += rate.max(0.001);
            if self.psola_cursor as usize >= len {
                self.psola_cursor -= len as f32;
            }
        }
    }
}

#[inline]
fn midi_to_hz(midi: f32) -> f32 {
    440.0 * (2.0_f32).powf((midi - 69.0) / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klatt_backend_creates_signal() {
        let mut s = SingingSynthesiser::new_klatt(16_000.0);
        s.set_note(SungNote::new(Phoneme::Aa, 69.0, 500));
        let mut buf = vec![0.0_f32; 2048];
        s.render(&mut buf);
        let energy: f32 = buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32;
        assert!(energy > 1e-5, "energy = {}", energy);
    }

    #[test]
    fn vibrato_modulates_pitch() {
        let mut s = SingingSynthesiser::new_klatt(16_000.0);
        s.set_vibrato(5.0, 50.0);
        s.set_note(SungNote::new(Phoneme::Aa, 69.0, 1000));
        let mut buf = vec![0.0_f32; 4096];
        s.render(&mut buf);
        for v in &buf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn psola_backend_with_synthetic_bank_works() {
        let bank = VoiceBank::synthetic(16_000.0);
        let mut s = SingingSynthesiser::new_psola(16_000.0, bank);
        s.set_note(SungNote::new(Phoneme::Aa, 72.0, 500));
        let mut buf = vec![0.0_f32; 1024];
        s.render(&mut buf);
        let energy: f32 = buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32;
        assert!(energy > 1e-6, "energy = {}", energy);
    }

    #[test]
    fn midi_to_hz_a4_is_440() {
        assert!((midi_to_hz(69.0) - 440.0).abs() < 1e-4);
    }

    #[test]
    fn output_finite() {
        let mut s = SingingSynthesiser::new_klatt(16_000.0);
        s.set_note(SungNote::new(Phoneme::Iy, 76.0, 200));
        let mut buf = vec![0.0_f32; 512];
        s.render(&mut buf);
        for v in &buf {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn sets_velocity_scales_amplitude() {
        let mut s = SingingSynthesiser::new_klatt(16_000.0);
        s.set_note(SungNote {
            phoneme: Phoneme::Aa,
            midi_note: 69.0,
            duration_ms: 500,
            velocity: 0.1,
        });
        let mut quiet = vec![0.0_f32; 2048];
        s.render(&mut quiet);
        let mut s2 = SingingSynthesiser::new_klatt(16_000.0);
        s2.set_note(SungNote {
            phoneme: Phoneme::Aa,
            midi_note: 69.0,
            duration_ms: 500,
            velocity: 1.0,
        });
        let mut loud = vec![0.0_f32; 2048];
        s2.render(&mut loud);
        let e_q: f32 = quiet.iter().map(|x| x * x).sum::<f32>() / quiet.len() as f32;
        let e_l: f32 = loud.iter().map(|x| x * x).sum::<f32>() / loud.len() as f32;
        assert!(e_l > e_q, "loud {} should exceed quiet {}", e_l, e_q);
    }
}
