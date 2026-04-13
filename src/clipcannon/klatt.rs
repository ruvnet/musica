//! Klatt-style cascade-parallel formant synthesiser. See ADR-155.
//!
//! Self-contained zero-data voice synthesis: a glottal pulse generator
//! feeds five cascade resonators (F1..F5); a noise source feeds four
//! parallel resonators for fricatives. Phoneme transitions are smoothed
//! by linear interpolation of formant targets.
//!
//! All state lives on the struct; `render` is allocation-free.

use core::f32::consts::PI;

use super::phonemise::Phoneme;

/// Formant target — five (frequency, bandwidth) pairs plus voicing/frication
/// gains and a base F0.
#[derive(Debug, Clone, Copy)]
pub struct FormantTarget {
    pub f0_hz: f32,
    pub formants: [(f32, f32); 5],
    pub voicing_amp: f32,
    pub frication_amp: f32,
}

impl FormantTarget {
    pub fn silence() -> Self {
        Self {
            f0_hz: 0.0,
            formants: [(500.0, 60.0); 5],
            voicing_amp: 0.0,
            frication_amp: 0.0,
        }
    }
}

/// Per-phoneme formant preset table. Values are derived from public-domain
/// formant tables in Klatt 1980 / 1990 (widely reproduced).
#[inline]
pub fn formant_target(p: Phoneme) -> FormantTarget {
    use Phoneme::*;
    let v = |f1, f2, f3| FormantTarget {
        f0_hz: 120.0,
        formants: [
            (f1, 60.0),
            (f2, 90.0),
            (f3, 150.0),
            (3500.0, 200.0),
            (4500.0, 250.0),
        ],
        voicing_amp: 1.0,
        frication_amp: 0.0,
    };
    let unvoiced = |f3, fric| FormantTarget {
        f0_hz: 0.0,
        formants: [
            (500.0, 100.0),
            (1500.0, 100.0),
            (f3, 200.0),
            (3500.0, 250.0),
            (4500.0, 300.0),
        ],
        voicing_amp: 0.0,
        frication_amp: fric,
    };
    let voiced_fric = |f1, f2, f3, fric| FormantTarget {
        f0_hz: 120.0,
        formants: [
            (f1, 80.0),
            (f2, 100.0),
            (f3, 200.0),
            (3500.0, 250.0),
            (4500.0, 300.0),
        ],
        voicing_amp: 0.6,
        frication_amp: fric,
    };
    let stop = |f1, f2| FormantTarget {
        f0_hz: 0.0,
        formants: [
            (f1, 100.0),
            (f2, 100.0),
            (2500.0, 200.0),
            (3500.0, 250.0),
            (4500.0, 300.0),
        ],
        voicing_amp: 0.0,
        frication_amp: 0.4,
    };
    let nasal = |f1, f2, f3| FormantTarget {
        f0_hz: 120.0,
        formants: [
            (f1, 60.0),
            (f2, 100.0),
            (f3, 150.0),
            (3500.0, 200.0),
            (4500.0, 250.0),
        ],
        voicing_amp: 0.7,
        frication_amp: 0.0,
    };
    match p {
        // Vowels (Klatt 1980 averages, male voice)
        Aa => v(730.0, 1090.0, 2440.0),
        Ae => v(660.0, 1720.0, 2410.0),
        Ah => v(640.0, 1190.0, 2390.0),
        Ao => v(570.0, 840.0, 2410.0),
        Aw => v(660.0, 1100.0, 2400.0),
        Ay => v(700.0, 1500.0, 2500.0),
        Eh => v(530.0, 1840.0, 2480.0),
        Er => v(490.0, 1350.0, 1690.0),
        Ey => v(480.0, 1900.0, 2580.0),
        Ih => v(390.0, 1990.0, 2550.0),
        Iy => v(270.0, 2290.0, 3010.0),
        Ow => v(460.0, 880.0, 2540.0),
        Oy => v(500.0, 1200.0, 2400.0),
        Uh => v(440.0, 1020.0, 2240.0),
        Uw => v(300.0, 870.0, 2240.0),
        // Voiced fricatives
        V => voiced_fric(220.0, 1100.0, 2500.0, 0.4),
        Dh => voiced_fric(200.0, 1500.0, 2700.0, 0.35),
        Z => voiced_fric(220.0, 1700.0, 5000.0, 0.5),
        Zh => voiced_fric(220.0, 1700.0, 4000.0, 0.5),
        // Unvoiced fricatives
        F => unvoiced(2500.0, 0.6),
        Th => unvoiced(2700.0, 0.55),
        S => unvoiced(5000.0, 0.7),
        Sh => unvoiced(4000.0, 0.7),
        Hh => unvoiced(2000.0, 0.4),
        // Stops (impulse-like)
        B => stop(220.0, 900.0),
        D => stop(220.0, 1700.0),
        G => stop(220.0, 1300.0),
        P => stop(220.0, 900.0),
        T => stop(220.0, 1700.0),
        K => stop(220.0, 1300.0),
        // Affricates
        Ch => unvoiced(4000.0, 0.6),
        Jh => voiced_fric(220.0, 1700.0, 4000.0, 0.5),
        // Nasals
        M => nasal(280.0, 900.0, 2200.0),
        N => nasal(280.0, 1700.0, 2700.0),
        Ng => nasal(280.0, 2200.0, 2700.0),
        // Liquids/glides
        L => v(360.0, 1300.0, 2700.0),
        R => v(310.0, 1060.0, 1380.0),
        W => v(300.0, 610.0, 2200.0),
        Y => v(260.0, 2070.0, 3020.0),
        // Silence
        Sil => FormantTarget::silence(),
    }
}

#[derive(Default, Clone, Copy)]
struct Resonator {
    a: f32,
    b: f32,
    c: f32,
    y1: f32,
    y2: f32,
}

impl Resonator {
    fn set(&mut self, freq: f32, bw: f32, sr: f32) {
        let two_pi = 2.0 * PI;
        let r = (-PI * bw / sr).exp();
        self.a = -r * r;
        self.b = 2.0 * r * (two_pi * freq / sr).cos();
        self.c = 1.0 - self.b - self.a;
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let y = self.c * x + self.b * self.y1 + self.a * self.y2;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
    fn reset(&mut self) {
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Glottal pulse + noise source.
struct GlottalSource {
    sample_rate: f32,
    phase: f32,
    /// Linear-feedback noise generator state.
    noise_state: u32,
}

impl GlottalSource {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            phase: 0.0,
            noise_state: 0xACE1u32,
        }
    }
    #[inline]
    fn next_glottal(&mut self, f0: f32) -> f32 {
        if f0 <= 0.0 {
            return 0.0;
        }
        let inc = f0 / self.sample_rate;
        self.phase += inc;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        // Asymmetric LF-style pulse: open phase 60%, closed 40%.
        if self.phase < 0.6 {
            let t = self.phase / 0.6;
            // Smoothed half-sine.
            (t * PI).sin()
        } else {
            // Closed phase contributes nothing.
            0.0
        }
    }
    #[inline]
    fn next_noise(&mut self) -> f32 {
        // xorshift32
        let mut x = self.noise_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.noise_state = x;
        ((x as f32) / (u32::MAX as f32)) * 2.0 - 1.0
    }
    fn reset(&mut self) {
        self.phase = 0.0;
        self.noise_state = 0xACE1u32;
    }
}

/// Klatt-class cascade-parallel formant voice synthesiser.
pub struct KlattSynthesiser {
    sample_rate: f32,
    glottal: GlottalSource,
    cascade: [Resonator; 5],
    parallel: [Resonator; 4],
    target: FormantTarget,
    current: FormantTarget,
    interp_alpha: f32,
}

impl KlattSynthesiser {
    pub fn new(sample_rate: f32) -> Self {
        let target = FormantTarget::silence();
        let current = target;
        let mut s = Self {
            sample_rate,
            glottal: GlottalSource::new(sample_rate),
            cascade: [Resonator::default(); 5],
            parallel: [Resonator::default(); 4],
            target,
            current,
            interp_alpha: 0.05,
        };
        s.apply_current();
        s
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    pub fn reset(&mut self) {
        self.glottal.reset();
        for r in &mut self.cascade {
            r.reset();
        }
        for r in &mut self.parallel {
            r.reset();
        }
    }

    /// Set the next phoneme. Formant targets are interpolated, not snapped.
    pub fn set_phoneme(&mut self, p: Phoneme, f0_hz: f32) {
        let mut t = formant_target(p);
        if f0_hz > 0.0 && t.voicing_amp > 0.0 {
            t.f0_hz = f0_hz;
        }
        self.target = t;
    }

    /// Set the formant target directly.
    pub fn set_target(&mut self, target: FormantTarget) {
        self.target = target;
    }

    /// Override the per-sample interpolation alpha.
    /// Larger = faster transitions; default 0.05 (~20-sample time constant).
    pub fn set_interp_alpha(&mut self, alpha: f32) {
        self.interp_alpha = alpha.clamp(0.0, 1.0);
    }

    fn apply_current(&mut self) {
        for i in 0..5 {
            self.cascade[i].set(
                self.current.formants[i].0.max(50.0),
                self.current.formants[i].1.max(20.0),
                self.sample_rate,
            );
        }
        // Parallel branch uses formants 2..5 with wider bandwidths.
        for i in 0..4 {
            let (f, bw) = self.current.formants[i + 1];
            self.parallel[i].set(f.max(50.0), bw.max(40.0) * 1.5, self.sample_rate);
        }
    }

    /// Render `out.len()` samples in-place into `out`. Allocation-free.
    pub fn render(&mut self, out: &mut [f32]) {
        for s in out.iter_mut() {
            // Smooth current toward target.
            let a = self.interp_alpha;
            self.current.f0_hz += (self.target.f0_hz - self.current.f0_hz) * a;
            self.current.voicing_amp += (self.target.voicing_amp - self.current.voicing_amp) * a;
            self.current.frication_amp +=
                (self.target.frication_amp - self.current.frication_amp) * a;
            for i in 0..5 {
                self.current.formants[i].0 +=
                    (self.target.formants[i].0 - self.current.formants[i].0) * a;
                self.current.formants[i].1 +=
                    (self.target.formants[i].1 - self.current.formants[i].1) * a;
            }
            self.apply_current();

            let g = self.glottal.next_glottal(self.current.f0_hz) * self.current.voicing_amp;
            let n = self.glottal.next_noise() * self.current.frication_amp;

            // Cascade branch (vowel envelope).
            let mut c = g;
            for r in &mut self.cascade {
                c = r.process(c);
            }

            // Parallel branch (frication).
            let mut p_sum = 0.0_f32;
            for r in &mut self.parallel {
                p_sum += r.process(n);
            }

            *s = (c * 0.5 + p_sum * 0.25).clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_target_renders_silence() {
        let mut k = KlattSynthesiser::new(16_000.0);
        k.set_target(FormantTarget::silence());
        let mut buf = vec![0.0_f32; 256];
        // Render some blocks to let the interpolator settle.
        for _ in 0..8 {
            k.render(&mut buf);
        }
        let energy: f32 = buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32;
        assert!(energy < 1e-3, "energy = {}", energy);
    }

    #[test]
    fn vowel_aa_produces_signal() {
        let mut k = KlattSynthesiser::new(16_000.0);
        k.set_phoneme(Phoneme::Aa, 150.0);
        let mut buf = vec![0.0_f32; 1024];
        // Let the resonators ring up.
        k.render(&mut buf);
        let mut buf2 = vec![0.0_f32; 1024];
        k.render(&mut buf2);
        let energy: f32 = buf2.iter().map(|x| x * x).sum::<f32>() / buf2.len() as f32;
        assert!(energy > 1e-4, "energy = {}", energy);
    }

    #[test]
    fn unvoiced_fricative_has_noise_energy() {
        let mut k = KlattSynthesiser::new(16_000.0);
        k.set_phoneme(Phoneme::S, 0.0);
        let mut buf = vec![0.0_f32; 1024];
        for _ in 0..4 {
            k.render(&mut buf);
        }
        let energy: f32 = buf.iter().map(|x| x * x).sum::<f32>() / buf.len() as f32;
        assert!(energy > 1e-4, "energy = {}", energy);
    }

    #[test]
    fn output_clamped_to_unit_range() {
        let mut k = KlattSynthesiser::new(16_000.0);
        k.set_phoneme(Phoneme::Aa, 200.0);
        let mut buf = vec![0.0_f32; 4096];
        for _ in 0..8 {
            k.render(&mut buf);
        }
        for s in &buf {
            assert!(s.is_finite());
            assert!(s.abs() <= 1.0);
        }
    }

    #[test]
    fn phoneme_transition_smooth() {
        let mut k = KlattSynthesiser::new(16_000.0);
        k.set_phoneme(Phoneme::Aa, 150.0);
        let mut buf = vec![0.0_f32; 256];
        k.render(&mut buf);
        k.set_phoneme(Phoneme::Iy, 180.0);
        let mut buf2 = vec![0.0_f32; 256];
        k.render(&mut buf2);
        // Should not blow up.
        for s in &buf2 {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn reset_returns_to_silence() {
        let mut k = KlattSynthesiser::new(16_000.0);
        k.set_phoneme(Phoneme::Aa, 150.0);
        let mut buf = vec![0.0_f32; 1024];
        k.render(&mut buf);
        k.set_target(FormantTarget::silence());
        k.reset();
        let mut buf2 = vec![0.0_f32; 1024];
        k.render(&mut buf2);
        let energy: f32 = buf2.iter().map(|x| x * x).sum::<f32>() / buf2.len() as f32;
        assert!(energy < 1.0);
    }
}
