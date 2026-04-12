//! Avatar Driving bounded context — viseme classification & lip coefficients.
//!
//! See [ADR-147](../../../docs/adr/ADR-147-clipcannon-prosody-emotion-speaker.md)
//! for the formant-band heuristic and [the domain
//! model](../../../docs/ddd/clipcannon-domain-model.md#33-visemecoeffs) for
//! invariants.

use super::prosody::ProsodySnapshot;

/// Discrete mouth-shape class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Viseme {
    /// Mouth closed / silence / inter-utterance.
    Rest,
    /// Open vowel as in "fAther".
    Aa,
    /// Front vowel as in "bEE".
    Ee,
    /// Mid front vowel as in "bIt".
    Ih,
    /// Rounded back vowel as in "bOAt".
    Oh,
    /// Closed back vowel as in "bOOt".
    Uw,
    /// Labiodental fricative /f/, /v/.
    Fv,
    /// Bilabial /m/, /b/, /p/ — closed lips.
    Mbp,
}

impl Viseme {
    /// Index suitable for blendshape array lookup.
    pub fn index(self) -> u8 {
        match self {
            Viseme::Rest => 0,
            Viseme::Aa => 1,
            Viseme::Ee => 2,
            Viseme::Ih => 3,
            Viseme::Oh => 4,
            Viseme::Uw => 5,
            Viseme::Fv => 6,
            Viseme::Mbp => 7,
        }
    }
}

/// Continuous + categorical mouth shape coefficients for one block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisemeCoeffs {
    pub viseme: Viseme,
    /// Mouth opening, `[0, 1]`.
    pub jaw_open: f32,
    /// Lip rounding (UW-like), `[0, 1]`.
    pub lip_round: f32,
    /// Lip spreading (EE-like), `[0, 1]`. By construction `lip_round + lip_spread ≈ 1`.
    pub lip_spread: f32,
    /// Mapper confidence, `[0, 1]`.
    pub confidence: f32,
}

impl VisemeCoeffs {
    pub const fn rest() -> Self {
        Self {
            viseme: Viseme::Rest,
            jaw_open: 0.0,
            lip_round: 0.5,
            lip_spread: 0.5,
            confidence: 1.0,
        }
    }
}

/// Stateful viseme mapper. Holds a 3-frame median filter for flicker
/// suppression. Allocation-free in steady state.
pub struct VisemeMapper {
    history: [Viseme; 3],
    history_len: usize,
}

impl Default for VisemeMapper {
    fn default() -> Self {
        Self::new()
    }
}

impl VisemeMapper {
    pub fn new() -> Self {
        Self {
            history: [Viseme::Rest; 3],
            history_len: 0,
        }
    }

    /// Reset internal history. Called on stream restart.
    pub fn reset(&mut self) {
        self.history = [Viseme::Rest; 3];
        self.history_len = 0;
    }

    /// Map prosody + magnitude spectrum to viseme coefficients.
    ///
    /// `mags` may be empty: in that case the mapper falls back to a
    /// voicing-only classification (REST or AA).
    pub fn map(&mut self, snap: &ProsodySnapshot, mags: &[f32], sample_rate: f32) -> VisemeCoeffs {
        let raw_viseme = if mags.is_empty() {
            classify_no_spectrum(snap)
        } else {
            classify_with_spectrum(snap, mags, sample_rate)
        };

        // 3-frame median filter on viseme.
        let filtered = self.median_filter(raw_viseme);

        // Continuous coefficients.
        let energy_norm = ((snap.energy_db + 60.0) / 60.0).clamp(0.0, 1.0);
        let f1_ratio = if mags.is_empty() {
            0.5
        } else {
            band_ratio(mags, sample_rate, 300.0, 1000.0)
        };
        let f2_ratio = if mags.is_empty() {
            0.5
        } else {
            band_ratio(mags, sample_rate, 1000.0, 2500.0)
        };

        let jaw_open = if filtered == Viseme::Rest || filtered == Viseme::Mbp {
            0.0
        } else {
            (energy_norm * (0.4 + f1_ratio).min(1.0)).clamp(0.0, 1.0)
        };
        let lip_round = smoothstep(0.15, 0.45, 1.0 - f2_ratio);
        let lip_spread = (1.0 - lip_round).clamp(0.0, 1.0);
        let confidence = (snap.voicing * (1.0 - snap.flatness)).clamp(0.0, 1.0);

        VisemeCoeffs {
            viseme: filtered,
            jaw_open,
            lip_round,
            lip_spread,
            confidence,
        }
    }

    fn median_filter(&mut self, current: Viseme) -> Viseme {
        // shift history left
        self.history[0] = self.history[1];
        self.history[1] = self.history[2];
        self.history[2] = current;
        self.history_len = (self.history_len + 1).min(3);

        if self.history_len < 3 {
            return current;
        }
        // 3-element majority vote (or pick middle if all distinct).
        let a = self.history[0];
        let b = self.history[1];
        let c = self.history[2];
        if a == b || a == c {
            a
        } else if b == c {
            b
        } else {
            b // middle
        }
    }
}

fn classify_no_spectrum(snap: &ProsodySnapshot) -> Viseme {
    if snap.energy_db < -40.0 || snap.voicing < 0.25 {
        Viseme::Rest
    } else {
        Viseme::Aa
    }
}

fn classify_with_spectrum(snap: &ProsodySnapshot, mags: &[f32], sr: f32) -> Viseme {
    if snap.energy_db < -40.0 || snap.voicing < 0.25 {
        return Viseme::Rest;
    }
    let f1 = band_ratio(mags, sr, 300.0, 1000.0);
    let f2 = band_ratio(mags, sr, 1000.0, 2500.0);
    let f3 = band_ratio(mags, sr, 2500.0, 5000.0);
    let voiced = snap.voicing > 0.5;

    // Bilabial /m,b,p/ — voiced "murmur" with low absolute energy
    // (lips closed reduces overall radiation by ~10–15 dB).
    if voiced && (-35.0..=-22.0).contains(&snap.energy_db) {
        return Viseme::Mbp;
    }
    // Labiodental /f,v/ — high f3, noisy (high flatness)
    if !voiced && f3 > 0.35 && snap.flatness > 0.35 {
        return Viseme::Fv;
    }
    if !voiced {
        return Viseme::Rest;
    }
    // Vowels
    if f1 > 0.55 {
        return Viseme::Aa;
    }
    if f1 > 0.40 && f2 < 0.30 {
        return Viseme::Oh;
    }
    if f2 < 0.25 && f1 < 0.35 {
        return Viseme::Uw;
    }
    if f2 > 0.50 {
        return Viseme::Ee;
    }
    if f2 > 0.35 && f1 < 0.30 {
        return Viseme::Ih;
    }
    Viseme::Aa
}

/// Power ratio inside `[lo_hz, hi_hz]` over total power.
fn band_ratio(mags: &[f32], sample_rate: f32, lo_hz: f32, hi_hz: f32) -> f32 {
    let b = mags.len();
    if b < 2 {
        return 0.0;
    }
    let nyq = sample_rate * 0.5;
    let bin_hz = nyq / (b as f32 - 1.0);
    let lo = (lo_hz / bin_hz).floor() as usize;
    let hi = ((hi_hz / bin_hz).ceil() as usize).min(b - 1);
    if hi <= lo {
        return 0.0;
    }
    let mut band = 0.0_f32;
    let mut total = 0.0_f32;
    for (i, &m) in mags.iter().enumerate() {
        let p = m * m;
        total += p;
        if i >= lo && i <= hi {
            band += p;
        }
    }
    if total > 1e-12 {
        (band / total).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::super::prosody::ProsodySnapshot;
    use super::*;

    fn voiced_snap(energy: f32, voicing: f32) -> ProsodySnapshot {
        ProsodySnapshot {
            f0_hz: 200.0,
            voicing,
            energy_db: energy,
            centroid_hz: 1500.0,
            rolloff_hz: 4000.0,
            zcr: 0.05,
            flatness: 0.1,
        }
    }

    fn fake_mags(emphasis_lo: f32, emphasis_hi: f32, n: usize) -> Vec<f32> {
        let nyq = 8000.0_f32;
        let bin_hz = nyq / (n as f32 - 1.0);
        (0..n)
            .map(|i| {
                let f = i as f32 * bin_hz;
                if f >= emphasis_lo && f <= emphasis_hi {
                    1.0
                } else {
                    0.05
                }
            })
            .collect()
    }

    #[test]
    fn silence_yields_rest() {
        let mut m = VisemeMapper::new();
        let snap = ProsodySnapshot::silent();
        let mags = vec![0.0_f32; 129];
        for _ in 0..3 {
            let v = m.map(&snap, &mags, 16_000.0);
            assert_eq!(v.viseme, Viseme::Rest);
            assert_eq!(v.jaw_open, 0.0);
        }
    }

    #[test]
    fn open_vowel_yields_aa() {
        let mut m = VisemeMapper::new();
        let snap = voiced_snap(-12.0, 0.9);
        let mags = fake_mags(300.0, 1000.0, 129);
        let mut last = VisemeCoeffs::rest();
        // 3 frames so the median filter saturates.
        for _ in 0..3 {
            last = m.map(&snap, &mags, 16_000.0);
        }
        assert_eq!(last.viseme, Viseme::Aa);
        assert!(last.jaw_open > 0.3);
    }

    #[test]
    fn front_vowel_yields_ee() {
        let mut m = VisemeMapper::new();
        let snap = voiced_snap(-12.0, 0.9);
        let mags = fake_mags(1000.0, 2500.0, 129);
        let mut last = VisemeCoeffs::rest();
        for _ in 0..3 {
            last = m.map(&snap, &mags, 16_000.0);
        }
        assert_eq!(last.viseme, Viseme::Ee);
        assert!(last.lip_spread > last.lip_round);
    }

    #[test]
    fn round_round_plus_spread_close_to_one() {
        let mut m = VisemeMapper::new();
        let snap = voiced_snap(-12.0, 0.9);
        let mags = fake_mags(300.0, 1000.0, 129);
        let v = m.map(&snap, &mags, 16_000.0);
        assert!((v.lip_round + v.lip_spread - 1.0).abs() < 1e-3);
    }

    #[test]
    fn coeffs_in_unit_interval() {
        let mut m = VisemeMapper::new();
        let snap = voiced_snap(-6.0, 0.95);
        let mags = fake_mags(2500.0, 5000.0, 129);
        let v = m.map(&snap, &mags, 16_000.0);
        assert!((0.0..=1.0).contains(&v.jaw_open));
        assert!((0.0..=1.0).contains(&v.lip_round));
        assert!((0.0..=1.0).contains(&v.lip_spread));
        assert!((0.0..=1.0).contains(&v.confidence));
    }

    #[test]
    fn median_filter_suppresses_single_frame_flicker() {
        let mut m = VisemeMapper::new();
        let snap_voiced = voiced_snap(-12.0, 0.9);
        let snap_silent = ProsodySnapshot::silent();
        let mags_aa = fake_mags(300.0, 1000.0, 129);
        let mags_silent = vec![0.0_f32; 129];

        // prime with two AA frames
        let _ = m.map(&snap_voiced, &mags_aa, 16_000.0);
        let _ = m.map(&snap_voiced, &mags_aa, 16_000.0);
        // single REST glitch
        let v = m.map(&snap_silent, &mags_silent, 16_000.0);
        // median of [AA, AA, REST] should be AA
        assert_eq!(v.viseme, Viseme::Aa);
    }
}
