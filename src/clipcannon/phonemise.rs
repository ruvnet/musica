//! Grapheme-to-phoneme conversion for English. See ADR-156.
//!
//! Rule-based, no neural net, no on-disk weights. Output is a `Vec` of
//! ARPAbet phonemes with per-phoneme duration estimates.

/// ARPAbet phoneme set (39 + silence).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phoneme {
    // Vowels (15)
    Aa, Ae, Ah, Ao, Aw, Ay, Eh, Er, Ey, Ih, Iy, Ow, Oy, Uh, Uw,
    // Consonants (24)
    B, Ch, D, Dh, F, G, Hh, Jh, K, L, M, N, Ng, P, R, S, Sh, T, Th,
    V, W, Y, Z, Zh,
    // Special
    Sil,
}

impl Phoneme {
    /// True if this is a vowel phoneme.
    pub fn is_vowel(self) -> bool {
        use Phoneme::*;
        matches!(
            self,
            Aa | Ae | Ah | Ao | Aw | Ay | Eh | Er | Ey | Ih | Iy | Ow | Oy | Uh | Uw
        )
    }

    /// True if this is a voiced phoneme.
    pub fn is_voiced(self) -> bool {
        use Phoneme::*;
        if self.is_vowel() {
            return true;
        }
        matches!(
            self,
            B | D | Dh | G | Jh | L | M | N | Ng | R | V | W | Y | Z | Zh
        )
    }

    /// Default duration in milliseconds for this phoneme.
    pub fn default_duration_ms(self) -> u32 {
        if self == Phoneme::Sil {
            30
        } else if self.is_vowel() {
            90
        } else {
            55
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TimedPhoneme {
    pub phoneme: Phoneme,
    pub duration_ms: u32,
    pub stress: u8,
}

impl TimedPhoneme {
    pub fn new(phoneme: Phoneme) -> Self {
        Self {
            phoneme,
            duration_ms: phoneme.default_duration_ms(),
            stress: 0,
        }
    }
}

/// Tiny built-in exception lexicon for the most common irregular English words.
const EXCEPTIONS: &[(&str, &[Phoneme])] = {
    use Phoneme::*;
    &[
        ("the", &[Dh, Ah]),
        ("a", &[Ah]),
        ("to", &[T, Uw]),
        ("of", &[Ah, V]),
        ("and", &[Ae, N, D]),
        ("in", &[Ih, N]),
        ("is", &[Ih, Z]),
        ("it", &[Ih, T]),
        ("you", &[Y, Uw]),
        ("that", &[Dh, Ae, T]),
        ("he", &[Hh, Iy]),
        ("was", &[W, Ah, Z]),
        ("for", &[F, Ao, R]),
        ("on", &[Aa, N]),
        ("are", &[Aa, R]),
        ("as", &[Ae, Z]),
        ("with", &[W, Ih, Th]),
        ("his", &[Hh, Ih, Z]),
        ("they", &[Dh, Ey]),
        ("be", &[B, Iy]),
        ("at", &[Ae, T]),
        ("one", &[W, Ah, N]),
        ("have", &[Hh, Ae, V]),
        ("this", &[Dh, Ih, S]),
        ("from", &[F, R, Ah, M]),
        ("or", &[Ao, R]),
        ("had", &[Hh, Ae, D]),
        ("by", &[B, Ay]),
        ("but", &[B, Ah, T]),
        ("what", &[W, Ah, T]),
        ("all", &[Ao, L]),
        ("were", &[W, Er]),
        ("we", &[W, Iy]),
        ("when", &[W, Eh, N]),
        ("your", &[Y, Ao, R]),
        ("can", &[K, Ae, N]),
        ("said", &[S, Eh, D]),
        ("there", &[Dh, Eh, R]),
        ("use", &[Y, Uw, Z]),
        ("an", &[Ae, N]),
        ("each", &[Iy, Ch]),
        ("which", &[W, Ih, Ch]),
        ("she", &[Sh, Iy]),
        ("do", &[D, Uw]),
        ("how", &[Hh, Aw]),
        ("their", &[Dh, Eh, R]),
        ("if", &[Ih, F]),
        ("will", &[W, Ih, L]),
        ("up", &[Ah, P]),
        ("other", &[Ah, Dh, Er]),
        ("about", &[Ah, B, Aw, T]),
        ("out", &[Aw, T]),
        ("many", &[M, Eh, N, Iy]),
        ("then", &[Dh, Eh, N]),
        ("them", &[Dh, Eh, M]),
        ("these", &[Dh, Iy, Z]),
        ("so", &[S, Ow]),
        ("some", &[S, Ah, M]),
        ("her", &[Hh, Er]),
        ("would", &[W, Uh, D]),
        ("make", &[M, Ey, K]),
        ("like", &[L, Ay, K]),
        ("him", &[Hh, Ih, M]),
        ("into", &[Ih, N, T, Uw]),
        ("time", &[T, Ay, M]),
        ("has", &[Hh, Ae, Z]),
        ("look", &[L, Uh, K]),
        ("two", &[T, Uw]),
        ("more", &[M, Ao, R]),
        ("write", &[R, Ay, T]),
        ("go", &[G, Ow]),
        ("see", &[S, Iy]),
        ("number", &[N, Ah, M, B, Er]),
        ("no", &[N, Ow]),
        ("way", &[W, Ey]),
        ("could", &[K, Uh, D]),
        ("people", &[P, Iy, P, Ah, L]),
        ("my", &[M, Ay]),
        ("than", &[Dh, Ae, N]),
        ("first", &[F, Er, S, T]),
        ("water", &[W, Ao, T, Er]),
        ("been", &[B, Ih, N]),
        ("call", &[K, Ao, L]),
        ("who", &[Hh, Uw]),
        ("its", &[Ih, T, S]),
        ("now", &[N, Aw]),
        ("find", &[F, Ay, N, D]),
        ("long", &[L, Ao, Ng]),
        ("down", &[D, Aw, N]),
        ("day", &[D, Ey]),
        ("did", &[D, Ih, D]),
        ("get", &[G, Eh, T]),
        ("come", &[K, Ah, M]),
        ("made", &[M, Ey, D]),
        ("may", &[M, Ey]),
        ("part", &[P, Aa, R, T]),
        ("hello", &[Hh, Eh, L, Ow]),
        ("world", &[W, Er, L, D]),
        ("musica", &[M, Y, Uw, Z, Ih, K, Ah]),
    ]
};

pub struct Phonemiser;

impl Default for Phonemiser {
    fn default() -> Self {
        Self::english()
    }
}

impl Phonemiser {
    pub fn english() -> Self {
        Self
    }

    /// Phonemise a UTF-8 string into a `Vec<TimedPhoneme>`.
    pub fn phonemise(&self, text: &str) -> Vec<TimedPhoneme> {
        let mut out = Vec::with_capacity(text.len());
        self.phonemise_into(text, &mut out);
        out
    }

    /// Append phonemes for `text` into `out`. Allocation only if `out` grows.
    pub fn phonemise_into(&self, text: &str, out: &mut Vec<TimedPhoneme>) {
        let mut current = String::with_capacity(16);
        for ch in text.chars() {
            if ch.is_ascii_alphabetic() {
                current.push(ch.to_ascii_lowercase());
            } else {
                if !current.is_empty() {
                    self.phonemise_word(&current, out);
                    current.clear();
                }
                match ch {
                    ' ' | '\t' | '\n' => {
                        out.push(TimedPhoneme {
                            phoneme: Phoneme::Sil,
                            duration_ms: 30,
                            stress: 0,
                        });
                    }
                    ',' | ';' | ':' => {
                        out.push(TimedPhoneme {
                            phoneme: Phoneme::Sil,
                            duration_ms: 80,
                            stress: 0,
                        });
                    }
                    '.' | '!' | '?' => {
                        out.push(TimedPhoneme {
                            phoneme: Phoneme::Sil,
                            duration_ms: 200,
                            stress: 0,
                        });
                    }
                    _ => {}
                }
            }
        }
        if !current.is_empty() {
            self.phonemise_word(&current, out);
        }
    }

    fn phonemise_word(&self, word: &str, out: &mut Vec<TimedPhoneme>) {
        // Lookup exception lexicon (linear scan, ≤120 entries — fits in cache).
        for (w, ph) in EXCEPTIONS {
            if *w == word {
                for &p in *ph {
                    out.push(TimedPhoneme::new(p));
                }
                return;
            }
        }
        // Fall back to letter-to-sound rules.
        let bytes = word.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let (consumed, phonemes) = letter_rules(bytes, i);
            for &p in phonemes {
                out.push(TimedPhoneme::new(p));
            }
            i += consumed;
        }
    }
}

/// Greedy letter-to-sound rule engine. Returns (chars_consumed, phonemes).
fn letter_rules(bytes: &[u8], pos: usize) -> (usize, &'static [Phoneme]) {
    use Phoneme::*;
    let rest = &bytes[pos..];
    let prev = if pos > 0 { bytes[pos - 1] } else { 0 };
    let prev_is_vowel = matches!(prev, b'a' | b'e' | b'i' | b'o' | b'u');

    // Multi-letter digraphs first (longest match wins).
    if rest.starts_with(b"tion") {
        return (4, &[Sh, Ah, N]);
    }
    if rest.starts_with(b"sion") {
        return (4, &[Zh, Ah, N]);
    }
    if rest.starts_with(b"ough") {
        return (4, &[Ah, F]);
    }
    if rest.starts_with(b"augh") {
        return (4, &[Ao]);
    }
    if rest.starts_with(b"eigh") {
        return (4, &[Ey]);
    }
    if rest.starts_with(b"igh") {
        return (3, &[Ay]);
    }
    if rest.starts_with(b"ing") {
        return (3, &[Ih, Ng]);
    }
    if rest.starts_with(b"sch") {
        return (3, &[S, K]);
    }
    if rest.starts_with(b"ch") {
        return (2, &[Ch]);
    }
    if rest.starts_with(b"sh") {
        return (2, &[Sh]);
    }
    if rest.starts_with(b"th") {
        return (2, &[Th]);
    }
    if rest.starts_with(b"ph") {
        return (2, &[F]);
    }
    if rest.starts_with(b"ck") {
        return (2, &[K]);
    }
    if rest.starts_with(b"qu") {
        return (2, &[K, W]);
    }
    if rest.starts_with(b"wh") {
        return (2, &[W]);
    }
    if rest.starts_with(b"ng") {
        return (2, &[Ng]);
    }
    if rest.starts_with(b"oo") {
        return (2, &[Uw]);
    }
    if rest.starts_with(b"ee") {
        return (2, &[Iy]);
    }
    if rest.starts_with(b"ea") {
        return (2, &[Iy]);
    }
    if rest.starts_with(b"ai") {
        return (2, &[Ey]);
    }
    if rest.starts_with(b"ay") {
        return (2, &[Ey]);
    }
    if rest.starts_with(b"ou") {
        return (2, &[Aw]);
    }
    if rest.starts_with(b"ow") {
        return (2, &[Ow]);
    }
    if rest.starts_with(b"oa") {
        return (2, &[Ow]);
    }
    if rest.starts_with(b"oi") || rest.starts_with(b"oy") {
        return (2, &[Oy]);
    }
    if rest.starts_with(b"er") {
        return (2, &[Er]);
    }
    if rest.starts_with(b"ar") {
        return (2, &[Aa, R]);
    }
    if rest.starts_with(b"or") {
        return (2, &[Ao, R]);
    }
    if rest.starts_with(b"ir") || rest.starts_with(b"ur") {
        return (2, &[Er]);
    }

    // Single-letter rules.
    let c = rest[0];
    match c {
        b'a' => {
            // 'a' followed by consonant + 'e' (silent e) → EY
            if rest.len() >= 3 && !is_vowel(rest[1]) && rest[2] == b'e' {
                (1, &[Ey])
            } else {
                (1, &[Ae])
            }
        }
        b'e' => {
            if rest.len() == 1 && pos > 0 && !prev_is_vowel {
                (1, &[]) // silent terminal e
            } else {
                (1, &[Eh])
            }
        }
        b'i' => {
            if rest.len() >= 3 && !is_vowel(rest[1]) && rest[2] == b'e' {
                (1, &[Ay])
            } else {
                (1, &[Ih])
            }
        }
        b'o' => {
            if rest.len() >= 3 && !is_vowel(rest[1]) && rest[2] == b'e' {
                (1, &[Ow])
            } else {
                (1, &[Aa])
            }
        }
        b'u' => {
            if rest.len() >= 3 && !is_vowel(rest[1]) && rest[2] == b'e' {
                (1, &[Y, Uw])
            } else {
                (1, &[Ah])
            }
        }
        b'y' => {
            if pos == 0 {
                (1, &[Y])
            } else {
                (1, &[Ay])
            }
        }
        b'b' => (1, &[B]),
        b'c' => {
            // c before e/i/y → S
            if rest.len() > 1 && matches!(rest[1], b'e' | b'i' | b'y') {
                (1, &[S])
            } else {
                (1, &[K])
            }
        }
        b'd' => (1, &[D]),
        b'f' => (1, &[F]),
        b'g' => {
            if rest.len() > 1 && matches!(rest[1], b'e' | b'i' | b'y') {
                (1, &[Jh])
            } else {
                (1, &[G])
            }
        }
        b'h' => (1, &[Hh]),
        b'j' => (1, &[Jh]),
        b'k' => (1, &[K]),
        b'l' => (1, &[L]),
        b'm' => (1, &[M]),
        b'n' => (1, &[N]),
        b'p' => (1, &[P]),
        b'q' => (1, &[K]),
        b'r' => (1, &[R]),
        b's' => {
            if pos > 0 && prev_is_vowel && rest.len() == 1 {
                (1, &[Z])
            } else {
                (1, &[S])
            }
        }
        b't' => (1, &[T]),
        b'v' => (1, &[V]),
        b'w' => (1, &[W]),
        b'x' => (1, &[K, S]),
        b'z' => (1, &[Z]),
        _ => (1, &[]),
    }
}

#[inline]
fn is_vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_yields_empty() {
        let p = Phonemiser::english();
        assert!(p.phonemise("").is_empty());
    }

    #[test]
    fn exception_word_uses_lexicon() {
        let p = Phonemiser::english();
        let r = p.phonemise("hello");
        let phs: Vec<Phoneme> = r.iter().map(|t| t.phoneme).collect();
        assert_eq!(phs, vec![Phoneme::Hh, Phoneme::Eh, Phoneme::L, Phoneme::Ow]);
    }

    #[test]
    fn punctuation_inserts_silence() {
        let p = Phonemiser::english();
        let r = p.phonemise("hi.");
        // Last phoneme should be silence.
        assert!(matches!(r.last().unwrap().phoneme, Phoneme::Sil));
        assert_eq!(r.last().unwrap().duration_ms, 200);
    }

    #[test]
    fn whitespace_separates_words() {
        let p = Phonemiser::english();
        let r = p.phonemise("the world");
        // Should contain "the" + Sil + "world".
        let phs: Vec<Phoneme> = r.iter().map(|t| t.phoneme).collect();
        assert!(phs.contains(&Phoneme::Dh));
        assert!(phs.contains(&Phoneme::W));
        assert!(phs.contains(&Phoneme::Sil));
    }

    #[test]
    fn rule_engine_handles_unknown_word() {
        let p = Phonemiser::english();
        let r = p.phonemise("blip");
        let phs: Vec<Phoneme> = r.iter().map(|t| t.phoneme).collect();
        // Should at least contain B, L, P
        assert!(phs.contains(&Phoneme::B));
        assert!(phs.contains(&Phoneme::L));
        assert!(phs.contains(&Phoneme::P));
    }

    #[test]
    fn duration_defaults_set() {
        let p = Phonemiser::english();
        let r = p.phonemise("the");
        for tp in &r {
            assert!(tp.duration_ms > 0);
        }
    }

    #[test]
    fn vowel_classifier() {
        assert!(Phoneme::Aa.is_vowel());
        assert!(Phoneme::Iy.is_vowel());
        assert!(!Phoneme::B.is_vowel());
        assert!(!Phoneme::Sil.is_vowel());
    }

    #[test]
    fn voicing_classifier() {
        assert!(Phoneme::B.is_voiced());
        assert!(!Phoneme::P.is_voiced());
        assert!(Phoneme::Aa.is_voiced());
        assert!(!Phoneme::S.is_voiced());
    }
}
