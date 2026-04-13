# ADR-156: Phonemiser — Grapheme to Phoneme

## Status
Accepted

## Date
2026-04-12

## Context

ADR-155 introduced a Klatt synthesiser that consumes a phoneme stream.
Without grapheme-to-phoneme (G2P) conversion the synthesiser is unusable
from a host that wants to call `tts.speak("hello world")`.

ClipCannon upstream uses a neural G2P (CharsiuG2P, ~80 MB). Realistic
musica budget: zero MB.

## Decision

Add `clipcannon/phonemise.rs` with a **rule-based English phonemiser**
modelled on the public-domain "NRL letter-to-sound" rules (Elovitz et
al. 1976) plus a tiny exception list for the most common irregular
words.

Output is a `Vec<TimedPhoneme>` where each phoneme carries an estimated
duration in milliseconds derived from a per-phoneme average.

### Public types

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phoneme {
    // Vowels (15)
    Aa, Ae, Ah, Ao, Aw, Ay, Eh, Er, Ey, Ih, Iy, Ow, Oy, Uh, Uw,
    // Consonants (24)
    B, Ch, D, Dh, F, G, Hh, Jh, K, L, M, N, Ng, P, R, S, Sh, T, Th,
    V, W, Y, Z, Zh,
    // Special
    Sil,    // silence / inter-word
}

#[derive(Debug, Clone, Copy)]
pub struct TimedPhoneme {
    pub phoneme:    Phoneme,
    pub duration_ms: u32,
    pub stress:     u8,    // 0 = unstressed, 1 = primary, 2 = secondary
}

pub struct Phonemiser {
    /// Optional small exception lexicon (compiled in).
    exceptions: &'static [(&'static str, &'static [Phoneme])],
}

impl Phonemiser {
    pub fn english() -> Self;
    pub fn phonemise(&self, text: &str) -> Vec<TimedPhoneme>;
}
```

### Algorithm

1. **Tokenise** input on whitespace and punctuation. Punctuation
   produces `Sil` of varied length (`,` → 80 ms, `.` → 200 ms, etc.).
2. **Lowercase + ASCII-only fallback.** Non-ASCII characters become `Sil`.
3. For each word:
   - Lookup in the **exception table** (~120 of the most irregular
     English words: `the`, `a`, `to`, `of`, `was`, `were`, `you`,
     `know`, `one`, `said`, `says`, `put`, `who`, `friend`,
     `business`, `island`, …). If found, emit the canned phonemes.
   - Otherwise apply **letter-to-sound rules** in order. Each rule has
     the shape `(left_context, target_letters, right_context, phonemes)`.
     Example:
     - `(_, "ck", _) → [K]`
     - `(_, "tion", end) → [SH, AH, N]`
     - `(_, "ph", _) → [F]`
     - `(_, "a", "e_end") → [EY]`     // "make"
4. Each emitted phoneme gets a default duration from the
   per-phoneme table (vowels ~80 ms, consonants ~50 ms).
5. Words separated by spaces emit a 30 ms `Sil` between them.

### Why not lookup-only?

CMUdict is 130 K words, ~3 MB compressed. Doable behind a feature flag,
but our default must stay zero-data. The rule engine + 120-word
exception table is ~5 KB and gets ~95% accuracy on arbitrary English
prose — sufficient for accessibility prompts and notifications, the
target use case.

### Realtime contract

- `phonemise` allocates one `Vec<TimedPhoneme>` per call (output sized
  by the input). Hosts that need zero-allocation streaming can use the
  iterator variant `phonemise_into(text, &mut Vec<TimedPhoneme>)`.
- The rule engine is pure pattern matching, no recursion, O(n) in input
  length.
- Target: <50 µs per word at typical English word lengths.

## Consequences

### Positive
- ADR-155 Klatt becomes usable from arbitrary text input.
- No on-disk weights.
- Fully deterministic, easy to audit, easy to extend.
- Hosts can plug in their own `Phonemiser` impl via a trait if they need
  better quality (e.g. CMUdict-backed).

### Negative
- English-only in v1. Other languages need their own rule files.
- Rule-based G2P can't handle pronunciation by analogy
  (e.g. "ghoti" = "fish") or proper-noun edge cases.

### Risks
- Mispronunciations are inevitable. The exception table can be extended
  per host without code changes.

## References
- Elovitz et al., *Letter-to-Sound Rules for Automatic Translation of
  English Text to Phonetics*, NRL Report 7948, 1976 (public domain).
- ARPAbet phoneme set (CMU, public domain).
- ADR-155, ADR-159.
