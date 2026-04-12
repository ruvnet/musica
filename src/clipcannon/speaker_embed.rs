//! Speaker Identity bounded context — compact spectral fingerprint tracker.
//!
//! See [ADR-147](../../../docs/adr/ADR-147-clipcannon-prosody-emotion-speaker.md)
//! for the rationale: 64-dim log-mel-ish + temporal-difference embedding,
//! cosine-matched against a bounded set of enrolled speakers.
//!
//! No external dependencies. Allocation only at enrolment time.

/// Number of bands per "static" half of the embedding.
pub const SPEAKER_EMBED_BANDS: usize = 32;
/// Total embedding dimensionality (= 32 static bands + 32 first-order deltas).
pub const SPEAKER_EMBED_DIM: usize = SPEAKER_EMBED_BANDS * 2;

/// One enrolled speaker. Identity is `id` (stable for the lifetime of the
/// tracker); `embedding` is updated by exponential moving average.
#[derive(Debug, Clone)]
pub struct SpeakerFingerprint {
    pub id: u32,
    pub embedding: [f32; SPEAKER_EMBED_DIM],
    pub frames: u32,
}

impl SpeakerFingerprint {
    fn from_vector(id: u32, vec: &[f32; SPEAKER_EMBED_DIM]) -> Self {
        Self {
            id,
            embedding: *vec,
            frames: 1,
        }
    }
}

/// Bounded set of enrolled speakers. Aggregate root of the Speaker Identity
/// context.
pub struct SpeakerTracker {
    speakers: Vec<SpeakerFingerprint>,
    next_id: u32,
    max_speakers: usize,
    cosine_threshold: f32,
    /// EMA learning rate when updating an existing fingerprint.
    update_alpha: f32,
    /// Last static-band buffer, used to compute the temporal difference half.
    prev_bands: [f32; SPEAKER_EMBED_BANDS],
    /// Whether `prev_bands` has any meaningful content yet.
    has_prev: bool,
    /// Scratch embedding buffer (avoids any per-call allocation).
    scratch: [f32; SPEAKER_EMBED_DIM],
}

impl SpeakerTracker {
    /// Create a new tracker with the given capacity and cosine match threshold.
    ///
    /// `cosine_threshold` is typically 0.85; raise it to be stricter.
    pub fn new(max_speakers: usize, cosine_threshold: f32) -> Self {
        assert!(max_speakers > 0, "max_speakers must be > 0");
        assert!(
            (0.0..1.0).contains(&cosine_threshold),
            "cosine_threshold must be in (0,1)"
        );
        Self {
            speakers: Vec::with_capacity(max_speakers),
            next_id: 1,
            max_speakers,
            cosine_threshold,
            update_alpha: 0.05,
            prev_bands: [0.0_f32; SPEAKER_EMBED_BANDS],
            has_prev: false,
            scratch: [0.0_f32; SPEAKER_EMBED_DIM],
        }
    }

    /// Number of currently enrolled speakers.
    pub fn len(&self) -> usize {
        self.speakers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.speakers.is_empty()
    }

    /// Iterate enrolled speakers.
    pub fn speakers(&self) -> &[SpeakerFingerprint] {
        &self.speakers
    }

    /// Reset all state. Clears enrolled speakers and the temporal buffer.
    pub fn reset(&mut self) {
        self.speakers.clear();
        self.next_id = 1;
        self.has_prev = false;
        for b in &mut self.prev_bands {
            *b = 0.0;
        }
    }

    /// Observe one frame of magnitudes and return the matching (or
    /// newly-enrolled) speaker id, or `None` if the frame is silent.
    ///
    /// `mags` is the magnitude spectrum (length `window/2 + 1`).
    /// `sample_rate` is the audio sample rate.
    pub fn observe(&mut self, mags: &[f32], sample_rate: f32, energy_db: f32) -> Option<u32> {
        if mags.is_empty() || energy_db < -45.0 {
            // silence — no enrol, no match
            return None;
        }

        // 1. Build the static band half.
        let mut bands = [0.0_f32; SPEAKER_EMBED_BANDS];
        log_mel_bands(mags, sample_rate, &mut bands);

        // 2. Build the embedding into scratch (32 static + 32 deltas).
        for i in 0..SPEAKER_EMBED_BANDS {
            self.scratch[i] = bands[i];
        }
        if self.has_prev {
            for i in 0..SPEAKER_EMBED_BANDS {
                self.scratch[SPEAKER_EMBED_BANDS + i] = bands[i] - self.prev_bands[i];
            }
        } else {
            for i in 0..SPEAKER_EMBED_BANDS {
                self.scratch[SPEAKER_EMBED_BANDS + i] = 0.0;
            }
        }
        self.prev_bands = bands;
        self.has_prev = true;
        l2_normalise(&mut self.scratch);

        // 3. Find best cosine match.
        let mut best_id = None;
        let mut best_idx = 0_usize;
        let mut best_sim = -2.0_f32;
        for (idx, sp) in self.speakers.iter().enumerate() {
            let sim = cosine(&sp.embedding, &self.scratch);
            if sim > best_sim {
                best_sim = sim;
                best_id = Some(sp.id);
                best_idx = idx;
            }
        }

        // 4. Match or enrol.
        if let Some(id) = best_id {
            if best_sim >= self.cosine_threshold {
                // EMA update
                let alpha = self.update_alpha;
                let sp = &mut self.speakers[best_idx];
                for i in 0..SPEAKER_EMBED_DIM {
                    sp.embedding[i] = (1.0 - alpha) * sp.embedding[i] + alpha * self.scratch[i];
                }
                l2_normalise(&mut sp.embedding);
                sp.frames = sp.frames.saturating_add(1);
                return Some(id);
            }
        }

        // No match: try to enrol.
        if self.speakers.len() < self.max_speakers {
            let id = self.next_id;
            self.next_id = self.next_id.saturating_add(1);
            self.speakers
                .push(SpeakerFingerprint::from_vector(id, &self.scratch));
            return Some(id);
        }

        // At capacity — return the closest match anyway.
        best_id
    }

    /// Pure cosine similarity between two embeddings.
    pub fn cosine(a: &[f32; SPEAKER_EMBED_DIM], b: &[f32; SPEAKER_EMBED_DIM]) -> f32 {
        cosine(a, b)
    }
}

/// Project a magnitude spectrum onto `SPEAKER_EMBED_BANDS` log-spaced bands
/// over `[0, sample_rate/2]` and log-compress.
fn log_mel_bands(mags: &[f32], sample_rate: f32, out: &mut [f32; SPEAKER_EMBED_BANDS]) {
    let b = mags.len();
    if b < 2 {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }
    let nyq = sample_rate * 0.5;
    let bin_hz = nyq / (b as f32 - 1.0);
    // Log-spaced edges from 30 Hz to nyquist.
    let lo = 30.0_f32.max(bin_hz);
    let hi = nyq.max(lo + 1.0);
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    let n_bands = SPEAKER_EMBED_BANDS;
    let step = (log_hi - log_lo) / n_bands as f32;

    for band in 0..n_bands {
        let f0 = (log_lo + step * band as f32).exp();
        let f1 = (log_lo + step * (band + 1) as f32).exp();
        let i0 = (f0 / bin_hz).floor() as usize;
        let i1 = ((f1 / bin_hz).ceil() as usize).min(b - 1);
        let i1 = i1.max(i0 + 1).min(b - 1);
        let mut sum = 0.0_f32;
        for i in i0..=i1 {
            let m = mags[i].max(0.0);
            sum += m * m;
        }
        out[band] = (1.0 + sum).ln();
    }
}

fn l2_normalise(v: &mut [f32]) {
    let mut sum = 0.0_f32;
    for &x in v.iter() {
        sum += x * x;
    }
    let norm = sum.sqrt().max(1e-9);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

fn cosine(a: &[f32; SPEAKER_EMBED_DIM], b: &[f32; SPEAKER_EMBED_DIM]) -> f32 {
    // Both L2-normalised → cosine == dot product. 8-way ILP for AVX2/NEON.
    let mut s0 = 0.0_f32;
    let mut s1 = 0.0_f32;
    let mut s2 = 0.0_f32;
    let mut s3 = 0.0_f32;
    let mut s4 = 0.0_f32;
    let mut s5 = 0.0_f32;
    let mut s6 = 0.0_f32;
    let mut s7 = 0.0_f32;
    let mut i = 0;
    while i + 8 <= SPEAKER_EMBED_DIM {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4];
        s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6];
        s7 += a[i + 7] * b[i + 7];
        i += 8;
    }
    let mut sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    while i < SPEAKER_EMBED_DIM {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_mags(emphasis_band: (f32, f32), n: usize, scale: f32) -> Vec<f32> {
        let nyq = 8000.0_f32;
        let bin_hz = nyq / (n as f32 - 1.0);
        (0..n)
            .map(|i| {
                let f = i as f32 * bin_hz;
                if f >= emphasis_band.0 && f <= emphasis_band.1 {
                    1.0 * scale
                } else {
                    0.05 * scale
                }
            })
            .collect()
    }

    #[test]
    fn silent_frame_returns_none() {
        let mut t = SpeakerTracker::new(8, 0.85);
        let mags = vec![0.0_f32; 129];
        assert_eq!(t.observe(&mags, 16_000.0, -80.0), None);
        assert_eq!(t.len(), 0);
    }

    #[test]
    fn first_voiced_frame_enrols_speaker_one() {
        let mut t = SpeakerTracker::new(8, 0.85);
        let mags = synth_mags((300.0, 1500.0), 129, 1.0);
        let id = t.observe(&mags, 16_000.0, -10.0);
        assert_eq!(id, Some(1));
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn same_speaker_matched_repeatedly() {
        let mut t = SpeakerTracker::new(8, 0.80);
        let mags = synth_mags((300.0, 1500.0), 129, 1.0);
        let mut ids = Vec::new();
        for _ in 0..20 {
            ids.push(t.observe(&mags, 16_000.0, -10.0).unwrap());
        }
        assert!(ids.iter().all(|&i| i == 1), "ids = {:?}", ids);
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn distinct_speakers_get_distinct_ids() {
        let mut t = SpeakerTracker::new(8, 0.85);
        let speaker_a = synth_mags((300.0, 800.0), 129, 1.0);
        let speaker_b = synth_mags((2500.0, 5000.0), 129, 1.0);
        // Warm up A
        for _ in 0..5 {
            assert_eq!(t.observe(&speaker_a, 16_000.0, -10.0), Some(1));
        }
        // Switch to B
        let id_b = t.observe(&speaker_b, 16_000.0, -10.0).unwrap();
        assert_ne!(id_b, 1);
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn capacity_is_respected() {
        let mut t = SpeakerTracker::new(2, 0.99);
        // Different spectra to force enrolment.
        let s1 = synth_mags((300.0, 600.0), 129, 1.0);
        let s2 = synth_mags((1500.0, 2000.0), 129, 1.0);
        let s3 = synth_mags((4000.0, 5000.0), 129, 1.0);
        let _ = t.observe(&s1, 16_000.0, -10.0);
        let _ = t.observe(&s2, 16_000.0, -10.0);
        let _ = t.observe(&s3, 16_000.0, -10.0);
        assert!(t.len() <= 2);
    }

    #[test]
    fn embeddings_stay_unit_norm() {
        let mut t = SpeakerTracker::new(4, 0.85);
        let mags = synth_mags((300.0, 1500.0), 129, 1.0);
        for _ in 0..10 {
            t.observe(&mags, 16_000.0, -10.0);
        }
        for sp in t.speakers() {
            let n: f32 = sp.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((n - 1.0).abs() < 1e-3, "norm = {}", n);
        }
    }
}
