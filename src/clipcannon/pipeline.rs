//! Anti-corruption layer between `hearmusica::AudioProcessor` and the four
//! ClipCannon bounded contexts.
//!
//! `RealtimeAvatarAnalyzer` is the only place in the crate where Signal
//! Analysis, Avatar Driving, Speaker Identity, and Analysis DAG are composed.
//! Downstream code talks to it via the `AudioProcessor` trait or its
//! [`last_frame`](RealtimeAvatarAnalyzer::last_frame) accessor.
//!
//! This module owns its own minimal STFT (radix-2 Cooley-Tukey + Hann window)
//! to keep the realtime contract: zero allocation in steady state and no
//! cross-context mutable state. The STFT here is intentionally an `f32`
//! single-frame variant separate from `crate::stft` (which is `f64`) so the
//! analyser can stay aligned with `AudioBlock`'s `f32` channels without
//! conversion.

use core::f32::consts::PI;

use crate::hearmusica::{AudioBlock, AudioProcessor};

use super::analysis::{AnalysisFrame, Analyzer};
use super::prosody::{ProsodyExtractor, ProsodySnapshot};
use super::speaker_embed::SpeakerTracker;
use super::viseme::{VisemeCoeffs, VisemeMapper};

/// Realtime analyser implementing [`AudioProcessor`]. See ADR-145/146.
pub struct RealtimeAvatarAnalyzer {
    sample_rate: f32,
    block_size: usize,
    window_size: usize,
    /// FFT scratch — real and imaginary parts kept separate for cache friendliness.
    real: Vec<f32>,
    imag: Vec<f32>,
    /// Bit-reverse table for the radix-2 FFT, sized to `window_size`.
    bitrev: Vec<usize>,
    /// Precomputed FFT twiddles, one stage per power-of-two.
    twiddles: Vec<Vec<(f32, f32)>>,
    /// Hann window for the STFT.
    hann: Vec<f32>,
    /// Frame buffer (windowed time-domain frame).
    frame_buf: Vec<f32>,
    /// Rolling history buffer of length `window_size`. Contains the most
    /// recent `window_size` samples of the L channel so STFT frames are
    /// always full even when blocks are smaller than the window.
    history: Vec<f32>,
    /// Magnitude scratch (length window/2+1).
    mags: Vec<f32>,

    extractor: ProsodyExtractor,
    viseme: VisemeMapper,
    speaker: SpeakerTracker,
    analyzer: Analyzer,

    last_frame: Option<AnalysisFrame>,
    prepared: bool,
}

impl Default for RealtimeAvatarAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl RealtimeAvatarAnalyzer {
    /// Create a new analyser at default settings (16 kHz, 128-sample blocks,
    /// 256-sample window, 16-speaker tracker, cosine threshold 0.85).
    pub fn new() -> Self {
        Self::with_capacity(super::DEFAULT_SAMPLE_RATE, super::DEFAULT_HOP, 16)
    }

    /// Create a new analyser with explicit configuration.
    pub fn with_capacity(sample_rate: f32, block_size: usize, max_speakers: usize) -> Self {
        let window_size = (block_size * 2).next_power_of_two().max(64);
        let extractor = ProsodyExtractor::new(sample_rate, window_size);
        let speaker = SpeakerTracker::new(max_speakers, 0.85);
        let analyzer = Analyzer::new(default_session_id());

        let bitrev = build_bitrev(window_size);
        let twiddles = build_twiddles(window_size);
        let hann: Vec<f32> = (0..window_size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / window_size as f32).cos()))
            .collect();

        Self {
            sample_rate,
            block_size,
            window_size,
            real: vec![0.0; window_size],
            imag: vec![0.0; window_size],
            bitrev,
            twiddles,
            hann,
            frame_buf: vec![0.0; window_size],
            history: vec![0.0; window_size],
            mags: vec![0.0; window_size / 2 + 1],
            extractor,
            viseme: VisemeMapper::new(),
            speaker,
            analyzer,
            last_frame: None,
            prepared: false,
        }
    }

    /// Latest analysis frame, if any. Updated on every `process` call.
    pub fn last_frame(&self) -> Option<&AnalysisFrame> {
        self.last_frame.as_ref()
    }

    /// Borrow the speaker tracker (read-only) for diagnostics.
    pub fn speaker_tracker(&self) -> &SpeakerTracker {
        &self.speaker
    }

    /// Reset all analyser state. Intended for stream restart between sessions.
    pub fn reset(&mut self) {
        self.viseme.reset();
        self.speaker.reset();
        self.analyzer = Analyzer::new(default_session_id());
        self.last_frame = None;
    }

    /// Push the L channel of `block` into the rolling history buffer, then
    /// compute STFT magnitudes over the most recent `window_size` samples
    /// into `self.mags` and the windowed time-domain frame into
    /// `self.frame_buf`. Allocation-free.
    fn compute_mags_and_frame(&mut self, block: &AudioBlock) {
        let n = self.window_size;
        let bn = block.left.len();

        // Shift history left by `bn` and append the new block at the end.
        if bn >= n {
            // New block fully replaces history.
            let off = bn - n;
            for i in 0..n {
                self.history[i] = block.left[off + i];
            }
        } else {
            // history[..n-bn] = old history[bn..]
            self.history.copy_within(bn..n, 0);
            for i in 0..bn {
                self.history[n - bn + i] = block.left[i];
            }
        }

        // Window into frame_buf and copy to FFT scratch.
        for i in 0..n {
            let s = self.history[i];
            self.frame_buf[i] = s;
            self.real[i] = s * self.hann[i];
            self.imag[i] = 0.0;
        }

        // In-place radix-2 FFT.
        fft_radix2(
            &mut self.real,
            &mut self.imag,
            &self.bitrev,
            &self.twiddles,
        );

        let b = n / 2 + 1;
        for k in 0..b {
            let re = self.real[k];
            let im = self.imag[k];
            self.mags[k] = (re * re + im * im).sqrt();
        }
    }
}

impl AudioProcessor for RealtimeAvatarAnalyzer {
    fn prepare(&mut self, sample_rate: f32, block_size: usize) {
        // If parameters change, fully re-allocate (one-shot, not realtime).
        if !self.prepared || sample_rate != self.sample_rate || block_size != self.block_size {
            *self = Self::with_capacity(sample_rate, block_size, self.speaker.speakers().len().max(16));
        }
        self.analyzer.prepare(sample_rate, block_size);
        self.prepared = true;
    }

    fn process(&mut self, block: &mut AudioBlock) {
        if !self.prepared {
            self.prepare(block.sample_rate, block.block_size);
        }

        // 1. STFT magnitudes from the L channel.
        self.compute_mags_and_frame(block);

        // 2. Prosody features.
        let snap: ProsodySnapshot = self.extractor.extract(&self.frame_buf, &self.mags);

        // 3. Speaker identity.
        let speaker_id = self
            .speaker
            .observe(&self.mags, self.sample_rate, snap.energy_db);

        // 4. Viseme classification.
        let viseme: VisemeCoeffs = self.viseme.map(&snap, &self.mags, self.sample_rate);

        // 5. Aggregate into AnalysisFrame.
        let frame = self.analyzer.analyse(snap, viseme, speaker_id);
        self.last_frame = Some(frame);

        // 6. Stamp the block metadata so downstream blocks can read prosody.
        block.metadata.frame_index = frame.frame_index;
        block.metadata.timestamp_us = frame.timestamp_us;
    }

    fn name(&self) -> &str {
        "RealtimeAvatarAnalyzer"
    }

    fn latency_samples(&self) -> usize {
        // Centred-window STFT has window/2 latency.
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

// ---------- Tiny f32 radix-2 FFT (zero deps) ----------

fn build_bitrev(n: usize) -> Vec<usize> {
    let mut v = vec![0_usize; n];
    let bits = (n as u64).trailing_zeros();
    for i in 0..n {
        let mut x = i;
        let mut r = 0;
        for _ in 0..bits {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        v[i] = r;
    }
    v
}

fn build_twiddles(n: usize) -> Vec<Vec<(f32, f32)>> {
    let mut stages = Vec::new();
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f32;
        let twiddles: Vec<(f32, f32)> = (0..half)
            .map(|k| {
                let a = angle * k as f32;
                (a.cos(), a.sin())
            })
            .collect();
        stages.push(twiddles);
        len <<= 1;
    }
    stages
}

fn fft_radix2(re: &mut [f32], im: &mut [f32], bitrev: &[usize], twiddles: &[Vec<(f32, f32)>]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reverse permutation.
    for i in 0..n {
        let j = bitrev[i];
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // Iterative butterfly stages.
    let mut len = 2;
    let mut stage = 0;
    while len <= n {
        let half = len / 2;
        let tw = &twiddles[stage];
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let (cos_t, sin_t) = tw[k];
                let i_a = start + k;
                let i_b = i_a + half;
                let xr = re[i_b] * cos_t - im[i_b] * sin_t;
                let xi = re[i_b] * sin_t + im[i_b] * cos_t;
                re[i_b] = re[i_a] - xr;
                im[i_b] = im[i_a] - xi;
                re[i_a] += xr;
                im[i_a] += xi;
            }
            start += len;
        }
        len <<= 1;
        stage += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let names = p.block_names();
        assert_eq!(names, vec!["RealtimeAvatarAnalyzer"]);
    }

    #[test]
    fn audio_buffers_unchanged_after_process() {
        let mut a = RealtimeAvatarAnalyzer::new();
        a.prepare(16_000.0, 128);
        let mut block = AudioBlock::new(128, 16_000.0);
        fill_sine(&mut block, 440.0, 16_000.0, 0.4);
        let saved_left = block.left.clone();
        let saved_right = block.right.clone();
        a.process(&mut block);
        assert_eq!(block.left, saved_left);
        assert_eq!(block.right, saved_right);
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

    #[test]
    fn voiced_frame_recovers_pitch() {
        let mut a = RealtimeAvatarAnalyzer::new();
        a.prepare(16_000.0, 128);
        let mut block = AudioBlock::new(128, 16_000.0);
        fill_sine(&mut block, 220.0, 16_000.0, 0.7);
        // Run several blocks so the speaker tracker enrols and viseme stabilises.
        for _ in 0..4 {
            a.process(&mut block);
        }
        let f = a.last_frame().unwrap();
        assert!(f.prosody.voicing > 0.5, "voicing = {}", f.prosody.voicing);
        assert!(f.prosody.f0_hz > 100.0 && f.prosody.f0_hz < 400.0);
        assert!(f.speaker_id.is_some());
    }

    #[test]
    fn fft_matches_naive_dft_on_small_input() {
        // Build an 8-pt FFT and compare to naive DFT.
        let n = 8;
        let bitrev = build_bitrev(n);
        let tw = build_twiddles(n);
        let mut re = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut im = [0.0_f32; 8];
        fft_radix2(&mut re, &mut im, &bitrev, &tw);

        // Naive DFT for comparison.
        let input: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for k in 0..n {
            let mut nr = 0.0_f32;
            let mut ni = 0.0_f32;
            for i in 0..n {
                let a = -2.0 * PI * (k * i) as f32 / n as f32;
                nr += input[i] * a.cos();
                ni += input[i] * a.sin();
            }
            assert!((re[k] - nr).abs() < 1e-3, "k={}: {} vs {}", k, re[k], nr);
            assert!((im[k] - ni).abs() < 1e-3, "k={}: {} vs {}", k, im[k], ni);
        }
    }
}
