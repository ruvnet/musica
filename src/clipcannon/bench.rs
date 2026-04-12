//! Custom micro-benchmark harness for the clipcannon subsystem.
//!
//! Follows the convention set by [`crate::benchmark`]: library functions, not
//! a `criterion` dependency. See [ADR-148](../../../docs/adr/ADR-148-clipcannon-benchmark-methodology.md)
//! for the methodology and performance targets.
//!
//! Run from a binary or test like:
//!
//! ```ignore
//! use musica::clipcannon::bench::*;
//! let r = bench_prosody_frame();
//! println!("{}", r);
//! ```

use core::f32::consts::PI;
use std::time::Instant;

use crate::hearmusica::{AudioBlock, AudioProcessor};

use super::emotion::EmotionEstimator;
use super::localize::{Localizer, DEFAULT_MIC_SPACING_M};
use super::music_speech::MusicSpeechDetector;
use super::pipeline::RealtimeAvatarAnalyzer;
use super::prosody::{ProsodyExtractor, ProsodySnapshot};
use super::singing::{PitchTracker, StyleClassifier, StyleMatch};
use super::spectrum::SharedSpectrum;
use super::speaker_embed::SpeakerTracker;
use super::vad::VadDetector;
use super::viseme::VisemeMapper;

/// Default warm-up iterations.
pub const DEFAULT_WARMUP: u32 = 256;
/// Default measurement iterations.
pub const DEFAULT_ITERS: u32 = 4096;

/// One benchmark result.
#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: &'static str,
    pub iterations: u32,
    pub total_ms: f64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

impl std::fmt::Display for BenchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:30}  iters={:>5}  mean={:>7.2}µs  p50={:>7.2}µs  p95={:>7.2}µs  p99={:>7.2}µs  max={:>7.2}µs",
            self.name,
            self.iterations,
            self.mean_us,
            self.p50_us,
            self.p95_us,
            self.p99_us,
            self.max_us
        )
    }
}

fn summarise(name: &'static str, mut samples_us: Vec<f64>) -> BenchResult {
    let iterations = samples_us.len() as u32;
    let total_ms = samples_us.iter().sum::<f64>() / 1000.0;
    let mean_us = samples_us.iter().sum::<f64>() / iterations.max(1) as f64;
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |q: f64| -> f64 {
        let idx = ((q * (samples_us.len() as f64 - 1.0)).round() as usize)
            .min(samples_us.len() - 1);
        samples_us[idx]
    };
    BenchResult {
        name,
        iterations,
        total_ms,
        mean_us,
        p50_us: p(0.50),
        p95_us: p(0.95),
        p99_us: p(0.99),
        max_us: *samples_us.last().unwrap_or(&0.0),
    }
}

fn voiced_frame(window: usize, sr: f32, f0: f32) -> Vec<f32> {
    (0..window)
        .map(|i| {
            let t = i as f32 / sr;
            (2.0 * PI * f0 * t).sin() * 0.5
                + (2.0 * PI * 2.0 * f0 * t).sin() * 0.2
                + (2.0 * PI * 3.0 * f0 * t).sin() * 0.1
        })
        .collect()
}

fn naive_mags(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let b = n / 2 + 1;
    let mut mags = vec![0.0_f32; b];
    for k in 0..b {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for i in 0..n {
            let a = -2.0 * PI * (k * i) as f32 / n as f32;
            re += frame[i] * a.cos();
            im += frame[i] * a.sin();
        }
        mags[k] = (re * re + im * im).sqrt();
    }
    mags
}

/// Bench `ProsodyExtractor::extract` on a 256-sample voiced frame.
pub fn bench_prosody_frame() -> BenchResult {
    bench_prosody_frame_with(DEFAULT_WARMUP, DEFAULT_ITERS)
}

pub fn bench_prosody_frame_with(warmup: u32, iters: u32) -> BenchResult {
    // Bench the SOTA path: prosody from a precomputed SharedSpectrum
    // (Wiener-Khinchin ACF, no redundant FFT). This is what the analyser
    // uses in production.
    let sr = 16_000.0_f32;
    let win = 256_usize;
    let mut spec = SharedSpectrum::new(sr, win);
    let l = voiced_frame(win, sr, 220.0);
    let r = voiced_frame(win, sr, 220.0);
    spec.compute(&l, &r);
    let mut ext = ProsodyExtractor::new(sr, win);
    for _ in 0..warmup {
        std::hint::black_box(ext.extract_from_spectrum(&spec));
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        let r = ext.extract_from_spectrum(&spec);
        std::hint::black_box(r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("prosody_frame", samples)
}

/// Bench `VisemeMapper::map` on a precomputed snapshot + spectrum.
pub fn bench_viseme_map() -> BenchResult {
    bench_viseme_map_with(DEFAULT_WARMUP, DEFAULT_ITERS)
}

pub fn bench_viseme_map_with(warmup: u32, iters: u32) -> BenchResult {
    let sr = 16_000.0_f32;
    let win = 256_usize;
    let mut ext = ProsodyExtractor::new(sr, win);
    let mut mapper = VisemeMapper::new();
    let frame = voiced_frame(win, sr, 220.0);
    let mags = naive_mags(&frame);
    let snap = ext.extract(&frame, &mags);

    for _ in 0..warmup {
        std::hint::black_box(mapper.map(&snap, &mags, sr));
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        let r = mapper.map(&snap, &mags, sr);
        std::hint::black_box(r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("viseme_map", samples)
}

/// Bench `SpeakerTracker::observe` in steady-state (one enrolled speaker).
pub fn bench_speaker_observe() -> BenchResult {
    bench_speaker_observe_with(DEFAULT_WARMUP, DEFAULT_ITERS)
}

pub fn bench_speaker_observe_with(warmup: u32, iters: u32) -> BenchResult {
    let sr = 16_000.0_f32;
    let win = 256_usize;
    let frame = voiced_frame(win, sr, 220.0);
    let mags = naive_mags(&frame);
    let mut tracker = SpeakerTracker::new(8, 0.85);
    // Enrol once.
    for _ in 0..16 {
        tracker.observe(&mags, sr, -10.0);
    }

    for _ in 0..warmup {
        std::hint::black_box(tracker.observe(&mags, sr, -10.0));
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        let r = tracker.observe(&mags, sr, -10.0);
        std::hint::black_box(r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("speaker_observe", samples)
}

/// Bench full `RealtimeAvatarAnalyzer::process` on one 128-sample block.
pub fn bench_analyzer_block() -> BenchResult {
    bench_analyzer_block_with(DEFAULT_WARMUP, DEFAULT_ITERS)
}

pub fn bench_analyzer_block_with(warmup: u32, iters: u32) -> BenchResult {
    let sr = 16_000.0_f32;
    let bs = 128_usize;
    let mut analyzer = RealtimeAvatarAnalyzer::with_capacity(sr, bs, 8);
    analyzer.prepare(sr, bs);
    let mut block = AudioBlock::new(bs, sr);
    let frame = voiced_frame(bs, sr, 220.0);
    block.left.copy_from_slice(&frame);
    block.right.copy_from_slice(&frame);

    for _ in 0..warmup {
        analyzer.process(&mut block);
    }
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        analyzer.process(&mut block);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("analyzer_block", samples)
}

/// Bench composite throughput: 1 second of audio (125 × 128-sample blocks).
pub fn bench_analyzer_composite() -> BenchResult {
    let sr = 16_000.0_f32;
    let bs = 128_usize;
    let n_blocks = (sr as usize) / bs;
    let mut analyzer = RealtimeAvatarAnalyzer::with_capacity(sr, bs, 8);
    analyzer.prepare(sr, bs);
    let mut block = AudioBlock::new(bs, sr);
    let frame = voiced_frame(bs, sr, 220.0);
    block.left.copy_from_slice(&frame);
    block.right.copy_from_slice(&frame);

    // warmup pass
    for _ in 0..4 {
        for _ in 0..n_blocks {
            analyzer.process(&mut block);
        }
    }

    let iters = 32_u32;
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        for _ in 0..n_blocks {
            analyzer.process(&mut block);
        }
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("analyzer_composite_1s", samples)
}

// ────────────────────────────────────────────────────────────────────────
// New benches for ADRs 149-154
// ────────────────────────────────────────────────────────────────────────

/// Bench `SharedSpectrum::compute` — the one FFT per block.
pub fn bench_shared_spectrum() -> BenchResult {
    let sr = 16_000.0_f32;
    let win = 256_usize;
    let mut spec = SharedSpectrum::new(sr, win);
    let l = voiced_frame(win, sr, 220.0);
    let r = voiced_frame(win, sr, 220.0);
    for _ in 0..DEFAULT_WARMUP {
        spec.compute(&l, &r);
    }
    let mut samples = Vec::with_capacity(DEFAULT_ITERS as usize);
    for _ in 0..DEFAULT_ITERS {
        let t = Instant::now();
        spec.compute(&l, &r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("shared_spectrum", samples)
}

/// Bench `Localizer::locate` against a precomputed shared spectrum.
pub fn bench_localize_block() -> BenchResult {
    let sr = 16_000.0_f32;
    let win = 256_usize;
    let mut spec = SharedSpectrum::new(sr, win);
    let l = voiced_frame(win, sr, 220.0);
    let r = voiced_frame(win, sr, 220.0);
    spec.compute(&l, &r);
    let mut loc = Localizer::new(sr, win, DEFAULT_MIC_SPACING_M);
    for _ in 0..DEFAULT_WARMUP {
        std::hint::black_box(loc.locate(&spec));
    }
    let mut samples = Vec::with_capacity(DEFAULT_ITERS as usize);
    for _ in 0..DEFAULT_ITERS {
        let t = Instant::now();
        let r = loc.locate(&spec);
        std::hint::black_box(r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("localize_block", samples)
}

/// Bench the VAD state machine on a precomputed snapshot.
pub fn bench_vad_observe() -> BenchResult {
    let mut vad = VadDetector::new();
    let snap = ProsodySnapshot {
        f0_hz: 220.0,
        voicing: 0.85,
        energy_db: -10.0,
        centroid_hz: 1500.0,
        rolloff_hz: 4000.0,
        zcr: 0.05,
        flatness: 0.10,
    };
    for _ in 0..DEFAULT_WARMUP {
        std::hint::black_box(vad.observe(&snap, 8.0));
    }
    let mut samples = Vec::with_capacity(DEFAULT_ITERS as usize);
    for _ in 0..DEFAULT_ITERS {
        let t = Instant::now();
        let r = vad.observe(&snap, 8.0);
        std::hint::black_box(r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("vad_observe", samples)
}

/// Bench the emotion estimator.
pub fn bench_emotion_observe() -> BenchResult {
    let mut e = EmotionEstimator::new();
    let snap = ProsodySnapshot {
        f0_hz: 220.0,
        voicing: 0.85,
        energy_db: -10.0,
        centroid_hz: 1500.0,
        rolloff_hz: 4000.0,
        zcr: 0.05,
        flatness: 0.10,
    };
    for _ in 0..DEFAULT_WARMUP {
        std::hint::black_box(e.observe(&snap));
    }
    let mut samples = Vec::with_capacity(DEFAULT_ITERS as usize);
    for _ in 0..DEFAULT_ITERS {
        let t = Instant::now();
        let r = e.observe(&snap);
        std::hint::black_box(r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("emotion_observe", samples)
}

/// Bench the music/speech discriminator.
pub fn bench_music_speech_observe() -> BenchResult {
    let mut d = MusicSpeechDetector::new();
    let snap = ProsodySnapshot {
        f0_hz: 220.0,
        voicing: 0.85,
        energy_db: -10.0,
        centroid_hz: 1500.0,
        rolloff_hz: 4000.0,
        zcr: 0.05,
        flatness: 0.10,
    };
    // warm history
    for _ in 0..16 {
        d.observe(&snap, true);
    }
    for _ in 0..DEFAULT_WARMUP {
        std::hint::black_box(d.observe(&snap, true));
    }
    let mut samples = Vec::with_capacity(DEFAULT_ITERS as usize);
    for _ in 0..DEFAULT_ITERS {
        let t = Instant::now();
        let r = d.observe(&snap, true);
        std::hint::black_box(r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("music_speech_observe", samples)
}

/// Bench the singing pitch tracker.
pub fn bench_pitch_track() -> BenchResult {
    let sr = 16_000.0_f32;
    let win = 256_usize;
    let mut spec = SharedSpectrum::new(sr, win);
    let l = voiced_frame(win, sr, 440.0);
    let r = vec![0.0_f32; win];
    spec.compute(&l, &r);
    let snap = ProsodySnapshot {
        f0_hz: 440.0,
        voicing: 0.9,
        energy_db: -10.0,
        centroid_hz: 2200.0,
        rolloff_hz: 5500.0,
        zcr: 0.10,
        flatness: 0.18,
    };
    let mut pt = PitchTracker::new(sr);
    for _ in 0..DEFAULT_WARMUP {
        std::hint::black_box(pt.track(&spec, &snap));
    }
    let mut samples = Vec::with_capacity(DEFAULT_ITERS as usize);
    for _ in 0..DEFAULT_ITERS {
        let t = Instant::now();
        let r = pt.track(&spec, &snap);
        std::hint::black_box(r);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("pitch_track", samples)
}

/// Bench the singing style classifier `top_k` lookup over the default library.
pub fn bench_style_top_k() -> BenchResult {
    let mut sc = StyleClassifier::default_library();
    let snap = ProsodySnapshot {
        f0_hz: 220.0,
        voicing: 0.9,
        energy_db: -10.0,
        centroid_hz: 2200.0,
        rolloff_hz: 5500.0,
        zcr: 0.10,
        flatness: 0.18,
    };
    let pitch = crate::clipcannon::PitchSnapshot {
        f0_hz: 220.0,
        cents: 0.0,
        midi_note: 69.0,
        stability: 0.9,
        voiced: true,
    };
    let vib = crate::clipcannon::VibratoSnapshot {
        rate_hz: 5.5,
        depth_cents: 50.0,
        presence: 0.6,
    };
    for _ in 0..32 {
        sc.observe(&snap, &pitch, &vib);
    }
    let mut out = [StyleMatch {
        name: "",
        similarity: 0.0,
    }; 3];
    for _ in 0..DEFAULT_WARMUP {
        sc.top_k(3, &mut out);
    }
    let mut samples = Vec::with_capacity(DEFAULT_ITERS as usize);
    for _ in 0..DEFAULT_ITERS {
        let t = Instant::now();
        sc.top_k(3, &mut out);
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    summarise("style_top_k", samples)
}

/// Run all benches and return their results.
pub fn run_all() -> Vec<BenchResult> {
    vec![
        bench_shared_spectrum(),
        bench_prosody_frame(),
        bench_viseme_map(),
        bench_speaker_observe(),
        bench_localize_block(),
        bench_vad_observe(),
        bench_emotion_observe(),
        bench_music_speech_observe(),
        bench_pitch_track(),
        bench_style_top_k(),
        bench_analyzer_block(),
        bench_analyzer_composite(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: every bench produces a positive mean.
    #[test]
    fn benches_run_and_report() {
        let r = bench_prosody_frame_with(8, 32);
        assert!(r.mean_us > 0.0);
        let r = bench_viseme_map_with(8, 32);
        assert!(r.mean_us > 0.0);
        let r = bench_speaker_observe_with(8, 32);
        assert!(r.mean_us > 0.0);
        let r = bench_analyzer_block_with(8, 32);
        assert!(r.mean_us > 0.0);
    }
}
