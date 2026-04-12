//! ClipCannon — realtime analysis & avatar-driving subsystem.
//!
//! This module is **not** a port of [ClipCannon](https://github.com/mrjcleaver/clipcannon)
//! but a Rust realtime analysis layer inspired by it. It implements the
//! sub-millisecond audio side of an AI avatar / meeting bot pipeline:
//! prosody extraction, viseme mapping, lightweight speaker fingerprinting,
//! and an aggregating analysis DAG that mirrors a subset of ClipCannon's
//! 23-stage pipeline.
//!
//! ## Bounded contexts
//!
//! - [`prosody`] — Signal Analysis. Pure functional `ProsodyExtractor` →
//!   `ProsodySnapshot` value object.
//! - [`viseme`] — Avatar Driving. `VisemeMapper` → `VisemeCoeffs`.
//! - [`speaker_embed`] — Speaker Identity. `SpeakerTracker` aggregate root.
//! - [`analysis`] — Analysis DAG. `Analyzer` → `AnalysisFrame` aggregate root.
//! - [`pipeline`] — Anti-corruption layer. `RealtimeAvatarAnalyzer` implements
//!   `hearmusica::AudioProcessor` and is the only place all four contexts
//!   are composed.
//!
//! ## Realtime contract
//!
//! - All hot-path methods (`extract`, `map`, `observe`, `analyse`, `process`)
//!   are allocation-free in steady state. The only allocations occur during
//!   `prepare()` (sizing internal buffers) or when a brand-new speaker is
//!   enrolled.
//! - All operations on `f32` to match the upstream `AudioBlock` channel type.
//! - Performance targets: see [ADR-148](../../../docs/adr/ADR-148-clipcannon-benchmark-methodology.md).
//!
//! ## Quick start
//!
//! ```ignore
//! use musica::clipcannon::pipeline::RealtimeAvatarAnalyzer;
//! use musica::hearmusica::{AudioBlock, AudioProcessor};
//!
//! let mut analyzer = RealtimeAvatarAnalyzer::new();
//! analyzer.prepare(16_000.0, 128);
//!
//! let mut block = AudioBlock::new(128, 16_000.0);
//! // ... fill block.left / block.right ...
//! analyzer.process(&mut block);
//!
//! if let Some(frame) = analyzer.last_frame() {
//!     println!("viseme = {:?}, jaw_open = {}", frame.viseme.viseme, frame.viseme.jaw_open);
//! }
//! ```

pub mod analysis;
pub mod bench;
pub mod pipeline;
pub mod prosody;
pub mod speaker_embed;
pub mod viseme;

pub use analysis::{AnalysisFrame, Analyzer, EmotionBucket, HighlightScorer, SafeCutDetector};
pub use pipeline::RealtimeAvatarAnalyzer;
pub use prosody::{ProsodyExtractor, ProsodySnapshot};
pub use speaker_embed::{SpeakerFingerprint, SpeakerTracker, SPEAKER_EMBED_DIM};
pub use viseme::{Viseme, VisemeCoeffs, VisemeMapper};

/// Default sample rate used by the realtime subsystem when none is supplied.
pub const DEFAULT_SAMPLE_RATE: f32 = 16_000.0;

/// Default analysis window size (samples).
pub const DEFAULT_WINDOW: usize = 256;

/// Default hop size (samples). Equal to the typical realtime block size.
pub const DEFAULT_HOP: usize = 128;
