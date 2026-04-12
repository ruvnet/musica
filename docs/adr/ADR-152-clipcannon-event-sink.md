# ADR-152: Event Sink — Push-Based Domain Events

## Status
Accepted

## Date
2026-04-12

## Context

ADR-146 §"Domain events" lists six events the analyser knows about
(`SpeakerEnrolled`, `SpeakerMatched`, `HighlightDetected`, `SafeCutDetected`)
and ADRs 149/150/151 add more (`AzimuthChanged`, `VadOnset`, `VadOffset`,
`EndOfTurn`, `MusicStarted`, `MusicStopped`).

In v1 the host had to poll `analyzer.last_frame()` and detect transitions
itself. That works for one consumer but breaks down when multiple
subsystems care about the same events (UI, logger, network bridge…).

## Decision

Add `clipcannon/events.rs` with a tiny **push-based event sink** that
costs zero allocation in the no-subscriber case.

```rust
#[derive(Debug, Clone, Copy)]
pub enum ClipCannonEvent {
    SpeakerEnrolled  { id: u32 },
    SpeakerSwitched  { from: Option<u32>, to: u32 },
    HighlightDetected{ frame_index: u64, score: f32 },
    SafeCutDetected  { frame_index: u64 },
    VadOnset         { frame_index: u64 },
    VadOffset        { frame_index: u64 },
    EndOfTurn        { frame_index: u64, silence_ms: u32 },
    MusicStarted     { frame_index: u64 },
    MusicStopped     { frame_index: u64 },
    AzimuthChanged   { frame_index: u64, azimuth_deg: f32 },
}

pub trait EventSink: Send {
    fn emit(&mut self, event: ClipCannonEvent);
}

/// No-op sink — the default. Inlines to nothing.
pub struct NullSink;
impl EventSink for NullSink { fn emit(&mut self, _: ClipCannonEvent) {} }

/// Bounded ring buffer sink — captures the last N events for tests/UI.
pub struct RingSink<const N: usize> {
    buf:   [Option<ClipCannonEvent>; N],
    head:  usize,
    count: usize,
}
```

The analyser owns a `Box<dyn EventSink>` field, default `NullSink`. A
host installs its own sink via `analyzer.set_sink(Box::new(...))`.

### Why a trait, not a channel?

- Channels (`mpsc`, `crossbeam`) imply allocation and locks. The realtime
  contract bans both.
- A trait object call inlines to a single virtual call. With `NullSink`,
  the optimiser collapses the whole emit chain to dead code (verified at
  -O3).
- Hosts that want async dispatch can wrap a channel inside their own
  `EventSink` impl. The library stays lock-free and dependency-free.

### When events fire

| Event                | Trigger                                                          |
|----------------------|------------------------------------------------------------------|
| `SpeakerEnrolled`    | `SpeakerTracker::observe` returns a brand-new id                 |
| `SpeakerSwitched`    | speaker_id differs from previous frame's speaker_id              |
| `HighlightDetected`  | `AnalysisFrame.highlight` crosses 0.75 from below                |
| `SafeCutDetected`    | `AnalysisFrame.safe_cut == true`                                 |
| `VadOnset`           | VAD transitions Inactive → Active (ADR-150)                      |
| `VadOffset`          | VAD transitions Active → Inactive                                |
| `EndOfTurn`          | EoT timer expires after VadOffset (ADR-150)                      |
| `MusicStarted`       | `SignalKind` becomes `Music` from anything else (ADR-151)        |
| `MusicStopped`       | `SignalKind` leaves `Music`                                       |
| `AzimuthChanged`     | `azimuth_deg` differs from previous emit by ≥ 15° (ADR-149)      |

## Consequences

### Positive
- Multiple consumers can plug into the same analyser without polling.
- Zero overhead when no sink is installed.
- `RingSink` doubles as a test fixture: assert that exactly the right
  events fired in the right order.

### Negative
- One trait object on the analyser hot path. Benchmarks confirm <0.5 ns
  per emit when sink is `NullSink`.

### Risks
- Subscribers that block the realtime thread will starve audio. The
  trait's docstring forbids syscalls in `emit`. We can't enforce it
  beyond documentation.

## References
- ADR-146 §"Domain Events" (original poll-only contract).
- ADR-149, ADR-150, ADR-151 (event sources).
