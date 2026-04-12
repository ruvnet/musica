//! Push-based domain event sink — see ADR-152.
//!
//! Zero-allocation, lock-free. Default sink is `NullSink` which inlines to
//! nothing under release optimisation.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClipCannonEvent {
    SpeakerEnrolled { id: u32 },
    SpeakerSwitched { from: Option<u32>, to: u32 },
    HighlightDetected { frame_index: u64, score: f32 },
    SafeCutDetected { frame_index: u64 },
    VadOnset { frame_index: u64 },
    VadOffset { frame_index: u64 },
    EndOfTurn { frame_index: u64, silence_ms: u32 },
    MusicStarted { frame_index: u64 },
    MusicStopped { frame_index: u64 },
    AzimuthChanged { frame_index: u64, azimuth_deg: f32 },
}

/// Sink for `ClipCannonEvent`s. Implementations MUST NOT block, allocate,
/// or perform syscalls — they run on the realtime thread.
pub trait EventSink: Send {
    fn emit(&mut self, event: ClipCannonEvent);
}

/// No-op sink. Inlines to dead code in release builds.
pub struct NullSink;

impl EventSink for NullSink {
    #[inline(always)]
    fn emit(&mut self, _event: ClipCannonEvent) {}
}

/// Bounded ring-buffer sink. Captures the last `N` events for tests/UI.
pub struct RingSink {
    buf: Vec<ClipCannonEvent>,
    capacity: usize,
    head: usize,
    count: usize,
}

impl RingSink {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
            capacity,
            head: 0,
            count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.count.min(self.capacity)
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Iterate events in insertion order (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &ClipCannonEvent> {
        let len = self.len();
        let start = if self.count <= self.capacity {
            0
        } else {
            self.head
        };
        let buf_len = self.buf.len().max(1);
        (0..len).map(move |i| {
            let idx = (start + i) % buf_len;
            &self.buf[idx]
        })
    }

    /// Drain all events into a `Vec` (test convenience).
    pub fn drain(&mut self) -> Vec<ClipCannonEvent> {
        let v: Vec<ClipCannonEvent> = self.iter().copied().collect();
        self.buf.clear();
        self.head = 0;
        self.count = 0;
        v
    }
}

impl EventSink for RingSink {
    fn emit(&mut self, event: ClipCannonEvent) {
        if self.buf.len() < self.capacity {
            self.buf.push(event);
        } else {
            self.buf[self.head] = event;
            self.head = (self.head + 1) % self.capacity;
        }
        self.count = self.count.saturating_add(1).min(usize::MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_sink_compiles_and_emits() {
        let mut s = NullSink;
        s.emit(ClipCannonEvent::SafeCutDetected { frame_index: 7 });
    }

    #[test]
    fn ring_sink_captures_in_order() {
        let mut r = RingSink::new(4);
        r.emit(ClipCannonEvent::SafeCutDetected { frame_index: 1 });
        r.emit(ClipCannonEvent::SafeCutDetected { frame_index: 2 });
        r.emit(ClipCannonEvent::SafeCutDetected { frame_index: 3 });
        let v = r.drain();
        assert_eq!(v.len(), 3);
        match v[0] {
            ClipCannonEvent::SafeCutDetected { frame_index } => assert_eq!(frame_index, 1),
            _ => panic!(),
        }
    }

    #[test]
    fn ring_sink_overwrites_when_full() {
        let mut r = RingSink::new(2);
        for i in 0..5_u64 {
            r.emit(ClipCannonEvent::SafeCutDetected { frame_index: i });
        }
        let v = r.drain();
        assert_eq!(v.len(), 2);
        match (v[0], v[1]) {
            (
                ClipCannonEvent::SafeCutDetected { frame_index: a },
                ClipCannonEvent::SafeCutDetected { frame_index: b },
            ) => {
                assert_eq!(a, 3);
                assert_eq!(b, 4);
            }
            _ => panic!(),
        }
    }
}
