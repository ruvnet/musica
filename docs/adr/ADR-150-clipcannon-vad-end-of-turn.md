# ADR-150: Voice Activity Detection & End-of-Turn Detection

## Status
Accepted

## Date
2026-04-12

## Context

A voice agent ("Jarvis"-style) needs two binary signals from the audio
stream: **is the user speaking?** (VAD) and **has the user finished their
turn?** (EoT). Without VAD the agent burns ASR cycles on silence; without
EoT it either talks over the user or waits awkwardly long.

ClipCannon upstream uses Silero VAD (a small neural net). We can do
better-than-Silero on the *latency* axis with classical features: we
already compute voicing, energy, ZCR, and spectral flatness at zero
incremental cost.

## Decision

Add `clipcannon/vad.rs` with:

```rust
pub enum VadState { Inactive, Active }

pub struct VadDetector {
    state:           VadState,
    onset_thresh:    f32,    // voicing+energy score to enter Active
    offset_thresh:   f32,    // score to leave Active (hysteresis)
    onset_frames:    u32,    // consecutive frames above onset_thresh required
    offset_frames:   u32,    // consecutive frames below offset_thresh required
    above_count:     u32,
    below_count:     u32,
    eot_silence_ms:  u32,    // ms of continuous silence to declare EoT
    silence_ms_acc:  u32,
}

#[derive(Debug, Clone, Copy)]
pub struct VadDecision {
    pub state:        VadState,
    pub speech_score: f32,    // [0,1]
    pub onset_edge:   bool,   // true exactly on the frame we entered Active
    pub offset_edge:  bool,   // true exactly on the frame we left Active
    pub end_of_turn:  bool,   // true exactly once after eot_silence_ms of silence
}

impl VadDetector {
    pub fn new() -> Self;       // sensible defaults for 16 kHz speech
    pub fn observe(&mut self, snap: &ProsodySnapshot, block_ms: f32) -> VadDecision;
    pub fn reset(&mut self);
}
```

### Speech score

Combines four features into a [0,1] score:

```text
voicing_score   = clamp(voicing, 0, 1)
energy_score    = sigmoid((energy_db + 38) / 6)         // -38 dB → 0.5
flatness_score  = clamp(1 - flatness, 0, 1)             // tonal > noisy
zcr_score       = 1 - smoothstep(0.20, 0.40, zcr)       // pitch < noise

speech_score    = 0.45·voicing + 0.30·energy + 0.15·flatness + 0.10·zcr
```

### Hysteresis

- Enter `Active` after `onset_frames` (default 3) consecutive frames with
  `speech_score ≥ onset_thresh` (default 0.55).
- Leave `Active` after `offset_frames` (default 6) consecutive frames with
  `speech_score < offset_thresh` (default 0.35).

Two thresholds + two counters give clean state transitions and immunity
to single-frame glitches.

### End-of-turn

After leaving `Active`, accumulate silence ms. Once
`silence_ms_acc ≥ eot_silence_ms` (default 700 ms), emit `end_of_turn = true`
**exactly once** and reset the accumulator. 700 ms matches typical
conversational turn-taking gaps.

### Realtime contract
- Pure state machine over a `ProsodySnapshot`.
- Allocation-free.
- Target: <0.5 µs / call (essentially free).

## Consequences

### Positive
- Voice agents can wake/sleep ASR with ~30 ms latency to onset and a
  configurable end-of-turn timeout.
- No model file, no GPU, no neural net.
- Hysteresis kills false-positives from single-frame energy spikes.

### Negative
- Whispered speech has low voicing and low energy → may miss it. We
  expose `onset_thresh` so a hosting app can tune.

### Risks
- Music can score above the speech threshold. Hosts that need to
  discriminate should also consume the music/speech bit from ADR-151.

## References
- Sohn et al., *A Statistical Model-Based Voice Activity Detection*, 1999.
- Silero VAD (architectural reference, not used).
- ADR-145, ADR-147, ADR-151.
