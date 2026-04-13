# ADR-159: Streaming TTS Pipeline — `RealtimeVoiceSynthesiser`

## Status
Accepted

## Date
2026-04-12

## Context

ADRs 155–158 ship four self-contained synthesis pieces:

- **Klatt** formant synthesiser (ADR-155)
- **Phonemiser** for English text → phonemes (ADR-156)
- **Singing synthesiser** with Klatt + PSOLA backends (ADR-158)

A host needs a single front door that composes them into a streaming
TTS engine, mirroring the structure of `RealtimeAvatarAnalyzer` but in
the *opposite* direction (audio out, not audio in).

## Decision

Add `clipcannon/tts_pipeline.rs` exposing `RealtimeVoiceSynthesiser`,
the **anti-corruption layer** between the synthesis bounded contexts
and the rest of the system.

```rust
pub struct RealtimeVoiceSynthesiser {
    sample_rate:    f32,
    block_size:     usize,
    klatt:          KlattSynthesiser,
    phonemiser:     Phonemiser,
    singer:         SingingSynthesiser,
    /// FIFO of pending phonemes to render.
    queue:          VecDeque<TimedPhoneme>,
    /// How many samples remain to render for the *current* phoneme.
    remaining:      usize,
    /// Optional sung-melody overlay.
    melody:         Option<Vec<SungNote>>,
    melody_cursor:  usize,
    /// Event sink (shared with the analyser when desired).
    sink:           Box<dyn EventSink>,
    /// Mode: speak vs sing.
    mode:           VoiceMode,
}

#[derive(Debug, Clone, Copy)]
pub enum VoiceMode {
    Speak,
    Sing,
}

impl RealtimeVoiceSynthesiser {
    pub fn new(sample_rate: f32, block_size: usize) -> Self;

    /// Push speech text. Phonemised internally and queued.
    pub fn speak(&mut self, text: &str);

    /// Push a sung melody (lyrics + notes).
    pub fn sing(&mut self, lyrics: &str, melody: Vec<SungNote>);

    /// Render one block of audio into `block.left` / `block.right`.
    /// Both channels carry the same mono signal.
    pub fn render_block(&mut self, block: &mut AudioBlock);

    /// True if the queue is empty and the engine has nothing left to say.
    pub fn is_idle(&self) -> bool;

    pub fn set_sink(&mut self, sink: Box<dyn EventSink>);
    pub fn reset(&mut self);
}
```

### Streaming contract

`render_block` always renders exactly `block.block_size` samples.
Behaviour:

1. If the queue is empty, fill with silence.
2. Otherwise, peek the head phoneme. If it's the start of a new
   phoneme, push it into the relevant backend (`KlattSynthesiser` for
   speech, `SingingSynthesiser` for singing).
3. Render up to `min(remaining, block_size)` samples from the current
   phoneme.
4. If we run out of samples for the current phoneme mid-block, pop it
   from the queue and continue with the next one. A single block can
   span multiple phonemes.
5. Emit `TtsBoundary` events at every phoneme boundary, `TtsStarted`
   on the first sample of an utterance, `TtsFinished` when the queue
   drains.

### Closed-loop integration

`RealtimeVoiceSynthesiser` is `Send` and uses the same `EventSink`
trait as `RealtimeAvatarAnalyzer`. A host can wire both to the same
sink:

```text
mic ──► RealtimeAvatarAnalyzer ──► (events) ──┐
                                              │
                                              ▼
                                            host LLM / agent
                                              │
                                              ▼
speaker ◄── RealtimeVoiceSynthesiser ◄── (text)
```

Both ends use `AudioBlock` and `BlockMetadata`, so the loop slots
straight into a `hearmusica::Pipeline`.

### New events (ADR-152 extension)

Add three event variants to `ClipCannonEvent`:

```rust
TtsStarted   { utterance_id: u32 }
TtsBoundary  { utterance_id: u32, phoneme_index: u32 }
TtsFinished  { utterance_id: u32 }
```

### Realtime contract

- Per-block render: `<5 µs` for speech (Klatt), `<15 µs` for singing
  (PSOLA depends on grain size). Both at 16 kHz, 128-sample blocks.
- Allocation: only `speak()`/`sing()` allocate (the queue grows by the
  number of phonemes in the input). `render_block` never allocates.

## Consequences

### Positive
- Single front-door API for hosts.
- Same `AudioBlock` / `EventSink` shape as the analyser → closes the
  agent loop with no glue code.
- Speech and singing share the same engine and the same queue.
- Hosts that want a different synth backend implement
  `KlattSynthesiser`'s trait surface.

### Negative
- Two backends in one struct (Klatt always present, PSOLA optional)
  inflate the type slightly. Acceptable.

### Risks
- `speak()` allocates a `Vec<TimedPhoneme>`. Hosts that need fully
  zero-allocation streaming use `phonemise_into` + manual queue push.

## References
- ADR-145 (the analyser's mirror image).
- ADR-152 (event sink — extended here).
- ADR-155 (Klatt), ADR-156 (phonemiser), ADR-158 (singing).
