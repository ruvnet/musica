# ADR-170: Lyria RealTime Live Performance Provider

## Status

Accepted

## Date

2026-07-19

## Context

As of July 2026, Google's Lyria surfaces have separate product shapes. Lyria 3 Clip and Lyria 3 Pro are batch Interactions API models for generated tracks, while Lyria RealTime is an experimental low-latency streaming model for continuous, steerable instrumental music. The Gemini API documentation was last updated on 2026-07-16 and describes Lyria RealTime as a persistent bidirectional WebSocket session using weighted prompts, realtime music generation config, and playback commands.

The relevant Lyria RealTime facts for Musica are:

- output is raw 16-bit PCM audio;
- output sample rate is 48 kHz;
- output is stereo;
- the model is instrumental-only;
- weighted prompts can steer instrument, genre, mood, and performance characteristics;
- config controls include BPM, density, brightness, guidance, scale, bass/drum muting, generation mode, temperature, topK, and seed;
- BPM and scale changes require a context reset or stop/play transition to take full effect;
- prompts can be safety-filtered and drastic prompt changes can create abrupt transitions.

Musica is a live VJ instrument, not only a batch song generator. A preview batch model that returns a finished MP3 cannot be the core of a realtime performance system. The live path must therefore model Lyria RealTime directly and keep batch Lyria 3 Pro as a separate song-generation/export feature from ADR-168 and ADR-169.

## Decision

Add a dedicated Lyria RealTime provider boundary with provider identity `lyria_realtime` and model `models/lyria-realtime-exp`. It is enabled only when:

```text
MUSICA_LYRIA_REALTIME_ENABLED=true
GEMINI_API_KEY=<credential available only to the Tauri process>
```

The React app may edit weighted prompts, realtime music config, keyboard gestures, and Auto DJ state, but it must not receive the Gemini key. Native code owns provider readiness, session identity, WebSocket setup, playback commands, and validation of all realtime controls.

The implementation exposes a native command surface for status, start, update, stop, and bounded PCM polling. It validates the documented control envelope, opens the official v1alpha WebSocket endpoint from Rust, sends setup, weighted prompt, generation config, and playback messages, receives Base64 PCM chunks, bounds the native queue, and lets React poll chunks into Web Audio. Lyria audio is routed through Musica's texture track so it improves the quality bed while local sequencer/MIDI/piano layers remain deterministic and visual analysis continues to see the combined mix.

## Consequences

### Positive

- Musica's live architecture now targets the correct July 2026 Google music surface.
- Prompt and config controls are typed, testable, and independent from batch generation.
- The frontend can ship a realtime performance deck and local piano preview without exposing credentials.
- The key and WebSocket transport remain out of React while streamed PCM is mixed into Musica.

### Negative

- Reconnect and long-session recovery are still minimal.
- Realtime billing is not reconciled to provider account data.
- Lyria RealTime is experimental and can change model, schema, or availability.

## Acceptance Tests

1. Provider status is unavailable unless `MUSICA_LYRIA_REALTIME_ENABLED=true` and `GEMINI_API_KEY` are present.
2. Realtime requests require one to four weighted prompts, non-empty text, finite non-zero weights, BPM 60 through 200, density and brightness 0 through 1, guidance 0 through 6, temperature 0 through 3, topK 1 through 1000, and a valid bass/drum mute combination.
3. React can start, update, stop, and poll a typed realtime session through Tauri commands without bundling provider credentials.
4. The piano keyboard remains locally playable in browser preview and desktop fallback.
5. Auto DJ mode changes prompt weights/config gradually rather than replacing the entire performance with batch audio.

## References

- [Google AI for Developers: Real-time music generation using Lyria RealTime](https://ai.google.dev/gemini-api/docs/realtime-music-generation)
- [Google AI for Developers: Generate music with Lyria 3](https://ai.google.dev/gemini-api/docs/music-generation)
- [Gemini API release notes](https://ai.google.dev/gemini-api/docs/changelog)
- ADR-168: Lyria 3 Pro provider, routing, and prompt contract
- ADR-169: Lyria paid job lifecycle, assets, analysis, and provenance
