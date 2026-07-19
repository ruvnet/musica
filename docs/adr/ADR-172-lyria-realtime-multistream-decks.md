# ADR-172: Lyria RealTime multistream decks

- Status: Accepted
- Date: 2026-07-19

## Context

A single full-arrangement Lyria RealTime stream did not give Musica's editable sequencer or mixer enough audible authority. Live sets also need an independently controllable voice-like layer without misrepresenting Lyria RealTime as a lyrical singing API.

Lyria RealTime accepts persistent WebSocket sessions, realtime weighted prompts, BPM/key controls, and `VOCALIZATION` generation mode. Its documented realtime output remains non-lyrical 48 kHz stereo PCM16. Lyria 3 Clip and Pro remain the separate batch path for intelligible vocals and timed lyrics.

## Decision

Musica runs three named Lyria RealTime sessions:

| Deck | Role | Generation mode |
|---|---|---|
| `main` | Style-directed performance bed | `QUALITY` |
| `sequence` | Dedicated interpretation of step density, mute, solo, volume, and master BPM | `QUALITY` |
| `vocal` | Wordless choir vowels and rhythmic vocal textures | `VOCALIZATION` |

The native provider owns an independent WebSocket, command channel, bounded PCM queue, warning state, and lifecycle for each deck. React never receives the Gemini key. It polls PCM by deck and schedules each queue through an independent Web Audio gain and clock path.

Each deck exposes mute, volume, semitone pitch, and beat nudge. All decks inherit the master key and audible BPM. Because Web Audio resampling changes pitch and tempo together, Musica compensates the BPM sent to Lyria by the inverse pitch ratio before applying the playback-rate shift. Beat nudge handles small network and content-phase offsets without changing the musical BPM.

Play starts all sessions concurrently and uses a finite startup deadline. The main deck is required; a late auxiliary deck is reported without leaving the transport in an indefinite buffering state. Stop closes every named session.

## Consequences

- Sequencer edits and mixer moves have a dedicated generated layer instead of only prompting the full backing stream.
- Operators can balance, mute, pitch, and align each AI layer during a set.
- Three sessions consume more quota, bandwidth, CPU, and provider concurrency than one session.
- Lyria's generated musical phase is probabilistic. Shared BPM/key plus beat nudge improves cohesion but does not provide sample-accurate stem synchronization.
- Realtime wordless vocalization is supported; realtime lyrical singing is not. Lyrical material must be generated with Lyria 3 or imported as a separate clip.
