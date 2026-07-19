# ADR-171: AI Instrument Bank and Auto DJ Sequencing

## Status

Accepted

## Date

2026-07-19

## Context

Using a single mixed generated song as a sampler source produced poor musical results. A mixed track contains overlapping transients, harmony, ambience, mastering compression, and phase relationships that become smeared when sliced and pitch-shifted. That approach is not state of the art and should not be presented as Musica's production instrument model.

The stronger architecture is closer to a sampler, groovebox, and live DJ system:

- AI generates isolated tones, one-shots, stems, loops, or continuous realtime beds;
- deterministic MIDI, notation, sequencer steps, and human/controller gestures decide the notes and timing;
- realtime AI is steered gradually with prompt weights and music config;
- local synthesis remains the fallback and clock authority;
- generated assets are labeled by role and quality instead of treated as interchangeable songs.

## Decision

Musica will treat AI music models as sound-design and performance collaborators, not the only source of musical truth.

The production path is:

1. Lyria RealTime supplies live instrumental beds and responsive music direction.
2. AI instrument banks supply isolated playable material: multisamples, one-shots, stems, and short loops.
3. Musica's sequencer, MIDI import, controllers, piano keyboard, and Auto DJ mode assemble and steer the performance.
4. Mixed generated clips may remain as experimental references but must not become default production instruments.

The app includes a realtime piano/Auto DJ deck that edits weighted prompts and realtime config while also triggering local notes for immediate performance feedback. Auto DJ mode may change prompt sets, density, brightness, BPM, and local note gestures, but it must keep changes bounded and gradual.

## Consequences

### Positive

- The architecture avoids the audible failure mode of granular-slicing mixed songs.
- Musica remains playable without network access.
- Lyria RealTime becomes the live layer while deterministic notes keep musical intent controllable.
- Future Stable Audio, ACE-Step, or other model adapters can produce bank assets without changing the sequencer.

### Negative

- High-quality AI instrument banks require better asset generation and curation than one prompt.
- Full realtime streaming still needs native WebSocket/PCM buffering work.
- The current Moonlight source is retained only as an experimental asset until replaced by isolated tones or stems.

## Acceptance Tests

1. The default live deck exposes weighted prompts, BPM, density, brightness, guidance, muting, piano notes, and Auto DJ mode.
2. Local note triggering works independently of the Lyria RealTime provider.
3. Mixed AI source assets are documented as experimental and not claimed as notation-faithful repertoire.
4. Realtime provider configuration is separate from batch Lyria 3 Pro song generation.
5. New AI-generated bank assets must record role, source model, prompt provenance, and whether they are suitable for production playback.

## References

- ADR-170: Lyria RealTime live performance provider
- ADR-168: Lyria 3 Pro provider, routing, and prompt contract
- ADR-169: Lyria paid job lifecycle, assets, analysis, and provenance
