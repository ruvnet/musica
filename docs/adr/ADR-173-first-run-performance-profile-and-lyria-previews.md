# ADR-173: First-run performance profile and Lyria preview bank

- Status: Accepted
- Date: 2026-07-19

## Context

Musica previously opened directly into a dense live-performance surface. New operators had no guided way to choose genre, tempo, vocal role, or visual direction, and style names alone were not enough to judge the musical interpretation from Lyria.

Browser and desktop autoplay policies also prevent a reliable audible brand cue before a user gesture. Generating a new realtime audition for every genre click adds startup latency, consumes quota, and makes comparisons inconsistent.

## Decision

Musica presents a first-run, four-step performance setup:

1. Music format and genre.
2. Pace, exact BPM, experimental character, and personal direction.
3. Optional voice type and vocal role.
4. Visual scene, color palette, motion model, and intensity.

The normalized profile is stored under a versioned local-storage key and can be reopened from the top-bar settings icon. Applying a profile writes directly to the main Lyria request, vocal companion guidance, master BPM, and Three.js visual controls.

After first-run setup, Musica shows an animated technical welcome screen. `Start Session` is the required user gesture. It adds a short four-note Musica opening direction to the selected Lyria prompt, buffers the enabled realtime decks, then enters the live workspace. Saving setup does not start transport.

Each genre and voice choice has a curated 30-second MP3 preview generated from `models/lyria-realtime-exp`. Preview generation:

- captures a verified 32 seconds of 48 kHz stereo PCM;
- uses a genre-specific prompt and compact four-phrase composition form;
- wraps the last two seconds into the opening material with a crossfade;
- encodes a deterministic 30-second 160 kbps MP3;
- writes a manifest with model, BPM, prompt, byte size, and generation time;
- never writes the Gemini key or raw provider response.

The preview bank is regenerated explicitly with `npm run render:onboarding-previews`. Existing files are retained unless `LYRIA_PREVIEW_FORCE=true` is set.

## Consequences

- First launch has a clear user gesture and does not jump directly from the last wizard choice into playback.
- Genre comparisons are fast, repeatable, and do not consume realtime quota.
- Preview MP3s increase the application bundle and repository size.
- A 30-second Lyria result is an authored style example, not a guarantee that every realtime session will produce identical instrumentation.
- Voice previews use Lyria RealTime vocalization and are wordless. They do not claim intelligible realtime lyrics or reliable singer identity.

## Acceptance tests

1. Missing or malformed stored profiles normalize to the Rock defaults.
2. The wizard cannot start transport while navigating or saving.
3. Every visible genre and vocal preview resolves to a verified 30-second MP3.
4. Completing setup updates both audio and visual engine state.
5. The welcome cue is requested only after `Start Session`.

## References

- ADR-163: Three.js WebGL2 visual engine
- ADR-170: Lyria RealTime live performance provider
- ADR-172: Lyria RealTime multistream decks

