# Lyria tone sources

This directory contains generated tone-source assets for Musica's MIDI-driven AI synth layer. These files are not treated as authoritative compositions. The app slices short grains from them and pitches those grains from deterministic sequencer notes, like an old-school sampler/sequencer workflow.

`moonlight-sonata-ai-timbre.mp3` was generated with Gemini/Lyria as synthetic source material for a public-domain Beethoven Moonlight Sonata inspired tone bank. The recognizable note order is encoded in the `moonlight-sequencer` performance template, not inferred from the generated audio.

The adjacent receipt records prompt provenance and model metadata for local audit. Do not place provider credentials in this directory.

`lyria-realtime-programmed-stream.mp3` is a direct Lyria RealTime capture driven by `npm run render:lyria-realtime-sample`. The script opens the Live Music WebSocket, sends timed samba, rock, and cinematic prompt/config updates, captures raw 48 kHz stereo PCM, and encodes a 20-second MP3 preview with `ffmpeg`.
