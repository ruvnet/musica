# Musica VJ sample renders

These six second vertical H.264 and AAC clips are reproducible smoke fixtures for the social export contract. They use procedurally generated stereo audio and audio reactive FFmpeg scopes, so the repository does not depend on copyrighted source material. They are not Lyria outputs or captures from the live Three.js recorder, and they are not evidence of physical Logitech control, 1080 by 1920 output, MediaRecorder frame delivery, timestamp continuity, or A/V drift on any supported macOS release.

Regenerate them from this directory's parent:

```bash
npm run render:samples
```

Validation:

```bash
ffprobe -v error -show_entries format=duration,size -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels -of json samples/*.mp4
```

The checked-in preview resolution is 360 by 640 to keep repository size bounded. The application presets capture at 1080 by 1920.
