# ADR-174: Restream native live output

- Status: Accepted
- Date: 2026-07-19

## Context

Musica can save social captures and expose a browser-source URL, but a live set also needs direct delivery to external broadcast systems. Restream accepts encoder input over RTMP or RTMPS and expects H.264 video, AAC audio, and a two-second keyframe interval for broad destination compatibility.

Web views do not provide a portable RTMPS publisher. Sending high-rate media through JSON IPC would also add unacceptable allocation and serialization overhead.

## Decision

The desktop application provides a session-scoped Restream encoder with two sources:

| Source | Video | Audio |
|---|---|---|
| `Visual + Audio` | Direct 1920x1080 Three.js canvas capture | Musica master mix |
| `Entire UI + Audio` | User-approved operating-system window capture | Musica master mix |

The frontend records the selected source with `MediaRecorder` and sends one-second chunks to Rust using Tauri raw binary IPC. The native provider writes those bytes to an FFmpeg stdin pipe. FFmpeg decodes the webview container and publishes:

- H.264 `yuv420p` video using a low-latency preset;
- 4.5 Mbps CBR-style video rate control;
- keyframes every two seconds;
- 48 kHz stereo AAC at 192 kbps;
- FLV over an operator-supplied official `restream.io` RTMP or RTMPS ingest URL.

The operator pastes the server URL and stream key from the Restream Encoder/RTMP setup. The key:

- is masked in the UI;
- is never persisted;
- is cleared from React state after the encoder starts;
- is not returned in status or error payloads;
- is not logged;
- is accepted only with an official `restream.io` host.

Musica requires FFmpeg on the desktop PATH. The native status command reports availability before `Go Live`. The browser build does not attempt direct RTMPS publishing and retains the clean OBS/browser-source handoff.

## Consequences

- Operators can stream the clean visual program or the complete control surface without an intermediate compositor.
- Restream handles downstream multistream destinations, events, monitoring, and recordings.
- Full-UI capture depends on operating-system screen-recording permission and webview `getDisplayMedia` support.
- FFmpeg is an explicit runtime dependency until signed platform-specific sidecars are added to release bundles.
- MediaRecorder input is transcoded once by FFmpeg; this costs CPU but gives Restream-compatible H.264/AAC output across webview container differences.
- Abrupt process termination may end a stream without a final encoder flush. Restream disconnect protection remains an account-side concern.

## Acceptance tests

1. Non-Restream hosts and malformed stream keys are rejected before process creation.
2. A second encoder cannot start while one is active.
3. Media chunks use raw binary IPC and have a bounded per-chunk size.
4. Stop closes the input pipe and terminates the FFmpeg process.
5. Status never contains the stream key or destination URL.
6. Program capture includes exactly one video track and the Musica master audio track.

## References

- [Restream RTMPS encoder setup](https://support.restream.io/en/articles/8523770-encrypt-your-stream-with-rtmps)
- [Restream encoder settings](https://support.restream.io/en/articles/111656-connect-obs-to-restream)
- [Restream API guide](https://developers.restream.io/guide)
- [Tauri request body and raw IPC](https://docs.rs/tauri/latest/tauri/ipc/struct.Request.html)
- ADR-165: Live social capture and deterministic export
- ADR-166: Desktop threat model and security boundaries

