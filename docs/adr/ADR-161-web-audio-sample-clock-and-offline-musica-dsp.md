# ADR-161: Web Audio Sample-Clock Scheduling and Offline Musica DSP

## Status
Accepted

## Date
2026-07-18

## Context

A VJ instrument must stay musically stable while the UI renders complex scenes, records video, receives controller events, and performs background work. JavaScript timers, wall-clock timestamps, animation frames, and Tauri IPC all have variable latency. None is suitable as the authoritative musical clock.

The existing Musica algorithms are batch-oriented and use allocations and graph operations. They are valuable for imported-audio analysis and stem preparation but do not yet satisfy a real-time callback contract. Treating them as real-time DSP would risk audible dropouts and make performance impossible to bound.

## Decision

Use a 48 kHz `AudioContext` and its sample timeline as the sole authority for live transport, synthesis, track playback, automation, and capture audio.

The v1 engine has six canonical tracks: drums, bass, chords, lead, voice, and texture. Each track has an input, gain, stereo pan, and analyser feeding a master gain, dynamics compressor, master analyser, hardware destination, and `MediaStreamAudioDestinationNode` for recording.

### Implemented scheduling invariant

The v1 engine does not implement its own audio callback. A main-thread scheduler wakes every 25 ms and keeps 120 ms of native Web Audio events queued. Notes use `AudioScheduledSourceNode.start(time)` and timestamped `AudioParam` automation against `AudioContext.currentTime`. The browser's audio renderer, not the JavaScript timer, determines when a queued event becomes audible.

Sixteenth-note spacing is calculated as:

```text
secondsPerStep = 60 / (bpm * 4)
nextStepTime = previousStepTime + secondsPerStep
```

`Date.now()`, `performance.now()`, React state, controller timestamps, and `requestAnimationFrame()` MUST NOT determine note placement. They may request an action, update the display, or wake the look-ahead scheduler. UI state may lag the sound; it must never pull queued sound back to the UI clock. A resume or `AudioContext` state transition establishes a new transport epoch.

The 25 ms scheduler callback performs only bounded graph scheduling and state publication. It does not call Tauri, access a socket or filesystem, poll a provider, await a Promise, or decode media. Those operations happen in separate event handlers and can only change future control state.

If the scheduler wakes more than 20 ms after an unscheduled step, it advances over every missed step and resumes with at least 10 ms of positive lead. Missed notes are counted and shown in the performance UI; they are never burst into the present. The visible playhead is derived from `AudioContext.currentTime` rather than the next event in the look-ahead queue.

### AudioWorklet and native callback roadmap

An integer sample-frame event ledger and rational phase accumulator are the next timing step if long arrangements, tempo automation, streaming DSP, or native Core Audio require explicit sample positions:

```text
framesPerStep = sampleRate * 60 / (bpm * 4)
eventTime = transportEpoch + eventFrame / sampleRate
```

They are roadmap contracts, not a claim that the v1 main-thread scheduler already stores every event as an integer frame.

Any future custom `AudioWorkletProcessor.process()` or native audio callback MUST obey all of these rules after initialization:

- no Tauri `invoke`, socket, HTTP, filesystem, or other IPC;
- no Promise or `async` continuation;
- no lock or blocking wait;
- no logging;
- no unbounded loop;
- no heap allocation, buffer growth, or graph construction;
- no assumption that every render quantum has a fixed length; process the arrays supplied by the host.

Control changes would reach that future engine through preallocated messages or a bounded ring buffer and would be applied at a specified sample frame. Telemetry would travel in the opposite direction at no more than 60 updates per second and could be dropped under pressure. Audio always wins over telemetry.

**There is no IPC in a real-time audio callback.** V1 has no application-authored callback. Any future AudioWorklet or native callback keeps Tauri events, Logitech messages, provider polling, media recording, and React rendering on the control side.

### Implemented v1 import path

V1 imports a user-selected browser `File`, rejects it above 250 MB before `arrayBuffer()`, decodes it with `AudioContext.decodeAudioData()`, and then enforces 10 minutes, eight channels, and 192 kHz before installing the resulting `AudioBuffer` on one of the six tracks. Imported buffers begin at transport start or the next bar when loaded during playback, then loop on the same `AudioContext` timeline as synthesized sources. Arbitrary clip lengths are not time-stretched. No Musica Rust algorithm or Tauri offline-job command is invoked by this path.

### Offline Musica DSP roadmap

Musica separation remains deliberately offline when added. The target design is:

1. The user selects an input through a native dialog.
2. The Rust backend validates and decodes it outside the audio thread.
3. A bounded background task calls Musica analysis or separation.
4. Results are written to an application cache with a manifest containing source and output hashes, sample rate, channel count, and algorithm version.
5. The frontend loads completed stems into `AudioBuffer` objects and schedules them on the existing Web Audio clock.

That future command must not serialize large PCM arrays as JSON through Tauri. It returns job status and scoped asset identifiers, supports cancellation between algorithm stages, and defaults to one concurrent separation. None of those job commands is claimed as implemented in v1.

A later streaming Musica path requires a separate real-time-safe API with all memory allocated during `prepare`, a bounded worst-case processing time, and an AudioWorklet-compatible WASM or native audio transport. It cannot reuse the current batch API in a callback merely because it compiles to WASM.

### Performance targets

Targets use the built-in output on an Apple M1 at 48 kHz. Bluetooth latency is outside the application budget and is reported separately.

| Metric | Target |
|---|---|
| Scheduled event placement in an offline render | within 1 sample of the expected frame |
| Controller action to audible response, wired output | less than 35 ms p95 and 60 ms p99 |
| Future custom real-time processing cost | less than 40% of one render quantum p95 before promotion |
| Scheduler lateness | fewer than 1 late event per 10,000 scheduled events under the stress profile |
| Audible underruns | zero during a continuous 60-minute reference session |
| Six-track master output | no sample outside `[-1, 1]`; no NaN or infinity |
| Future offline job impact on live frame timing | less than 5 ms increase in audio action latency p95 before promotion |

## Alternatives Considered

### Use JavaScript timers as the transport

Timers are delayed by UI work, background throttling, and garbage collection. Rejected as a clock; retained only to fill the Web Audio look-ahead queue.

### Put synthesis and mixing in Rust through Tauri commands

Per-event or per-buffer IPC adds nondeterministic serialization and thread scheduling. Rejected. A future native Core Audio engine would require a dedicated lock-free audio bridge, not Tauri command traffic.

### Run current Musica separation in an AudioWorklet

The current algorithms have no proven allocation-free, bounded-time contract. Rejected until the explicit real-time gate above is met.

### Adopt Tone.js as the transport

Tone.js offers a capable abstraction but adds another timing model over the small six-track engine and makes a future sample-frame ledger harder to audit. Rejected for v1. Reconsider if arrangement and automation requirements exceed the local scheduler.

### Native AVAudioEngine or CPAL for all audio

This can reduce latency and improve device routing, but it splits the visual application from its audio graph and makes browser preview behavior diverge. Deferred behind the application boundary in ADR-160.

## Consequences

### Positive

- Musical timing is independent of render and React load.
- Live and directly imported tracks share one Web Audio transport.
- The offline roadmap can add Musica separation without placing it in live scheduling.
- The capture graph receives the exact master mix rather than microphone loopback.

### Negative

- No Musica analysis or separation command ships in v1; imported files are decoded directly in the webview.
- WKWebView controls the final hardware buffer size and output latency.
- The current floating-point step accumulator is less explicit than a future integer sample-frame ledger.

## Risks and Mitigations

The main failure mode is scheduler starvation during shader compilation, recording finalization, or a long JavaScript task. The 120 ms queue protects events already submitted to Web Audio. V1 skips unscheduled late steps instead of playing a burst and exposes the drop count. Automatic visual degradation beyond pixel ratio and dynamic look-ahead expansion remain release-hardening work.

Background suspension is handled as a transport interruption, not by attempting to catch up a backlog of notes. The UI requires a user gesture to initialize or resume the `AudioContext`, as required by browser autoplay policy.

## Rollback

The future offline Musica module can be omitted or disabled without changing the implemented local synth or direct imported-loop playback. If Web Audio cannot meet the reference latency on supported macOS versions, the roadmap integer sample-frame event ledger becomes the input contract for a future native engine.

## Acceptance Tests

1. An `OfflineAudioContext` render at 60, 112, and 200 BPM confirms scheduled transients occur at the requested Web Audio times; the future integer-frame engine must tighten this to one-sample placement.
2. A 60-minute six-track session on the reference M1 has zero audible underruns and fewer than one late event per 10,000 events while the visual stress scene runs.
3. Injecting a 50 ms main-thread block once per second for five minutes does not move already scheduled notes; any unscheduled missed steps increment telemetry and are skipped rather than burst together.
4. A source test confirms the v1 scheduler tick contains no Tauri `invoke`, fetch, socket, filesystem, decode, or awaited operation. The same test is mandatory for any future AudioWorklet callback.
5. Mute, solo, gain, pan, tempo change, stop, and retrigger are tested against requested `AudioContext` times.
6. Directly imported audio rejects the byte, duration, channel, and sample-rate limits; accepted clips start on a transport or bar boundary, loop, mute, solo, clear, and stop on the existing Web Audio graph without a Tauri or Musica job.
7. Before offline Musica DSP is promoted from roadmap, its separate gate proves cancellation, cache quota enforcement, bounded concurrency, manifest hashes, and less than 5 ms p95 impact on live action latency.
8. The output scan rejects NaN and infinity and confirms all master samples remain within `[-1, 1]` after the limiter/compressor path.

## References

- [W3C Web Audio API 1.1](https://www.w3.org/TR/webaudio-1.1/)
- [W3C AudioWorklet processing model](https://www.w3.org/TR/webaudio-1.1/#AudioWorklet)
- [Web Audio automatic scheduling recommendation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices)
- [Web Audio autoplay policy guidance](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Autoplay)
- ADR-143: HEARmusica real-time processing contract
