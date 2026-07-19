# ADR-162: Logitech Actions SDK Companion and Authenticated Controller Bridge

## Status
Accepted

## Date
2026-07-18

## Context

The target desk controller is the Logitech MX Creative Console: an LCD Keypad plus a Dialpad with a rotary dial, roller, and buttons. Logitech exposes these controls through the Logi Actions SDK and Logi Plugin Service installed with Options+. The official SDK is not a Rust or Tauri API. Direct USB or HID access is undocumented, would compete with Options+, and would forfeit LCD action and Marketplace integration.

The application also needs a zero-install fallback so it remains usable while the official plugin is absent, under review, or incompatible with a future Options+ release.

## Decision

Build a separate .NET 8 C# companion using the official Logi Actions SDK. It is a thin controller adapter, not an audio process and not a general command runner.

The companion owns Logitech action declarations, adjustment events, and LCD symbols. The Tauri application owns all semantic actions. They communicate over an installation-token-authenticated Unix domain socket on macOS.

### Why C#

Logitech documents both C# and Node.js SDKs and describes C# as the advanced-feature path. The official C# setup targets .NET 8, packages and verifies through `LogiPluginTool`, and is hosted by Logi Plugin Service. Selecting C# avoids embedding a second Node runtime beside the Tauri webview and gives explicit typed adjustment and plugin lifecycle APIs.

### Protocol

The application listens on a per-user socket path:

```text
/tmp/musica-vj-<USER>.sock
```

The application validates `USER`, rejects an unsafe existing filesystem object at that path, binds a Unix stream socket, and sets it to mode `0600`. A 256-bit random installation token is created on first run and encoded as exactly 64 lowercase hexadecimal characters at:

```text
~/Library/Application Support/one.cognitum.musica.vj/controller.token
```

The token is a regular file owned by the effective user, mode `0600`, with one hard link. Both sides reject a symlink, unsafe permissions, wrong owner, wrong length, or invalid encoding. The token is never hard-coded, checked into the repository, or logged.

The protocol is versioned UTF-8 JSON with a maximum message size of 4 KiB. The companion opens a fresh connection for each action, writes one message, and closes it. A message is:

```json
{"v":1,"seq":42,"ts_ms":1721337600123,"action":"track.mute","value":0,"token":"64-lowercase-hex-characters"}
```

The Rust receiver validates the strict schema, protocol version, action-specific value range, exact installation-token equality using a constant-time comparison, a strictly increasing positive sequence, and no more than 30 seconds clock skew. Invalid UTF-8, unknown field, stale timestamp, duplicate or decreasing sequence, oversized message, wrong token, or unknown action closes that one connection without emitting an event.

Allowed actions are transport toggle, record, tap tempo, track selection, trigger, mute, solo, master delta, visual selection, visual intensity delta, and tempo delta. The companion cannot send shell commands, file paths, URLs, prompts, or provider requests.

V1 is one-way. Dynamic app-to-device state and LCD refresh require a versioned protocol extension and are roadmap work. No audio, project content, provider secret, or user media crosses the bridge.

The C# hardware callback attempts a non-blocking insertion into a 64-action FIFO and returns immediately. A background sender enforces a 100 ms total connect-and-send deadline. A full or contended queue drops the new event and increments a counter. Failed sends are not retried, because retrying a toggle could duplicate a transport or mute action.

Rust accepts and processes one complete message at a time in listener order. Spawning concurrent handlers could let sequence `N+1` reach replay validation before sequence `N`, incorrectly dropping a legitimate action. The Unix socket's bounded OS backlog absorbs the companion's short connection bursts, and every accepted connection has a 150 ms read deadline. With the companion's 100 ms total deadline and no retries, an incomplete client can delay but cannot indefinitely stall later valid actions.

### Options+ fallback

The application registers these function-key shortcuts while running:

| Key | Action |
|---|---|
| F13 | Transport toggle |
| F14 | Start or stop recording |
| F15 / F16 | Previous / next track |
| F17 / F18 | Mute / solo selected track |
| F19 / F20 | Previous / next visual |
| F21 / F22 | Visual intensity down / up |
| F23 / F24 | Master volume down / up |

Users can assign those keys to any MX Creative Console control in Options+. The fallback carries no dynamic LCD state and maps analog controls to repeated key presses, but it covers every critical live action without the companion. In-app keyboard and Web MIDI controls remain separate fallbacks.

### Performance targets

| Metric | Target |
|---|---|
| Hardware event to normalized app event | less than 20 ms p95, 40 ms p99 |
| Companion operation deadline | 100 ms maximum per message |
| Rust accepted-message read deadline | 150 ms maximum |
| Hardware callback blocking | zero blocking waits; bounded enqueue only |
| Queue capacity | 64 actions; drops observable |
| Reliability soak | 100,000 mixed events with zero unauthorized or replayed actions |

These numbers exclude the audio scheduling latency defined in ADR-161.

## Alternatives Considered

### Direct HID or USB access

This depends on an undocumented protocol, risks exclusive-device conflicts, loses Options+ and Marketplace behavior, and expands the native attack surface. Rejected.

### Function keys only

This is simple and robust but cannot expose high-resolution dial values, dynamic labels, stateful icons, or device feedback. Accepted only as the fallback.

### Node.js Actions SDK companion

The Node.js SDK is official and may reduce plugin boilerplate. It introduces another JavaScript runtime and is positioned by Logitech as the simpler path while C# exposes advanced features. Rejected for the primary companion; it remains a viable replacement behind the same protocol.

### Local TCP or unauthenticated WebSocket

Loopback ports are discoverable by other local processes and browser pages, and an unauthenticated endpoint could trigger recording or loud output. Rejected.

### Put .NET inside the Tauri process

Embedding the CLR or using fragile FFI would couple Options+ plugin lifecycle to audio and UI stability. Rejected.

## Consequences

### Positive

- Uses Logitech's supported customization and distribution path.
- Delivers high-resolution analog controls and branded LCD action symbols.
- A compromised or malformed plugin can invoke only an enumerated action set.
- The F13 through F24 profile prevents the plugin from becoming a launch blocker.

### Negative

- Development requires .NET 8, Options+, Logi Plugin Service, and physical hardware.
- The plugin has its own package and Marketplace review lifecycle.
- The installation token file is shared with the same-user companion and must remain mode `0600`.

## Risks and Mitigations

The main operational risk is an Options+ or Actions SDK update breaking the companion. The SDK package is pinned, a physical-device compatibility test is required before release, and the fallback profile is always documented and tested.

Same-user malware can potentially read the token from either process or from the user's account. Owner-only file and socket permissions, constant-time token comparison, sequence and timestamp replay defense, bounded messages, and a non-privileged action allowlist limit impact. The bridge never runs as root and never opens a network listener. Moving the token to a mutually accessible Keychain access group is a future hardening option, not part of v1.

Marketplace review time is outside the project's control. The plugin package is optional and may be distributed for development independently of the signed app, subject to Logitech's agreement.

## Rollback

Disable the socket listener and companion packaging with a build feature or runtime setting. The application continues with F13 through F24, keyboard, MIDI, and UI control. The normalized `ControlAction` contract remains unchanged if the companion later moves to Node.js or another supported SDK.

## Acceptance Tests

1. The official `LogiPluginTool` verifies the `.lplug4` package and Options+ lists it for both MX Creative Keypad and Dialpad.
2. All keypad buttons, dial rotation, roller rotation, and available Dialpad buttons produce the expected normalized actions on physical hardware.
3. A 100,000-event soak produces no unauthorized or replayed action, exposes any bounded-queue drops, and meets the latency budget.
4. Messages with the wrong installation token, stale timestamp, replayed sequence, unknown field or action, invalid value, or more than 4 KiB are rejected without reaching the frontend; an incomplete connection is closed at 150 ms.
5. The token file and socket are mode `0600`, owned by the current user, reject symlink substitution, and neither process runs with elevated privileges.
6. With the app closed, callbacks return within the 100 ms operation deadline and do not retry a failed toggle. Starting the app makes the next fresh action succeed without restarting Options+; a burst test confirms listener-order processing never rejects a valid increasing sequence through handler reordering.
7. With the plugin uninstalled, an Options+ F13 through F24 profile controls every critical live action.

## References

- [Logi Actions SDK documentation](https://logitech.github.io/actions-sdk-docs/)
- [Logi Actions SDK C# introduction and .NET 8 setup](https://logitech.github.io/actions-sdk-docs/csharp/plugin-development/introduction/)
- [Logi Actions SDK supported devices](https://logitech.github.io/actions-sdk-docs/supported-devices/)
- [Logitech Marketplace approval guidelines](https://logitech.github.io/actions-sdk-docs/marketplace-approval-guidelines/)
- [Microsoft .NET `UnixDomainSocketEndPoint`](https://learn.microsoft.com/en-us/dotnet/api/system.net.sockets.unixdomainsocketendpoint?view=net-8.0)
- ADR-166: Desktop threat model and secret boundaries
