# Musica VJ local controller protocol

The companion opens one fresh Unix stream socket per action, writes one UTF-8 JSON line, and closes the connection. The application listens at `/tmp/musica-vj-${USER}.sock` and owns the socket with mode `0600`.

```json
{"v":1,"seq":42,"ts_ms":1721337600123,"action":"visual.intensity.delta","value":-1,"token":"64-lowercase-hex-characters"}
```

The fields are deliberately strict:

| Field | Invariant |
| --- | --- |
| `v` | Integer `1` |
| `seq` | Positive, strictly increasing `Int64` for this plugin process |
| `ts_ms` | Unix epoch milliseconds; receiver permits at most 30 seconds clock skew |
| `action` | Exact value from `ControlActions.Allowed` |
| `value` | Discrete actions use `0`; `track.trigger` uses an integer `0..5`; delta actions use `[-1, 1]` |
| `token` | Exactly 64 lowercase hexadecimal characters |

The bounded delta actions include master level, tempo, visual intensity, and the `visual.sculpture.delta`, `visual.motion.delta`, `visual.atmosphere.delta`, and `visual.ribbon.delta` artist controls.

The token is read for every action from `~/Library/Application Support/one.cognitum.musica.vj/controller.token`. The companion rejects symlinks, files over 1 KiB, and files readable, writable, or executable by group or other users. It never logs the token.

The application receiver should reject unknown JSON properties, lines over 4096 bytes, invalid UTF-8, replayed sequence numbers, stale timestamps, unlisted actions, and invalid tokens. It should remove the socket on clean exit and before binding a stale path.

The hardware callback only performs validation plus a bounded queue insertion. Capacity is 64 actions. A full or contended queue rejects new input. The background sender uses a 100 ms total connect and send deadline and does not retry, preventing duplicated transport or track toggles.
