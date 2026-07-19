# ADR-169: Lyria Paid Job Lifecycle, Assets, Analysis, and Provenance

## Status

Accepted

## Date

2026-07-18

## Context

A Lyria 3 Pro submission is a paid, non-deterministic operation. The Gemini Interactions documentation does not provide a create idempotency key, a Lyria-specific background-job API, or a provider-confirmed cancellation operation. Retrying after a connection fails can therefore create a second billable song even when Musica never received the first response.

Provider audio and text are also untrusted remote data. Musica needs the original response for audit, locally measured media properties for playback and visualization, an explicit cost record, and provenance fields that distinguish provider claims from locally verified evidence.

## Decision

Run Lyria through a local asynchronous `GenerationManager` in the Tauri Rust process. Submission validates and reserves one published-price attempt, creates an opaque local task, and acknowledges the UI with `queued` state before the network operation completes. The local states are:

```text
queued -> processing -> complete
                    -> failed
queued -> cancelled
processing -> processing with cancellation requested -> complete after cancellation or failed
```

The registry allows at most two executing generations. It is intentionally process-local in V1: task state, reservations, and `clientRequestId` deduplication do not survive an app restart. Durable restart recovery and a transactional workspace budget ledger remain release-hardening work.

### Cost reservation and attempts

The V1 published unit price is recorded as USD 0.08 using integer microdollars. A request must provide `maxCostUsd`, exactly one candidate, exactly one maximum attempt, and a budget at least USD 0.08 but no larger than the installation ceiling. React requires an acknowledgement, and Rust independently opens a native warning dialog with maximum cost, duration, format, mode, and a fingerprint of the exact request before reserving. The process-local registry reserves before dispatch and rejects a request when cumulative reserved plus recorded cost would exceed the ceiling. The default ceiling is USD 0.32 and can be changed with `MUSICA_CREATIVE_MAX_GENERATION_USD`.

The adapter's ledger allows one paid dispatch for a local task, and its `reqwest` client explicitly disables retries. The HTTP request deadline defaults to 600 seconds and is configurable only from 60 through 900 seconds. It does not automatically retry rate limits, server errors, interrupted uploads, or interrupted responses because Google does not document create idempotency for this operation. A transient failure is shown as a new user decision. An aesthetically poor but technically complete output is always a new paid candidate. Unit tests cover the one-dispatch transition, but a mock or real HTTP transport test that counts outbound `POST` calls is still required before treating the one-POST claim as end-to-end evidence.

An ambiguous network failure records `ambiguous_paid_request_outcome` and conservatively reports the USD 0.08 potential cost. This is safer than a duplicate automatic call. The receipt records the versioned published price, not an account invoice. `providerBillingVerified=false` remains explicit until a billing reconciliation API or account export is integrated.

`clientRequestId` deduplicates repeated UI submission messages only within the running process. It is not claimed as upstream idempotency. The largest remaining spend risk is a user restarting Musica after an ambiguous response and manually resubmitting the same creative request.

### Cancellation

Cancellation before dispatch aborts the local job, marks it cancelled, and releases its reservation. Once provider dispatch starts, Musica sets `cancellationRequested=true` but deliberately leaves the task in `processing`; the frontend continues polling because aborting the local future would discard a potentially billable response without cancelling Google work. `providerCancelConfirmed` remains false. If a valid response later arrives, the immutable asset is retained, the task records `completedAfterCancel=true`, and the UI does not auto-load it.

Cancellation is never described as a guaranteed refund.

### Response validation and immutable assets

The adapter accepts a successful JSON response only from the fixed Gemini endpoint and bounds it to 96 MiB. It scans only `model_output` steps, permits up to 1 MiB of text, requires exactly one audio block, bounds decoded audio to 72 MiB, validates Base64, and requires MP3 or WAV MIME and signature agreement. It then inspects the encoded file for duration, sample rate, channels, and codec, requires stereo output, and rejects actual duration above 184 seconds. A valid result shorter than `max(75% of requested duration, 30 seconds)` is preserved with `output_shorter_than_requested`, which the UI presents as a warning rather than a provider failure.

For every successful response, Rust creates a private application-data directory with mode `0700` and new files with mode `0600`:

```text
generated/<local-task-id>/
  <sha256>.<mp3|wav>
  provider-response.json
  receipt.json
```

Files use create-new semantics and are not overwritten. The original successful provider response and audio are retained exactly as received, and the receipt records their separate SHA-256 hashes. A dispatched failure completes its create-and-sync attempt for `failure-receipt.json` before publishing terminal state. A stored receipt contains the typed error, validated Google request ID when available, dispatch/cancellation state, reservation, optional conservative charge, pricing version, and `providerBillingVerified=false`; a failed receipt write is reflected in the terminal error suffix. The raw provider error response is not preserved. Authentication, unavailable model, quota, rate limit, and safety failures release the reservation. Service, malformed response, unsupported media, oversize, storage, and ambiguous post-dispatch failures conservatively record a possible charge. The webview receives audio bytes through a scoped Tauri command only after the local task is complete. It never receives the provider URL, API key, raw response, or filesystem path.

This is immutable application asset staging, not yet a portable Musica project asset library. Project save, reopening after process restart, transform lineage, archive export, retention limits, and garbage collection require a later project schema decision.

The three successful bundle files use create-new writes and per-file sync, but V1 does not atomically commit the directory as one transaction. A late disk failure can therefore leave a private partial task directory beside the terminal storage error; automatic recovery and garbage collection are deferred.

### Local decode and analysis

The frontend checks MP3 or WAV encoded metadata again, decodes through Web Audio into 32-bit floating-point channel buffers, and measures:

- duration, decoded sample rate, and channel count;
- waveform buckets and phase-safe, BS.1770-style integrated loudness in LUFS;
- BPM, beat grid, and onset map;
- low, mid, and high-frequency energy plus spectral centroid;
- bounded chroma-based musical key estimation with confidence and tonal-contrast gates;
- probable contiguous sections;
- the implemented mapping of bass to camera displacement, beats to radial pulse, high-frequency energy to particle count, and measured sections to terrain/bloom/tunnel scene transitions.

Analysis values come from decoded PCM, not the prompt or provider text. The V1 pipeline does not persist a resampled 48 kHz PCM working file and does not isolate media decoding in a constrained subprocess. Those two gaps must remain visible in the implementation specification.

### Provenance and rights

`receipt.json` records schema version, provider, model, provider request ID, local task ID, prompt or prompt hash according to retention policy, output and response hashes, UTC generation time, requested duration, requested and detected media properties, requested language, fixed-price reservation and charge, pricing and optional terms versions, rights declaration, expected SynthID and C2PA status, and input-asset hashes.

`MUSICA_CREATIVE_RETAIN_PROMPTS=false` omits prompt text from the success receipt while retaining its SHA-256 hash. It does not redact `provider-response.json`, which may include generated lyrics, structure, safety details, or provider-echoed sensitive content. The request always sets `store: false` at Google. Musica does not log prompt or response bodies through the provider command surface. The frontend also stores a mutable, summarized terminal receipt for success, failure, or cancellation, omits waveform/onset/beat arrays, and enforces 500-entry and 2 MiB limits; it is not authoritative provenance.

Google states that Lyria outputs include SynthID and support C2PA. V1 records `synthidExpected=true`, `c2paExpected=true`, and `c2paStatus=preserved_unverified`. Retaining the provider response is not C2PA extraction or cryptographic validation, and Musica does not locally detect SynthID. The app must not label an asset authenticated, copyrightable, royalty-free, or free of infringement risk. User-supplied non-empty lyrics require an affirmative rights declaration.

## Alternatives Considered

### Retry transient provider errors twice

Rejected for V1. A retry limit alone does not prevent duplicate paid output when the first response is ambiguous. Automatic retries require documented upstream idempotency or a durable reconciliation mechanism.

### Keep a long HTTP request in React

Rejected. It couples navigation and webview lifetime to paid work and exposes too much provider state to an untrusted boundary.

### Store only decoded PCM

Rejected. It discards the provider's original encoded artifact, exact response, and embedded provenance material.

### Trust provider metadata instead of inspecting media

Rejected because current Google pages disagree on sample rate and provider text is not measured audio evidence.

### Claim C2PA verified when expected

Rejected. A provider capability statement is different from a parsed, cryptographically valid manifest on a specific asset.

## Consequences

### Positive

- The UI remains responsive while long generation runs outside the render path.
- One paid submission produces at most one automatic provider POST.
- Original bytes, response, cost basis, and hashes remain available for audit.
- Visualization derives from measured sound rather than prompt intent.

### Negative

- Rate limits and server failures require explicit paid resubmission.
- App restart loses the in-memory task registry and deduplication map.
- Local receipt cost is a published-price record, not verified provider billing.
- Web Audio remains part of the hostile-media decode boundary.

## Risks and Mitigations

The largest failure mode is duplicate spend after an ambiguous network result, especially after restart because the ledger and deduplication map are process-local. Unit tests prove that a task can enter paid dispatch once and `reqwest` retries are disabled; an injected HTTP transport test must still count real outbound `POST` attempts across success and error paths. The UI describes the ambiguous USD 0.08 as potentially charged and requires a fresh explicit budget for another candidate.

The largest integrity risk is treating a mutable UI receipt as authoritative. The private create-new Rust receipt and raw-response hash are the authoritative V1 evidence. The local-storage summary is a convenience index and cannot establish provenance by itself.

## Rollback

Disabling the adapter prevents new paid calls without deleting any completed private assets. A model outage changes provider status to unavailable while local synth, imported audio, visual performance, live recording, and export continue. Retained assets are not automatically deleted during rollback.

## Acceptance Tests

1. A submit call returns a queued local task without waiting for provider completion, and at most two jobs execute concurrently.
2. Repeated `clientRequestId` values return the same running-process task, the ledger starts paid dispatch once, and `reqwest` has retries disabled. A transport-level fixture must still prove exactly one outbound provider `POST` on success, rate limit, server failure, and ambiguous network failure.
3. The adapter rejects a missing budget, budget below USD 0.08, budget above the installation ceiling, candidate count other than one, and maximum attempts other than one before dispatch.
4. Queued cancellation releases the reservation; processing cancellation keeps the task processing and the UI polling, never claims provider confirmation or refund, and a late response records completion after cancellation and preserves the asset without auto-loading it.
5. Empty or multiple audio blocks, invalid Base64, oversized JSON/text/72 MiB audio, MIME-signature mismatch, invalid MP3/WAV, non-stereo media, and duration above 184 seconds fail closed; the internal short-output threshold produces a warning instead of rejection.
6. A successful fixture creates private, non-overwriting audio, raw-response, and receipt files with matching SHA-256 values.
7. Prompt-retention-off fixtures omit plaintext prompt and keep its hash; bundle and log scans contain no Gemini key.
8. Receipt fixtures distinguish expected SynthID/C2PA from unverified presence and set provider billing verification false.
9. Decoded PCM tests measure a 120 BPM fixture within two BPM and produce finite waveform, loudness, beat, onset, spectral, section, and visualization data. A separate 180-second, 48 kHz stereo fixture must complete the full local analysis in under five seconds on the CI runner.
10. Real paid MP3/WAV generation, account billing reconciliation, C2PA cryptographic validation, decode-process isolation, restart recovery, external generation latency, and physical-Mac performance remain credentialed or future gates rather than pull-request CI claims.

## References

- [Google AI for Developers: Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Google AI for Developers: Generate music with Lyria 3](https://ai.google.dev/gemini-api/docs/music-generation)
- [Google AI for Developers: Lyria 3 Pro Preview model](https://ai.google.dev/gemini-api/docs/models/lyria-3-pro-preview)
- [Google AI for Developers: Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Google DeepMind: Lyria 3 model card](https://deepmind.google/models/model-cards/lyria-3/)
- ADR-164: Governed creative AI provider boundary and asset provenance
- ADR-166: Desktop threat model and security boundaries
- ADR-168: Lyria provider, routing, and prompt contract
