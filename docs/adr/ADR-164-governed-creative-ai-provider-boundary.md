# ADR-164: Governed Creative AI Provider Boundary and Asset Provenance

## Status
Accepted

## Date
2026-07-18

## Lyria 3 Pro amendment

ADRs 168 and 169 supersede this record's generic V1 limitations for the `lyria_3_pro` path only. The generic async partner and disabled Suno paths remain as described below.

The Lyria adapter is now implemented but remains off by default. It uses a fixed Gemini Interactions endpoint and `lyria-3-pro-preview`, consumes `GEMINI_API_KEY` only in Rust application code, requires a native Rust-owned USD 0.08 confirmation dialog, returns a local asynchronous task, provides conservative cancellation semantics, validates MP3/WAV bytes, and stages original audio, the exact provider response, and create-new private success or failure receipts. Its `reqwest` client has retries explicitly disabled and its in-process ledger permits one paid dispatch because Lyria create idempotency is undocumented.

This is not completion of every target-governance item in this ADR. The Lyria job registry, cost reservation, and `clientRequestId` deduplication are process-local. They do not survive restart and do not form a transactional daily or workspace ledger. C2PA and SynthID are recorded as expected but unverified, provider billing is not reconciled, image/PDF input is rejected, and Web Audio decoding is not an isolated media process. The frontend local-storage receipt remains a convenience index; the private Rust receipt is the authoritative V1 record.

## Context

Creative music APIs can accelerate a performance from a text concept to playable source material. They also introduce material business risk: credentials, variable cost, provider outages, changing models, output-rights conditions, geographic restrictions, artist-imitation policy, and uncertain provenance.

The live instrument must not depend on a provider being online. Provider-specific response formats must not leak into projects or the Web Audio engine. In particular, a third-party service that mimics a consumer product is not equivalent to an authorized developer API.

As of this decision, Google documents Lyria 3 and experimental Lyria RealTime through the Gemini API, and ElevenLabs documents a paid Eleven Music API. No first-party Suno developer contract or public developer API documentation has been recorded for this repository. Suno integration therefore cannot be represented as generally available or supported.

## Decision

Create a creative-provider boundary in the Tauri Rust backend. The application remains fully functional with every cloud provider disabled.

### Implemented generic-partner v1 subset

The generic async partner originally implements three Tauri commands: `creative_provider_status`, `creative_generate`, and `creative_generation_status`. A request contains a bounded prompt, duration, instrumental flag, and optional integer seed. A task contains opaque ID, provider, model when reported, state, optional title, and an allowlisted HTTPS audio URL. The shared request DTO now also carries the optional fields required by ADR-168; the generic partner ignores fields it does not support.

The standard build is disabled until `MUSICA_CREATIVE_ENABLED=true` and a valid Rust-process environment token are supplied. It permits two concurrent calls, uses HTTPS port 443 only, disables redirects and system proxies, bounds connect time to five seconds, total request time to 30 seconds, and JSON responses to 512 KiB. Runtime configuration cannot expand the compile-time exact-host allowlist. The built-in generic async partner host is `api.cognitum.one`.

The generic partner does not implement cancel, persistent download staging, cost estimates, progress percentages, idempotent retry, project manifests, output-content hashing, C2PA validation, or rights assertions. A completed task is repolled by ID in Rust; its allowlisted URL is fetched with the proxy-disabled, redirect-disabled client, capped at 64 MB while streaming, required to have an `audio/*` content type and supported file signature, and returned to the frontend as raw IPC bytes. Generic binary MIME types such as `application/octet-stream` fail closed. The webview never fetches the URL directly.

The frontend stores a mutable convenience index of at most 500 receipts and 2 MiB in local storage. It writes a terminal summary for successes, failures, and cancellations. Lyria analysis is summarized with counts instead of retaining full waveform, onset, and beat arrays. A summary can contain task ID, provider/model, SHA-256 prompt hash, creation time, cost/provenance flags, encoded metadata, analysis summary, optional SHA-256 URL hash for the generic provider, and optional terms version. Signed URLs are never persisted. This index is not append-only project provenance and does not prove that output bytes were retrieved or remained unchanged.

### Target provider contract

Before official Lyria, Eleven Music, or a user-facing partner provider is promoted, extend the boundary to these versioned operations:

```text
provider_status(provider)
generation_submit(request)
generation_status(task_id)
generation_cancel(task_id)
generation_import(task_id)
manual_asset_import(path_token, provenance)
```

A target normalized request adds requested output class and explicit maximum cost. A target task adds progress when reported, estimated cost, output metadata, and typed failure. Provider-native fields stay in a private receipt extension and never become required track fields.

The target ingestion path moves audio retrieval, size validation, hashing, provenance preservation, and local staging into Rust so the webview receives a scoped local asset rather than a remote URL. Provider keys remain Rust-only in both the implemented and target paths.

### Provider policy

| Provider path | Release state | Enablement condition |
|---|---|---|
| Local synth and user-created/imported audio | Enabled | Always available offline |
| Generic async partner | Implemented, off by default | Explicit environment opt-in, token, exact compiled host allowlist |
| Google Lyria 3 Pro | Implemented preview, disabled by default | `MUSICA_CREATIVE_ENABLED=true`, `MUSICA_CREATIVE_PROVIDER=lyria_3_pro`, Rust-only `GEMINI_API_KEY`, explicit USD 0.08 request budget |
| Google Lyria RealTime | Experimental future | All Lyria 3 gates plus disconnect, steering, and session-cost controls |
| Eleven Music | Future, disabled by default | Official paid API, plan rights review, output validation, budget and retry tests |
| Suno partner adapter | Compiled and runtime disabled | Written partner authorization, official endpoint specification, credential flow, and legal approval |
| Manual Suno export import | Generic file import is enabled | V1 records filename only; provenance and rights assertion are target work |

The Suno partner adapter MUST NOT call an unofficial wrapper, scrape `suno.com`, automate a browser session, reuse consumer cookies, or ask for a user's Suno password. Until all enablement conditions are present, status returns `Unavailable("partner integration not configured")` and no Suno hostname is in the production network allowlist.

Manual import is deliberately provider-neutral. V1 can load a file exported from Suno, a DAW, a field recorder, or another service into a track, but it does not assert ownership or collect a rights declaration. Source, rights assertion, content hash, and embedded-provenance preservation are required before the target project provenance contract is called complete.

### Governance controls

Implemented generic-partner controls are provider opt-in, bounded duration, two concurrent calls, fixed timeouts, no redirects, exact hosts, strict task IDs, bounded JSON, and a 64 MB streamed audio cap with signature checking. The target governance gate adds:

- maximum downloaded response: 250 MB;
- maximum two submit attempts using a provider idempotency key when available;
- submit timeout: 10 seconds;
- polling backs off from 1 to 30 seconds with a 15-minute task deadline;
- confirmation required above an estimated USD 2 per request;
- default daily local budget: USD 25, configurable downward or upward by the user;
- at most two cloud generation tasks in flight;
- provider kill switch available in local configuration without a new project migration.

For Lyria only, ADR-169 implements a React acknowledgement plus native Rust cost confirmation, local cancellation semantics, persistent private staging, success and failure receipts, and content hashes. Post-dispatch cancellation leaves the paid task processing and the UI polling because there is no provider cancel operation. Durable daily/workspace budgets, restart recovery, provider-confirmed cancellation, verified billing, and safe automatic retry remain unimplemented. The generic partner contract does not inherit Lyria's stronger controls. Raw prompts and generated audio are never sent to Musica telemetry; telemetry is off by default.

### Provenance receipt

The target provenance design gives every generated or manually imported asset an append-only project receipt and exportable `.musica-receipt.json` sidecar. It includes:

| Field | Purpose |
|---|---|
| Receipt schema version and UUID | Migration and stable reference |
| Source kind | `local`, `manual-import`, or official provider identifier |
| Provider task ID and model | Audit and support |
| UTC creation and retrieval timestamps | Timeline |
| SHA-256 prompt hash | Correlation without forcing prompt disclosure |
| Optional encrypted/local prompt | Reproduction when the user opts in |
| Normalized request and seed | Creative settings |
| Provider terms/version reference when available | Decision context, not a legal guarantee |
| Source URI host, never embedded credentials | Origin |
| Input and output SHA-256 hashes | Integrity and lineage |
| User rights assertion and intended use | Governance record |
| Transform history and parent asset hashes | Derived-work lineage |
| Preserved C2PA data, if present | Standards-based provenance |

This table remains a roadmap project schema, not the lightweight local-storage receipt or Lyria's provider-specific private receipt. Musica VJ never fabricates a C2PA signature or labels an unsigned asset as authenticated. When C2PA validation is added, a provider or imported manifest is preserved and its validation status is recorded. A receipt is evidence of application workflow, not proof of copyright ownership.

## Alternatives Considered

### Call providers directly from the React frontend

This exposes long-lived credentials to the webview and permits CSP or injected-code failures to spend money. Rejected.

### Build a generic OpenAI-compatible music endpoint

Music generation has provider-specific asynchronous jobs, streaming, formats, policy, rights, and costs. A nominally generic HTTP shape would hide rather than remove those differences. Rejected; normalize lifecycle and provenance, not every parameter.

### Ship an unofficial Suno API wrapper

Unofficial wrappers commonly depend on cookies, reverse-engineered endpoints, or an intermediary service. This creates credential, continuity, and contractual risk disproportionate to optional functionality. Rejected.

### Make one cloud provider mandatory

This would turn a live instrument into a network-dependent client and make old projects contingent on one vendor. Rejected.

### Store only the generated audio

This is simplest but cannot answer which model, prompt, source, rights assertion, or terms produced a clip. Rejected for generated content.

## Consequences

### Positive

- Local performance and project playback remain sovereign and offline.
- Official providers can be added without changing track or project APIs.
- The target contract makes cost, terms, and provenance visible before content enters a social export; v1 records only lightweight generation metadata.
- A Suno partnership can be enabled later without normalizing unauthorized access today.

### Negative

- Initial releases may expose no one-click cloud generator.
- Completing the target receipts adds project metadata and requires schema migration discipline.
- A normalized provider surface cannot expose every vendor feature immediately.

## Risks and Mitigations

The largest uncertainty is commercial-use scope as provider plans and terms change. Adapters are disabled by default, and official-provider promotion requires a dated terms review plus the target receipt. The UI must say that lightweight v1 metadata is not provenance or legal advice and must not claim universal commercial clearance.

Provider model and endpoint drift is contained in adapters and contract fixtures, but Lyria provider status is a static local-configuration result and does not prove live model, account, region, quota, or preview availability. The generic partner path treats remote bytes as hostile in Rust but does not persist them in a hashed private cache. Lyria stages create-new private audio, raw provider response, hashes, and a receipt, but decoding still occurs in Web Audio. Disabling plaintext prompt retention does not redact the raw provider response, which can contain generated lyrics, structure, safety detail, or provider-echoed confidential content. The generated directory therefore remains privacy-sensitive. Published-price records can be wrong when pricing changes, so provider account quota and billing remain authoritative.

## Rollback

Any adapter can be disabled without modifying local tracks. Existing decoded buffers remain playable for the current session, while persistable hashed assets are target work. If a provider changes terms or security posture, new submissions stop while manual import and the local synth continue.

## Acceptance Tests

1. With no credentials and no network, synth, direct file import, performance, and live export work; provider status reports a typed unavailable reason.
2. A frontend bundle and memory/log scan contains no provider secret.
3. Implemented validation rejects prompt, duration, task ID, model, token, JSON or audio size, audio signature, content type, redirect, non-HTTPS, non-443, credentialed URL, and non-allowlisted-host violations.
4. The two-permit concurrency limit and five/30-second network timeouts are covered by Rust tests.
5. The lightweight receipt hashes the prompt, summarizes large analysis arrays, and enforces both 500-entry and 2 MiB limits without storing the provider token.
6. The Suno adapter makes zero network call in a standard build and remains unavailable unless partner-only compile-time host and runtime opt-in gates are both supplied.
7. Lyria-specific tests prove an explicit USD 0.08 budget, native-dialog request fingerprinting, one paid-dispatch state transition, process-local request deduplication, pre- and post-dispatch cancellation state, a 72 MiB audio cap, output and response hashes, immutable success/failure receipts, rights declaration for supplied lyrics, and expected-but-unverified C2PA state. A mock or real HTTP transport test that counts outbound `POST` calls, durable budgets, restart recovery, billing reconciliation, verified C2PA, manual-import rights assertions, and provider-safe retry remain promotion gates.

## References

- [Google Gemini API: Generate music with Lyria 3](https://ai.google.dev/gemini-api/docs/music-generation)
- [Google Gemini API: Lyria RealTime music generation](https://ai.google.dev/gemini-api/docs/realtime-music-generation)
- [Google Gemini API: Live Music WebSocket reference](https://ai.google.dev/api/live_music)
- [ElevenLabs Music API quickstart](https://elevenlabs.io/docs/eleven-api/guides/cookbooks/music)
- [ElevenLabs Music compose endpoint](https://elevenlabs.io/docs/api-reference/music/compose)
- [C2PA technical specification 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html)
- [Suno product and plan information](https://suno.com/)
- ADR-166: Desktop threat model and secret boundaries
- ADR-168: Lyria 3 Pro provider, routing, and prompt contract
- ADR-169: Lyria paid job lifecycle, assets, analysis, and provenance
