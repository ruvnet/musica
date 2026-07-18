# ADR-168: Lyria 3 Pro Provider, Routing, and Prompt Contract

## Status

Accepted

## Date

2026-07-18

## Context

Musica needs an optional provider for complete songs while retaining its offline synthesizer, imported-audio playback, Three.js visuals, performance controls, recording, and export. Google documents `lyria-3-pro-preview` as a public-preview Gemini model for complete songs, `lyria-3-clip-preview` for 30-second clips, and `lyria-realtime-exp` for interactive instrumental streaming. These surfaces have different capabilities, costs, and lifecycle semantics and must not be presented as interchangeable.

Google's current documentation describes Lyria 3 Pro text and image input, up to approximately three minutes of output, MP3 and WAV output on the Gemini Developer API surface, generated textual content alongside audio, SynthID watermarking, and C2PA support. The generation guide and model page disagree on whether provider audio is 44.1 or 48 kHz. Musica therefore cannot infer media facts from a prompt or model name.

The model is a preview. Its identifier, price, duration ceiling, availability, response schema, terms, and regional access can change without a Musica release. Provider functionality must fail closed and remain separable from the project and performance engines.

## Decision

Implement a dedicated, disabled-by-default Lyria 3 Pro adapter behind the Rust provider boundary from ADR-164. Select it only when all three startup conditions are present:

```text
MUSICA_CREATIVE_ENABLED=true
MUSICA_CREATIVE_PROVIDER=lyria_3_pro
GEMINI_API_KEY=<credential available only to the Tauri process>
```

The adapter uses the exact endpoint `https://generativelanguage.googleapis.com/v1beta/interactions`, model `lyria-3-pro-preview`, and `x-goog-api-key` header. It sends `store: false`, does not accept a runtime host override, requires HTTPS, disables redirects, system proxies, and `reqwest` retries, and never returns the key or upstream headers to the webview. The request deadline defaults to 600 seconds and `MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS` may set it only within 60 through 900 seconds. Status reports configuration readiness; it is not a live provider capability check and cannot prove current model, account, region, or quota availability.

### V1 capability envelope

| Capability | V1 decision |
|---|---|
| Text to complete song | Implemented for 31 to 180 seconds in the UI, with a 184-second defensive provider ceiling |
| Instrumental mode | Implemented through explicit prompt constraints |
| User-supplied lyrics | Implemented; non-empty lyrics require `rightsDeclared=true` |
| Timed song structure | Implemented; up to 32 strictly increasing sections within requested duration |
| Tempo | Implemented as an optional, independently validated 60 to 200 BPM prompt constraint |
| Language | Implemented for the eight languages currently documented by Google: English, German, Spanish, French, Hindi, Japanese, Korean, and Portuguese |
| MP3 | Implemented as the default response |
| WAV | Implemented with an explicit audio response-format request, a WAV prompt constraint, and strict WAV response validation |
| Image conditioning | Deferred to V2; the V1 adapter rejects every reference asset before spending money |
| PDF conditioning | Deferred experimental Vertex capability; not verified for this Interactions API adapter and rejected in V1 |
| Lyria 3 Clip | Routing decision is modeled but the provider is disabled in V1 |
| Lyria RealTime | Deferred experimental WebSocket provider; no V1 streaming or session-cost claim |
| Multiple candidates | Deferred; V1 requires exactly one candidate per request |

Image conditioning will not be enabled merely because the shared request DTO permits reference metadata. Its implementation requires locally authorized asset reads, signature and MIME checks, hashing, bounded upload, capability detection, and new cost fixtures. PDF input requires a separate Vertex-specific decision because the Gemini Interactions behavior is not assumed to match Vertex documentation.

### Structured composition and prompt compilation

The frontend represents creative intent as a `StructuredComposition` rather than treating free-form prose as the durable contract. It can contain duration, genres, BPM, time signature, moods, instruments, vocals, lyrics, timed sections, social hook, production style, dynamic progression, visual synchronization cues, and output format.

The deterministic frontend compiler validates the structured composition and produces the provider-ready natural-language prompt exactly once. It preserves explicit timestamps, lyrics, instrumental constraints, language, response format, and the social hook window. Rust treats that prompt as the final provider input and does not append a second copy of lyrics, structure, tempo, duration, or vocal instructions. At the trust boundary it independently validates the typed duration, BPM, language, lyrics and rights declaration, section ordering, output format, candidate count, attempt count, and cost budget before submitting the prompt. The duplicated typed fields are validation and provenance evidence, not a second prompt compiler. Provider text is an output artifact, not a substitute for measured local audio analysis.

### Routing

The routing contract is:

| Request | Route | V1 availability |
|---|---|---|
| Complete song longer than 30 seconds, vocals, multiple sections, or a durable Musica project | Lyria 3 Pro | Available when explicitly configured |
| Exactly 30 seconds, loop, preview, or multiple low-cost social candidates | Lyria 3 Clip | Modeled, unavailable |
| Continuous interactive steering during performance | Lyria RealTime | Modeled, unavailable |
| No provider, provider unavailable, or offline session | Local synthesis/import | Always available |

Routing does not silently fall back from a paid request to a different paid model. The UI identifies the selected model, published unit price, and locally configured capability before submission, but must not call that status a live availability probe. The React budget acknowledgement is followed by a native Rust-owned confirmation dialog showing maximum charge, duration, format, mode, and a fingerprint of the exact request. The Lyria path does not accept a seed because Google does not document deterministic seed control for this model.

### Output normalization

The adapter walks all `model_output` content blocks and requires exactly one supported audio block. Text blocks are preserved and classified as lyrics or structure when recognizable. Missing lyrics or structure is not treated as an audio failure because these fields are not guaranteed as dedicated response properties.

Before import, Rust bounds decoded audio to 72 MiB, validates Base64, declared MIME, file signature, stereo channels, encoded sample rate, and encoded duration, and rejects output longer than 184 seconds. A result shorter than `max(75% of requested duration, 30 seconds)` is valid but is tagged `output_shorter_than_requested` for an internal UI warning. Rust inspects WAV headers or MP3 frames and does not assume 44.1 or 48 kHz. After webview decode, Musica analyzes decoded PCM locally. Waveform, BPM, beat, key, sections, and spectrum use the channel with greatest energy so anti-phase stereo does not collapse during analysis; BS.1770-style loudness K-weights each channel independently, combines energy, and applies absolute and relative gates. The resulting mapping is concrete: bass displaces the camera, detected beats create radial pulse, high-frequency energy changes particle count, and measured section types switch terrain, bloom, or tunnel scenes during one-shot playback. Web Audio supplies the 32-bit floating-point working buffers; an explicit offline 48 kHz archive conversion pipeline and a constrained native decoder are deferred.

## Alternatives Considered

### Call the Gemini API from React

Rejected. The key would be recoverable from the web bundle or webview and a frontend compromise could make paid calls.

### Put Lyria behind the generic partner response schema

Rejected. The Interactions content-block response, fixed pricing, preview capability detection, and paid-attempt semantics need a provider-specific adapter. The shared contract normalizes jobs and assets, not every provider field.

### Infer BPM, structure, or sample rate from the prompt

Rejected. Prompt constraints describe intent, not the generated bytes. Media facts and visualization input are measured locally.

### Enable image and PDF inputs immediately

Rejected for V1. Image support needs scoped binary input and upload controls. The documented PDF capability is not assumed to exist on the selected Gemini endpoint.

### Make Lyria the default source

Rejected. A preview model cannot become a runtime dependency of a performance instrument.

## Consequences

### Positive

- Musica can create a complete song while keeping the provider key and network authority out of the webview.
- A deterministic prompt compiler makes timed structures, social hooks, and rights constraints testable.
- Measured metadata handles Google's documented sample-rate discrepancy without guessing.
- Clip, RealTime, image, and PDF support can evolve without changing the offline engine.

### Negative

- The V1 router leaves lower-cost clips and interactive generation unavailable.
- Provider preview changes can disable generation until the adapter and fixtures are updated.
- Web Audio decoding is not an isolated media process, so hostile-decoder hardening remains incomplete under ADR-166.

## Risks and Mitigations

The largest compatibility risk is an undocumented preview response or model change. The adapter requires the expected model, one audio block, bounded JSON and audio, matching MIME and signature, and an inspectable provider request ID. Any mismatch fails closed without exposing the response body to the frontend.

The largest product risk is presenting a successful prompt as verified audio metadata. Tests deliberately use encoded-file inspection and decoded PCM analysis, and the UI must display measured values when available.

## Rollback

Unset `MUSICA_CREATIVE_ENABLED`, select a different authorized provider, or remove `GEMINI_API_KEY`. Existing local synthesis, imported tracks, visuals, Logitech controls, live capture, and saved local assets remain usable. Provider retirement requires no project migration.

## Acceptance Tests

1. Composition tests enforce the 31 to 180 second UI envelope and 184-second hard ceiling, ordered timestamps, supported languages, one candidate, and user-rights declaration for supplied lyrics.
2. A fixed structured composition compiles twice to identical text containing the exact timed sections, lyrics, instrumental or vocal constraint, social hook, and output format.
3. A Rust request fixture proves the already compiled prompt, after surrounding-whitespace normalization only, is the exact Interactions `input`; typed validation must not append duplicate lyrics, structure, tempo, duration, or vocal instructions.
4. Routing tests choose Pro for a complete 120-second song and report Clip and RealTime unavailable in V1.
5. WAV and MP3 fixtures report encoded sample rate, channels, duration, and codec without using prompt metadata, reject a mismatched MIME declaration, reject duration above 184 seconds, and retain a materially short valid output with a warning code.
6. A synthetic 120 BPM decoded track is measured within two BPM and yields a beat grid, onset map, contiguous sections, spectral profile, and the implemented camera/radial/particle/scene mapping. Anti-phase stereo and BS.1770-style loudness fixtures remain finite and do not cancel to silence. A generated 180-second, 48 kHz stereo fixture completes analysis in under five seconds on the CI runner.
7. Reference images and PDFs are rejected before the provider call in V1.
8. With Lyria disabled and network unavailable, local synthesis, import, visualization, recording, and export remain reachable.
9. A real 120 to 180 second MP3 and WAV generation is a credentialed, paid manual integration gate and is not claimed by secret-free pull-request CI.

## References

- [Google AI for Developers: Generate music with Lyria 3](https://ai.google.dev/gemini-api/docs/music-generation)
- [Google AI for Developers: Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview)
- [Google AI for Developers: Lyria 3 Pro Preview model](https://ai.google.dev/gemini-api/docs/models/lyria-3-pro-preview)
- [Google AI for Developers: Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Google Cloud: Lyria 3 and Lyria 3 Pro on Vertex AI](https://cloud.google.com/blog/products/ai-machine-learning/lyria-3-and-lyria-3-pro-on-vertex-ai)
- [Google Cloud: Prompting guide for Lyria 3 Pro](https://cloud.google.com/blog/products/ai-machine-learning/ultimate-prompting-guide-for-lyria-3-pro)
- [Google DeepMind: Lyria 3 model card](https://deepmind.google/models/model-cards/lyria-3/)
- [Google AI for Developers: Lyria RealTime](https://ai.google.dev/gemini-api/docs/realtime-music-generation)
- ADR-164: Governed creative AI provider boundary and asset provenance
- ADR-166: Desktop threat model and security boundaries
- ADR-169: Paid generation lifecycle, assets, analysis, and provenance
