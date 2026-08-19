# ADR-182: MiniMax-Music3 as a second generation provider, via a Cognitum gateway on Cloud Run GPU

- Status: Accepted (gateway + inference service implemented and deployed; Musica-side provider wiring is a documented follow-up, not yet built)
- Date: 2026-08-19

## Context

Musica's only music-generation provider today is Google Lyria (RealTime for live performance, batch for asset generation — ADR-168/170), reached either with a bring-your-own `GEMINI_API_KEY` or, per [ADR-178](ADR-178-metered-lyria-via-cognitum-proxy.md), a metered pass-through via Cognitum One sign-in.

[MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) is a different kind of provider: an **open-weights, self-hosted** model (not a hosted API) that generates a **complete song up to five minutes long** from lyrics + a structured style description, with explicit section tags (`[verse]`, `[chorus]`, `[bridge]`, ...) and long-range structural coherence Lyria's shorter-form generation doesn't target. It is a strict complement to Lyria, not a replacement: Lyria RealTime does live, continuous, low-latency generation; MiniMax-Music3 does batch, long-form, structurally-planned generation, and cannot do the former at all (confirmed — see Constraints).

### Constraints (verified, not assumed)

- **No real-time streaming generation.** The model produces a complete track in one autoregressive pass; upstream docs state this explicitly. A "streaming endpoint" for this provider can only mean *progressive delivery* of a batch-generated result (chunked HTTP/SSE), not Lyria-style continuous generation. This ADR does not claim otherwise.
- **GPU-only, no CPU inference.** Minimum 24GB VRAM (bfloat16, full pipeline); ~22GB with automatic CPU offload of the flow-matching/decoder stages; ~8GB is reachable by additionally group-offloading the 8B language model, at a real generation-speed cost.
- **Verified on real hardware, not just the model card.** Ran the actual `diffusers` `MiniMaxMusic3ModularPipeline` (merged into `diffusers` main via [huggingface/diffusers#14456](https://github.com/huggingface/diffusers/pull/14456)) on an RTX 5080 (16GB VRAM — below every documented tier) using the 8GB-tier group-offload path. It works: produced real, non-silent 44.1kHz stereo audio across four genres (acoustic pop, synthwave, orchestral cinematic, lo-fi hip-hop). Generation on that undersized, offloaded configuration ran at roughly 20x real time (a 15–20s clip took 5–6.5 minutes) — this is the *slow* tier the docs warn about, not a production number. A Cloud Run instance with a full 24GB+ GPU and no offloading will be substantially faster; this deployment does not yet have a measured production-hardware number (see Implementation).

## Decision

**Deploy MiniMax-Music3 as a new, independent Cognitum-hosted service — `cognitum-one/minimax-music` — with two components, and give Musica a second provider mode that reaches it through the same Cognitum-bearer pattern [ADR-175](ADR-175-cognitum-oauth-and-expanded-meta-llm-capabilities.md) already established for Meta-LLM.**

### Why Cloud Run GPU, and why a separate repo

- Cloud Run supports GPU-backed services (`--gpu`), which fits this workload's shape better than a standing GCE VM: scales to zero between requests (this is not a live/always-on service the way Lyria RealTime is), and the existing Cognitum deployment tooling (Secret Manager, dedicated least-privilege service accounts, Cloudflare DNS) already targets Cloud Run.
- The `cognitum-20260110` project's Cloud Run GPU quota is **`nvidia-l4`: 0** (every region) but **`nvidia-rtx-pro-6000`: available** (confirmed by a real trial deploy) — the RTX Pro 6000 has far more VRAM than the L4 and comfortably clears the model's 24GB full-precision floor, so the production service runs at full precision, not a memory-constrained tier.
- Mirrors the `cognitum-one/buzz` precedent: a dedicated deployment/ownership boundary for a self-hosted third-party model, separate from the app repo (Musica) and the platform repos (`meta-llm`, `api`), so its GPU cost, scaling, and lifecycle are managed independently.

### Two components in `cognitum-one/minimax-music`

1. **`services/inference`** — the GPU model server. A container running the `diffusers` `ModularPipeline` (not `sglang-omni`; see Implementation for why) behind a small FastAPI wrapper exposing the same request shape as the model's own reference server (`prompt`, `lyrics`, `audio_duration`, `seed`) plus a chunked-delivery variant. **Internal-only** — no public ingress; only the gateway can reach it (Cloud Run service-to-service IAM invoker binding, the same pattern used for `cog-gateway`).
2. **`services/gateway`** — a Rust (axum) authenticated proxy, publicly reachable. Validates a Cognitum bearer token (the same `cog_`-prefixed Secret-Manager-issued key scheme `meta-llm` and every other Cognitum-agent service already use in production — not a new OAuth server; ADR-178's aspirational full end-user OAuth token is a larger, separate integration this ADR does not attempt), meters requests, and proxies to the inference service. Exposes:
   - `POST /v1/music/generate` — batch: blocks until the full WAV is ready, returns it.
   - `POST /v1/music/stream` — **progressive delivery**, not real-time generation (see Constraints): starts generation server-side and streams the resulting audio to the client via chunked HTTP as soon as bytes are available, rather than making the client wait for the entire multi-minute file before any bytes arrive. Explicitly documented as not equivalent to Lyria RealTime.

### Musica-side integration (documented, not yet built)

A `creative_provider` mode (`minimax`) alongside the existing `gemini`/`cognitum` Lyria modes, calling `/v1/music/generate` for asset generation (mirrors how Lyria batch generation already works — ADR-169). This is real, scoped follow-up work — not built in this pass, which focused on proving the model runs and stands up an authenticated API in front of it. Tracked as this ADR's open item rather than silently left unmentioned.

## Consequences

- Musica gains a long-form, structurally-coherent generation option that Lyria doesn't offer, without touching Lyria's code path at all (additive, same shape as ADR-178).
- Real GPU cost: RTX Pro 6000 Cloud Run instances are not cheap and this model's generation is not fast even at full precision (autoregressive, frame-by-frame). Scale-to-zero keeps idle cost at zero, but each generation has real per-request cost and latency (single digit minutes, not seconds) that the client UI must set expectations for.
- No streaming *generation* is possible for this provider, full stop — this is a hard model-architecture constraint, not a missed feature. Any UI copy or documentation must say "progressive delivery," never "real-time," to avoid the exact wrong expectation Lyria RealTime sets.
- The gateway's auth is the same `cog_`-key bearer scheme used elsewhere today (proven, working), not the aspirational full OAuth-metering plane ADR-178 describes for Lyria RealTime — a smaller, honest scope than "OAuth + meta-proxy metering" might imply on first read; upgrading to full per-end-user OAuth is future work, called out explicitly rather than glossed over.

## Implementation

- Inference framework: `diffusers` (`ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-Music3")`), not `sglang-omni`. The model card recommends `sglang-omni` as the primary path, but `diffusers` is the officially documented, actively maintained alternative (its own docs page: `docs/source/en/api/pipelines/minimax_music3.md`) and needs far less container surface than a full inference-serving framework for a single-model, low-QPS internal service.
- Verified locally before writing any deployment code: `MiniMaxMusic3ModularPipeline` (merged upstream in huggingface/diffusers#14456) generated real audio in four styles on undersized local hardware (RTX 5080, 16GB) using `ComponentsManager.enable_auto_cpu_offload` + `apply_group_offloading` on the language model — the diffusers docs' own documented 8GB-tier path.
- Production inference container: same `diffusers` pipeline, `bfloat16`, full precision (no offloading needed on the RTX Pro 6000's VRAM headroom) — expected to be substantially faster than the local 8GB-tier measurement above, but not yet load-tested on the deployed Cloud Run GPU instance; treat the ~20x-realtime local number as a worst-case bound, not the production figure, until a real production timing run is done.
- Gateway auth reuses the existing Secret-Manager `cog_`-key issuance and validation already running in production for `meta-llm`/agent services — no new identity infrastructure.
- See `cognitum-one/minimax-music`'s own ADRs for the GCP topology, service accounts, and deploy commands (mirrors the `cognitum-one/buzz` deployment-reference pattern).

## Next steps

1. Build the `creative_provider` `minimax` mode in Musica's Rust provider layer (Musica-side integration, above).
2. Run a real production-hardware timing benchmark on the deployed RTX Pro 6000 instance and replace the "worst-case, undersized-hardware" number in this ADR with it.
3. Decide whether `/v1/music/stream`'s progressive-delivery framing needs surfacing in the UI as a distinct, slower-feeling operation from Lyria RealTime, so users don't carry the wrong mental model across providers.
