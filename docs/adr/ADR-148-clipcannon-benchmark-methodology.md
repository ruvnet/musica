# ADR-148: ClipCannon Subsystem — Benchmark Methodology & Perf Targets

## Status
Accepted

## Date
2026-04-12

## Context

ADRs 145–147 establish a realtime analysis & avatar-driving subsystem. Without
a published benchmark methodology the subsystem risks slipping into the
same "it works on my machine" trap as Python ML pipelines.

Existing musica performance (README):

- `hearing_aid`: 0.20 ms avg / 0.26 ms max per frame.
- Model size: 0 bytes.
- Total latency budget: <8 ms.

The new subsystem adds prosody, viseme, speaker fingerprint, and a
`RealtimeAvatarAnalyzer` pipeline block. We need numbers for each, separately
and composed.

## Decision

### Methodology

Benchmarks are implemented as **library functions** (matching `src/benchmark.rs`
precedent — no criterion dep) exposed from `src/clipcannon/bench.rs` and
called from both unit tests and `main.rs` via a CLI flag.

```rust
pub struct BenchResult {
    pub name:           &'static str,
    pub iterations:     u32,
    pub total_ms:       f64,
    pub mean_us:        f64,
    pub p50_us:         f64,
    pub p95_us:         f64,
    pub p99_us:         f64,
    pub max_us:         f64,
    pub alloc_free:     bool, // verified via a counting test harness
}
```

Each benchmark:

1. Preallocates all inputs & outputs.
2. Runs **N=256 warmup iterations** (discarded).
3. Runs **N=4096 measurement iterations**.
4. Records per-iteration wall-clock via `std::time::Instant`.
5. Reports p50/p95/p99/max and `mean_us`.
6. Verifies zero allocation by instrumenting the extractor with an
   `#[cfg(test)]` allocation counter.

### Coverage

| Benchmark ID                | What it measures                                                      |
|-----------------------------|-----------------------------------------------------------------------|
| `bench_prosody_frame`       | `ProsodyExtractor::extract` on one 256-sample frame                   |
| `bench_viseme_map`          | `VisemeMapper::map` from prosody + 129-bin magnitude spectrum         |
| `bench_speaker_observe`     | `SpeakerTracker::observe` steady-state (no new speaker)               |
| `bench_speaker_enrol`       | `SpeakerTracker::observe` on a novel speaker                          |
| `bench_analyzer_block`      | `RealtimeAvatarAnalyzer::process` on one 128-sample block             |
| `bench_analyzer_composite`  | 16000-sample end-to-end stream split into 125×128-sample blocks       |

### Performance Targets (16 kHz, 128-sample block, release build)

| Benchmark                   | Target mean | Target p99 | Rationale                                    |
|-----------------------------|-------------|------------|----------------------------------------------|
| `bench_prosody_frame`       | <80 µs      | <200 µs    | dominates composite budget                   |
| `bench_viseme_map`          | <15 µs      | <40 µs     | arithmetic-only, no FFT                      |
| `bench_speaker_observe`     | <25 µs      | <80 µs     | cosine against N≤16 enrolled speakers       |
| `bench_speaker_enrol`       | <40 µs      | <120 µs    | one-off allocation at enrol (accepted)       |
| `bench_analyzer_block`      | <250 µs     | <600 µs    | full pipeline per block                      |
| `bench_analyzer_composite`  | <32 ms      | <40 ms     | 1 s of audio in 125 blocks — ~31× realtime   |

The `bench_analyzer_block` 250 µs/block corresponds to **~32× realtime at 16 kHz**
(block = 8 ms audio, processed in ≤0.25 ms). This matches the existing
`hearing_aid` headroom.

### Quality Targets

Numeric correctness is enforced via *oracle tests* (not benchmarks):

| Oracle                                     | Threshold                        |
|--------------------------------------------|----------------------------------|
| Sinusoid at 220 Hz → `f0_hz` recovered     | within ±2 Hz                     |
| Silence → `voicing` close to 0             | <0.10                            |
| Full-scale 1 kHz → `energy_db` close to 0  | within 1 dB of −3                |
| Two distinct speakers → different IDs      | >95% agreement over 1 s          |
| Same speaker recorded twice → same ID      | >95% agreement over 1 s          |
| Vowel /a/ → viseme AA                      | majority vote over synthetic test|
| Silence → viseme REST                      | 100%                             |

### Optimisation Stopping Criteria

Optimisation iterates until:

1. All `mean_us` targets are met, **and**
2. p99 is within 3× of mean (tail acceptable), **and**
3. All oracle tests still pass.

Beyond that, optimisation is explicitly halted — don't trade correctness for
a few microseconds.

### Tooling

- Wall-clock via `std::time::Instant` (monotonic on Linux).
- Perf counters: none (would add a dep).
- Determinism: fixed RNG seed in test signal generation.
- CI: benchmarks do not gate CI — they print results and record them in
  `docs/bench/clipcannon-baseline.md` for human review.

## Consequences

### Positive
- Perf story is as documented and verifiable as correctness.
- Zero new dependencies (matches musica ethos).
- Oracle tests catch accuracy regressions even if speed regressions are absent.

### Negative
- Wall-clock benches are noisier than criterion's statistical framework; we
  mitigate by reporting p99 & max, not just mean.

### Risks
- Someone runs benches on a low-power laptop and sees the CI numbers as a
  floor, not a ceiling. The rationale column makes targets portable.

## References
- `src/benchmark.rs` — existing custom harness precedent.
- ADR-145, ADR-146, ADR-147.
