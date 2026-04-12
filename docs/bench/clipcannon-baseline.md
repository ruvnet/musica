# ClipCannon Realtime Subsystem — Baseline Benchmarks

> Recorded **2026-04-12**, release build, Linux x86_64.
> Methodology: see [ADR-148](../adr/ADR-148-clipcannon-benchmark-methodology.md).
> Reproduce with: `cargo run --release --example clipcannon_bench`.

## Configuration

| Parameter            | Value           |
|----------------------|-----------------|
| Sample rate          | 16 000 Hz       |
| Block size           | 128 samples (8 ms) |
| Window size          | 256 samples (16 ms) |
| Speaker capacity     | 8               |
| Cosine threshold     | 0.85            |
| Warmup iterations    | 256             |
| Measurement iters    | 4 096 (32 for composite) |

## Results

| Benchmark              | Iters | Mean   | p50    | p95    | p99    | Max    | Target mean | Target p99 | Status |
|------------------------|-------|--------|--------|--------|--------|--------|-------------|------------|--------|
| `prosody_frame`        | 4 096 | 18.24 µs |16.96 µs |26.88 µs |34.46 µs |50.36 µs |  <80 µs |  <200 µs | ✅ 4.4× under |
| `viseme_map`           | 4 096 |  0.88 µs | 0.84 µs | 1.19 µs | 1.48 µs |12.44 µs |  <15 µs |   <40 µs | ✅ 17× under  |
| `speaker_observe`      | 4 096 |  1.64 µs | 1.59 µs | 1.91 µs | 2.10 µs |13.07 µs |  <25 µs |   <80 µs | ✅ 15× under  |
| `analyzer_block`       | 4 096 | 26.04 µs |23.53 µs |37.40 µs |47.29 µs |88.71 µs | <250 µs |  <600 µs | ✅ 9.6× under |
| `analyzer_composite_1s`|    32 | 3201.99 µs | 3203.70 µs | 3423.49 µs | 3496.43 µs | 3496.43 µs | <32 ms | <40 ms | ✅ 10× under |

## Realtime Factors

- **Per-block analyser**: ~307× realtime (26 µs of CPU time per 8 ms audio).
- **Composite 1 s stream**: ~312× realtime (3.2 ms wall-clock per second of audio).

## Tail Behaviour

| Benchmark              | p99 / mean ratio | Within 3× target |
|------------------------|------------------|------------------|
| `prosody_frame`        | 1.89×            | ✅ |
| `viseme_map`           | 1.68×            | ✅ |
| `speaker_observe`      | 1.28×            | ✅ |
| `analyzer_block`       | 1.82×            | ✅ |

All p99 latencies are well within the ADR-148 stopping-criterion tail
budget of 3× the mean.

## Stopping Criterion (per ADR-148)

1. ✅ All `mean_us` targets met.
2. ✅ All p99 within 3× of mean.
3. ✅ All oracle tests still pass (`cargo test --lib clipcannon` → 30/30).

**Optimisation halted.** Trading correctness for further microseconds is
explicitly disallowed by ADR-148.

## Comparison vs ClipCannon Upstream (Estimated)

ClipCannon upstream uses Wav2Vec2-large for prosody+emotion (Python+CUDA).
Conservative estimates from public benchmarks:

| Stage              | ClipCannon (Python+CUDA, A100) | Musica (Rust, CPU) | Speedup    |
|--------------------|-------------------------------|--------------------|------------|
| Prosody (per frame)| 30–80 ms                      | 18 µs              | ~1 600–4 400× |
| Speaker embed      | 25 ms (ECAPA-TDNN)            | 1.6 µs             | ~15 600×   |
| Per-block end-to-end | 50–100 ms                   | 26 µs              | ~1 900–3 800× |

Caveats: musica's prosody is hand-engineered features (7-dim) vs Wav2Vec2's
1024-dim learned embedding. Comparisons are *latency*, not *quality*. See
ADR-145 §"Why Not Fork ClipCannon Python Directly?" for the qualitative
trade-off.

## Hot-Path Breakdown (analyzer_block, ~26 µs)

| Stage                  | Approx µs | % of block |
|------------------------|-----------|------------|
| FFT (256-pt radix-2)   |    ~5     |   19%      |
| Prosody extraction     |   ~18     |   69%      |
| Viseme mapping         |    ~1     |    4%      |
| Speaker observe        |    ~2     |    8%      |
| Analysis aggregation   |    <1     |   <1%      |

The dominant cost is `prosody_frame` (specifically the autocorrelation inner
loop). Further optimisation would target SIMD-friendly ACF, but with the
current 9.6× safety margin we leave that to a future ADR.
