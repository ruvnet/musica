//! Run the full clipcannon benchmark suite. Build with `--release`.
//!
//! ```text
//! cargo run --release --example clipcannon_bench
//! ```

use musica::clipcannon::bench::{
    bench_analyzer_block, bench_analyzer_composite, bench_prosody_frame, bench_speaker_observe,
    bench_viseme_map,
};

fn main() {
    println!("================================================================");
    println!("  ClipCannon realtime subsystem — micro-benchmark suite");
    println!("  16 kHz, 128-sample blocks, 256-sample windows");
    println!("================================================================\n");

    let results = vec![
        bench_prosody_frame(),
        bench_viseme_map(),
        bench_speaker_observe(),
        bench_analyzer_block(),
        bench_analyzer_composite(),
    ];

    for r in &results {
        println!("{}", r);
    }

    println!("\n----------------------------------------------------------------");
    if let Some(block_bench) = results.iter().find(|r| r.name == "analyzer_block") {
        let realtime_factor = (128.0 / 16_000.0) * 1_000_000.0 / block_bench.mean_us;
        println!(
            "  Realtime factor (analyzer_block): {:.1}× ({}μs per 8 ms audio)",
            realtime_factor, block_bench.mean_us as u32
        );
    }
    if let Some(comp) = results.iter().find(|r| r.name == "analyzer_composite_1s") {
        let realtime_factor = 1_000_000.0 / comp.mean_us;
        println!(
            "  Realtime factor (1 s composite):   {:.1}× ({:.2} ms wall-clock)",
            realtime_factor,
            comp.mean_us / 1000.0
        );
    }
    println!("================================================================");
}
