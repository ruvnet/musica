#!/usr/bin/env python3
"""One-off local validation script for MiniMax-Music3 (diffusers ModularPipeline),
tuned for a 16GB-VRAM consumer GPU (below the model's documented 24GB/22GB
tiers) via group-offloading the language model. Not the production inference
service -- that runs as a container on Cloud Run GPU. This is just proof the
model produces real audio on hardware we actually have.
"""
import argparse
import time

import soundfile as sf
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.hooks.group_offloading import apply_group_offloading


def build_pipeline():
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device="cuda")
    pipe = ModularPipeline.from_pretrained(
        "MiniMaxAI/MiniMax-Music3", components_manager=manager
    )
    pipe.load_components(dtype=torch.bfloat16)
    apply_group_offloading(
        pipe.language_model,
        onload_device=torch.device("cuda"),
        offload_type="leaf_level",
        use_stream=True,
    )
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    pipe = build_pipeline()
    print(f"pipeline loaded in {time.time() - t0:.1f}s")

    t1 = time.time()
    audio = pipe(
        prompt=args.prompt,
        lyrics=args.lyrics,
        audio_duration=args.duration,
        generator=torch.Generator("cuda").manual_seed(args.seed),
        output="audios",
    )[0]
    gen_s = time.time() - t1

    sf.write(args.out, audio.T, pipe.sampling_rate)
    print(f"generated {args.out} in {gen_s:.1f}s (sample_rate={pipe.sampling_rate})")


if __name__ == "__main__":
    main()
