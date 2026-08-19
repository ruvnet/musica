#!/usr/bin/env python3
"""Generate several short samples across different styles in one process,
reusing the loaded pipeline. Local hardware-validation script only."""
import time

import soundfile as sf
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.hooks.group_offloading import apply_group_offloading

STYLES = [
    {
        "name": "acoustic_pop",
        "lyrics": "[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe",
        "prompt": "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus.",
    },
    {
        "name": "synthwave",
        "lyrics": "[verse]\nNeon signs are burning through the rain\nRunning circuits down a midnight lane\n[chorus]\nWe are electric, we are alive tonight",
        "prompt": "Genre: synthwave. BPM: 118. Key: A minor. Driving and nostalgic, retro-futuristic energy. Vocals: confident male lead, layered with a robotic vocoder harmony in the chorus. Arrangement: analog synth arpeggios, gated reverb drums, deep sidechained bass, soaring lead synth in the chorus.",
    },
    {
        "name": "orchestral_cinematic",
        "lyrics": "[intro]\n[verse]\nRise above the ashes of the fall\nEvery broken piece will heal us all\n[chorus]\nThis is the moment we stand tall",
        "prompt": "Genre: orchestral cinematic. BPM: 84. Key: D minor rising to D major at the chorus. Epic, emotional, triumphant build. Vocals: powerful mixed choir with a solo soprano lead. Arrangement: swelling string section, French horns, timpani hits, choir crescendo into the chorus.",
    },
    {
        "name": "lofi_hiphop",
        "lyrics": "[verse]\nDust on the windowsill, coffee gone cold\nStories in the vinyl that we used to hold\n[chorus]\nJust let it play, let it play",
        "prompt": "Genre: lo-fi hip-hop. BPM: 78. Key: F major. Relaxed, nostalgic, warm tape saturation. Vocals: laid-back female lead, half-spoken half-sung, soft and close-mic'd. Arrangement: dusty boom-bap drums, mellow jazz piano chords, vinyl crackle, upright bass, light rain ambience.",
    },
]

DURATION_S = 20.0


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
    t0 = time.time()
    pipe = build_pipeline()
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    for style in STYLES:
        t1 = time.time()
        audio = pipe(
            prompt=style["prompt"],
            lyrics=style["lyrics"],
            audio_duration=DURATION_S,
            generator=torch.Generator("cuda").manual_seed(7),
            output="audios",
        )[0]
        gen_s = time.time() - t1
        out_path = f"/tmp/claude-1000/-home-ruvultra-projects-AgentBBS-1/c4217b5f-d961-452c-9b3d-c80828b3ac21/scratchpad/sample_{style['name']}.wav"
        sf.write(out_path, audio.T, pipe.sampling_rate)
        print(f"{style['name']}: generated {out_path} in {gen_s:.1f}s", flush=True)


if __name__ == "__main__":
    main()
