#!/usr/bin/env python3
"""Third validation batch: ad jingle, techno, EDM, classical, and a
1960s-British-Invasion-style rock genre piece (an era/genre treatment with
wholly original lyrics -- deliberately not styled after any specific named
artist, to avoid imitating a real performer's identifiable sound)."""
import time

import soundfile as sf
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.hooks.group_offloading import apply_group_offloading

NEW_TRACKS = [
    {
        "name": "ad_jingle",
        "duration": 15.0,
        "lyrics": "[verse]\nFresh and bright, start your day right\nSunny Cola, feel the light\n[chorus]\nSunny Cola, pour it up",
        "prompt": "Genre: upbeat commercial jingle. BPM: 128. Key: C major. Punchy, bright, instantly catchy, radio-ready. Vocals: energetic mixed group vocal, tight unison on the chorus hook, handclaps. Arrangement: bouncy ukulele, snappy pop drums, bright bell synth, short and memorable, no long intro.",
    },
    {
        "name": "techno",
        "duration": 20.0,
        "lyrics": "[intro]\n[instrumental]\n[verse]\nPulse and drive, machine alive\nCount the beat, survive, survive\n[instrumental]",
        "prompt": "Genre: driving techno. BPM: 132. Key: A minor. Hypnotic, relentless, warehouse energy. Vocals: sparse robotic vocal chop used as a rhythmic texture, not a lead melody. Arrangement: four-on-the-floor kick, rolling acid bassline, hi-hat and clap groove, filtered synth stabs building tension, minimal and repetitive by design.",
    },
    {
        "name": "edm_festival",
        "duration": 20.0,
        "lyrics": "[intro]\n[verse]\nHands up high, we touch the sky tonight\nEvery beat ignites the light\n[chorus]\nWe are one, under the sun and stars\nThis is who we are",
        "prompt": "Genre: festival EDM. BPM: 128. Key: F major. Euphoric, massive drop, arena-scale energy. Vocals: powerful anthemic female lead, big layered gang vocals on the drop. Arrangement: huge supersaw lead synth, sidechained pumping bass, riser and white-noise sweep into the drop, four-on-the-floor kick, crowd-energy production.",
    },
    {
        "name": "classical_chamber",
        "duration": 20.0,
        "lyrics": "[instrumental]",
        "prompt": "Genre: classical chamber music, no vocals. BPM: 72. Key: D major. Elegant, refined, delicate interplay between instruments. Arrangement: string quartet -- two violins, viola, and cello -- with a light piano accompaniment; expressive dynamics, a singing violin melody line, gentle rubato phrasing, warm concert-hall acoustic.",
    },
    {
        "name": "sixties_british_rock",
        "duration": 20.0,
        "lyrics": "[verse]\nDown the cobbled street we go tonight\nUnder a fading neon light\n[chorus]\nCome along, come along with me\nWe'll be young and wild and free",
        "prompt": "Genre: 1960s British Invasion rock, era-accurate original composition (not styled after any specific named artist). BPM: 132. Key: E major. Bright, jangly, optimistic energy typical of the era. Vocals: youthful male lead with light close-harmony backing vocals on the chorus. Arrangement: jangly 12-string electric guitar, driving quarter-note bass, tight simple drum kit, tambourine, period-accurate mono-leaning production with plate reverb.",
    },
]

DURATION_DEFAULT = 20.0


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


def generate_one(pipe, spec):
    t1 = time.time()
    audio = pipe(
        prompt=spec["prompt"],
        lyrics=spec["lyrics"],
        audio_duration=spec.get("duration", DURATION_DEFAULT),
        generator=torch.Generator("cuda").manual_seed(7),
        output="audios",
    )[0]
    gen_s = time.time() - t1
    out_path = f"/tmp/claude-1000/-home-ruvultra-projects-AgentBBS-1/c4217b5f-d961-452c-9b3d-c80828b3ac21/scratchpad/sample_{spec['name']}.wav"
    sf.write(out_path, audio.T, pipe.sampling_rate)
    print(f"{spec['name']}: generated {out_path} in {gen_s:.1f}s", flush=True)


def main():
    t0 = time.time()
    pipe = build_pipeline()
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    for spec in NEW_TRACKS:
        generate_one(pipe, spec)


if __name__ == "__main__":
    main()
