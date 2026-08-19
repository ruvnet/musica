#!/usr/bin/env python3
"""Second validation batch: more genre variety plus one longer, structurally
complex composition (intro/verse/chorus/verse/chorus/bridge/chorus/outro) to
demonstrate the model's actual differentiator -- long-range structural
coherence across a real song shape, not just a 20s clip."""
import time

import soundfile as sf
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.hooks.group_offloading import apply_group_offloading

NEW_GENRES = [
    {
        "name": "jazz_noir",
        "duration": 20.0,
        "lyrics": "[verse]\nSmoke curls up from an ashtray downtown\nRain on the window, city shuts down\n[chorus]\nMidnight confession, one more cigarette",
        "prompt": "Genre: jazz noir. BPM: 68. Key: C minor. Smoky, late-night, melancholic. Vocals: low sultry female alto, breathy, restrained vibrato. Arrangement: upright bass walking line, brushed jazz drums, muted trumpet solo, minor 7th piano chords, subtle room reverb.",
    },
    {
        "name": "afrobeat_pop",
        "duration": 20.0,
        "lyrics": "[verse]\nSunlight on the water, dancing feet\nEvery drum a heartbeat on the street\n[chorus]\nLift your hands up, feel the rhythm move",
        "prompt": "Genre: afrobeat pop. BPM: 108. Key: E major. Bright, communal, celebratory. Vocals: energetic male lead with call-and-response backing vocals. Arrangement: interlocking guitar riffs, congas and shakers, punchy horn stabs, syncopated bassline, layered percussion groove.",
    },
    {
        "name": "dark_ambient_electronic",
        "duration": 20.0,
        "lyrics": "[intro]\n[verse]\nStatic whispers through the wire\nSignals fading, climbing higher\n[instrumental]",
        "prompt": "Genre: dark ambient electronic. BPM: 60. Key: F minor. Tense, atmospheric, sparse. Vocals: distant processed female vocal, heavily reverbed, almost textural. Arrangement: deep sub bass drones, glitchy percussion, granular synth textures, slow filter sweeps, minimal and spacious mix.",
    },
    {
        "name": "bluegrass_folk",
        "duration": 20.0,
        "lyrics": "[verse]\nDown by the river where the willows bend\nWe'll ride 'til the road comes to an end\n[chorus]\nCarry me home on a summer wind",
        "prompt": "Genre: bluegrass folk. BPM: 128. Key: G major. Energetic, rootsy, joyful. Vocals: twangy male tenor lead with close-harmony trio backing. Arrangement: fast banjo rolls, fiddle lead lines, upright bass, flatpicked acoustic guitar, foot-stomp percussion.",
    },
]

COMPLEX_SONG = {
    "name": "complex_full_structure",
    "duration": 100.0,
    "lyrics": (
        "[intro]\n"
        "[verse]\n"
        "Started out with nothing but a spark\n"
        "Walking forward through the dark\n"
        "Every step a story left unsaid\n"
        "Chasing echoes of a dream ahead\n"
        "[chorus]\n"
        "We rise, we rise, above the tide\n"
        "Nothing left for us to hide\n"
        "We rise, we rise, into the light\n"
        "Holding on with all our might\n"
        "[verse]\n"
        "Cities burning bright below the stars\n"
        "Counting all our hidden scars\n"
        "Every scar a lesson that we learned\n"
        "Every bridge behind us slowly burned\n"
        "[chorus]\n"
        "We rise, we rise, above the tide\n"
        "Nothing left for us to hide\n"
        "We rise, we rise, into the light\n"
        "Holding on with all our might\n"
        "[bridge]\n"
        "And if the world forgets our name\n"
        "We'll carry on just the same\n"
        "[chorus]\n"
        "We rise, we rise, above the tide\n"
        "Nothing left for us to hide\n"
        "We rise, we rise, into the light\n"
        "Holding on with all our might\n"
        "[outro]"
    ),
    "prompt": (
        "Genre: anthemic indie rock. BPM: 100. Key: E minor rising to E major at the chorus. "
        "Emotional progression: introspective verses building into a triumphant, wide chorus, "
        "a stripped-back intimate bridge, then the biggest final chorus. "
        "Vocals: passionate male lead with a raspy edge, gang vocal backing harmonies doubling "
        "the chorus melody, ad-libbed vocal runs in the final chorus. "
        "Arrangement: verses are guitar and vocal only with a simple kick pulse; each chorus adds "
        "full drum kit, driving bassline, layered electric guitars, and a synth pad underneath; "
        "the bridge drops to just piano and vocal; the final chorus adds a soaring lead guitar line "
        "on top of the full arrangement. Production: wide stereo guitars, big room reverb on the "
        "chorus drums, tight and dry on the verses."
    ),
}


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
        audio_duration=spec["duration"],
        generator=torch.Generator("cuda").manual_seed(7),
        output="audios",
    )[0]
    gen_s = time.time() - t1
    out_path = f"/tmp/claude-1000/-home-ruvultra-projects-AgentBBS-1/c4217b5f-d961-452c-9b3d-c80828b3ac21/scratchpad/sample_{spec['name']}.wav"
    sf.write(out_path, audio.T, pipe.sampling_rate)
    print(f"{spec['name']}: generated {out_path} in {gen_s:.1f}s (duration={spec['duration']}s)", flush=True)


def main():
    t0 = time.time()
    pipe = build_pipeline()
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    for spec in NEW_GENRES:
        generate_one(pipe, spec)

    # The long structured composition last -- it takes the longest.
    generate_one(pipe, COMPLEX_SONG)


if __name__ == "__main__":
    main()
