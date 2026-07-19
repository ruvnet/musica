import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AudioEngine,
  createEngineSnapshotFromTemplate,
  importedAudioStartTime,
  performanceStepTime,
  rebaseTempoClock,
} from "../src/audio/AudioEngine";
import { defaultPerformanceTemplate } from "../src/core/presets";

class FakeAudioParam {
  value = 0;

  setTargetAtTime(value: number): this {
    this.value = value;
    return this;
  }

  setValueAtTime(value: number): this {
    this.value = value;
    return this;
  }

  linearRampToValueAtTime(value: number): this {
    this.value = value;
    return this;
  }

  exponentialRampToValueAtTime(value: number): this {
    this.value = value;
    return this;
  }
}

class FakeAudioNode {
  connect<T>(destination: T): T {
    return destination;
  }
}

class FakeGainNode extends FakeAudioNode {
  gain = new FakeAudioParam();
}

class FakeStereoPannerNode extends FakeAudioNode {
  pan = new FakeAudioParam();
}

class FakeAnalyserNode extends FakeAudioNode {
  fftSize = 2048;
  smoothingTimeConstant = 0;

  get frequencyBinCount(): number {
    return this.fftSize / 2;
  }
}

class FakeCompressorNode extends FakeAudioNode {
  threshold = new FakeAudioParam();
  knee = new FakeAudioParam();
  ratio = new FakeAudioParam();
  attack = new FakeAudioParam();
  release = new FakeAudioParam();
}

class FakeDelayNode extends FakeAudioNode {
  delayTime = new FakeAudioParam();
}

class FakeConvolverNode extends FakeAudioNode {
  buffer: AudioBuffer | null = null;
}

class FakeAudioBuffer {
  private readonly data: Float32Array[];

  constructor(
    readonly numberOfChannels: number,
    readonly length: number,
    readonly sampleRate: number,
  ) {
    this.data = Array.from({ length: numberOfChannels }, () => new Float32Array(length));
  }

  get duration(): number {
    return this.length / this.sampleRate;
  }

  getChannelData(channel: number): Float32Array {
    return this.data[channel]!;
  }
}

class FakeAudioContext {
  readonly currentTime = 0;
  readonly sampleRate = 48_000;
  readonly destination = new FakeAudioNode();
  readonly state = "running";

  createGain(): GainNode {
    return new FakeGainNode() as unknown as GainNode;
  }

  createDynamicsCompressor(): DynamicsCompressorNode {
    return new FakeCompressorNode() as unknown as DynamicsCompressorNode;
  }

  createDelay(): DelayNode {
    return new FakeDelayNode() as unknown as DelayNode;
  }

  createConvolver(): ConvolverNode {
    return new FakeConvolverNode() as unknown as ConvolverNode;
  }

  createAnalyser(): AnalyserNode {
    return new FakeAnalyserNode() as unknown as AnalyserNode;
  }

  createMediaStreamDestination(): MediaStreamAudioDestinationNode {
    return { ...new FakeAudioNode(), stream: {} as MediaStream } as unknown as MediaStreamAudioDestinationNode;
  }

  createStereoPanner(): StereoPannerNode {
    return new FakeStereoPannerNode() as unknown as StereoPannerNode;
  }

  createBuffer(channels: number, length: number, sampleRate: number): AudioBuffer {
    return new FakeAudioBuffer(channels, length, sampleRate) as unknown as AudioBuffer;
  }

  async decodeAudioData(): Promise<AudioBuffer> {
    return new FakeAudioBuffer(2, 48_000, 48_000) as unknown as AudioBuffer;
  }

  async resume(): Promise<void> {}
}

afterEach(() => vi.unstubAllGlobals());

describe("audio engine state application", () => {
  it("starts from the curated default MIDI song bank template", () => {
    const template = defaultPerformanceTemplate();
    const snapshot = new AudioEngine().getSnapshot();
    const expected = createEngineSnapshotFromTemplate(template);

    expect(snapshot.bpm).toBe(template.bpm);
    expect(snapshot.tracks.map((track) => track.pattern)).toEqual(expected.tracks.map((track) => track.pattern));
    expect(snapshot.tracks.find((track) => track.id === "lead")?.notes).toEqual(template.tracks.lead.notes.slice(0, 64));
  });

  it("preserves control state while keeping local synth tracks silent", async () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const engine = new AudioEngine();
    engine.setTrackPan("bass", 0.65);
    engine.toggleSolo("bass");

    await engine.initialize();

    const runtimes = (engine as unknown as {
      tracks: Map<string, { gain: FakeGainNode; pan: FakeStereoPannerNode }>;
    }).tracks;
    expect(runtimes.get("bass")?.pan.pan.value).toBe(0.65);
    expect(runtimes.get("bass")?.gain.gain.value).toBe(0);
    expect(runtimes.get("drums")?.gain.gain.value).toBe(0);
  });

  it("starts one-shot imports immediately while preserving bar quantization for loops", () => {
    expect(importedAudioStartTime(false, 10, 12.5)).toBeCloseTo(10.01, 8);
    expect(importedAudioStartTime(true, 10, 12.5)).toBe(12.5);
  });

  it("sets sequencer steps idempotently for drag painting", () => {
    const engine = new AudioEngine();
    engine.setStep("lead", 1, true);
    engine.setStep("lead", 2, true);
    engine.setStep("lead", 1, true);
    engine.setStep("lead", 2, false);

    const lead = engine.getSnapshot().tracks.find((track) => track.id === "lead");
    expect(lead?.pattern[1]).toBe(true);
    expect(lead?.pattern[2]).toBe(false);
  });

  it("keeps independent bounded controls for each realtime deck", () => {
    const engine = new AudioEngine();
    engine.setRealtimeDeckControl("sequence", { volume: 0.63, pitchSemitones: 4, beatNudgeMs: -85 });
    engine.setRealtimeDeckControl("vocal", { volume: 2, pitchSemitones: 20, beatNudgeMs: 900, muted: true });

    const controls = (engine as unknown as {
      realtimeDeckControls: Record<string, { volume: number; pitchSemitones: number; beatNudgeMs: number; muted: boolean }>;
    }).realtimeDeckControls;
    expect(controls.sequence).toMatchObject({ volume: 0.63, pitchSemitones: 4, beatNudgeMs: -85, muted: false });
    expect(controls.vocal).toMatchObject({ volume: 1, pitchSemitones: 12, beatNudgeMs: 500, muted: true });
    expect(controls.main).toMatchObject({ volume: 0.72, pitchSemitones: 0, beatNudgeMs: 0, muted: false });
  });

  it("releases realtime decks on one shared clock and preserves beat nudge offsets", async () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const engine = new AudioEngine();
    engine.setRealtimeDeckControl("sequence", { beatNudgeMs: 100 });

    const anchor = await engine.synchronizeRealtimeDeckClocks(0.45);
    const runtimes = (engine as unknown as {
      realtimeDecks: Map<string, { streamTime: number }>;
    }).realtimeDecks;

    expect(anchor).toBeCloseTo(0.45, 8);
    expect(runtimes.get("main")?.streamTime).toBeCloseTo(anchor, 8);
    expect(runtimes.get("sequence")?.streamTime).toBeCloseTo(anchor + 0.1, 8);
    expect(runtimes.get("vocal")?.streamTime).toBeCloseTo(anchor, 8);

    engine.setRealtimeDeckControl("sequence", { beatNudgeMs: -50 });
    expect(runtimes.get("sequence")?.streamTime).toBeCloseTo(anchor - 0.05, 8);
  });

  it("loads AI tone buffers without replacing MIDI sequencer patterns", async () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const engine = new AudioEngine();
    const before = engine.getSnapshot().tracks.find((track) => track.id === "lead");

    await engine.loadTrackToneFile("lead", new Uint8Array([1, 2, 3]).buffer, "Moonlight Glass", {
      baseNote: 73,
      grainSeconds: 0.44,
      level: 0.06,
      brightness: 0.82,
      windowStartSeconds: 10,
      windowDurationSeconds: 20,
    });

    const after = engine.getSnapshot().tracks.find((track) => track.id === "lead");
    expect(after?.aiToneFile).toBe("Moonlight Glass");
    expect(after?.loadedFile).toBeUndefined();
    expect(after?.pattern).toEqual(before?.pattern);
    expect(after?.notes).toEqual(before?.notes);

    engine.clearTrackToneFile("lead");
    expect(engine.getSnapshot().tracks.find((track) => track.id === "lead")?.aiToneFile).toBeUndefined();
  });

  it("adds bounded swing and humanization to synthesized steps", () => {
    const straight = performanceStepTime(0, 0, 4, 0.125);
    const swung = performanceStepTime(1, 1, 4.125, 0.125);
    expect(Math.abs(straight - 4)).toBeLessThanOrEqual(0.004);
    expect(swung).toBeGreaterThan(4.125);
    expect(swung).toBeLessThan(4.15);
  });

  it("preserves bar phase and aligns the next scheduler step after a live tempo change", () => {
    const currentTime = 1.0625;
    const rebased = rebaseTempoClock(currentTime, 0, 120, 60);
    const nextStepSeconds = 0.25;
    const phase = (currentTime - rebased.transportStartedAt) / nextStepSeconds;
    const nextStep = (rebased.nextStepTime - rebased.transportStartedAt) / nextStepSeconds;

    expect(phase).toBeCloseTo(8.5, 8);
    expect(rebased.nextStepTime).toBeGreaterThan(currentTime + 0.01);
    expect(nextStep).toBeCloseTo(Math.round(nextStep), 8);
    expect(rebased.currentStep).toBe(Math.round(nextStep) % 16);
  });

  it("cancels only future synthesized events when applying a live tempo change", async () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const engine = new AudioEngine();
    await engine.initialize();

    const futureSynth = { stop: vi.fn() } as unknown as AudioScheduledSourceNode;
    const imminentSynth = { stop: vi.fn() } as unknown as AudioScheduledSourceNode;
    const importedAudio = { stop: vi.fn() } as unknown as AudioScheduledSourceNode;
    const internals = engine as unknown as {
      playing: boolean;
      transportStartedAt: number;
      nextStepTime: number;
      scheduledSources: Map<AudioScheduledSourceNode, { startsAt: number; imported: boolean }>;
    };
    internals.playing = true;
    internals.transportStartedAt = 0;
    internals.nextStepTime = 0.12;
    internals.scheduledSources.set(futureSynth, { startsAt: 0.05, imported: false });
    internals.scheduledSources.set(imminentSynth, { startsAt: 0.005, imported: false });
    internals.scheduledSources.set(importedAudio, { startsAt: 0.05, imported: true });

    engine.setBpm(60);

    expect(futureSynth.stop).toHaveBeenCalledOnce();
    expect(imminentSynth.stop).not.toHaveBeenCalled();
    expect(importedAudio.stop).not.toHaveBeenCalled();
    expect(internals.scheduledSources.has(futureSynth)).toBe(false);
    expect(internals.nextStepTime).toBeCloseTo(0.25, 8);
  });

  it("rebases a tempo change during transport pre-roll to a bounded lead time", () => {
    expect(rebaseTempoClock(1, 1.06, 112, 128)).toEqual({
      transportStartedAt: 1.01,
      nextStepTime: 1.01,
      currentStep: 0,
    });
  });
});
