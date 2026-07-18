import { afterEach, describe, expect, it, vi } from "vitest";
import {
  AudioEngine,
  importedAudioStartTime,
  rebaseTempoClock,
} from "../src/audio/AudioEngine";

class FakeAudioParam {
  value = 0;

  setTargetAtTime(value: number): this {
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

  async resume(): Promise<void> {}
}

afterEach(() => vi.unstubAllGlobals());

describe("audio engine state application", () => {
  it("applies pan and solo state chosen before AudioContext initialization", async () => {
    vi.stubGlobal("AudioContext", FakeAudioContext);
    const engine = new AudioEngine();
    engine.setTrackPan("bass", 0.65);
    engine.toggleSolo("bass");

    await engine.initialize();

    const runtimes = (engine as unknown as {
      tracks: Map<string, { gain: FakeGainNode; pan: FakeStereoPannerNode }>;
    }).tracks;
    expect(runtimes.get("bass")?.pan.pan.value).toBe(0.65);
    expect(runtimes.get("bass")?.gain.gain.value).toBeCloseTo(0.78 ** 2, 8);
    expect(runtimes.get("drums")?.gain.gain.value).toBe(0);
  });

  it("starts one-shot imports immediately while preserving bar quantization for loops", () => {
    expect(importedAudioStartTime(false, 10, 12.5)).toBeCloseTo(10.01, 8);
    expect(importedAudioStartTime(true, 10, 12.5)).toBe(12.5);
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
