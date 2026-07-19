import { afterEach, describe, expect, it, vi } from "vitest";
import { chooseRecordingMime, inspectRecordingContainer, SocialRecorder } from "../src/export/SocialRecorder";
import { SOCIAL_PRESETS } from "../src/core/presets";

function concat(...parts: Array<Uint8Array<ArrayBufferLike>>): Uint8Array<ArrayBuffer> {
  const output = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function uint32(value: number): Uint8Array {
  const bytes = new Uint8Array(4);
  new DataView(bytes.buffer).setUint32(0, value, false);
  return bytes;
}

function isoBox(type: string, payload: Uint8Array<ArrayBufferLike> = new Uint8Array()): Uint8Array<ArrayBuffer> {
  return concat(uint32(payload.length + 8), new TextEncoder().encode(type), payload);
}

function mp4Fixture(
  sampleEntries: string[],
  extraPayload: Uint8Array<ArrayBufferLike> = new Uint8Array(),
): Uint8Array<ArrayBuffer> {
  const entries = sampleEntries.map((type) => isoBox(type));
  const stsd = isoBox("stsd", concat(new Uint8Array(4), uint32(entries.length), ...entries));
  const stbl = isoBox("stbl", stsd);
  const minf = isoBox("minf", stbl);
  const mdia = isoBox("mdia", minf);
  const trak = isoBox("trak", mdia);
  const moov = isoBox("moov", trak);
  return concat(isoBox("ftyp", new TextEncoder().encode("isom\0\0\0\0isom")), moov, extraPayload);
}

describe("social export contracts", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("prefers H.264 MP4 when the webview supports it", () => {
    expect(chooseRecordingMime((mime) => mime.includes("mp4"))).toBe("video/mp4;codecs=h264,aac");
  });

  it("falls back to VP9 and Opus", () => {
    expect(chooseRecordingMime((mime) => mime === "video/webm;codecs=vp9,opus")).toBe("video/webm;codecs=vp9,opus");
  });

  it("reports unavailable when neither MP4 nor WebM is supported", () => {
    expect(chooseRecordingMime(() => false)).toBe("");
  });

  it("keeps every reel preset vertical and social ready", () => {
    const reelPresets = SOCIAL_PRESETS.filter((preset) => preset.id.startsWith("reel"));
    expect(reelPresets.every((preset) => preset.width === 1080 && preset.height === 1920)).toBe(true);
    expect(reelPresets.every((preset) => preset.fps >= 30)).toBe(true);
    expect(reelPresets.map((preset) => preset.durationSeconds)).toEqual([6, 9, 15, 30]);
  });

  it("validates an H.264 and AAC MP4 container", () => {
    const bytes = mp4Fixture(["avc1", "mp4a"]);
    expect(inspectRecordingContainer(bytes, "video/mp4;codecs=h264,aac")).toEqual({
      container: "mp4",
      videoCodec: "h264",
      audioCodec: "aac",
    });
  });

  it("rejects an MP4 that does not prove H.264 and AAC", () => {
    const bytes = mp4Fixture([]);
    expect(() => inspectRecordingContainer(bytes, "video/mp4")).toThrow(/H\.264 video and AAC audio/);
  });

  it("does not trust codec text outside an stsd sample entry", () => {
    const bytes = mp4Fixture([], isoBox("free", new TextEncoder().encode("avc1 mp4a")));
    expect(() => inspectRecordingContainer(bytes, "video/mp4")).toThrow(/H\.264 video and AAC audio/);
  });

  it("validates the WebM fallback signature without claiming MP4", () => {
    const bytes = new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, ...new TextEncoder().encode("V_VP9A_OPUS")]);
    expect(inspectRecordingContainer(bytes, "video/webm;codecs=vp9,opus")).toEqual({
      container: "webm",
      videoCodec: "vp9",
      audioCodec: "opus",
    });
  });

  it("locks the recorder synchronously while startup awaits the first frame", async () => {
    let releaseFrame: FrameRequestCallback | undefined;
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      releaseFrame = callback;
      return 1;
    });

    const videoTrack = { stop: vi.fn() };
    const audioTrack = { stop: vi.fn() };
    class FakeStream {
      constructor(private readonly tracks: unknown[] = []) {}
      getVideoTracks() { return this.tracks.includes(videoTrack) ? [videoTrack] : []; }
      getAudioTracks() { return this.tracks.includes(audioTrack) ? [audioTrack] : []; }
    }
    class FakeMediaRecorder {
      static isTypeSupported(mime: string) { return mime.startsWith("video/mp4"); }
      state = "inactive";
      mimeType: string;
      ondataavailable: ((event: BlobEvent) => void) | null = null;
      onerror: ((event: Event & { error?: DOMException }) => void) | null = null;
      onstop: (() => void) | null = null;
      constructor(public readonly stream: FakeStream, options: MediaRecorderOptions) {
        this.mimeType = options.mimeType ?? "video/mp4";
      }
      start() { this.state = "recording"; }
      requestData() {}
      stop() {
        this.state = "inactive";
        queueMicrotask(() => this.onstop?.());
      }
    }
    vi.stubGlobal("MediaStream", FakeStream);
    vi.stubGlobal("MediaRecorder", FakeMediaRecorder);

    const recorder = new SocialRecorder(
      { captureStream: () => new FakeStream([videoTrack]) } as unknown as HTMLCanvasElement,
      { getCaptureStream: () => new FakeStream([audioTrack]) } as never,
      { lockResolution: vi.fn(), unlockResolution: vi.fn() } as never,
    );
    const firstStart = recorder.start(SOCIAL_PRESETS[0]);
    expect(recorder.getState()).toBe("starting");
    await expect(recorder.start(SOCIAL_PRESETS[0])).rejects.toThrow(/already active/);
    releaseFrame?.(performance.now());
    await firstStart;
    expect(recorder.getState()).toBe("recording");
    await expect(recorder.stop()).rejects.toThrow(/empty or unsupported/);
    expect(recorder.getState()).toBe("idle");
  });
});
