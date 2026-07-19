import { isTauri } from "@tauri-apps/api/core";
import type { AudioEngine } from "../audio/AudioEngine";
import type { SocialPreset } from "../core/types";
import type { VisualEngine } from "../visual/VisualEngine";

export interface RecordingResult {
  blob: Blob;
  mimeType: string;
  fileName: string;
  durationSeconds: number;
  bytes: number;
  container: "mp4" | "webm";
  videoCodec: "h264" | "vp8" | "vp9" | "unknown";
  audioCodec: "aac" | "opus" | "unknown";
  targetWidth: number;
  targetHeight: number;
  targetFps: number;
}

export type RecorderState = "idle" | "starting" | "recording" | "finalizing";

const MIME_CANDIDATES = [
  "video/mp4;codecs=h264,aac",
  "video/mp4",
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
];
const FINALIZE_TIMEOUT_MS = 15_000;

export function chooseRecordingMime(isSupported: (mime: string) => boolean): string {
  return MIME_CANDIDATES.find((candidate) => isSupported(candidate)) ?? "";
}

export interface RecordingContainerInspection {
  container: "mp4" | "webm";
  videoCodec: "h264" | "vp8" | "vp9" | "unknown";
  audioCodec: "aac" | "opus" | "unknown";
}

function containsAscii(bytes: Uint8Array, value: string): boolean {
  const needle = new TextEncoder().encode(value);
  outer: for (let offset = 0; offset <= bytes.length - needle.length; offset += 1) {
    for (let index = 0; index < needle.length; index += 1) {
      if (bytes[offset + index] !== needle[index]) continue outer;
    }
    return true;
  }
  return false;
}

interface IsoBox {
  type: string;
  payloadStart: number;
  end: number;
}

function readIsoBox(bytes: Uint8Array, offset: number, parentEnd: number): IsoBox {
  if (offset + 8 > parentEnd) throw new Error("Truncated ISO media box header");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const size32 = view.getUint32(offset, false);
  const type = new TextDecoder("ascii", { fatal: true }).decode(bytes.subarray(offset + 4, offset + 8));
  let headerBytes = 8;
  let size = size32;
  if (size32 === 1) {
    if (offset + 16 > parentEnd) throw new Error("Truncated extended ISO media box header");
    const extended = view.getBigUint64(offset + 8, false);
    if (extended > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("ISO media box is too large to inspect safely");
    size = Number(extended);
    headerBytes = 16;
  } else if (size32 === 0) {
    size = parentEnd - offset;
  }
  if (size < headerBytes || offset + size > parentEnd) throw new Error(`Invalid ISO media ${type || "unknown"} box`);
  return { type, payloadStart: offset + headerBytes, end: offset + size };
}

function inspectMp4SampleEntries(bytes: Uint8Array): { hasFtyp: boolean; entries: Set<string> } {
  const entries = new Set<string>();
  let hasFtyp = false;
  const containers = new Set(["moov", "trak", "mdia", "minf", "stbl"]);

  const walk = (start: number, end: number, depth: number): void => {
    if (depth > 8) throw new Error("ISO media box nesting exceeds the inspection limit");
    let offset = start;
    while (offset < end) {
      const box = readIsoBox(bytes, offset, end);
      if (depth === 0 && box.type === "ftyp") hasFtyp = true;
      if (box.type === "stsd") {
        if (box.payloadStart + 8 > box.end) throw new Error("Invalid ISO media sample description box");
        const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        const entryCount = view.getUint32(box.payloadStart + 4, false);
        if (entryCount > 64) throw new Error("ISO media contains too many sample descriptions");
        let entryOffset = box.payloadStart + 8;
        for (let index = 0; index < entryCount; index += 1) {
          const entry = readIsoBox(bytes, entryOffset, box.end);
          entries.add(entry.type);
          entryOffset = entry.end;
        }
        if (entryOffset > box.end) throw new Error("Invalid ISO media sample descriptions");
      } else if (containers.has(box.type)) {
        walk(box.payloadStart, box.end, depth + 1);
      }
      offset = box.end;
    }
  };

  walk(0, bytes.length, 0);
  return { hasFtyp, entries };
}

export function inspectRecordingContainer(
  input: ArrayBuffer | Uint8Array,
  declaredMimeType: string,
): RecordingContainerInspection {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (declaredMimeType.startsWith("video/mp4")) {
    const { hasFtyp, entries } = inspectMp4SampleEntries(bytes);
    if (!hasFtyp) {
      throw new Error("The recorder declared MP4 but returned an invalid ISO media container");
    }
    const videoCodec = entries.has("avc1") || entries.has("avc3") ? "h264" : "unknown";
    const audioCodec = entries.has("mp4a") ? "aac" : "unknown";
    if (videoCodec !== "h264" || audioCodec !== "aac") {
      throw new Error("Desktop social export requires H.264 video and AAC audio");
    }
    return { container: "mp4", videoCodec, audioCodec };
  }
  if (declaredMimeType.startsWith("video/webm")) {
    if (bytes.length < 4 || bytes[0] !== 0x1a || bytes[1] !== 0x45 || bytes[2] !== 0xdf || bytes[3] !== 0xa3) {
      throw new Error("The recorder declared WebM but returned an invalid EBML container");
    }
    const videoCodec = containsAscii(bytes, "V_VP9") ? "vp9" : containsAscii(bytes, "V_VP8") ? "vp8" : "unknown";
    const audioCodec = containsAscii(bytes, "A_OPUS") ? "opus" : "unknown";
    return { container: "webm", videoCodec, audioCodec };
  }
  throw new Error("The recorder returned an unsupported media type");
}

export class SocialRecorder {
  private recorder?: MediaRecorder;
  private chunks: Blob[] = [];
  private state: RecorderState = "idle";
  private preset?: SocialPreset;
  private startedAt = 0;
  private autoStopTimer?: number;
  private finalizeTimer?: number;
  private stopPromise?: Promise<RecordingResult>;
  private stopResolver?: (value: RecordingResult) => void;
  private stopRejecter?: (reason: unknown) => void;
  private resultListeners = new Set<(result: RecordingResult) => void>();
  private errorListeners = new Set<(error: Error) => void>();

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly audio: AudioEngine,
    private readonly visuals: VisualEngine,
  ) {}

  getState(): RecorderState {
    return this.state;
  }

  getProgress(): number {
    if (this.state !== "recording" || !this.preset) return 0;
    return Math.min(1, (performance.now() - this.startedAt) / (this.preset.durationSeconds * 1_000));
  }

  subscribeResults(listener: (result: RecordingResult) => void): () => void {
    this.resultListeners.add(listener);
    return () => this.resultListeners.delete(listener);
  }

  subscribeErrors(listener: (error: Error) => void): () => void {
    this.errorListeners.add(listener);
    return () => this.errorListeners.delete(listener);
  }

  async start(preset: SocialPreset): Promise<void> {
    if (this.state !== "idle") throw new Error("A recording is already active");
    if (typeof MediaRecorder === "undefined" || typeof this.canvas.captureStream !== "function") {
      throw new Error("This webview does not support canvas recording");
    }

    const mimeType = chooseRecordingMime((mime) => MediaRecorder.isTypeSupported(mime));
    if (!mimeType) throw new Error("This webview does not provide a supported MP4 or WebM recorder codec");
    if (isTauri() && !mimeType.startsWith("video/mp4")) {
      throw new Error("This Mac webview cannot provide the required H.264 and AAC MP4 export");
    }
    this.state = "starting";
    this.visuals.lockResolution(preset.width, preset.height);
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

    let canvasStream: MediaStream | undefined;
    try {
      canvasStream = this.canvas.captureStream(preset.fps);
      const audioStream = this.audio.getCaptureStream();
      const videoTracks = canvasStream.getVideoTracks();
      const audioTracks = audioStream?.getAudioTracks() ?? [];
      if (videoTracks.length === 0 || audioTracks.length === 0) {
        throw new Error("Recording requires one video track and one audio track");
      }
      const tracks = [...videoTracks, ...audioTracks];

      const stream = new MediaStream(tracks);
      const options: MediaRecorderOptions = {
        videoBitsPerSecond: preset.videoBitsPerSecond,
        audioBitsPerSecond: 256_000,
      };
      options.mimeType = mimeType;

      this.chunks = [];
      this.preset = preset;
      this.recorder = new MediaRecorder(stream, options);
      this.recorder.ondataavailable = (event) => {
        if (event.data.size > 0) this.chunks.push(event.data);
      };
      this.recorder.onerror = (event) => {
        this.finishWithError(event.error ?? new Error("Recording failed"));
      };
      this.recorder.onstop = () => void this.finishRecording();
      this.state = "recording";
      this.startedAt = performance.now();
      this.recorder.start(500);
      this.autoStopTimer = window.setTimeout(() => void this.stop().catch(() => undefined), preset.durationSeconds * 1_000);
    } catch (error) {
      for (const track of canvasStream?.getVideoTracks() ?? []) track.stop();
      this.cleanup();
      throw error;
    }
  }

  stop(): Promise<RecordingResult> {
    if (this.stopPromise) return this.stopPromise;
    if (this.state !== "recording" || !this.recorder) return Promise.reject(new Error("No recording is active"));
    this.state = "finalizing";
    if (this.autoStopTimer !== undefined) window.clearTimeout(this.autoStopTimer);
    this.autoStopTimer = undefined;
    this.stopPromise = new Promise<RecordingResult>((resolve, reject) => {
      this.stopResolver = resolve;
      this.stopRejecter = reject;
    });
    const stopPromise = this.stopPromise;
    this.finalizeTimer = window.setTimeout(() => {
      if (this.state === "finalizing") {
        this.finishWithError(new Error("The recorder did not finalize within 15 seconds"));
      }
    }, FINALIZE_TIMEOUT_MS);
    try {
      this.recorder.requestData();
      this.recorder.stop();
    } catch (error) {
      this.finishWithError(error);
    }
    return stopPromise;
  }

  async save(result: RecordingResult): Promise<string | undefined> {
    if (isTauri()) {
      const [{ save }, { writeFile }] = await Promise.all([import("@tauri-apps/plugin-dialog"), import("@tauri-apps/plugin-fs")]);
      const extension = result.mimeType.includes("mp4") ? "mp4" : "webm";
      const path = await save({
        defaultPath: result.fileName,
        filters: [{ name: extension.toUpperCase(), extensions: [extension] }],
      });
      if (!path) return undefined;
      await writeFile(path, new Uint8Array(await result.blob.arrayBuffer()));
      return path;
    }

    const href = URL.createObjectURL(result.blob);
    const anchor = document.createElement("a");
    anchor.href = href;
    anchor.download = result.fileName;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(href), 1_000);
    return result.fileName;
  }

  private async finishRecording(): Promise<void> {
    const preset = this.preset;
    const recorder = this.recorder;
    if (!preset || !recorder) {
      this.finishWithError(new Error("Recording state was lost"));
      return;
    }
    const mimeType = recorder.mimeType || this.chunks[0]?.type || "";
    const blob = new Blob(this.chunks, { type: mimeType });
    if (blob.size === 0 || (!mimeType.startsWith("video/mp4") && !mimeType.startsWith("video/webm"))) {
      this.finishWithError(new Error("The recorder returned an empty or unsupported media file"));
      return;
    }
    let inspection: RecordingContainerInspection;
    try {
      inspection = inspectRecordingContainer(await blob.arrayBuffer(), mimeType);
    } catch (error) {
      this.finishWithError(error);
      return;
    }
    const extension = inspection.container;
    const result: RecordingResult = {
      blob,
      mimeType,
      fileName: `musica-vj-${preset.id}-${new Date().toISOString().replace(/[:.]/g, "")}.${extension}`,
      durationSeconds: (performance.now() - this.startedAt) / 1_000,
      bytes: blob.size,
      ...inspection,
      targetWidth: preset.width,
      targetHeight: preset.height,
      targetFps: preset.fps,
    };
    this.cleanup();
    for (const listener of this.resultListeners) listener(result);
    this.stopResolver?.(result);
    this.stopResolver = undefined;
    this.stopRejecter = undefined;
  }

  private finishWithError(error: unknown): void {
    const normalized = error instanceof Error ? error : new Error(String(error));
    const recorder = this.recorder;
    if (recorder) {
      recorder.onerror = null;
      recorder.onstop = null;
      if (recorder.state !== "inactive") {
        try {
          recorder.stop();
        } catch {
          // The original recorder error is more useful than a secondary stop error.
        }
      }
    }
    this.cleanup();
    for (const listener of this.errorListeners) listener(normalized);
    this.stopRejecter?.(normalized);
    this.stopResolver = undefined;
    this.stopRejecter = undefined;
  }

  private cleanup(): void {
    if (this.autoStopTimer !== undefined) window.clearTimeout(this.autoStopTimer);
    this.autoStopTimer = undefined;
    if (this.finalizeTimer !== undefined) window.clearTimeout(this.finalizeTimer);
    this.finalizeTimer = undefined;
    for (const track of this.recorder?.stream.getVideoTracks() ?? []) track.stop();
    this.visuals.unlockResolution();
    this.recorder = undefined;
    this.preset = undefined;
    this.chunks = [];
    this.state = "idle";
    this.stopPromise = undefined;
  }
}
