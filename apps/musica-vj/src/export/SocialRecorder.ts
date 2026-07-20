import { isTauri } from "@tauri-apps/api/core";
import type { AudioEngine } from "../audio/AudioEngine";
import type { SocialPreset } from "../core/types";
import type { VisualEngine } from "../visual/VisualEngine";

export interface RecordingResult {
  mode: CaptureMode;
  blob: Blob;
  mimeType: string;
  fileName: string;
  durationSeconds: number;
  bytes: number;
  container: "mp4" | "webm";
  fileExtension: "mp4" | "m4a" | "webm";
  videoCodec: "h264" | "vp8" | "vp9" | "none" | "unknown";
  audioCodec: "aac" | "opus" | "unknown";
  targetWidth: number;
  targetHeight: number;
  targetFps: number;
}

export type CaptureMode = "video-audio" | "audio-only";
export type RecorderState = "idle" | "starting" | "recording" | "finalizing";

const VIDEO_MIME_CANDIDATES = [
  "video/mp4;codecs=h264,aac",
  "video/mp4",
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
];
const AUDIO_MIME_CANDIDATES = [
  "audio/mp4;codecs=mp4a.40.2",
  "audio/mp4",
  "audio/webm;codecs=opus",
  "audio/webm",
];
const FINALIZE_TIMEOUT_MS = 15_000;

export function chooseRecordingMime(
  isSupported: (mime: string) => boolean,
  mode: CaptureMode = "video-audio",
): string {
  const candidates = mode === "audio-only" ? AUDIO_MIME_CANDIDATES : VIDEO_MIME_CANDIDATES;
  return candidates.find((candidate) => isSupported(candidate)) ?? "";
}

/// Transcodes a WebM clip to H.264/AAC MP4 via the bundled FFmpeg and saves it
/// through a native dialog (desktop only). Returns the saved path, or undefined
/// if cancelled. Lets Windows/Linux captures export a portable MP4.
export async function transcodeWebmToMp4(blob: Blob): Promise<string | undefined> {
  if (!isTauri()) throw new Error("MP4 export requires the desktop app");
  const { invoke } = await import("@tauri-apps/api/core");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const path = await invoke<string | null>("transcode_to_mp4", bytes);
  return path ?? undefined;
}

/// Writes a media blob to disk: a native Save dialog in the desktop app, or a
/// browser download otherwise. Returns the chosen path/name, or undefined if
/// the user cancelled. Shared by live capture and the capture library.
export async function saveMediaBlob(blob: Blob, fileName: string, fileExtension: string): Promise<string | undefined> {
  if (isTauri()) {
    const [{ save }, { writeFile }] = await Promise.all([import("@tauri-apps/plugin-dialog"), import("@tauri-apps/plugin-fs")]);
    const path = await save({
      defaultPath: fileName,
      filters: [{ name: fileExtension.toUpperCase(), extensions: [fileExtension] }],
    });
    if (!path) return undefined;
    await writeFile(path, new Uint8Array(await blob.arrayBuffer()));
    return path;
  }
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = fileName;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(href), 1_000);
  return fileName;
}

export interface RecordingContainerInspection {
  container: "mp4" | "webm";
  videoCodec: "h264" | "vp8" | "vp9" | "none" | "unknown";
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
  mode: CaptureMode = "video-audio",
): RecordingContainerInspection {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  if (declaredMimeType.includes("mp4")) {
    const { hasFtyp, entries } = inspectMp4SampleEntries(bytes);
    if (!hasFtyp) {
      throw new Error("The recorder declared MP4 but returned an invalid ISO media container");
    }
    const videoCodec = entries.has("avc1") || entries.has("avc3") ? "h264" : "unknown";
    const audioCodec = entries.has("mp4a") ? "aac" : "unknown";
    if (audioCodec !== "aac" || (mode === "video-audio" && videoCodec !== "h264")) {
      throw new Error(mode === "audio-only" ? "Audio-only MP4 export requires AAC audio" : "Desktop social export requires H.264 video and AAC audio");
    }
    return { container: "mp4", videoCodec: mode === "audio-only" ? "none" : videoCodec, audioCodec };
  }
  if (declaredMimeType.includes("webm")) {
    if (bytes.length < 4 || bytes[0] !== 0x1a || bytes[1] !== 0x45 || bytes[2] !== 0xdf || bytes[3] !== 0xa3) {
      throw new Error("The recorder declared WebM but returned an invalid EBML container");
    }
    const detectedVideoCodec = containsAscii(bytes, "V_VP9") ? "vp9" : containsAscii(bytes, "V_VP8") ? "vp8" : "unknown";
    const audioCodec = containsAscii(bytes, "A_OPUS") ? "opus" : "unknown";
    if (mode === "audio-only" && audioCodec !== "opus") throw new Error("Audio-only WebM export requires Opus audio");
    return { container: "webm", videoCodec: mode === "audio-only" ? "none" : detectedVideoCodec, audioCodec };
  }
  throw new Error("The recorder returned an unsupported media type");
}

export class SocialRecorder {
  private recorder?: MediaRecorder;
  private chunks: Blob[] = [];
  private state: RecorderState = "idle";
  private preset?: SocialPreset;
  private mode: CaptureMode = "video-audio";
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

  async start(preset: SocialPreset, mode: CaptureMode = "video-audio"): Promise<void> {
    if (this.state !== "idle") throw new Error("A recording is already active");
    if (typeof MediaRecorder === "undefined" || (mode === "video-audio" && typeof this.canvas.captureStream !== "function")) {
      throw new Error(`This webview does not support ${mode === "audio-only" ? "audio" : "canvas"} recording`);
    }

    const mimeType = chooseRecordingMime((mime) => MediaRecorder.isTypeSupported(mime), mode);
    if (!mimeType) throw new Error(`This webview does not provide a supported ${mode === "audio-only" ? "AAC or Opus audio" : "MP4 or WebM video"} recorder codec`);
    // Only macOS (WKWebView) is expected to produce H.264/AAC MP4; if it falls
    // back to WebM there, something is wrong. Windows (WebView2) and Linux
    // (WebKitGTK) provide VP8/VP9 WebM via MediaRecorder, which is a fully
    // supported export container here — so accept it rather than blocking
    // capture entirely (the "no Save button" symptom on Windows/Linux).
    const isMacWebview = typeof navigator !== "undefined" && /Macintosh|Mac OS X/i.test(navigator.userAgent);
    if (mode === "video-audio" && isTauri() && isMacWebview && !mimeType.startsWith("video/mp4")) {
      throw new Error("This macOS webview cannot provide the required H.264 and AAC MP4 export");
    }
    this.state = "starting";
    this.mode = mode;
    if (mode === "video-audio") {
      this.visuals.lockResolution(preset.width, preset.height);
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    }

    let canvasStream: MediaStream | undefined;
    try {
      const audioStream = this.audio.getCaptureStream();
      const audioTracks = audioStream?.getAudioTracks() ?? [];
      if (audioTracks.length === 0) throw new Error("Recording requires the master audio track");
      let videoTracks: MediaStreamTrack[] = [];
      if (mode === "video-audio") {
        canvasStream = this.canvas.captureStream(preset.fps);
        videoTracks = canvasStream.getVideoTracks();
        if (videoTracks.length === 0) throw new Error("Video capture requires one canvas video track");
      }
      const tracks = [...videoTracks, ...audioTracks];

      const stream = new MediaStream(tracks);
      const options: MediaRecorderOptions = {
        audioBitsPerSecond: 256_000,
      };
      if (mode === "video-audio") options.videoBitsPerSecond = preset.videoBitsPerSecond;
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
    return saveMediaBlob(result.blob, result.fileName, result.fileExtension);
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
    if (blob.size === 0 || !["video/mp4", "video/webm", "audio/mp4", "audio/webm"].some((type) => mimeType.startsWith(type))) {
      this.finishWithError(new Error("The recorder returned an empty or unsupported media file"));
      return;
    }
    let inspection: RecordingContainerInspection;
    try {
      inspection = inspectRecordingContainer(await blob.arrayBuffer(), mimeType, this.mode);
    } catch (error) {
      this.finishWithError(error);
      return;
    }
    const extension: RecordingResult["fileExtension"] = this.mode === "audio-only" && inspection.container === "mp4" ? "m4a" : inspection.container;
    const result: RecordingResult = {
      mode: this.mode,
      blob,
      mimeType,
      fileName: `musica-${this.mode === "audio-only" ? "audio" : "vj"}-${preset.id}-${new Date().toISOString().replace(/[:.]/g, "")}.${extension}`,
      fileExtension: extension,
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
