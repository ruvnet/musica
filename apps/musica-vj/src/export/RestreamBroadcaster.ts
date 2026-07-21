import { invoke, isTauri } from "@tauri-apps/api/core";
import type { AudioEngine } from "../audio/AudioEngine";
import { supportsDisplayCapture } from "../core/platform";
import type { VisualEngine } from "../visual/VisualEngine";

export type RestreamSource = "program" | "window";

export interface RestreamStatus {
  available: boolean;
  active: boolean;
  source?: RestreamSource;
  encoder?: string;
  reason?: string;
}

export interface RestreamConfig {
  ingestUrl: string;
  streamKey: string;
  source: RestreamSource;
  videoBitrateKbps: number;
  fps: number;
}

const BROADCAST_MIME_CANDIDATES = [
  "video/webm;codecs=vp8,opus",
  "video/webm;codecs=vp9,opus",
  "video/webm",
  "video/mp4;codecs=h264,aac",
  "video/mp4",
];

function chooseBroadcastMime(): string {
  if (typeof MediaRecorder === "undefined") return "";
  return BROADCAST_MIME_CANDIDATES.find((mime) => MediaRecorder.isTypeSupported(mime)) ?? "";
}

export async function getRestreamStatus(): Promise<RestreamStatus> {
  if (!isTauri()) {
    return { available: false, active: false, reason: "Restream RTMPS output is available in the desktop app" };
  }
  return invoke<RestreamStatus>("restream_status");
}

export class RestreamBroadcaster {
  private recorder?: MediaRecorder;
  private sourceStream?: MediaStream;
  private pushChain: Promise<void> = Promise.resolve();
  private errorListeners = new Set<(error: Error) => void>();
  private currentSource?: RestreamSource;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly audio: AudioEngine,
    private readonly visuals: VisualEngine,
  ) {}

  subscribeErrors(listener: (error: Error) => void): () => void {
    this.errorListeners.add(listener);
    return () => this.errorListeners.delete(listener);
  }

  private emitError(error: unknown): void {
    const normalized = error instanceof Error ? error : new Error(String(error));
    for (const listener of this.errorListeners) listener(normalized);
  }

  async start(config: RestreamConfig): Promise<RestreamStatus> {
    if (!isTauri()) throw new Error("Restream RTMPS output requires the desktop app");
    if (this.recorder?.state === "recording") throw new Error("A Restream broadcast is already active");
    const mimeType = chooseBroadcastMime();
    if (!mimeType) throw new Error("This webview does not provide a live H.264/VP8 media source");
    const audioTracks = this.audio.getCaptureStream()?.getAudioTracks() ?? [];
    if (audioTracks.length === 0) throw new Error("Start the Lyria transport before going live so master audio is available");

    let sourceStream: MediaStream;
    if (config.source === "program") {
      if (typeof this.canvas.captureStream !== "function") throw new Error("Clean visual capture is unavailable in this webview");
      this.visuals.lockResolution(1920, 1080);
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
      sourceStream = this.canvas.captureStream(config.fps);
    } else {
      if (!supportsDisplayCapture()) throw new Error("Window capture is unavailable in this webview");
      sourceStream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: { ideal: config.fps, max: config.fps } },
        audio: false,
      });
    }

    const videoTracks = sourceStream.getVideoTracks();
    if (videoTracks.length === 0) {
      for (const track of sourceStream.getTracks()) track.stop();
      if (config.source === "program") this.visuals.unlockResolution();
      throw new Error("The selected Restream source did not provide a video track");
    }

    try {
      const status = await invoke<RestreamStatus>("restream_start", {
        request: {
          ingestUrl: config.ingestUrl,
          streamKey: config.streamKey,
          source: config.source,
          videoBitrateKbps: config.videoBitrateKbps,
          fps: config.fps,
        },
      });
      const mediaStream = new MediaStream([...videoTracks, ...audioTracks]);
      const recorder = new MediaRecorder(mediaStream, {
        mimeType,
        videoBitsPerSecond: config.videoBitrateKbps * 1_000,
        audioBitsPerSecond: 192_000,
      });
      recorder.ondataavailable = (event) => {
        if (event.data.size === 0) return;
        this.pushChain = this.pushChain
          .then(async () => {
            const chunk = new Uint8Array(await event.data.arrayBuffer());
            await invoke<void>("restream_push_chunk", chunk);
          })
          .catch((error) => this.emitError(error));
      };
      recorder.onerror = (event) => this.emitError(event.error ?? new Error("Live encoder capture failed"));
      recorder.start(1_000);
      this.recorder = recorder;
      this.sourceStream = sourceStream;
      this.currentSource = config.source;
      return status;
    } catch (error) {
      for (const track of sourceStream.getTracks()) track.stop();
      if (config.source === "program") this.visuals.unlockResolution();
      await invoke<RestreamStatus>("restream_stop").catch(() => undefined);
      throw error;
    }
  }

  async stop(): Promise<RestreamStatus> {
    const recorder = this.recorder;
    if (recorder && recorder.state !== "inactive") {
      await new Promise<void>((resolve) => {
        recorder.addEventListener("stop", () => resolve(), { once: true });
        recorder.requestData();
        recorder.stop();
      });
    }
    await this.pushChain;
    const status = isTauri()
      ? await invoke<RestreamStatus>("restream_stop")
      : { available: false, active: false, reason: "Desktop app required" };
    this.cleanup();
    return status;
  }

  async dispose(): Promise<void> {
    if (this.recorder?.state === "recording") await this.stop().catch(() => undefined);
    this.cleanup();
  }

  private cleanup(): void {
    for (const track of this.sourceStream?.getTracks() ?? []) track.stop();
    if (this.currentSource === "program") this.visuals.unlockResolution();
    this.recorder = undefined;
    this.sourceStream = undefined;
    this.currentSource = undefined;
    this.pushChain = Promise.resolve();
  }
}
