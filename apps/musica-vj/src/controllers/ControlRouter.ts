import { isTauri } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { ControlAction, ControlMessage } from "../core/types";

export interface ControllerStatus {
  keyboard: boolean;
  globalShortcuts: boolean;
  logitechBridge: boolean;
  midi: boolean;
  midiInputs: string[];
}

type ControlListener = (message: ControlMessage) => void;
type StatusListener = (status: ControllerStatus) => void;

const KEYBOARD_ACTIONS: Record<string, { action: ControlAction; value?: number }> = {
  Space: { action: "transport.toggle" },
  KeyR: { action: "transport.record" },
  KeyT: { action: "tempo.tap" },
  KeyM: { action: "track.mute" },
  KeyS: { action: "track.solo" },
  Enter: { action: "track.trigger" },
  ArrowUp: { action: "track.previous" },
  ArrowDown: { action: "track.next" },
  ArrowLeft: { action: "visual.previous" },
  ArrowRight: { action: "visual.next" },
  BracketLeft: { action: "visual.intensity.delta", value: -1 },
  BracketRight: { action: "visual.intensity.delta", value: 1 },
  Minus: { action: "master.delta", value: -1 },
  Equal: { action: "master.delta", value: 1 },
  Digit1: { action: "visual.scene.select", value: 0 },
  Digit2: { action: "visual.scene.select", value: 1 },
  Digit3: { action: "visual.scene.select", value: 2 },
  Digit4: { action: "visual.scene.select", value: 3 },
  Digit5: { action: "visual.scene.select", value: 4 },
  Digit6: { action: "visual.scene.select", value: 5 },
  Digit7: { action: "visual.scene.select", value: 6 },
  Digit8: { action: "visual.scene.select", value: 7 },
};

const GLOBAL_SHORTCUTS: Array<[string, ControlAction, number?]> = [
  ["F13", "transport.toggle"],
  ["F14", "transport.record"],
  ["F15", "track.previous"],
  ["F16", "track.next"],
  ["F17", "track.mute"],
  ["F18", "track.solo"],
  ["F19", "visual.previous"],
  ["F20", "visual.next"],
  ["F21", "visual.intensity.delta", -1],
  ["F22", "visual.intensity.delta", 1],
  ["F23", "master.delta", -1],
  ["F24", "master.delta", 1],
];

export class ControlRouter {
  private controlListeners = new Set<ControlListener>();
  private statusListeners = new Set<StatusListener>();
  private unlistenBridge?: UnlistenFn;
  private midi?: MIDIAccess;
  private started = false;
  private lifecycle: Promise<void> = Promise.resolve();
  private status: ControllerStatus = {
    keyboard: true,
    globalShortcuts: false,
    logitechBridge: false,
    midi: false,
    midiInputs: [],
  };

  start(): Promise<void> {
    const operation = this.lifecycle.then(() => this.startNow());
    this.lifecycle = operation.catch(() => undefined);
    return operation;
  }

  stop(): Promise<void> {
    const operation = this.lifecycle.then(() => this.stopNow());
    this.lifecycle = operation.catch(() => undefined);
    return operation;
  }

  private async startNow(): Promise<void> {
    if (this.started) return;
    this.started = true;
    window.addEventListener("keydown", this.onKeyDown);
    await Promise.allSettled([this.startTauriInputs(), this.startMidi()]);
  }

  private async stopNow(): Promise<void> {
    if (!this.started) return;
    this.started = false;
    window.removeEventListener("keydown", this.onKeyDown);
    this.unlistenBridge?.();
    this.unlistenBridge = undefined;
    if (this.status.globalShortcuts && isTauri()) {
      const { unregisterAll } = await import("@tauri-apps/plugin-global-shortcut");
      await unregisterAll().catch(() => undefined);
    }
    for (const input of this.midi?.inputs.values() ?? []) input.onmidimessage = null;
    if (this.midi) this.midi.onstatechange = null;
    this.midi = undefined;
    this.status = {
      keyboard: true,
      globalShortcuts: false,
      logitechBridge: false,
      midi: false,
      midiInputs: [],
    };
    this.emitStatus();
  }

  subscribe(listener: ControlListener): () => void {
    this.controlListeners.add(listener);
    return () => this.controlListeners.delete(listener);
  }

  subscribeStatus(listener: StatusListener): () => void {
    this.statusListeners.add(listener);
    listener({ ...this.status, midiInputs: [...this.status.midiInputs] });
    return () => this.statusListeners.delete(listener);
  }

  dispatch(action: ControlAction, value?: number, source: ControlMessage["source"] = "ui"): void {
    const message: ControlMessage = { action, value, source, timestamp: performance.now() };
    for (const listener of this.controlListeners) listener(message);
  }

  private onKeyDown = (event: KeyboardEvent): void => {
    if (event.repeat && !["BracketLeft", "BracketRight", "Minus", "Equal"].includes(event.code)) return;
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return;
    const mapping = KEYBOARD_ACTIONS[event.code];
    if (!mapping) return;
    event.preventDefault();
    this.dispatch(mapping.action, mapping.value, "keyboard");
  };

  private async startTauriInputs(): Promise<void> {
    if (!isTauri()) return;
    this.unlistenBridge = await listen<{ action: ControlAction; value?: number; source?: "logitech" | "shortcut" }>(
      "controller://action",
      (event) => {
        this.status.logitechBridge = event.payload.source === "logitech" || this.status.logitechBridge;
        this.emitStatus();
        this.dispatch(event.payload.action, event.payload.value, event.payload.source ?? "logitech");
      },
    );

    try {
      const { register } = await import("@tauri-apps/plugin-global-shortcut");
      for (const [shortcut, action, value] of GLOBAL_SHORTCUTS) {
        await register(shortcut, (event) => {
          if (event.state === "Pressed") this.dispatch(action, value, "shortcut");
        });
      }
      this.status.globalShortcuts = true;
      this.emitStatus();
    } catch (error) {
      console.warn("Global shortcuts are unavailable", error);
      const { unregisterAll } = await import("@tauri-apps/plugin-global-shortcut");
      await unregisterAll().catch(() => undefined);
      this.status.globalShortcuts = false;
      this.emitStatus();
    }
  }

  private async startMidi(): Promise<void> {
    if (!navigator.requestMIDIAccess) return;
    try {
      this.midi = await navigator.requestMIDIAccess({ sysex: false, software: false });
      this.bindMidiInputs();
      this.midi.onstatechange = () => this.bindMidiInputs();
    } catch (error) {
      console.info("MIDI access is unavailable in this webview", error);
    }
  }

  private bindMidiInputs(): void {
    if (!this.midi) return;
    const inputNames: string[] = [];
    for (const input of this.midi.inputs.values()) {
      inputNames.push(input.name ?? input.id);
      input.onmidimessage = (event) => this.onMidiMessage(event);
    }
    this.status.midiInputs = inputNames;
    this.status.midi = inputNames.length > 0;
    this.emitStatus();
  }

  private onMidiMessage(event: MIDIMessageEvent): void {
    const data = event.data;
    if (!data || data.length < 2) return;
    const status = data[0] & 0xf0;
    const key = data[1];
    const value = data[2] ?? 0;
    if (status === 0x90 && value > 0) {
      if (key >= 36 && key <= 41) this.dispatch("track.trigger", key - 36, "midi");
      else if (key === 42) this.dispatch("transport.toggle", 0, "midi");
      else if (key === 43) this.dispatch("transport.record", 0, "midi");
      else if (key >= 48 && key <= 55) this.dispatch("visual.scene.select", key - 48, "midi");
      else if (key >= 60 && key <= 71) this.dispatch("performance.template.select", key - 60, "midi");
    } else if (status === 0xb0) {
      const normalized = (value - 64) / 63;
      if (key === 1) this.dispatch("master.delta", normalized, "midi");
      if (key === 2) this.dispatch("visual.intensity.delta", normalized, "midi");
      if (key === 3) this.dispatch("tempo.delta", normalized, "midi");
      if (key === 4) this.dispatch("visual.sculpture.delta", normalized, "midi");
      if (key === 5) this.dispatch("visual.motion.delta", normalized, "midi");
      if (key === 6) this.dispatch("visual.atmosphere.delta", normalized, "midi");
      if (key === 7) this.dispatch("visual.ribbon.delta", normalized, "midi");
      if (key === 8) this.dispatch("visual.temporal.speed.delta", normalized, "midi");
      if (key === 9) this.dispatch("visual.temporal.strobe.delta", normalized, "midi");
      if (key === 10) this.dispatch("visual.temporal.trail.delta", normalized, "midi");
      if (key === 11) this.dispatch("visual.temporal.morph.delta", normalized, "midi");
      if (key === 12) this.dispatch("visual.temporal.camera.delta", normalized, "midi");
      if (key === 13) this.dispatch("visual.temporal.phase.delta", normalized, "midi");
    }
  }

  private emitStatus(): void {
    const snapshot = { ...this.status, midiInputs: [...this.status.midiInputs] };
    for (const listener of this.statusListeners) listener(snapshot);
  }
}

export class TapTempo {
  private taps: number[] = [];

  tap(at = performance.now()): number | undefined {
    this.taps = this.taps.filter((tap) => at - tap <= 3_000);
    this.taps.push(at);
    if (this.taps.length < 2) return undefined;
    const intervals = this.taps.slice(1).map((tap, index) => tap - this.taps[index]);
    const sorted = [...intervals].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    if (median <= 0) return undefined;
    return Math.round(Math.min(200, Math.max(60, 60_000 / median)));
  }
}
