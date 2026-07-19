import { isTauri } from "@tauri-apps/api/core";
import { emit, listen } from "@tauri-apps/api/event";
import type { ControlAction, VisualColorControls, VisualSceneId } from "./types";
import type { LyriaRealtimeDeckId } from "./lyriaRealtime";
import type { LyriaDeckControl, LyriaDeckScene } from "./lyriaDeckScenes";

export type DjControlProfileId = "mixer" | "launcher" | "visual";
export type DjControlWidgetId = "transport" | "deck-scenes" | "deck-mixer" | "styles" | "visuals" | "color" | "master";

export interface DjControlWidgetLayout {
  id: DjControlWidgetId;
  visible: boolean;
  wide: boolean;
}

export interface DjControlState {
  playing: boolean;
  bpm: number;
  masterVolume: number;
  styleId: string;
  activeDeckSceneId?: string;
  deckScenes: LyriaDeckScene[];
  deckEnabled: Record<LyriaRealtimeDeckId, boolean>;
  deckControls: Record<LyriaRealtimeDeckId, LyriaDeckControl>;
  visualScene: VisualSceneId;
  visualIntensity: number;
  visualColor: VisualColorControls;
}

export type DjControlCommand =
  | { type: "action"; action: ControlAction; value?: number }
  | { type: "deck-control"; deck: LyriaRealtimeDeckId; update: Partial<LyriaDeckControl> }
  | { type: "deck-enabled"; deck: LyriaRealtimeDeckId; enabled: boolean }
  | { type: "style"; styleId: string }
  | { type: "bpm"; value: number }
  | { type: "visual-color"; update: Partial<VisualColorControls> }
  | { type: "request-state" };

export const DJ_CONTROL_PROFILES: Record<DjControlProfileId, { label: string; widgets: DjControlWidgetLayout[] }> = {
  mixer: {
    label: "Mixer",
    widgets: [
      { id: "transport", visible: true, wide: true },
      { id: "deck-scenes", visible: true, wide: true },
      { id: "deck-mixer", visible: true, wide: true },
      { id: "styles", visible: true, wide: false },
      { id: "master", visible: true, wide: false },
      { id: "visuals", visible: false, wide: true },
      { id: "color", visible: false, wide: true },
    ],
  },
  launcher: {
    label: "Launcher",
    widgets: [
      { id: "transport", visible: true, wide: true },
      { id: "deck-scenes", visible: true, wide: true },
      { id: "styles", visible: true, wide: true },
      { id: "visuals", visible: true, wide: true },
      { id: "color", visible: true, wide: true },
      { id: "master", visible: true, wide: false },
      { id: "deck-mixer", visible: false, wide: true },
    ],
  },
  visual: {
    label: "Visual",
    widgets: [
      { id: "transport", visible: true, wide: true },
      { id: "visuals", visible: true, wide: true },
      { id: "color", visible: true, wide: true },
      { id: "master", visible: true, wide: true },
      { id: "deck-scenes", visible: true, wide: true },
      { id: "styles", visible: false, wide: true },
      { id: "deck-mixer", visible: false, wide: true },
    ],
  },
};

const COMMAND_EVENT = "dj-control://command";
const STATE_EVENT = "dj-control://state";
const BROWSER_CHANNEL = "musica-dj-control-v1";
const ALL_WIDGETS: DjControlWidgetId[] = ["transport", "deck-scenes", "deck-mixer", "styles", "visuals", "color", "master"];

export function normalizeDjControlLayout(profile: DjControlProfileId, value?: unknown): DjControlWidgetLayout[] {
  const fallback = DJ_CONTROL_PROFILES[profile].widgets;
  if (!Array.isArray(value)) return fallback.map((widget) => ({ ...widget }));
  const normalized = value
    .filter((candidate): candidate is Partial<DjControlWidgetLayout> => typeof candidate === "object" && candidate !== null)
    .filter((candidate) => ALL_WIDGETS.includes(candidate.id as DjControlWidgetId))
    .map((candidate) => {
      const base = fallback.find((widget) => widget.id === candidate.id) ?? { id: candidate.id as DjControlWidgetId, visible: true, wide: true };
      return { id: base.id, visible: candidate.visible ?? base.visible, wide: candidate.wide ?? base.wide };
    });
  for (const widget of fallback) {
    if (!normalized.some((candidate) => candidate.id === widget.id)) normalized.push({ ...widget });
  }
  return normalized;
}

function browserChannel(): BroadcastChannel | undefined {
  return "BroadcastChannel" in window ? new BroadcastChannel(BROWSER_CHANNEL) : undefined;
}

export async function sendDjControlCommand(command: DjControlCommand): Promise<void> {
  if (isTauri()) {
    await emit(COMMAND_EVENT, command);
    return;
  }
  const channel = browserChannel();
  channel?.postMessage({ kind: "command", payload: command });
  channel?.close();
}

export async function broadcastDjControlState(state: DjControlState): Promise<void> {
  if (isTauri()) {
    await emit(STATE_EVENT, state);
    return;
  }
  const channel = browserChannel();
  channel?.postMessage({ kind: "state", payload: state });
  channel?.close();
}

export async function subscribeDjControlCommands(listener: (command: DjControlCommand) => void): Promise<() => void> {
  if (isTauri()) {
    return listen<DjControlCommand>(COMMAND_EVENT, (event) => listener(event.payload));
  }
  const channel = browserChannel();
  if (!channel) return () => undefined;
  channel.onmessage = (event) => {
    if (event.data?.kind === "command") listener(event.data.payload as DjControlCommand);
  };
  return () => channel.close();
}

export async function subscribeDjControlState(listener: (state: DjControlState) => void): Promise<() => void> {
  if (isTauri()) {
    return listen<DjControlState>(STATE_EVENT, (event) => listener(event.payload));
  }
  const channel = browserChannel();
  if (!channel) return () => undefined;
  channel.onmessage = (event) => {
    if (event.data?.kind === "state") listener(event.data.payload as DjControlState);
  };
  return () => channel.close();
}
