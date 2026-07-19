import { isTauri } from "@tauri-apps/api/core";
import type { DjControlProfileId } from "./djControls";

const WINDOW_WIDTH = 430;
const WINDOW_HEIGHT = 720;
const SNAP_GAP = 8;
const SNAP_THRESHOLD = 24;

export async function openDjControlWindow(profile: DjControlProfileId): Promise<void> {
  const label = `dj-${profile}`;
  const url = `/?dj-control=${profile}`;
  if (!isTauri()) {
    const popup = window.open(url, label, `popup,width=${WINDOW_WIDTH},height=${WINDOW_HEIGHT},resizable=yes`);
    if (!popup) throw new Error("The browser blocked the DJ control window");
    popup.focus();
    return;
  }

  const [{ WebviewWindow }, { getCurrentWindow, currentMonitor, PhysicalPosition }] = await Promise.all([
    import("@tauri-apps/api/webviewWindow"),
    import("@tauri-apps/api/window"),
  ]);
  const existing = await WebviewWindow.getByLabel(label);
  if (existing) {
    await existing.show();
    await existing.setFocus();
    return;
  }

  const owner = getCurrentWindow();
  const [ownerPosition, ownerSize, scaleFactor, monitor] = await Promise.all([
    owner.outerPosition(),
    owner.outerSize(),
    owner.scaleFactor(),
    currentMonitor(),
  ]);
  const controlWidth = Math.round(WINDOW_WIDTH * scaleFactor);
  const monitorRight = monitor ? monitor.position.x + monitor.size.width : Number.POSITIVE_INFINITY;
  const preferredRight = ownerPosition.x + ownerSize.width + Math.round(SNAP_GAP * scaleFactor);
  const x = preferredRight + controlWidth <= monitorRight
    ? preferredRight
    : ownerPosition.x - controlWidth - Math.round(SNAP_GAP * scaleFactor);

  const controlWindow = new WebviewWindow(label, {
    url,
    title: `Musica DJ · ${profile.toUpperCase()}`,
    width: WINDOW_WIDTH,
    height: WINDOW_HEIGHT,
    minWidth: 340,
    minHeight: 480,
    resizable: true,
    decorations: true,
    alwaysOnTop: false,
    skipTaskbar: false,
    preventOverflow: true,
  });
  await new Promise<void>((resolve, reject) => {
    void controlWindow.once("tauri://created", () => resolve());
    void controlWindow.once("tauri://error", (event) => reject(new Error(String(event.payload))));
  });
  await controlWindow.setPosition(new PhysicalPosition(x, ownerPosition.y));
}

function rangesOverlap(startA: number, endA: number, startB: number, endB: number, margin: number): boolean {
  return startA <= endB + margin && startB <= endA + margin;
}

export async function installMagneticWindowSnapping(): Promise<() => void> {
  if (!isTauri()) return () => undefined;
  const [{ getAllWebviewWindows, getCurrentWebviewWindow }, { PhysicalPosition }] = await Promise.all([
    import("@tauri-apps/api/webviewWindow"),
    import("@tauri-apps/api/window"),
  ]);
  const current = getCurrentWebviewWindow();
  const scaleFactor = await current.scaleFactor();
  const threshold = Math.round(SNAP_THRESHOLD * scaleFactor);
  let applyingSnap = false;
  let timer: number | undefined;

  const unlisten = await current.onMoved(({ payload }) => {
    if (applyingSnap) return;
    if (timer !== undefined) window.clearTimeout(timer);
    timer = window.setTimeout(() => {
      void (async () => {
        const size = await current.outerSize();
        const others = (await getAllWebviewWindows()).filter((candidate) => candidate.label !== current.label);
        let bestX = payload.x;
        let bestY = payload.y;
        let xDistance = threshold + 1;
        let yDistance = threshold + 1;
        for (const other of others) {
          const [position, otherSize] = await Promise.all([other.outerPosition(), other.outerSize()]);
          const verticalOverlap = rangesOverlap(payload.y, payload.y + size.height, position.y, position.y + otherSize.height, threshold);
          const horizontalOverlap = rangesOverlap(payload.x, payload.x + size.width, position.x, position.x + otherSize.width, threshold);
          const xCandidates = verticalOverlap
            ? [position.x + otherSize.width, position.x - size.width, position.x, position.x + otherSize.width - size.width]
            : [position.x, position.x + otherSize.width - size.width];
          const yCandidates = horizontalOverlap
            ? [position.y + otherSize.height, position.y - size.height, position.y, position.y + otherSize.height - size.height]
            : [position.y, position.y + otherSize.height - size.height];
          for (const candidate of xCandidates) {
            const distance = Math.abs(candidate - payload.x);
            if (distance <= threshold && distance < xDistance) {
              bestX = candidate;
              xDistance = distance;
            }
          }
          for (const candidate of yCandidates) {
            const distance = Math.abs(candidate - payload.y);
            if (distance <= threshold && distance < yDistance) {
              bestY = candidate;
              yDistance = distance;
            }
          }
        }
        if (bestX === payload.x && bestY === payload.y) return;
        applyingSnap = true;
        await current.setPosition(new PhysicalPosition(bestX, bestY));
        window.setTimeout(() => { applyingSnap = false; }, 120);
      })().catch(() => undefined);
    }, 45);
  });

  return () => {
    if (timer !== undefined) window.clearTimeout(timer);
    unlisten();
  };
}
