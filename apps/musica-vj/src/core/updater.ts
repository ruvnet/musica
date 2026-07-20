import { isTauri } from "@tauri-apps/api/core";

export interface UpdateInfo {
  available: boolean;
  version?: string;
  currentVersion?: string;
  notes?: string;
  reason?: string;
}

/// Checks the configured update endpoint for a newer signed release. Never
/// throws — returns a reason on failure so the UI can show it. No-op (not
/// available) outside the desktop app.
export async function checkForUpdate(): Promise<UpdateInfo> {
  if (!isTauri()) return { available: false, reason: "Updates require the desktop app" };
  try {
    const { check } = await import("@tauri-apps/plugin-updater");
    const update = await check();
    if (!update) return { available: false };
    return {
      available: true,
      version: update.version,
      currentVersion: update.currentVersion,
      notes: update.body,
    };
  } catch (error) {
    return { available: false, reason: error instanceof Error ? error.message : "Update check failed" };
  }
}

/// Downloads and installs the pending update (signature-verified by the updater
/// plugin against the bundled public key), reporting 0–100% progress, then
/// relaunches into the new version.
export async function downloadAndInstallUpdate(onProgress?: (percent: number) => void): Promise<void> {
  if (!isTauri()) throw new Error("Updates require the desktop app");
  const { check } = await import("@tauri-apps/plugin-updater");
  const { relaunch } = await import("@tauri-apps/plugin-process");
  const update = await check();
  if (!update) throw new Error("No update is available");
  let downloaded = 0;
  let total = 0;
  await update.downloadAndInstall((event) => {
    if (event.event === "Started") {
      total = event.data.contentLength ?? 0;
      onProgress?.(0);
    } else if (event.event === "Progress") {
      downloaded += event.data.chunkLength;
      if (total > 0) onProgress?.(Math.min(99, Math.round((downloaded / total) * 100)));
    } else if (event.event === "Finished") {
      onProgress?.(100);
    }
  });
  await relaunch();
}
