// Copy the ffmpeg-static binary into the Tauri external-bin (sidecar) path for
// the current Rust host target, so `tauri build`/`tauri dev` can bundle FFmpeg
// next to the app. Runs automatically via beforeBuild/beforeDev, and in CI.
//
// Tauri resolves external binaries by `<name>-<target-triple>[.exe]`, so the
// filename must carry the host triple (e.g. ffmpeg-aarch64-apple-darwin).

import { execSync } from "node:child_process";
import { chmodSync, copyFileSync, existsSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = dirname(fileURLToPath(import.meta.url));
const tauriRoot = join(here, "..", "src-tauri");
const binariesDir = join(tauriRoot, "binaries");

function hostTriple() {
  const out = execSync("rustc -vV", { encoding: "utf8" });
  const match = out.match(/host:\s*(\S+)/);
  if (!match) throw new Error("Could not determine rustc host triple");
  return match[1];
}

const ffmpegSrc = require("ffmpeg-static");
if (!ffmpegSrc || !existsSync(ffmpegSrc)) {
  throw new Error(`ffmpeg-static binary not found (${ffmpegSrc}); run npm install`);
}

const triple = hostTriple();
const ext = process.platform === "win32" ? ".exe" : "";
const dest = join(binariesDir, `ffmpeg-${triple}${ext}`);

mkdirSync(binariesDir, { recursive: true });
copyFileSync(ffmpegSrc, dest);
if (ext === "") chmodSync(dest, 0o755);
console.log(`FFmpeg sidecar ready: ${dest}`);
