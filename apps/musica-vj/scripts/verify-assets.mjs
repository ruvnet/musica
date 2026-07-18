import { readFileSync, readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import { parse as parseYaml } from "yaml";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const read = (path) => readFileSync(resolve(root, path), "utf8");

for (const path of [
  "package.json",
  "src-tauri/tauri.conf.json",
  "src-tauri/capabilities/default.json",
  "logitech/options-plus-fallback.json",
]) {
  JSON.parse(read(path));
}

const tauriConfiguration = JSON.parse(read("src-tauri/tauri.conf.json"));
const csp = tauriConfiguration.app?.security?.csp;
if (typeof csp !== "string" || !csp.includes("form-action 'none'") || !csp.includes("connect-src 'self' ipc: http://ipc.localhost")) {
  throw new Error("Tauri CSP must prohibit form submissions and keep provider networking out of the webview");
}
if (csp.includes("googleapis.com") || csp.includes("generativelanguage")) {
  throw new Error("Provider origins must not be available to the Tauri webview");
}
const mediaPolicy = /media-src\s+([^;]+)/.exec(csp)?.[1] ?? "";
if (mediaPolicy.includes("data:") || !mediaPolicy.includes("blob:")) {
  throw new Error("Tauri media CSP must allow bounded blob playback without enabling data URLs");
}

const metadata = parseYaml(read("logitech/src/MusicaVj.Logitech.Plugin/package/metadata/LoupedeckPackage.yaml"));
if (metadata.type !== "plugin4" || metadata.pluginFileName !== "MusicaVj.Logitech.Plugin.dll") {
  throw new Error("Logitech package metadata does not identify the plugin4 assembly");
}
if (!Array.isArray(metadata.supportedDevices) || !metadata.supportedDevices.includes("LoupedeckExtendedFamily")) {
  throw new Error("Logitech package metadata does not declare MX Creative Console compatibility");
}

const svgDirectory = resolve(root, "logitech/src/MusicaVj.Logitech.Plugin/package/actionsymbols");
const svgFiles = readdirSync(svgDirectory).filter((file) => file.endsWith(".svg"));
if (svgFiles.length !== 17) throw new Error(`Expected 17 Logitech action symbols, received ${svgFiles.length}`);

const window = new JSDOM("").window;
for (const file of svgFiles) {
  const document = new window.DOMParser().parseFromString(readFileSync(resolve(svgDirectory, file), "utf8"), "image/svg+xml");
  if (document.querySelector("parsererror") || document.documentElement.localName !== "svg") {
    throw new Error(`${file} is not valid SVG`);
  }
}
window.close();

const icon = readFileSync(resolve(root, "logitech/src/MusicaVj.Logitech.Plugin/package/metadata/Icon256x256.png"));
const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
if (!icon.subarray(0, 8).equals(pngSignature) || icon.readUInt32BE(16) !== 256 || icon.readUInt32BE(20) !== 256) {
  throw new Error("Logitech package icon must be a 256 by 256 PNG");
}

console.log(`Verified JSON, YAML, ${svgFiles.length} Logitech symbols, and the package icon`);
