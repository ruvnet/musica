// Build the Tauri updater `latest.json` from the signed release artifacts.
//
// Scans a directory tree for the platform `.sig` files produced by
// `tauri build` with `createUpdaterArtifacts: true`, reads each signature, and
// emits a manifest whose per-platform `url` points at the matching GitHub
// release asset. GitHub replaces spaces in asset names with `.`, so URLs use
// the space→dot basename to match the uploaded asset exactly.
//
// Usage:
//   node build-updater-manifest.mjs \
//     --dir release-artifacts --repo ruvnet/musica \
//     --tag musica-vj-v0.5.0 --version 0.5.0 \
//     --notes "..." --date 2026-07-20T00:00:00Z --out latest.json

import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, basename } from "node:path";

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : fallback;
}

const dir = arg("dir", "release-artifacts");
const repo = arg("repo", "ruvnet/musica");
const tag = arg("tag");
const version = arg("version");
const notes = arg("notes", "See the release notes.");
const date = arg("date", "");
const out = arg("out", "latest.json");
if (!tag || !version) {
  console.error("Missing --tag or --version");
  process.exit(1);
}

// sig suffix → Tauri platform key + how to find the bundle it signs.
const MATCHERS = [
  { suffix: ".app.tar.gz.sig", platform: "darwin-aarch64" },
  { suffix: "-setup.exe.sig", platform: "windows-x86_64" },
  { suffix: ".AppImage.sig", platform: "linux-x86_64" },
];

function walk(root) {
  const files = [];
  for (const entry of readdirSync(root)) {
    const full = join(root, entry);
    if (statSync(full).isDirectory()) files.push(...walk(full));
    else files.push(full);
  }
  return files;
}

const assetUrl = (file) =>
  `https://github.com/${repo}/releases/download/${tag}/${basename(file).replace(/ /g, ".")}`;

const files = walk(dir);
const platforms = {};
for (const { suffix, platform } of MATCHERS) {
  const sig = files.find((f) => f.endsWith(suffix));
  if (!sig) {
    console.warn(`No signature found for ${platform} (${suffix}) — skipping`);
    continue;
  }
  const bundle = sig.slice(0, -4); // strip ".sig"
  platforms[platform] = {
    signature: readFileSync(sig, "utf8").trim(),
    url: assetUrl(bundle),
  };
  console.log(`${platform}: ${basename(bundle)}`);
}

if (Object.keys(platforms).length === 0) {
  console.error("No updater signatures found — refusing to write an empty manifest");
  process.exit(1);
}

const manifest = { version, notes, pub_date: date || undefined, platforms };
writeFileSync(out, JSON.stringify(manifest, null, 2));
console.log(`Wrote ${out} for v${version} with ${Object.keys(platforms).length} platform(s).`);
