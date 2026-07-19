import { createHash } from "node:crypto";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { dirname, extname, join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function hasDrawtextFilter() {
  const result = spawnSync("ffmpeg", ["-hide_banner", "-filters"], { encoding: "utf8" });
  return result.status === 0 && result.stdout.includes("drawtext");
}

function hashes(sampleDirectory) {
  return Object.fromEntries(
    readdirSync(sampleDirectory)
      .filter((file) => extname(file) === ".mp4")
      .sort()
      .map((file) => [file, createHash("sha256").update(readFileSync(resolve(sampleDirectory, file))).digest("hex")]),
  );
}

if (!hasDrawtextFilter()) {
  console.warn("Skipping sample reproducibility check because this FFmpeg build lacks the drawtext filter");
  process.exit(0);
}

const firstDirectory = mkdtempSync(join(tmpdir(), "musica-vj-samples-a-"));
const secondDirectory = mkdtempSync(join(tmpdir(), "musica-vj-samples-b-"));

try {
  for (const outputDirectory of [firstDirectory, secondDirectory]) {
    const render = spawnSync(process.execPath, [resolve(root, "scripts/render-samples.mjs")], {
      stdio: "inherit",
      env: { ...process.env, MUSICA_VJ_SAMPLE_OUTPUT: outputDirectory },
    });
    if (render.status !== 0) process.exit(render.status ?? 1);
  }

  const first = hashes(firstDirectory);
  const second = hashes(secondDirectory);

  if (JSON.stringify(first) !== JSON.stringify(second)) {
    throw new Error("Sample renderer is not deterministic across two runs on this runner");
  }

  console.log(`Verified runner-local bit reproducibility for ${Object.keys(second).length} sample fixtures`);
} finally {
  rmSync(firstDirectory, { recursive: true, force: true });
  rmSync(secondDirectory, { recursive: true, force: true });
}
