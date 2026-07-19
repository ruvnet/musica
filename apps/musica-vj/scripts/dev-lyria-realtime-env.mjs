import { existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawn } from "node:child_process";

const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";

function parseEnvFile(path) {
  const values = {};
  const text = readFileSync(path, "utf8");
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const match = /^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/.exec(line);
    if (!match) continue;
    const key = match[1];
    let value = match[2].trim();
    if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    values[key] = value;
  }
  return values;
}

function findEnvFiles() {
  const found = [];
  let current = resolve(process.cwd());
  while (true) {
    const candidate = join(current, ".env");
    if (existsSync(candidate)) found.unshift(candidate);
    const parent = dirname(current);
    if (parent === current) return found;
    current = parent;
  }
}

const fileEnv = Object.assign({}, ...findEnvFiles().map(parseEnvFile));
const geminiApiKey = process.env.GEMINI_API_KEY ?? fileEnv.GEMINI_API_KEY;

if (!geminiApiKey || geminiApiKey.trim().length < 20) {
  console.error("GEMINI_API_KEY was not found in the shell environment or a parent .env file.");
  console.error("Add GEMINI_API_KEY=... to /Users/cohen/Projects/musica/.env, then retry:");
  console.error("  npm run dev:lyria:realtime");
  process.exit(1);
}

const modelsResponse = await fetch("https://generativelanguage.googleapis.com/v1beta/models", {
  headers: { "x-goog-api-key": geminiApiKey },
});
if (!modelsResponse.ok) {
  let reason = `${modelsResponse.status} ${modelsResponse.statusText}`;
  try {
    const body = await modelsResponse.json();
    reason = body.error?.message || reason;
  } catch {
    // Keep the status line when the error body is not JSON.
  }
  console.error(`Gemini API key probe failed: ${reason}`);
  process.exit(1);
}

console.log("Starting Musica VJ with Lyria RealTime controls enabled.");

const child = spawn(npmCommand, ["run", "tauri", "dev"], {
  stdio: "inherit",
  env: {
    ...process.env,
    ...fileEnv,
    GEMINI_API_KEY: geminiApiKey,
    MUSICA_LYRIA_REALTIME_ENABLED: "true",
    MUSICA_CREATIVE_ENABLED: process.env.MUSICA_CREATIVE_ENABLED ?? fileEnv.MUSICA_CREATIVE_ENABLED ?? "false",
  },
});

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 0);
});
