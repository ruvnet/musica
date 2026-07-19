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
    if (
      (value.startsWith("\"") && value.endsWith("\"")) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
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
  console.error("  npm run dev:lyria");
  process.exit(1);
}

if (typeof fetch !== "function") {
  console.error("Node.js fetch support is required to verify Gemini API key access before launch.");
  process.exit(1);
}

const modelsResponse = await fetch("https://generativelanguage.googleapis.com/v1beta/models", {
  headers: {
    "x-goog-api-key": geminiApiKey,
  },
});
if (!modelsResponse.ok) {
  let reason = `${modelsResponse.status} ${modelsResponse.statusText}`;
  try {
    const body = await modelsResponse.json();
    const error = Array.isArray(body) ? body[0]?.error : body.error;
    reason = error?.message || reason;
  } catch {
    // Keep the status line when the error body is not JSON.
  }
  console.error(`Gemini API key probe failed: ${reason}`);
  console.error("Replace GEMINI_API_KEY in /Users/cohen/Projects/musica/.env with a valid Google AI Studio API key.");
  console.error("If the key is restricted, allow generativelanguage.googleapis.com and server-side use from this dev machine.");
  process.exit(1);
}

console.log("Starting Musica VJ with Lyria/Gemini via GEMINI_API_KEY from local environment.");

const child = spawn(npmCommand, ["run", "tauri", "dev"], {
  stdio: "inherit",
  env: {
    ...process.env,
    ...fileEnv,
    GEMINI_API_KEY: geminiApiKey,
    MUSICA_CREATIVE_ENABLED: "true",
    MUSICA_CREATIVE_PROVIDER: "lyria_3_pro",
    MUSICA_CREATIVE_MAX_GENERATION_USD: process.env.MUSICA_CREATIVE_MAX_GENERATION_USD ?? fileEnv.MUSICA_CREATIVE_MAX_GENERATION_USD ?? "0.32",
    MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS: process.env.MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS ?? fileEnv.MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS ?? "600",
    MUSICA_CREATIVE_RETAIN_PROMPTS: process.env.MUSICA_CREATIVE_RETAIN_PROMPTS ?? fileEnv.MUSICA_CREATIVE_RETAIN_PROMPTS ?? "false",
  },
});

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 0);
});
