import { spawn, spawnSync } from "node:child_process";

const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const REQUIRED_SCOPES = "https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.retriever";

function runCheck(command, args) {
  return spawnSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

const gcloudVersion = runCheck("gcloud", ["--version"]);
if (gcloudVersion.error || gcloudVersion.status !== 0) {
  console.error("gcloud CLI is required for dev:lyria:gcloud.");
  console.error(`Install it, then run: gcloud auth application-default login --scopes=${REQUIRED_SCOPES}`);
  process.exit(1);
}

const adc = runCheck("gcloud", ["auth", "application-default", "print-access-token"]);
if (adc.error || adc.status !== 0 || adc.stdout.trim().length === 0) {
  console.error("Google application-default credentials are not available.");
  console.error("Run:");
  console.error(`  gcloud auth application-default login --scopes=${REQUIRED_SCOPES}`);
  console.error("Optionally set the quota project:");
  console.error("  gcloud auth application-default set-quota-project ruv-dev");
  console.error("Then retry: npm run dev:lyria:gcloud");
  process.exit(1);
}

const project = runCheck("gcloud", ["config", "get-value", "project"]);
const projectName = project.status === 0 ? project.stdout.trim() : "";
const token = adc.stdout.trim();

if (typeof fetch !== "function") {
  console.error("Node.js fetch support is required to verify Gemini OAuth access before launch.");
  process.exit(1);
}

const modelsResponse = await fetch("https://generativelanguage.googleapis.com/v1/models", {
  headers: {
    Authorization: `Bearer ${token}`,
    "x-goog-user-project": projectName || "ruv-dev",
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
  console.error(`Gemini API OAuth probe failed: ${reason}`);
  console.error("Refresh ADC with the Gemini OAuth scope:");
  console.error(`  gcloud auth application-default login --scopes=${REQUIRED_SCOPES}`);
  console.error("Then retry: npm run dev:lyria:gcloud");
  process.exit(1);
}

console.log(`Starting Musica VJ with Lyria/Gemini via application-default credentials${projectName ? ` for project ${projectName}` : ""}.`);

const child = spawn(npmCommand, ["run", "tauri", "dev"], {
  stdio: "inherit",
  env: {
    ...process.env,
    MUSICA_CREATIVE_ENABLED: "true",
    MUSICA_CREATIVE_PROVIDER: "lyria_3_pro",
    MUSICA_GCP_AUTH: "gcloud",
    MUSICA_CREATIVE_MAX_GENERATION_USD: process.env.MUSICA_CREATIVE_MAX_GENERATION_USD ?? "0.32",
    MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS: process.env.MUSICA_CREATIVE_REQUEST_TIMEOUT_SECONDS ?? "600",
    MUSICA_CREATIVE_RETAIN_PROMPTS: process.env.MUSICA_CREATIVE_RETAIN_PROMPTS ?? "false",
  },
});

child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  process.exit(code ?? 0);
});
