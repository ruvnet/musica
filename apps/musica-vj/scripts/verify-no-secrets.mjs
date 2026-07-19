import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dist = resolve(root, "dist");
const expectedCanaries = [
  process.env.GEMINI_API_KEY,
  process.env.VITE_GEMINI_API_KEY,
  process.env.MUSICA_CREATIVE_API_TOKEN,
  process.env.MUSICA_META_LLM_API_TOKEN,
].filter(Boolean);

const files = [];
const visit = (directory) => {
  for (const entry of readdirSync(directory)) {
    const path = resolve(directory, entry);
    if (statSync(path).isDirectory()) visit(path);
    else files.push(path);
  }
};

visit(dist);

const forbiddenLiterals = [
  "GEMINI_API_KEY",
  "VITE_GEMINI_API_KEY",
  "MUSICA_CREATIVE_API_TOKEN",
  "MUSICA_META_LLM_API_TOKEN",
  "x-goog-api-key",
  "generativelanguage.googleapis.com",
];
const cognitumToken = /cog_[0-9a-f]{64}/g;
const googleApiKey = /AIza[0-9A-Za-z_-]{35}/g;
const findings = [];

for (const file of files) {
  const body = readFileSync(file);
  const text = body.toString("utf8");
  for (const value of [...forbiddenLiterals, ...expectedCanaries]) {
    if (text.includes(value)) findings.push(`${relative(dist, file)} contains ${value}`);
  }
  for (const match of text.matchAll(googleApiKey)) {
    findings.push(`${relative(dist, file)} contains a Google API key pattern at byte ${match.index}`);
  }
  for (const match of text.matchAll(cognitumToken)) {
    findings.push(`${relative(dist, file)} contains a Cognitum token pattern at byte ${match.index}`);
  }
}

if (findings.length > 0) {
  throw new Error(`Frontend secret isolation failed:\n${findings.join("\n")}`);
}

console.log(`Verified ${files.length} frontend files contain no provider credentials or endpoints`);
