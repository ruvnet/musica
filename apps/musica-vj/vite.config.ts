import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// Tauri's webview resolves assets from a root-absolute base; the browser PWA
// build (docs/app/, see ADR-182) is served from a GitHub Pages subpath and
// needs relative asset URLs instead. MUSICA_WEB_BUILD=true switches only
// that, set by `npm run build:web` — the desktop build is unaffected.
const isWebBuild = process.env.MUSICA_WEB_BUILD === "true";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  base: isWebBuild ? "./" : "/",
  server: {
    strictPort: true,
    host: "127.0.0.1",
    port: 1420,
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    target: ["es2022", "safari15"],
    sourcemap: true,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        manualChunks: (id) => (id.includes("/three/") ? "three" : undefined),
      },
    },
  },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.ts"],
  },
});
