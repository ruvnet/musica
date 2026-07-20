import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { DjControlWindow } from "./DjControlWindow";
import type { DjControlProfileId } from "./core/djControls";
// Self-hosted display + mono fonts. These were previously assumed to exist on
// the system, which only held on the design Mac; bundling the woff2 makes the
// UI render identically on Windows and Linux with no CDN dependency (Tauri CSP
// blocks remote fonts anyway).
import "@fontsource/space-grotesk/latin-400.css";
import "@fontsource/space-grotesk/latin-500.css";
import "@fontsource/space-grotesk/latin-600.css";
import "@fontsource/space-grotesk/latin-700.css";
import "@fontsource/dm-mono/latin-400.css";
import "@fontsource/dm-mono/latin-500.css";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Missing root element");

const controlProfile = new URLSearchParams(window.location.search).get("dj-control") as DjControlProfileId | null;
const content = controlProfile && ["mixer", "launcher", "visual"].includes(controlProfile)
  ? <DjControlWindow profile={controlProfile} />
  : <App />;

createRoot(root).render(
  <StrictMode>
    {content}
  </StrictMode>,
);
