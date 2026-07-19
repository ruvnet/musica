import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { DjControlWindow } from "./DjControlWindow";
import type { DjControlProfileId } from "./core/djControls";
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
