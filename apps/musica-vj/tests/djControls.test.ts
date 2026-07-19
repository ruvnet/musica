import { describe, expect, it } from "vitest";
import { DJ_CONTROL_PROFILES, normalizeDjControlLayout } from "../src/core/djControls";

describe("detachable DJ control layouts", () => {
  it("preserves custom order, visibility, and width while restoring new widgets", () => {
    const layout = normalizeDjControlLayout("mixer", [
      { id: "master", visible: false, wide: true },
      { id: "transport", visible: true, wide: false },
    ]);

    expect(layout[0]).toEqual({ id: "master", visible: false, wide: true });
    expect(layout[1]).toEqual({ id: "transport", visible: true, wide: false });
    expect(layout.map((widget) => widget.id)).toEqual(expect.arrayContaining(DJ_CONTROL_PROFILES.mixer.widgets.map((widget) => widget.id)));
  });

  it("falls back to a complete visual control surface", () => {
    const layout = normalizeDjControlLayout("visual", "invalid");
    expect(layout.find((widget) => widget.id === "color")).toMatchObject({ visible: true, wide: true });
    expect(layout.find((widget) => widget.id === "visuals")).toMatchObject({ visible: true });
  });
});
