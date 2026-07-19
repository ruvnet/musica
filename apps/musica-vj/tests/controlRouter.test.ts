import { describe, expect, it } from "vitest";
import { keyboardControlFor } from "../src/controllers/ControlRouter";

describe("keyboard control mapping", () => {
  it("keeps bare number keys for visuals and reserves Shift+1 through Shift+4 for deck scenes", () => {
    expect(keyboardControlFor("Digit1")).toEqual({ action: "visual.scene.select", value: 0 });
    expect(keyboardControlFor("Digit4", true)).toEqual({ action: "lyria.deck-scene.select", value: 3 });
    expect(keyboardControlFor("Digit5", true)).toEqual({ action: "visual.scene.select", value: 4 });
  });
});
