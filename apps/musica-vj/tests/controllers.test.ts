import { describe, expect, it, vi } from "vitest";
import { ControlRouter, TapTempo } from "../src/controllers/ControlRouter";

describe("control routing", () => {
  it("dispatches a normalized message to subscribers", () => {
    const router = new ControlRouter();
    const listener = vi.fn();
    const unsubscribe = router.subscribe(listener);
    router.dispatch("master.delta", -1, "logitech");
    expect(listener).toHaveBeenCalledTimes(1);
    expect(listener.mock.calls[0][0]).toMatchObject({ action: "master.delta", value: -1, source: "logitech" });
    unsubscribe();
    router.dispatch("transport.toggle");
    expect(listener).toHaveBeenCalledTimes(1);
  });

  it("preserves an absent keyboard trigger value so Enter targets the selected track", async () => {
    const router = new ControlRouter();
    const listener = vi.fn();
    router.subscribe(listener);
    await router.start();
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "Enter" }));
    expect(listener).toHaveBeenCalledWith(expect.objectContaining({
      action: "track.trigger",
      value: undefined,
      source: "keyboard",
    }));
    await router.stop();
  });

  it("maps the hardware play/pause keyboard key to transport toggle", async () => {
    const router = new ControlRouter();
    const listener = vi.fn();
    router.subscribe(listener);
    await router.start();
    window.dispatchEvent(new KeyboardEvent("keydown", { code: "MediaPlayPause" }));
    expect(listener).toHaveBeenCalledWith(expect.objectContaining({
      action: "transport.toggle",
      source: "keyboard",
    }));
    await router.stop();
  });

  it("calculates tap tempo from the median interval", () => {
    const tap = new TapTempo();
    expect(tap.tap(1_000)).toBeUndefined();
    expect(tap.tap(1_500)).toBe(120);
    expect(tap.tap(2_010)).toBe(118);
    expect(tap.tap(2_500)).toBe(120);
  });

  it("discards stale taps", () => {
    const tap = new TapTempo();
    tap.tap(0);
    expect(tap.tap(4_000)).toBeUndefined();
  });

  it("bounds tempo to the performance range", () => {
    const fast = new TapTempo();
    fast.tap(0);
    expect(fast.tap(100)).toBe(200);
    const slow = new TapTempo();
    slow.tap(0);
    expect(slow.tap(2_000)).toBe(60);
  });
});
