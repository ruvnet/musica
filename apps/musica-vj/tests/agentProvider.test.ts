import { describe, expect, it } from "vitest";
import { localAgentPlan } from "../src/core/agentProvider";

describe("local agent planner", () => {
  it("maps musical intent to a full performance plan", () => {
    const plan = localAgentPlan({
      goal: "fast liquid drum and bass with glassy blue visuals",
      currentPrompt: "melodic techno",
      bpm: 112,
      scene: "bloom",
      selectedTrack: "drums",
    });

    expect(plan.templateId).toBe("liquid-breaks");
    expect(plan.bpm).toBeGreaterThan(150);
    expect(plan.arrangementNotes.length).toBeGreaterThan(1);
  });
});
