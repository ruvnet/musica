import { invoke, isTauri } from "@tauri-apps/api/core";
import { PERFORMANCE_TEMPLATES, performanceTemplateById } from "./presets";
import type { PerformanceTemplate, TrackId, VisualSceneId } from "./types";

export interface AgentStatus {
  available: boolean;
  provider: string;
  endpointHost?: string;
  model?: string;
  reason?: string;
}

export interface AgentPlanRequest {
  goal: string;
  currentPrompt: string;
  bpm: number;
  scene: VisualSceneId;
  selectedTrack: TrackId;
}

export interface AgentPlan {
  title: string;
  rationale: string;
  prompt: string;
  templateId: string;
  scene: VisualSceneId;
  bpm: number;
  intensity: number;
  artDirection: PerformanceTemplate["artDirection"];
  temporal?: PerformanceTemplate["temporal"];
  arrangementNotes: string[];
}

export async function getAgentStatus(): Promise<AgentStatus> {
  if (!isTauri()) {
    return {
      available: true,
      provider: "local_agent",
      model: "deterministic-template-director",
      reason: "Browser preview uses deterministic local planning",
    };
  }
  return invoke<AgentStatus>("meta_llm_status");
}

export async function createAgentPlan(request: AgentPlanRequest): Promise<AgentPlan> {
  if (!isTauri()) return localAgentPlan(request);
  return invoke<AgentPlan>("meta_llm_plan", { request });
}

export function localAgentPlan(request: AgentPlanRequest): AgentPlan {
  const text = `${request.goal} ${request.currentPrompt}`.toLowerCase();
  const candidates = PERFORMANCE_TEMPLATES.map((template) => ({
    template,
    score: scoreTemplate(text, template),
  })).sort((left, right) => right.score - left.score);
  const template = candidates[0]?.template ?? performanceTemplateById("warehouse-techno");
  return {
    title: template.name,
    rationale: `Matched the direction to ${template.name.toLowerCase()} and prepared a full transport, pattern, mix, and realtime visual state.`,
    prompt: request.goal.trim() || template.prompt,
    templateId: template.id,
    scene: template.scene,
    bpm: template.bpm,
    intensity: template.intensity,
    artDirection: template.artDirection,
    temporal: template.temporal,
    arrangementNotes: [
      template.description,
      `Sets ${template.scene} as the realtime visual theme.`,
      "Applies six track patterns and note sets together so the groove changes coherently.",
    ],
  };
}

function scoreTemplate(text: string, template: PerformanceTemplate): number {
  const haystack = `${template.name} ${template.description} ${template.prompt}`.toLowerCase();
  let score = 0;
  for (const word of text.split(/[^a-z0-9]+/).filter((item) => item.length > 2)) {
    if (haystack.includes(word)) score += 2;
  }
  if (text.includes("fast") || text.includes("break") || text.includes("drum")) {
    score += template.bpm >= 150 ? 4 : 0;
  }
  if (text.includes("ambient") || text.includes("slow") || text.includes("dub")) {
    score += template.bpm <= 105 ? 4 : 0;
  }
  if (text.includes("cinematic") || text.includes("dramatic")) {
    score += template.id.includes("cinematic") ? 5 : 0;
  }
  if (text.includes("techno") || text.includes("warehouse")) {
    score += template.id.includes("techno") ? 5 : 0;
  }
  return score;
}
