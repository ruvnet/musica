export const LYRIA_PRO_MODEL = "lyria-3-pro-preview" as const;
export const LYRIA_PRO_PRICE_USD = 0.08;
export const LYRIA_CLIP_PRICE_USD = 0.04;
export const LYRIA_PRO_UI_MIN_SECONDS = 31;
export const LYRIA_PRO_UI_MAX_SECONDS = 180;
export const LYRIA_PRO_PROVIDER_MAX_SECONDS = 184;
export const LYRIA_VOCAL_LANGUAGES = [
  "English",
  "German",
  "Spanish",
  "French",
  "Hindi",
  "Japanese",
  "Korean",
  "Portuguese",
] as const;

export type AudioOutputFormat = "mp3" | "wav";
export type LyriaRoute = "clip" | "pro" | "realtime";
export type DurationEnvelope = "router" | "pro-ui" | "pro-provider";

export interface CompositionSection {
  time: string | number;
  section: string;
  direction?: string;
}

export interface CompositionVocals {
  enabled: boolean;
  type?: string;
  language?: string;
  lyrics?: string;
}

export interface SocialHook {
  startSeconds: number;
  durationSeconds: number;
}

export interface LoopIntent {
  enabled: boolean;
  bars?: number;
  seamless?: boolean;
}

export interface TonalControls {
  key?: string;
  tonalCenter?: string;
  intensity?: number;
  negativePrompt?: string;
}

export interface StructuredComposition {
  durationSeconds: number;
  genre: string[];
  bpm?: number;
  timeSignature?: string;
  mood?: string[];
  instruments?: string[];
  vocals: CompositionVocals;
  structure?: CompositionSection[];
  socialHook?: SocialHook;
  productionStyle?: string;
  dynamicProgression?: string;
  visualSyncCues?: string[];
  outputFormat?: AudioOutputFormat;
  loop?: LoopIntent;
  tonal?: TonalControls;
}

export interface CompositionValidationIssue {
  path: string;
  code: string;
  message: string;
}

export interface CompositionValidationResult {
  valid: boolean;
  errors: CompositionValidationIssue[];
}

export interface RouteOptions {
  interactive?: boolean;
  loopOrPreview?: boolean;
  multipleCandidates?: boolean;
  costSensitive?: boolean;
}

export interface GenerationRouteDecision {
  route: LyriaRoute;
  model: typeof LYRIA_PRO_MODEL | undefined;
  availableInV1: boolean;
  reason: string;
}

export interface CostReservation {
  route: Exclude<LyriaRoute, "realtime">;
  unitCostUsd: number;
  candidateCount: number;
  reservedCostUsd: number;
  maximumPaidAttempts: number;
}

function issue(path: string, code: string, message: string): CompositionValidationIssue {
  return { path, code, message };
}

function cleanInline(value: string): string {
  return value.trim().replace(/\s+/g, " ");
}

function finiteNumber(value: number): boolean {
  return Number.isFinite(value);
}

export function timestampToSeconds(timestamp: string | number): number | undefined {
  if (typeof timestamp === "number") return finiteNumber(timestamp) && timestamp >= 0 ? timestamp : undefined;
  const match = /^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$/.exec(timestamp.trim());
  if (!match) return undefined;
  const hours = Number(match[1] ?? 0);
  const minutes = Number(match[2]);
  const seconds = Number(match[3]);
  const milliseconds = Number(`0.${match[4] ?? 0}`);
  if (minutes > 59 && match[1] !== undefined) return undefined;
  if (seconds > 59) return undefined;
  return hours * 3600 + minutes * 60 + seconds + milliseconds;
}

export function formatTimestamp(seconds: number): string {
  const wholeSeconds = Math.max(0, Math.round(seconds));
  const hours = Math.floor(wholeSeconds / 3600);
  const minutes = Math.floor((wholeSeconds % 3600) / 60);
  const remainder = wholeSeconds % 60;
  return hours > 0
    ? `${hours}:${minutes.toString().padStart(2, "0")}:${remainder.toString().padStart(2, "0")}`
    : `${minutes}:${remainder.toString().padStart(2, "0")}`;
}

export function validateProDuration(
  durationSeconds: number,
  envelope: "ui" | "provider" = "ui",
): CompositionValidationResult {
  const maximum = envelope === "ui" ? LYRIA_PRO_UI_MAX_SECONDS : LYRIA_PRO_PROVIDER_MAX_SECONDS;
  const errors: CompositionValidationIssue[] = [];
  if (!Number.isInteger(durationSeconds)) {
    errors.push(issue("durationSeconds", "integer", "Duration must be a whole number of seconds"));
  }
  if (!finiteNumber(durationSeconds) || durationSeconds < LYRIA_PRO_UI_MIN_SECONDS || durationSeconds > maximum) {
    errors.push(
      issue(
        "durationSeconds",
        "range",
        `Lyria 3 Pro duration must be between ${LYRIA_PRO_UI_MIN_SECONDS} and ${maximum} seconds`,
      ),
    );
  }
  return { valid: errors.length === 0, errors };
}

export function validateCompositionSpec(
  specification: StructuredComposition,
  envelope: DurationEnvelope = "pro-ui",
): CompositionValidationResult {
  const errors: CompositionValidationIssue[] = [];
  const durationMinimum = envelope === "router" ? 30 : LYRIA_PRO_UI_MIN_SECONDS;
  const durationMaximum = envelope === "pro-provider" ? LYRIA_PRO_PROVIDER_MAX_SECONDS : LYRIA_PRO_UI_MAX_SECONDS;

  if (!Number.isInteger(specification.durationSeconds)) {
    errors.push(issue("durationSeconds", "integer", "Duration must be a whole number of seconds"));
  }
  if (
    !finiteNumber(specification.durationSeconds) ||
    specification.durationSeconds < durationMinimum ||
    specification.durationSeconds > durationMaximum
  ) {
    errors.push(
      issue(
        "durationSeconds",
        "range",
        `Duration must be between ${durationMinimum} and ${durationMaximum} seconds for ${envelope}`,
      ),
    );
  }
  if (specification.genre.length === 0 || specification.genre.some((value) => cleanInline(value).length === 0)) {
    errors.push(issue("genre", "required", "At least one non-empty genre is required"));
  }
  if (specification.genre.length > 8) {
    errors.push(issue("genre", "limit", "At most eight genres may be supplied"));
  }
  if (specification.bpm !== undefined && (!finiteNumber(specification.bpm) || specification.bpm < 60 || specification.bpm > 200)) {
    errors.push(issue("bpm", "range", "BPM must be between 60 and 200"));
  }
  if (specification.timeSignature !== undefined && !/^\d{1,2}\/\d{1,2}$/.test(specification.timeSignature.trim())) {
    errors.push(issue("timeSignature", "format", "Time signature must use a value such as 4/4"));
  }
  if ((specification.vocals.lyrics?.length ?? 0) > 12_000) {
    errors.push(issue("vocals.lyrics", "limit", "Lyrics must not exceed 12,000 characters"));
  }
  if (
    specification.loop?.bars !== undefined &&
    (!Number.isInteger(specification.loop.bars) || specification.loop.bars < 1 || specification.loop.bars > 64)
  ) {
    errors.push(issue("loop.bars", "range", "Loop bars must be a whole number between 1 and 64"));
  }
  if (
    specification.tonal?.intensity !== undefined &&
    (!finiteNumber(specification.tonal.intensity) || specification.tonal.intensity < 0 || specification.tonal.intensity > 1)
  ) {
    errors.push(issue("tonal.intensity", "range", "Production intensity must be between 0 and 1"));
  }
  if ((specification.tonal?.negativePrompt?.length ?? 0) > 800) {
    errors.push(issue("tonal.negativePrompt", "limit", "Negative prompt must not exceed 800 characters"));
  }
  if (!specification.vocals.enabled && specification.vocals.lyrics?.trim()) {
    errors.push(issue("vocals.lyrics", "conflict", "Lyrics require vocals to be enabled"));
  }
  if (
    specification.vocals.language !== undefined &&
    !LYRIA_VOCAL_LANGUAGES.some(
      (language) => language.toLowerCase() === cleanInline(specification.vocals.language ?? "").toLowerCase(),
    )
  ) {
    errors.push(
      issue(
        "vocals.language",
        "unsupported",
        `Vocal language must be one of: ${LYRIA_VOCAL_LANGUAGES.join(", ")}`,
      ),
    );
  }

  let previousTime = -1;
  for (const [index, section] of (specification.structure ?? []).entries()) {
    const seconds = timestampToSeconds(section.time);
    if (seconds === undefined) {
      errors.push(issue(`structure.${index}.time`, "format", "Section time must be seconds or an MM:SS timestamp"));
    } else {
      if (seconds <= previousTime) {
        errors.push(issue(`structure.${index}.time`, "order", "Section timestamps must be strictly increasing"));
      }
      if (seconds >= specification.durationSeconds) {
        errors.push(issue(`structure.${index}.time`, "range", "Section must begin before the requested duration"));
      }
      previousTime = seconds;
    }
    if (cleanInline(section.section).length === 0) {
      errors.push(issue(`structure.${index}.section`, "required", "Section name must not be empty"));
    }
  }

  if (specification.socialHook !== undefined) {
    const { startSeconds, durationSeconds } = specification.socialHook;
    if (!finiteNumber(startSeconds) || startSeconds < 0) {
      errors.push(issue("socialHook.startSeconds", "range", "Social hook start must be zero or greater"));
    }
    if (!finiteNumber(durationSeconds) || durationSeconds <= 0 || durationSeconds > 30) {
      errors.push(issue("socialHook.durationSeconds", "range", "Social hook duration must be greater than zero and at most 30 seconds"));
    }
    if (startSeconds + durationSeconds > specification.durationSeconds) {
      errors.push(issue("socialHook", "range", "Social hook must fit inside the requested duration"));
    }
  }

  return { valid: errors.length === 0, errors };
}

export function assertValidComposition(
  specification: StructuredComposition,
  envelope: DurationEnvelope = "pro-ui",
): void {
  const result = validateCompositionSpec(specification, envelope);
  if (!result.valid) {
    throw new Error(result.errors.map((entry) => `${entry.path}: ${entry.message}`).join("; "));
  }
}

export function selectGenerationRoute(
  specification: StructuredComposition,
  options: RouteOptions = {},
): GenerationRouteDecision {
  if (options.interactive) {
    return {
      route: "realtime",
      model: undefined,
      availableInV1: false,
      reason: "Interactive continuous generation is reserved for the experimental realtime adapter",
    };
  }
  if (
    specification.durationSeconds === 30 ||
    options.loopOrPreview ||
    options.multipleCandidates ||
    options.costSensitive
  ) {
    return {
      route: "clip",
      model: undefined,
      availableInV1: false,
      reason: "Thirty second previews and candidate batches route to the lower-cost clip capability",
    };
  }
  const validation = validateCompositionSpec(specification, "pro-ui");
  if (!validation.valid) {
    throw new Error(validation.errors.map((entry) => entry.message).join("; "));
  }
  return {
    route: "pro",
    model: LYRIA_PRO_MODEL,
    availableInV1: true,
    reason: "Complete songs longer than 30 seconds route to Lyria 3 Pro",
  };
}

export function reserveGenerationCost(
  route: Exclude<LyriaRoute, "realtime">,
  candidateCount = 1,
  explicitBudgetUsd?: number,
): CostReservation {
  if (!Number.isInteger(candidateCount) || candidateCount < 1 || candidateCount > 16) {
    throw new Error("Candidate count must be an integer between 1 and 16");
  }
  const unitCostUsd = route === "pro" ? LYRIA_PRO_PRICE_USD : LYRIA_CLIP_PRICE_USD;
  const reservedCostUsd = Math.round(unitCostUsd * candidateCount * 100) / 100;
  if (explicitBudgetUsd === undefined || !finiteNumber(explicitBudgetUsd) || explicitBudgetUsd < reservedCostUsd) {
    throw new Error(`An explicit generation budget of at least $${reservedCostUsd.toFixed(2)} is required`);
  }
  return {
    route,
    unitCostUsd,
    candidateCount,
    reservedCostUsd,
    maximumPaidAttempts: candidateCount,
  };
}

export function acceptedOutputCost(totalGenerationCostUsd: number, acceptedTracks: number): number {
  if (!finiteNumber(totalGenerationCostUsd) || totalGenerationCostUsd < 0) {
    throw new Error("Total generation cost must be a non-negative number");
  }
  if (!Number.isInteger(acceptedTracks) || acceptedTracks < 1) {
    throw new Error("Accepted track count must be a positive integer");
  }
  return Math.round((totalGenerationCostUsd / acceptedTracks) * 10_000) / 10_000;
}

export function acceptedTrackCost(candidateCostUsd: number, acceptanceRate: number): number {
  if (!finiteNumber(candidateCostUsd) || candidateCostUsd < 0) {
    throw new Error("Candidate cost must be a non-negative number");
  }
  if (!finiteNumber(acceptanceRate) || acceptanceRate <= 0 || acceptanceRate > 1) {
    throw new Error("Acceptance rate must be greater than zero and at most one");
  }
  return Math.round((candidateCostUsd / acceptanceRate) * 10_000) / 10_000;
}

export function compileLyriaPrompt(
  specification: StructuredComposition,
  envelope: DurationEnvelope = "pro-ui",
): string {
  assertValidComposition(specification, envelope);
  const lines: string[] = [
    `Create a ${specification.durationSeconds} second ${specification.vocals.enabled ? "song" : "instrumental track"}.`,
    `Genre: ${specification.genre.map(cleanInline).join(", ")}.`,
  ];

  if (specification.mood?.length) lines.push(`Mood: ${specification.mood.map(cleanInline).join(", ")}.`);
  if (specification.bpm !== undefined) lines.push(`Tempo: ${specification.bpm} BPM.`);
  if (specification.timeSignature) lines.push(`Time signature: ${cleanInline(specification.timeSignature)}.`);
  if (specification.tonal?.key) lines.push(`Key: ${cleanInline(specification.tonal.key)}.`);
  if (specification.tonal?.tonalCenter) lines.push(`Tonal center: ${cleanInline(specification.tonal.tonalCenter)}.`);
  if (specification.loop?.enabled) {
    const loopBars = specification.loop.bars ?? 16;
    const seamless = specification.loop.seamless ?? true;
    lines.push(
      `Loop intent: create a ${loopBars} bar ${seamless ? "seamless, DJ-loopable" : "loop-ready"} phrase with a clean downbeat, no long fade-in, no long fade-out, and an ending that returns naturally to bar 1.`,
    );
  }
  if (specification.instruments?.length) {
    lines.push(`Instrumentation: ${specification.instruments.map(cleanInline).join(", ")}.`);
  }
  if (specification.productionStyle) lines.push(`Production style: ${cleanInline(specification.productionStyle)}.`);
  if (specification.tonal?.intensity !== undefined) {
    lines.push(`Production intensity: ${Math.round(specification.tonal.intensity * 100)}%.`);
  }
  if (specification.dynamicProgression) lines.push(`Dynamic progression: ${cleanInline(specification.dynamicProgression)}.`);
  if (specification.tonal?.negativePrompt?.trim()) {
    lines.push(`Avoid: ${cleanInline(specification.tonal.negativePrompt)}.`);
  }

  if (specification.vocals.enabled) {
    const details = [specification.vocals.type, specification.vocals.language].filter(
      (value): value is string => value !== undefined && cleanInline(value).length > 0,
    );
    lines.push(`Vocals: enabled${details.length ? `; ${details.map(cleanInline).join("; ")}` : ""}.`);
  } else {
    lines.push("Vocals: disabled. Do not generate lead vocals or sung lyrics.");
  }

  if (specification.structure?.length) {
    lines.push("Structure:");
    for (const section of specification.structure) {
      const seconds = timestampToSeconds(section.time);
      if (seconds === undefined) throw new Error("Invalid section timestamp");
      const direction = section.direction ? `: ${cleanInline(section.direction)}` : "";
      lines.push(`${formatTimestamp(seconds)} ${cleanInline(section.section)}${direction}`);
    }
  }

  if (specification.vocals.enabled && specification.vocals.lyrics?.trim()) {
    lines.push("User supplied lyrics:");
    lines.push(specification.vocals.lyrics.trim().replace(/\r\n/g, "\n"));
    lines.push("Use the supplied lyrics verbatim unless pronunciation requires a minimal phonetic adjustment.");
  }

  if (specification.socialHook) {
    const end = specification.socialHook.startSeconds + specification.socialHook.durationSeconds;
    lines.push(
      `Social hook: optimize ${formatTimestamp(specification.socialHook.startSeconds)} to ${formatTimestamp(end)} for a vertical social video.`,
    );
  }
  const visualCues = specification.visualSyncCues?.length
    ? specification.visualSyncCues.map(cleanInline).join("; ")
    : "clear beat transients; audible section changes; dynamic contrast between sections";
  lines.push(`Visual synchronization cues: ${visualCues}.`);
  lines.push(`Requested response audio format: ${(specification.outputFormat ?? "wav").toUpperCase()}.`);
  return lines.join("\n");
}
