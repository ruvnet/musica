// Browser-native Cognitum One sign-in — the web equivalent of the manual
// (paste-a-code) flow in src-tauri/src/cognitum_provider.rs. The desktop
// app's primary flow binds a loopback TCP listener for the OAuth redirect,
// which a browser can't do; this uses the RFC 8252 out-of-band sentinel
// (`urn:ietf:wg:oauth:2.0:oob`) instead — Cognitum shows a code on screen
// that the user copies back in, no redirect needed at all (ADR-182).
//
// Verified against production before building this (see ADR-182): both
// `https://auth.cognitum.one/v1/oauth/code-exchange` and `/oauth/token`
// already send `Access-Control-Allow-Origin: *`, so this needs zero changes
// to Cognitum's infrastructure. `/v1/capabilities` does NOT have CORS
// enabled — same as the Rust side's own fallback, a failed/unreachable
// capabilities call just grants the full known set rather than blocking.
import { configureLyriaRealtimeWebKey } from "./lyriaRealtimeWeb";
import type { CognitumCapability, CognitumStatus } from "./cognitum";

const AUTH_BASE = "https://auth.cognitum.one";
const API_BASE = "https://api.cognitum.one";
const CLIENT_ID = "meta-proxy";
const OAUTH_SCOPES = "inference";
const OOB_REDIRECT_URI = "urn:ietf:wg:oauth:2.0:oob";
const LYRIA_BROKER_URL = "https://us-central1-cognitum-20260110.cloudfunctions.net/lyriaBroker";
const KNOWN_CAPABILITIES: CognitumCapability[] = ["advanced-prompting", "autopilot", "learning", "realtime-vocals"];
const SESSION_STORAGE_KEY = "musica.cognitumWebSession.v1";

interface TokenResponse {
  access_token: string;
  account_email?: string;
  refresh_token?: string;
  expires_in?: number;
}

interface WebAuthState {
  accessToken?: string;
  refreshToken?: string;
  expiresAt?: number;
  account?: string;
  capabilities: CognitumCapability[];
  lastError?: string;
}

interface PersistedSession {
  refreshToken: string;
  account?: string;
  capabilities: CognitumCapability[];
}

function loadPersisted(): WebAuthState {
  try {
    const raw = localStorage.getItem(SESSION_STORAGE_KEY);
    if (!raw) return { capabilities: [] };
    const parsed = JSON.parse(raw) as Partial<PersistedSession>;
    if (typeof parsed.refreshToken !== "string") return { capabilities: [] };
    // A restored session has no access token yet — mark it pre-expired so the
    // first authenticated call transparently refreshes via the stored
    // refresh token, mirroring the desktop keychain-restore behavior.
    return {
      refreshToken: parsed.refreshToken,
      account: parsed.account,
      capabilities: Array.isArray(parsed.capabilities) ? parsed.capabilities : [],
      expiresAt: 0,
    };
  } catch {
    return { capabilities: [] };
  }
}

let state: WebAuthState = loadPersisted();
let manualVerifier: string | null = null;

function persist(): void {
  try {
    if (state.refreshToken) {
      const persisted: PersistedSession = {
        refreshToken: state.refreshToken,
        account: state.account,
        capabilities: state.capabilities,
      };
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(persisted));
    } else {
      localStorage.removeItem(SESSION_STORAGE_KEY);
    }
  } catch {
    // localStorage unavailable (private mode, quota) — session just won't
    // survive a reload; not fatal to the current tab's sign-in.
  }
}

function base64UrlEncode(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function randomUrlSafe(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes);
}

async function sha256Base64Url(input: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(input));
  return base64UrlEncode(new Uint8Array(digest));
}

function storeTokenResponse(token: TokenResponse, account: string | undefined, capabilities: CognitumCapability[]): void {
  state = {
    accessToken: token.access_token,
    refreshToken: token.refresh_token,
    // Identity issues 15-minute access tokens; refresh one minute early —
    // mirrors the desktop provider's own margin.
    expiresAt:
      typeof token.expires_in === "number" ? Date.now() + Math.max(0, token.expires_in) * 1_000 - 60_000 : undefined,
    account: token.account_email ?? account,
    capabilities,
    lastError: undefined,
  };
  persist();
}

async function fetchCapabilities(accessToken: string): Promise<{ account?: string; capabilities: CognitumCapability[] }> {
  try {
    const response = await fetch(`${API_BASE}/v1/capabilities`, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (response.ok) {
      const json = (await response.json()) as { account?: unknown; capabilities?: unknown };
      const capabilities = Array.isArray(json.capabilities)
        ? json.capabilities.filter((value): value is CognitumCapability => KNOWN_CAPABILITIES.includes(value as CognitumCapability))
        : [];
      return { account: typeof json.account === "string" ? json.account : undefined, capabilities };
    }
  } catch {
    // No CORS on this endpoint today — falls through to the same "reachable
    // token, no capabilities endpoint" default the Rust side already uses.
  }
  return { account: undefined, capabilities: [...KNOWN_CAPABILITIES] };
}

export function getCognitumWebStatus(): CognitumStatus {
  return {
    signedIn: state.accessToken !== undefined || state.refreshToken !== undefined,
    pending: false,
    account: state.account,
    capabilities: state.capabilities,
    authHost: "auth.cognitum.one",
    reason: state.lastError,
  };
}

export async function startCognitumManualSignInWeb(): Promise<{ authUrl: string }> {
  const verifier = randomUrlSafe(48);
  const challenge = await sha256Base64Url(verifier);
  const oauthState = randomUrlSafe(24);
  const authUrl =
    `${AUTH_BASE}/oauth/authorize?response_type=code` +
    `&client_id=${encodeURIComponent(CLIENT_ID)}` +
    `&redirect_uri=${encodeURIComponent(OOB_REDIRECT_URI)}` +
    `&code_challenge=${challenge}` +
    `&code_challenge_method=S256` +
    `&state=${oauthState}` +
    `&scope=${encodeURIComponent(OAUTH_SCOPES)}`;
  manualVerifier = verifier;
  state.lastError = undefined;
  window.open(authUrl, "_blank", "noopener,noreferrer");
  return { authUrl };
}

export async function completeCognitumManualSignInWeb(code: string): Promise<void> {
  const trimmed = code.trim().toUpperCase();
  if (!trimmed || trimmed.length > 64) throw new Error("Paste the CGN- code shown in the browser");
  const verifier = manualVerifier;
  if (!verifier) throw new Error("Start the paste-code sign-in first");
  const response = await fetch(`${AUTH_BASE}/v1/oauth/code-exchange`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: trimmed, code_verifier: verifier, client_id: CLIENT_ID }),
  });
  if (!response.ok) throw new Error("Cognitum rejected the pasted code");
  const token = (await response.json()) as TokenResponse;
  manualVerifier = null;
  const { account, capabilities } = await fetchCapabilities(token.access_token);
  storeTokenResponse(token, account, capabilities);
}

export function signOutCognitumWeb(): void {
  state = { capabilities: [] };
  manualVerifier = null;
  persist();
}

/// Returns a currently valid access token, refreshing when expired — the
/// browser equivalent of `fresh_access_token` in cognitum_provider.rs.
async function freshCognitumAccessTokenWeb(): Promise<string> {
  const token = state.accessToken;
  const expired = !state.expiresAt || Date.now() >= state.expiresAt;
  if (token && !expired) return token;
  const refresh = state.refreshToken;
  if (!refresh) {
    state.accessToken = undefined;
    throw new Error("Sign in to Cognitum One first");
  }
  // Consumed before use: a rotated refresh token must never be presented twice.
  state.refreshToken = undefined;
  const response = await fetch(`${AUTH_BASE}/oauth/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ grant_type: "refresh_token", refresh_token: refresh, client_id: CLIENT_ID }).toString(),
  });
  if (!response.ok) {
    state = { capabilities: [] };
    persist();
    throw new Error("Cognitum session expired — sign in again");
  }
  const json = (await response.json()) as TokenResponse;
  storeTokenResponse(json, state.account, state.capabilities);
  return json.access_token;
}

export async function activateCognitumLyriaWeb(): Promise<{ ok: boolean; reason?: string }> {
  try {
    const token = await freshCognitumAccessTokenWeb();
    const response = await fetch(LYRIA_BROKER_URL, { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) return { ok: false, reason: "Lyria broker rejected the request" };
    const json = (await response.json()) as { api_key?: unknown; key?: unknown };
    const raw = typeof json.api_key === "string" ? json.api_key : typeof json.key === "string" ? json.key : "";
    const trimmed = raw.trim();
    if (trimmed.length < 20 || trimmed.length > 512) return { ok: false, reason: "Lyria broker returned no usable key" };
    configureLyriaRealtimeWebKey(trimmed);
    return { ok: true };
  } catch (error) {
    return { ok: false, reason: error instanceof Error ? error.message : String(error) };
  }
}
