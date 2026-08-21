import { createRemoteJWKSet, jwtVerify } from "jose";

/// Cognitum OAuth gate (ADR-182, mirroring ADR-179's lyria-broker).
///
/// This verifies the *Cognitum* token Musica holds, not a Firebase ID token:
/// the api gateway's `verifyAuthToken` is Firebase-based and would reject it,
/// so this service does its own JWKS verification. Fail-closed — no route does
/// anything without a signature-verified, unexpired, in-scope token.

const DEFAULT_ISSUER = "https://auth.cognitum.one";
const REQUIRED_SCOPE = "inference";

let cachedJwks;
let cachedJwksUrl;

function issuer() {
  return (process.env.COGNITUM_ISSUER || DEFAULT_ISSUER).replace(/\/$/, "");
}

function jwks() {
  const url = `${issuer()}/.well-known/jwks.json`;
  // createRemoteJWKSet caches and rotates internally; rebuild only if the
  // configured issuer changed.
  if (!cachedJwks || cachedJwksUrl !== url) {
    cachedJwks = createRemoteJWKSet(new URL(url));
    cachedJwksUrl = url;
  }
  return cachedJwks;
}

export function bearerFrom(request) {
  const header = request.get?.("authorization") ?? request.headers?.authorization ?? "";
  const match = /^Bearer\s+(.+)$/i.exec(String(header).trim());
  return match ? match[1].trim() : undefined;
}

function hasRequiredScope(payload) {
  const raw = payload.scope ?? payload.scp ?? "";
  const scopes = Array.isArray(raw) ? raw : String(raw).split(/\s+/);
  return scopes.includes(REQUIRED_SCOPE);
}

/// Resolves a request to a verified subject, or throws with an HTTP status.
/// `keyResolver` is injectable so tests can verify against a locally generated
/// ES256 key instead of reaching the network.
export async function verifyRequest(request, keyResolver = jwks()) {
  const token = bearerFrom(request);
  if (!token) {
    throw Object.assign(new Error("Missing bearer token"), { status: 401 });
  }
  let payload;
  try {
    ({ payload } = await jwtVerify(token, keyResolver, { issuer: issuer() }));
  } catch {
    throw Object.assign(new Error("Invalid or expired token"), { status: 401 });
  }
  if (!hasRequiredScope(payload)) {
    throw Object.assign(new Error("Token is missing the required scope"), { status: 403 });
  }
  const subject = typeof payload.sub === "string" ? payload.sub.trim() : "";
  if (!subject) {
    throw Object.assign(new Error("Token has no subject"), { status: 401 });
  }
  return { subject, payload };
}
