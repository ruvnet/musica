import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { SignJWT, exportJWK, generateKeyPair } from "jose";
import { bearerFrom, verifyRequest } from "./auth.mjs";

/// The gate, exercised against a real ES256 keypair rather than a stub, so a
/// signature that does not verify genuinely fails (ADR-179's broker is verified
/// the same way).

const ISSUER = "https://auth.cognitum.one";

const keys = await generateKeyPair("ES256");
const other = await generateKeyPair("ES256");
// A local resolver standing in for the remote JWKS.
const resolver = async () => keys.publicKey;

async function token({
  issuer = ISSUER,
  scope = "inference",
  subject = "auth0|user-1",
  expiresIn = "5m",
  key = keys.privateKey,
} = {}) {
  return new SignJWT({ scope })
    .setProtectedHeader({ alg: "ES256" })
    .setIssuer(issuer)
    .setSubject(subject)
    .setIssuedAt()
    .setExpirationTime(expiresIn)
    .sign(key);
}

const request = (authorization) => ({ headers: { authorization } });

describe("bearer extraction", () => {
  it("reads a bearer header in any casing and ignores anything else", () => {
    assert.equal(bearerFrom(request("Bearer abc")), "abc");
    assert.equal(bearerFrom(request("bearer abc")), "abc");
    assert.equal(bearerFrom(request("Basic abc")), undefined);
    assert.equal(bearerFrom(request("")), undefined);
    assert.equal(bearerFrom({ headers: {} }), undefined);
  });
});

describe("Cognitum token gate", () => {
  it("accepts a valid, in-scope, unexpired token and returns its subject", async () => {
    const { subject } = await verifyRequest(request(`Bearer ${await token()}`), resolver);
    assert.equal(subject, "auth0|user-1");
  });

  it("accepts an array-valued scope claim", async () => {
    const jwt = await new SignJWT({ scope: ["inference", "other"] })
      .setProtectedHeader({ alg: "ES256" })
      .setIssuer(ISSUER)
      .setSubject("auth0|user-2")
      .setIssuedAt()
      .setExpirationTime("5m")
      .sign(keys.privateKey);
    const { subject } = await verifyRequest(request(`Bearer ${jwt}`), resolver);
    assert.equal(subject, "auth0|user-2");
  });

  it("rejects a missing token", async () => {
    await assert.rejects(verifyRequest(request(undefined), resolver), (error) => error.status === 401);
  });

  it("rejects garbage", async () => {
    await assert.rejects(verifyRequest(request("Bearer not-a-jwt"), resolver), (error) => error.status === 401);
  });

  it("rejects a forged signature", async () => {
    // Correct issuer and scope, wrong key: only the signature check catches
    // this, which is exactly what must not be skippable.
    const forged = await token({ key: other.privateKey });
    await assert.rejects(verifyRequest(request(`Bearer ${forged}`), resolver), (error) => error.status === 401);
  });

  it("rejects an expired token", async () => {
    const expired = await token({ expiresIn: "-1s" });
    await assert.rejects(verifyRequest(request(`Bearer ${expired}`), resolver), (error) => error.status === 401);
  });

  it("rejects a wrong issuer", async () => {
    const wrong = await token({ issuer: "https://evil.example.com" });
    await assert.rejects(verifyRequest(request(`Bearer ${wrong}`), resolver), (error) => error.status === 401);
  });

  it("rejects a token without the inference scope", async () => {
    const scopeless = await token({ scope: "" });
    await assert.rejects(verifyRequest(request(`Bearer ${scopeless}`), resolver), (error) => error.status === 403);
    const wrongScope = await token({ scope: "billing" });
    await assert.rejects(verifyRequest(request(`Bearer ${wrongScope}`), resolver), (error) => error.status === 403);
  });

  it("rejects a token with no subject", async () => {
    const anonymous = await new SignJWT({ scope: "inference" })
      .setProtectedHeader({ alg: "ES256" })
      .setIssuer(ISSUER)
      .setIssuedAt()
      .setExpirationTime("5m")
      .sign(keys.privateKey);
    await assert.rejects(verifyRequest(request(`Bearer ${anonymous}`), resolver), (error) => error.status === 401);
  });

  it("publishes an ES256 key, confirming the algorithm the gate expects", async () => {
    const jwk = await exportJWK(keys.publicKey);
    assert.equal(jwk.kty, "EC");
    assert.equal(jwk.crv, "P-256");
  });
});
