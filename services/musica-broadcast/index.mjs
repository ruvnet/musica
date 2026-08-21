import * as functions from "@google-cloud/functions-framework";
import { verifyRequest } from "./auth.mjs";
import { handle } from "./handler.mjs";
import { createFirestoreStore } from "./store.mjs";

/// `musicaBroadcast` — the settings broadcast plane for Musica (ADR-182).
///
/// Public-invokable because it does its own auth: the JWT gate, not a network
/// ACL, is the boundary — the same posture as the ADR-179 Lyria broker. Every
/// route requires a signature-verified, unexpired, `inference`-scoped Cognitum
/// token, and nothing happens without one.

let store;

function lazyStore() {
  if (!store) store = createFirestoreStore();
  return store;
}

functions.http("musicaBroadcast", async (request, response) => {
  // Snapshots are live performance state; nothing here should ever be cached
  // by an intermediary.
  response.set("Cache-Control", "no-store");

  if (request.method === "OPTIONS") {
    response.status(405).send("");
    return;
  }

  try {
    const { subject } = await verifyRequest(request);
    const result = await handle({
      store: lazyStore(),
      subject,
      method: request.method,
      path: request.path,
      query: request.query,
      body: request.body,
      nowSeconds: Math.floor(Date.now() / 1000),
    });
    response.status(result.status).json(result.body);
  } catch (error) {
    const status = Number.isInteger(error?.status) ? error.status : 500;
    // Never echo internals: the client only needs to know which gate it hit.
    const message = status === 500 ? "Internal error" : error.message;
    response.status(status).json({ error: message });
  }
});
