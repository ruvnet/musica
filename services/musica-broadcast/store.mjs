import { Firestore } from "@google-cloud/firestore";

/// Firestore implementation of the store interface `handler.mjs` expects.
///
/// Two collections: `musica_broadcasts/{broadcastId}` holds the durable
/// snapshot and its denormalized `listenerCount`; the `listeners` subcollection
/// holds one presence document per listener, keyed by their OAuth subject so an
/// upsert can never become an append.

const BROADCASTS = "musica_broadcasts";
const LISTENERS = "listeners";

export function createFirestoreStore(firestore = new Firestore()) {
  const broadcasts = firestore.collection(BROADCASTS);

  return {
    async getBroadcast(id) {
      const snapshot = await broadcasts.doc(id).get();
      return snapshot.exists ? snapshot.data() : undefined;
    },

    async putBroadcast(id, doc) {
      await broadcasts.doc(id).set(doc, { merge: true });
    },

    /// Ordered by the denormalized count so the directory is one indexed query
    /// rather than a per-broadcaster count fan-out.
    async listBroadcasts(limit) {
      const query = await broadcasts.orderBy("listenerCount", "desc").limit(limit).get();
      return query.docs.map((doc) => ({ id: doc.id, doc: doc.data() }));
    },

    async getListener(id, subject) {
      const snapshot = await broadcasts.doc(id).collection(LISTENERS).doc(subject).get();
      return snapshot.exists ? snapshot.data() : undefined;
    },

    async putListener(id, subject, doc) {
      await broadcasts.doc(id).collection(LISTENERS).doc(subject).set({
        ...doc,
        // A Firestore TTL policy on this field reclaims storage. The count
        // query below does not rely on it: TTL deletion can lag by hours, so
        // correctness comes from the range filter, not from the sweeper.
        expiresAtTimestamp: new Date(doc.expiresAt * 1000),
      });
    },

    async deleteListener(id, subject) {
      await broadcasts.doc(id).collection(LISTENERS).doc(subject).delete();
    },

    async countListeners(id, seenSinceSeconds) {
      const query = broadcasts
        .doc(id)
        .collection(LISTENERS)
        .where("seenAt", ">=", seenSinceSeconds);
      // An aggregation query, so the cost is one read regardless of how many
      // listeners a broadcaster has.
      const result = await query.count().get();
      return result.data().count ?? 0;
    },
  };
}
