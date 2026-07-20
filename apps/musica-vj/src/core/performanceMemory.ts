/// Performance memory backing the LEARNING capability: remembers which AI
/// directions the operator actually used and recalls the closest past
/// direction for a new request. Storage is IndexedDB; matching is bounded
/// token-overlap scoring so it works offline. The interface is deliberately
/// vector-shaped (record → recall by similarity) so a ruvector-backed
/// implementation can replace the scorer without touching call sites
/// (docs/adr/ADR-176).

const DB_NAME = "musica-performance-memory";
const DB_VERSION = 1;
const STORE_NAME = "directions";
const MAX_ENTRIES = 400;

export type MemoryKind = "ai-look" | "fx-mood" | "ai-style" | "set-arc";

export interface MemoryEntry {
  id: string;
  kind: MemoryKind;
  query: string;
  payload: unknown;
  note: string;
  at: string;
}

export function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 2);
}

/// Jaccard-style overlap with a recency tiebreaker handled by the caller.
export function similarity(a: string, b: string): number {
  const tokensA = new Set(tokenize(a));
  const tokensB = new Set(tokenize(b));
  if (tokensA.size === 0 || tokensB.size === 0) return 0;
  let shared = 0;
  for (const token of tokensA) if (tokensB.has(token)) shared += 1;
  return shared / (tokensA.size + tokensB.size - shared);
}

function openMemoryDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Performance memory is unavailable"));
  });
}

async function withStore<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const db = await openMemoryDb();
  try {
    return await new Promise<T>((resolve, reject) => {
      const transaction = db.transaction(STORE_NAME, mode);
      const request = run(transaction.objectStore(STORE_NAME));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error("Performance memory operation failed"));
    });
  } finally {
    db.close();
  }
}

export async function recordDirection(kind: MemoryKind, query: string, payload: unknown, note: string): Promise<void> {
  const trimmed = query.trim();
  if (!trimmed) return;
  const entry: MemoryEntry = {
    id: `${kind}:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`,
    kind,
    query: trimmed.slice(0, 300),
    payload,
    note: note.slice(0, 120),
    at: new Date().toISOString(),
  };
  try {
    await withStore("readwrite", (store) => store.put(entry));
    const all = await withStore<MemoryEntry[]>("readonly", (store) => store.getAll() as IDBRequest<MemoryEntry[]>);
    if (all.length > MAX_ENTRIES) {
      const excess = all
        .sort((a, b) => a.at.localeCompare(b.at))
        .slice(0, all.length - MAX_ENTRIES);
      for (const stale of excess) {
        await withStore("readwrite", (store) => store.delete(stale.id));
      }
    }
  } catch {
    // Memory is an enhancement; failures must never break the performance.
  }
}

export async function recallDirection(kind: MemoryKind, query: string, threshold = 0.45): Promise<MemoryEntry | undefined> {
  try {
    const all = await withStore<MemoryEntry[]>("readonly", (store) => store.getAll() as IDBRequest<MemoryEntry[]>);
    const candidates = all
      .filter((entry) => entry.kind === kind)
      .map((entry) => ({ entry, score: similarity(query, entry.query) }))
      .filter(({ score }) => score >= threshold)
      .sort((a, b) => b.score - a.score || b.entry.at.localeCompare(a.entry.at));
    return candidates[0]?.entry;
  } catch {
    return undefined;
  }
}

export async function memoryCount(): Promise<number> {
  try {
    return await withStore<number>("readonly", (store) => store.count());
  } catch {
    return 0;
  }
}
