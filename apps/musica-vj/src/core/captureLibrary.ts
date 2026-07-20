import type { RecordingResult } from "../export/SocialRecorder";

// Persistent, on-device library of captured clips. Recordings are auto-saved
// here the moment they finish, so a clip is never lost just because it wasn't
// exported to disk before the next capture. Blobs live in IndexedDB (which
// structured-clones them natively); metadata rides alongside for the gallery.

export interface CaptureEntry {
  id: string;
  fileName: string;
  mode: RecordingResult["mode"];
  mimeType: string;
  container: RecordingResult["container"];
  fileExtension: RecordingResult["fileExtension"];
  bytes: number;
  durationSeconds: number;
  createdAt: number;
  blob: Blob;
}

const DB_NAME = "musica-captures";
const STORE = "captures";
const MAX_ENTRIES = 24; // keep the most recent; prune the rest

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: "id" });
        store.createIndex("createdAt", "createdAt");
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not open capture library"));
  });
}

function tx(db: IDBDatabase, mode: IDBTransactionMode): IDBObjectStore {
  return db.transaction(STORE, mode).objectStore(STORE);
}

function idFor(result: RecordingResult): string {
  return `cap-${result.mode}-${Date.now().toString(36)}-${Math.floor(performance.now() * 1000).toString(36)}`;
}

/// Auto-saves a finished recording to the library and returns the stored entry.
/// Prunes to the most recent MAX_ENTRIES. Best-effort: returns null on failure
/// (an unavailable library must never break capture).
export async function addCapture(result: RecordingResult): Promise<CaptureEntry | null> {
  try {
    const entry: CaptureEntry = {
      id: idFor(result),
      fileName: result.fileName,
      mode: result.mode,
      mimeType: result.mimeType,
      container: result.container,
      fileExtension: result.fileExtension,
      bytes: result.bytes,
      durationSeconds: result.durationSeconds,
      createdAt: Date.now(),
      blob: result.blob,
    };
    const db = await openDb();
    await new Promise<void>((resolve, reject) => {
      const store = tx(db, "readwrite");
      store.put(entry);
      store.transaction.oncomplete = () => resolve();
      store.transaction.onerror = () => reject(store.transaction.error);
    });
    await pruneToLimit(db);
    db.close();
    return entry;
  } catch {
    return null;
  }
}

async function pruneToLimit(db: IDBDatabase): Promise<void> {
  const all = await new Promise<CaptureEntry[]>((resolve) => {
    const request = tx(db, "readonly").getAll();
    request.onsuccess = () => resolve((request.result as CaptureEntry[]) ?? []);
    request.onerror = () => resolve([]);
  });
  if (all.length <= MAX_ENTRIES) return;
  const doomed = all.sort((a, b) => b.createdAt - a.createdAt).slice(MAX_ENTRIES);
  await new Promise<void>((resolve) => {
    const store = tx(db, "readwrite");
    for (const entry of doomed) store.delete(entry.id);
    store.transaction.oncomplete = () => resolve();
    store.transaction.onerror = () => resolve();
  });
}

/// Lists stored captures, newest first.
export async function listCaptures(): Promise<CaptureEntry[]> {
  try {
    const db = await openDb();
    const entries = await new Promise<CaptureEntry[]>((resolve, reject) => {
      const request = tx(db, "readonly").getAll();
      request.onsuccess = () => resolve((request.result as CaptureEntry[]) ?? []);
      request.onerror = () => reject(request.error);
    });
    db.close();
    return entries.sort((a, b) => b.createdAt - a.createdAt);
  } catch {
    return [];
  }
}

export async function deleteCapture(id: string): Promise<void> {
  try {
    const db = await openDb();
    await new Promise<void>((resolve) => {
      const store = tx(db, "readwrite");
      store.delete(id);
      store.transaction.oncomplete = () => resolve();
      store.transaction.onerror = () => resolve();
    });
    db.close();
  } catch {
    // best-effort
  }
}
