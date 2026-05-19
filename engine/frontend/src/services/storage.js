/**
 * Two-tier local persistence:
 *   - prefs:   localStorage (small JSON, synchronous)
 *   - library: IndexedDB via idb-keyval (large/structured, Blobs OK)
 *
 * Schemas live in the plan; we keep them duck-typed here. The library is
 * single-origin and only the current user's; nothing leaves the device unless
 * they explicitly export.
 */

import { createStore, set, get, del, entries, clear } from 'idb-keyval'

const PREF_PREFIX = 'biotuner_pref_'
const TUNINGS_STORE = createStore('biotuner-tunings', 'kv')
const RECORDINGS_STORE = createStore('biotuner-recordings', 'kv')

// ---------------------------------------------------------------------------
// Preferences (localStorage)
// ---------------------------------------------------------------------------

export const prefs = {
  get(key, fallback = null) {
    try {
      const raw = localStorage.getItem(PREF_PREFIX + key)
      if (raw == null) return fallback
      return JSON.parse(raw)
    } catch {
      return fallback
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(PREF_PREFIX + key, JSON.stringify(value))
    } catch (err) {
      console.warn('prefs.set failed:', err)
    }
  },
  remove(key) {
    try { localStorage.removeItem(PREF_PREFIX + key) } catch { /* ignore */ }
  },
}

// ---------------------------------------------------------------------------
// Library (IndexedDB)
// ---------------------------------------------------------------------------

function uuid() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `id_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`
}

async function listEntries(store) {
  const all = await entries(store)
  return all.map(([, value]) => value).sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0))
}

export const library = {
  // --- Tunings (Harmonic Portraits) ---

  async listTunings() {
    return listEntries(TUNINGS_STORE)
  },
  async getTuning(id) {
    return get(id, TUNINGS_STORE)
  },
  async saveTuning(tuning) {
    const id = tuning.id || uuid()
    const value = {
      kind: 'plain',
      ...tuning,
      id,
      createdAt: tuning.createdAt || Date.now(),
    }
    await set(id, value, TUNINGS_STORE)
    requestPersistence().catch(() => {})
    return id
  },
  async deleteTuning(id) {
    return del(id, TUNINGS_STORE)
  },

  // --- Recordings (Blob storage) ---

  async listRecordings() {
    const all = await entries(RECORDINGS_STORE)
    return all
      .map(([, value]) => ({ ...value, blob: undefined }))   // metadata only
      .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0))
  },
  async getRecording(id) {
    return get(id, RECORDINGS_STORE)
  },
  async saveRecording(recording) {
    const id = recording.id || uuid()
    const value = {
      ...recording,
      id,
      createdAt: recording.createdAt || Date.now(),
    }
    await set(id, value, RECORDINGS_STORE)
    requestPersistence().catch(() => {})
    return id
  },
  async deleteRecording(id) {
    return del(id, RECORDINGS_STORE)
  },

  // --- Bulk import/export ---

  async exportAll() {
    const tunings = await listEntries(TUNINGS_STORE)
    const recRaw = await entries(RECORDINGS_STORE)
    const recordings = await Promise.all(
      recRaw.map(async ([, r]) => ({
        ...r,
        blob: undefined,
        blobBase64: r.blob ? await blobToBase64(r.blob) : null,
      }))
    )
    return {
      biotuner_library_version: 1,
      exported_at: new Date().toISOString(),
      tunings,
      recordings,
    }
  },

  async importAll(bundle, { mode = 'merge' } = {}) {
    if (!bundle || typeof bundle !== 'object') throw new Error('Invalid bundle')
    if (mode === 'replace') {
      await clear(TUNINGS_STORE)
      await clear(RECORDINGS_STORE)
    }
    const tunings = bundle.tunings || []
    const recordings = bundle.recordings || []
    for (const t of tunings) {
      const next = { ...t, id: mode === 'merge' ? uuid() : (t.id || uuid()) }
      await set(next.id, next, TUNINGS_STORE)
    }
    for (const r of recordings) {
      const blob = r.blobBase64 ? base64ToBlob(r.blobBase64, r.mimeType) : null
      const next = {
        ...r,
        blob,
        id: mode === 'merge' ? uuid() : (r.id || uuid()),
      }
      delete next.blobBase64
      await set(next.id, next, RECORDINGS_STORE)
    }
    return {
      tunings: tunings.length,
      recordings: recordings.length,
    }
  },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => {
      const s = fr.result
      const i = String(s).indexOf(',')
      resolve(i >= 0 ? String(s).slice(i + 1) : String(s))
    }
    fr.onerror = reject
    fr.readAsDataURL(blob)
  })
}

function base64ToBlob(b64, mimeType = 'application/octet-stream') {
  const bin = atob(b64)
  const arr = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i)
  return new Blob([arr], { type: mimeType })
}

let persistenceRequested = false
async function requestPersistence() {
  if (persistenceRequested) return
  persistenceRequested = true
  if (typeof navigator !== 'undefined' && navigator.storage?.persist) {
    try { await navigator.storage.persist() } catch { /* ignore */ }
  }
}
