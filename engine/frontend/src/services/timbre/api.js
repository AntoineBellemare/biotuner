/**
 * Timbre API — thin wrappers around the backend timbre endpoints.
 *
 * The compute endpoint returns a JSON snapshot (partials, amps, phases,
 * modulator routings, voicing) that the frontend feeds directly into
 * the TimbreSynth and the SpectrumViz. The export endpoint streams a
 * file we download on the user's behalf.
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

/**
 * Derive the set of available scales from an analysis result.
 *
 * The /api/analyze endpoint puts the user's chosen tuning into a single
 * ``tuning`` field, labelled by ``tuning_method``. It does NOT populate
 * individual fields like ``diss_scale`` / ``HE_scale`` separately. So
 * we have to map the chosen tuning back to whichever SCALE_KEYS slot it
 * belongs to, based on the tuning method. Plus ``peaks_ratios`` is
 * always trivially derivable from ``peaks`` (peaks / min(peaks)).
 *
 * Returns a dict keyed by SCALE_KEYS (vocabulary used by HarmonicInput),
 * containing only the scales that are actually available. Exported so
 * both the request builder AND the UI's "what scales are available"
 * dropdown agree on the same source of truth.
 */
export function deriveScales(analysisResult, extras = {}) {
  if (!analysisResult) return {}
  const r = analysisResult
  const out = {}
  // peaks_ratios is always free: peaks / min(peaks).
  if (Array.isArray(r.peaks) && r.peaks.length > 0) {
    const minP = Math.min(...r.peaks)
    if (minP > 0) out.peaks_ratios = r.peaks.map((p) => p / minP)
  }
  // Map the chosen tuning to its corresponding SCALE_KEYS slot.
  // tuning_method comes from the analyze endpoint response and tells
  // us which biotuner computation produced ``r.tuning``.
  if (Array.isArray(r.tuning) && r.tuning.length > 0) {
    const method = (r.tuning_method || '').toLowerCase()
    if (method === 'diss_curve')           out.diss_scale = r.tuning
    else if (method === 'harmonic_fit')    out.harm_fit   = r.tuning
    else if (method === 'harm_fit')        out.harm_fit   = r.tuning
    else if (method === 'harm_tuning')     out.harm_tuning = r.tuning
    else if (method === 'euler_fokker')    out.euler_fokker = r.tuning
    else if (method === 'harmonic_entropy' || method === 'he') out.HE = r.tuning
    // Pure peaks-based tunings don't add a new entry — peaks_ratios
    // already covers it.
  }
  // Forward-compat: if the backend ever starts populating explicit
  // scale fields, pick them up too (overrides the chosen-tuning slot
  // when both are present).
  const explicitMap = {
    peaks_ratios:               'peaks_ratios',
    peaks_ratios_cons:          'peaks_ratios_cons',
    extended_peaks_ratios:      'extended_peaks_ratios',
    extended_peaks_ratios_cons: 'extended_peaks_ratios_cons',
    diss_scale:                 'diss_scale',
    HE_scale:                   'HE',
    euler_fokker:               'euler_fokker',
    harm_tuning_scale:          'harm_tuning',
    harm_fit_tuning_scale:      'harm_fit',
  }
  for (const [bt_attr, hi_key] of Object.entries(explicitMap)) {
    if (Array.isArray(r[bt_attr]) && r[bt_attr].length > 0) {
      out[hi_key] = r[bt_attr]
    }
  }
  // Caller-supplied extras (e.g. extended ratios computed on demand
  // via /api/timbre/extended-ratios) merge in last so they override.
  // Keys must already match SCALE_KEYS.
  for (const [hi_key, vals] of Object.entries(extras || {})) {
    if (Array.isArray(vals) && vals.length > 0) {
      out[hi_key] = vals
    }
  }
  return out
}

/**
 * Build the request payload from an analysis result + user design choices.
 *
 * Centralises the field mapping so the components don't all need to
 * know which analysis-result keys map to which TimbreComputeRequest
 * fields. Callers pass the full `analysisResult` from app state plus
 * a `design` dict with their UI choices.
 */
export function buildTimbreRequest(analysisResult, design = {}) {
  if (!analysisResult) return null
  const r = analysisResult
  const req = {
    peaks: r.peaks || [],
    amps:  r.amps  || r.powers || null,
    // Optional Tier-A fields — only forward when actually present
    phases:             r.phases             || null,
    linewidths:         r.linewidth          || r.peaks_linewidth || null,
    aperiodic_exponent: r.aperiodic_exponent ?? null,
    spectral_flatness:  r.spectral_flatness  ?? r.spectral_entropy ?? null,
    // Scale variants — only the ones the analysis actually produced,
    // plus any extras (e.g. on-demand extended ratios) merged in by
    // the caller. Derived in one place so the UI's availability
    // dropdown and this request stay in sync.
    scales: deriveScales(analysisResult, design.scale_extras || {}),
    // Modulator sources
    pac_freqs:        r.pac_freqs    || null,
    pac_coupling:     r.pac_coupling || null,
    cfc_freqs:        r.cfc_freqs    || null,
    cfc_coupling:     r.cfc_coupling || null,
    // Intermods: prefer the user's on-demand cache from
    // /api/timbre/intermods; fall back to whatever the analysis
    // result happened to include (rare — only set when analyze
    // explicitly populated it).
    intermodulations: (design.intermods_override && design.intermods_override.length)
      ? design.intermods_override
      : (r.endogenous_intermodulations || null),
    // Design choices
    scale_priority:    design.scale_priority   || null,
    matching_method:   design.matching_method  || 'harmonic_input',
    voicing:           design.voicing          || {},
    enabled_modulators: design.enabled_modulators || {},
    export_config:     design.export_config    || null,
    enrichment:        design.enrichment        || null,
  }
  // Drop empty `scales` keys so the backend sees None for absent variants
  req.scales = Object.fromEntries(
    Object.entries(req.scales).filter(([, v]) => v && v.length > 0),
  )
  if (Object.keys(req.scales).length === 0) req.scales = null
  return req
}

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

/**
 * POST /api/timbre/compute — returns the Timbre snapshot for viz + synth.
 */
export async function computeTimbre(payload) {
  const { data } = await client.post('/api/timbre/compute', payload)
  return data
}

/**
 * POST /api/timbre/export/{format} — downloads a file. Returns the Blob
 * + filename so the caller can decide what to do (we save-as by default).
 */
export async function exportTimbre(format, payload) {
  const res = await client.post(`/api/timbre/export/${format}`, payload, {
    responseType: 'blob',
  })
  // Extract filename from Content-Disposition; fall back to a sensible default.
  const dispo = res.headers?.['content-disposition'] || ''
  const match = /filename="?([^";]+)"?/i.exec(dispo)
  const filename = match?.[1] || `timbre.${format}`
  return { blob: res.data, filename }
}

/**
 * POST /api/timbre/extended-ratios — run peaks_extension on the
 * current peak list and return the extended ratios. The result is a
 * dict { extended_peaks, extended_peaks_ratios, extended_peaks_ratios_cons }
 * that the caller can stash and pass into the next buildTimbreRequest
 * to unlock the "Extended raw ratios" / "Extended cons. ratios" scale
 * options without re-running the whole analysis.
 */
export async function computeExtendedRatios(payload) {
  const { data } = await client.post('/api/timbre/extended-ratios', payload)
  return data
}

/**
 * POST /api/timbre/compute-scale/{name} — populate any of the
 * Scale-Source dropdown options on demand. Backend builds a fresh
 * compute_biotuner from the peaks, runs the relevant scale
 * construction method, and returns the resulting ratios.
 */
export async function computeScale(scaleName, payload) {
  const { data } = await client.post(
    `/api/timbre/compute-scale/${scaleName}`,
    payload,
  )
  return data
}

/**
 * POST /api/timbre/intermods — detect endogenous intermodulation pairs
 * from the analysis's peak set. The analyze endpoint doesn't run this
 * by default, so the Timbre tab's enrichment toggle stays disabled
 * until the user opts in here.
 */
export async function computeIntermods(payload) {
  const { data } = await client.post('/api/timbre/intermods', payload)
  return data
}

/** Trigger a browser download of an exported file. */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
