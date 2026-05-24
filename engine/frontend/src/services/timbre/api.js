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
    // Scale variants — only the ones the analysis actually produced
    scales: {
      peaks_ratios:               r.peaks_ratios,
      peaks_ratios_cons:          r.peaks_ratios_cons,
      extended_peaks_ratios:      r.extended_peaks_ratios,
      extended_peaks_ratios_cons: r.extended_peaks_ratios_cons,
      diss_scale:                 r.diss_scale,
      HE:                         r.HE_scale,
      euler_fokker:               r.euler_fokker,
      harm_tuning:                r.harm_tuning_scale,
      harm_fit:                   r.harm_fit_tuning_scale,
    },
    // Modulator sources
    pac_freqs:        r.pac_freqs    || null,
    pac_coupling:     r.pac_coupling || null,
    cfc_freqs:        r.cfc_freqs    || null,
    cfc_coupling:     r.cfc_coupling || null,
    intermodulations: r.endogenous_intermodulations || null,
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

/** Trigger a browser download of an exported file. */
export function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
