/**
 * Parse a backend analysis error message into a list of actionable
 * suggestions. Each suggestion knows how to mutate the current
 * analysisConfig so the user can tap a chip and re-run.
 *
 * The backend (biotuner_service.py) already produces structured hints in its
 * error text — phrases like "Reduce precision (try 0.1 Hz)" or
 * "Try a different peak extraction method like 'EMD'". This module turns
 * those hints into UI suggestions.
 *
 * Returned suggestions: { label, patch } where `patch` is a partial
 * AnalysisConfig override (applied with `{ ...current, ...patch }`).
 */

const METHOD_ALIASES = {
  EMD: 'EMD',
  HARMONIC_RECURRENCE: 'harmonic_recurrence',
  'HARMONIC RECURRENCE': 'harmonic_recurrence',
  FOOOF: 'FOOOF',
  EIMC: 'EIMC',
  FIXED: 'fixed',
  CEPSTRUM: 'cepstrum',
  SMS: 'SMS',
}

/**
 * @param {string} message  – error.response?.data?.detail or similar
 * @param {object} config   – the current analysisConfig
 * @returns {{label: string, patch: object}[]}
 */
export function suggestionsFor(message, config = {}) {
  if (!message || typeof message !== 'string') return []
  const text = message
  const lower = text.toLowerCase()
  const out = []

  // 1) "try 0.1 hz" / "precision = 0.1" / "reduce precision"
  const precMatch =
    lower.match(/precision[^0-9]*([0-9]*\.?[0-9]+)\s*hz/i) ||
    lower.match(/try\s*([0-9]*\.?[0-9]+)\s*hz/i)
  if (precMatch) {
    const v = parseFloat(precMatch[1])
    if (Number.isFinite(v) && v > 0 && v !== config.precision) {
      out.push({ label: `Precision → ${v} Hz`, patch: { precision: v } })
    }
  } else if (/reduce precision|lower(?:ing)?\s+precision|precision(?:\s+is)?\s+too\s+(?:high|large)/i.test(text)) {
    // Generic "lower precision" without a number → halve it.
    const cur = Number(config.precision) || 1
    const next = Math.max(0.01, +(cur / 2).toFixed(2))
    if (next < cur) {
      out.push({ label: `Precision → ${next} Hz`, patch: { precision: next } })
    }
  }

  // 2) "increase max frequency" / "max_freq"
  if (/increase\s+max\s*(?:frequency|freq)|max[\s_]*freq(?:uency)?\s+too\s+low/i.test(text)) {
    const cur = Number(config.max_freq) || 100
    const next = Math.min(20000, Math.round(cur * 2))
    if (next > cur) {
      out.push({ label: `Max freq → ${next} Hz`, patch: { max_freq: next } })
    }
  }

  // 3) Suggested method name in single-quotes ('EMD', 'harmonic_recurrence', ...)
  const methodRe = /['"]([A-Za-z_ ]+?)['"]/g
  let m
  const seen = new Set()
  while ((m = methodRe.exec(text)) !== null) {
    const key = m[1].toUpperCase().trim()
    const resolved = METHOD_ALIASES[key]
    if (resolved && resolved !== config.method && !seen.has(resolved)) {
      seen.add(resolved)
      out.push({ label: `Try ${resolved}`, patch: { method: resolved } })
    }
  }

  // 4) "longer signal segment" — can't fix from config alone, but flag it
  //    via a no-op suggestion so the user knows what to do.
  // (Intentionally not rendered as a chip; only actionable items become chips.)

  // 5) "different peak extraction method" with no specific name — propose
  //    a sensible alternative based on current method.
  if (
    out.every((s) => !('method' in s.patch)) &&
    /different\s+(?:peak\s+extraction\s+)?method|try\s+a\s+different\s+method/i.test(text)
  ) {
    const fallback = config.method === 'EMD' ? 'harmonic_recurrence' : 'EMD'
    out.push({ label: `Try ${fallback}`, patch: { method: fallback } })
  }

  // 6) Deduplicate by label
  const dedup = []
  const labels = new Set()
  for (const s of out) {
    if (!labels.has(s.label)) {
      labels.add(s.label)
      dedup.push(s)
    }
  }
  return dedup
}
