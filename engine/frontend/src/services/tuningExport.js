/**
 * Client-side tuning export.
 *
 * Mirrors backend/services/tuning_export_service.py so .scl / .txt / .json
 * downloads work offline and survive a stale backend deploy.
 */

function ratiosToCents(ratios) {
  return ratios
    .filter((r) => r != null && r > 0)
    .map((r) => 1200 * Math.log2(r))
}

function sanitizeDescription(desc) {
  return (desc || 'Biotuner-derived tuning').trim().replace(/[\r\n]+/g, ' ')
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  // Give the click handler time to fire before we revoke.
  setTimeout(() => URL.revokeObjectURL(url), 0)
}

/**
 * Emit a Scala (.scl) file as a Blob. Skips a leading 1/1 (Scala excludes
 * the unison) and appends "2/1" as the closing octave if the tuning doesn't
 * already end there. Pitches are written in cents (with one decimal), except
 * the closing octave which uses the literal ratio "2/1".
 */
export function toSclBlob(tuning, description = 'Biotuner-derived tuning',
                          filename = 'biotuner_tuning') {
  description = sanitizeDescription(description)
  let ratios = (tuning || []).filter((r) => r && r > 0)
  if (ratios.length && Math.abs(ratios[0] - 1.0) < 1e-9) ratios = ratios.slice(1)
  const hasOctave = ratios.length && Math.abs(ratios[ratios.length - 1] - 2.0) < 1e-6
  if (!hasOctave) ratios = [...ratios, 2.0]

  const lines = [
    `! ${filename}.scl`,
    '!',
    description,
    ` ${ratios.length}`,
    '!',
  ]
  for (const r of ratios) {
    if (Math.abs(r - 2.0) < 1e-9) lines.push(' 2/1')
    else lines.push(` ${(1200 * Math.log2(r)).toFixed(4)}`)
  }
  return new Blob([lines.join('\n') + '\n'], { type: 'application/octet-stream' })
}

/**
 * Plain-text dump: one ratio per line with cents in a tab column.
 */
export function toTxtBlob(tuning, description = 'Biotuner-derived tuning',
                          filename = 'biotuner_tuning') {
  description = sanitizeDescription(description)
  const lines = [`# ${filename}`, `# ${description}`, '# ratio\tcents']
  for (const r of tuning || []) {
    if (r == null || r <= 0) continue
    lines.push(`${r.toFixed(6)}\t${(1200 * Math.log2(r)).toFixed(4)}`)
  }
  return new Blob([lines.join('\n') + '\n'], { type: 'text/plain' })
}

/**
 * JSON dump: name, description, ratios, cents — useful for tooling.
 */
export function toJsonBlob(tuning, description = 'Biotuner-derived tuning',
                           filename = 'biotuner_tuning') {
  const clean = (tuning || []).filter((r) => r && r > 0)
  const payload = {
    name: filename,
    description: sanitizeDescription(description),
    ratios: clean.map((r) => Number(r)),
    cents: ratiosToCents(clean),
  }
  return new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
}

const EXPORTERS = { scl: toSclBlob, txt: toTxtBlob, json: toJsonBlob }

/**
 * One-shot helper: build the blob and trigger a browser download.
 */
export function exportTuning(format, tuning, {
  description = 'Biotuner-derived tuning',
  filename = 'biotuner_tuning',
} = {}) {
  const fn = EXPORTERS[format.toLowerCase()]
  if (!fn) throw new Error(`Unsupported export format: ${format}`)
  const blob = fn(tuning, description, filename)
  downloadBlob(blob, `${filename}.${format.toLowerCase()}`)
}
