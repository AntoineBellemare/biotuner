/**
 * Source-mode derivation for the JS-engine geometries.
 *
 * The user picks a *source mode* per geometry, and we expand the analysis's
 * tuning ratios into a list of candidate ratios. The geometry's
 * `fromDerivedRatios(derived)` then maps the first N of those into its
 * frequency-defining params.
 *
 * Modes:
 *   direct       - the analysis ratios themselves (the "raw" tuning).
 *   harmonics    - each ratio × 1..depth.
 *   subharmonics - each ratio / 1..depth.
 *   intermod     - pairwise sum / product / |difference|.
 *   manual       - ignore the analysis; the user's sliders take over.
 */

function meaningful(arr, eps = 5e-4) {
  const out = []
  for (const v of arr) {
    if (!Number.isFinite(v) || v <= 0) continue
    if (out.every((s) => Math.abs(v - s) > eps)) out.push(v)
  }
  return out
}

const directDerive = (ratios) => meaningful(ratios || [])

const harmonicsDerive = (ratios, depth = 4) => {
  if (!ratios?.length) return []
  const out = []
  for (const r of ratios) {
    for (let n = 1; n <= depth; n++) out.push(r * n)
  }
  return meaningful(out)
}

const subharmonicsDerive = (ratios, depth = 4) => {
  if (!ratios?.length) return []
  const out = []
  for (const r of ratios) {
    for (let n = 1; n <= depth; n++) out.push(r / n)
  }
  return meaningful(out)
}

const intermodDerive = (ratios) => {
  if (!ratios?.length) return []
  const out = []
  for (let i = 0; i < ratios.length; i++) {
    for (let j = i + 1; j < ratios.length; j++) {
      const a = ratios[i]
      const b = ratios[j]
      out.push(a + b)
      out.push(a * b)
      out.push(Math.abs(a - b))
    }
  }
  return meaningful(out)
}

export const SOURCE_MODES = {
  direct: {
    label: 'Direct ratios',
    description: 'Use the analysis ratios as-is (decimal, not rounded).',
    derive: directDerive,
  },
  harmonics: {
    label: 'Harmonics (×n)',
    description: 'Integer multiples of each ratio: ×1, ×2, ×3, ×4.',
    derive: harmonicsDerive,
  },
  subharmonics: {
    label: 'Subharmonics (÷n)',
    description: 'Integer divisions of each ratio: ÷1, ÷2, ÷3, ÷4.',
    derive: subharmonicsDerive,
  },
  intermod: {
    label: 'Intermodulation',
    description: 'r_i + r_j, r_i × r_j, |r_i − r_j| of every pair.',
    derive: intermodDerive,
  },
  manual: {
    label: 'Manual',
    description: 'Free editing — ignore the analysis ratios.',
    derive: () => null,
  },
}

export const SOURCE_MODE_ORDER = ['direct', 'harmonics', 'subharmonics', 'intermod', 'manual']

export function deriveFromMode(ratios, mode) {
  const m = SOURCE_MODES[mode] || SOURCE_MODES.direct
  return m.derive(ratios) || []
}
