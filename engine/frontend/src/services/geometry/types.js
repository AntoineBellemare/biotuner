/**
 * Registry of harmonic-geometry types renderable in the GeometryTab.
 *
 * Each entry conforms to:
 *   {
 *     key, label, description,
 *     defaultParams: { ... },
 *     paramSchema: [ { key, label, type, min, max, step, format? }, ... ],
 *     fromRatios: (ratios) => Partial<params>,    // map a derived tuning in
 *     render: (params, tSec) => GeometryOutput,
 *   }
 *
 * GeometryOutput is either { kind: 'path', points } or
 * { kind: 'field', data, width, height }.
 */

import { findFraction, gcd } from './utils'

// Pick the first ratio that's at least `minCents` from the unison; falls back
// to ratios[0] * 1.5 (or 1.5) if nothing qualifies. Used by every JS-engine
// fromRatios so a tuning whose ratios are all clustered near 1.0 doesn't
// collapse to the trivial 1/1 fraction.
function firstMeaningfulRatio(ratios, minCents = 30) {
  if (!ratios?.length) return 1.5
  for (const r of ratios) {
    if (!Number.isFinite(r) || r <= 0) continue
    if (Math.abs(1200 * Math.log2(r)) >= minCents) return r
  }
  return (ratios[0] || 1) * 1.5
}

// Small-integer ratio pairs used as fallbacks when a tight tuning
// collapses every pairwise ratio to ~1:1. Picked from binding indices so
// every slot choice produces a distinct pattern.
const NICE_PAIRS = [
  [1, 2], [2, 3], [3, 4], [3, 5], [4, 5], [5, 6],
  [2, 5], [3, 7], [4, 7], [5, 7], [2, 7], [5, 8],
]

export function nicePairFor(ia, ib) {
  const i = (ia * 13 + ib * 5) % NICE_PAIRS.length
  return NICE_PAIRS[((i % NICE_PAIRS.length) + NICE_PAIRS.length) % NICE_PAIRS.length]
}

/**
 * Picks the integer fraction n/d from `ratio_b / ratio_a` capped at a
 * small denominator. Falls back to `nicePairFor` when the fraction
 * collapses to ~1:1 or to a dense near-square ratio.
 */
export function smallIntFractionForPair(derived, ia, ib, maxDenom = 6) {
  const ra = derived[ia] || 1
  const rb = derived[ib] || 1.5
  let k = rb / ra
  if (k < 1) k = 1 / k
  let { n, d } = findFraction(k, maxDenom)
  if (n === d || (n + d > 9 && Math.abs(n - d) <= 1)) {
    const [pn, pd] = nicePairFor(ia, ib)
    n = pn
    d = pd
  }
  return { n, d }
}

// Pick the first N ratios that are at least `minCents` apart from the unison
// and from each other. Pads with 1.5× the last value if fewer than N exist.
function meaningfulRatios(ratios, n, minCents = 30) {
  const out = []
  if (ratios?.length) {
    for (const r of ratios) {
      if (!Number.isFinite(r) || r <= 0) continue
      const isFar =
        Math.abs(1200 * Math.log2(r)) >= minCents &&
        out.every((p) => Math.abs(1200 * Math.log2(r / p)) >= minCents)
      if (isFar) out.push(r)
      if (out.length >= n) break
    }
  }
  while (out.length < n) {
    out.push(out.length ? out[out.length - 1] * 1.5 : 1.5)
  }
  return out
}

// ---------------------------------------------------------------------------
// Lissajous — two perpendicular sinusoids, the canonical harmonic curve.
// ---------------------------------------------------------------------------

const lissajous = {
  key: 'lissajous',
  label: 'Lissajous',
  description:
    'Curve traced by two perpendicular sinusoids. The ratio a:b sets the ' +
    'shape; phase δ sweeps it through closed and open states.',
  defaultParams: {
    a: 3,
    b: 2,
    delta: Math.PI / 4,
    cycles: 1,
    lineWidth: 1.5,
  },
  paramSchema: [
    // a, b are integer winding numbers — what makes Lissajous patterns close.
    { key: 'a',         label: 'Frequency a', type: 'int',    min: 1,   max: 24, step: 1, derived: true },
    { key: 'b',         label: 'Frequency b', type: 'int',    min: 1,   max: 24, step: 1, derived: true },
    { key: 'delta',     label: 'Phase δ',     type: 'slider', min: 0,   max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'cycles',    label: 'Cycles',      type: 'slider', min: 0.5, max: 8,  step: 0.1 },
    { key: 'lineWidth', label: 'Line width',  type: 'slider', min: 0.4, max: 4,  step: 0.1 },
  ],
  fromRatios(ratios) {
    const r = firstMeaningfulRatio(ratios)
    const { n, d } = findFraction(r, 12)
    return { a: n, b: d }
  },
  slots: [
    { key: 'a' },
    { key: 'b' },
  ],
  fromDerivedRatios(derived, bindings = {}) {
    if (!derived?.length) return {}
    const ia = Math.min(derived.length - 1, bindings.a ?? 0)
    const ib = Math.min(derived.length - 1, bindings.b ?? 1)
    const { n, d } = smallIntFractionForPair(derived, ia, ib, 6)
    return { a: n, b: d }
  },
  render(params, t) {
    const { a, b, delta, cycles } = params
    const N = 1400
    const totalAngle = cycles * 2 * Math.PI
    const animDelta = delta + (t || 0) * 0.4
    const points = new Array(N + 1)
    for (let i = 0; i <= N; i++) {
      const u = (i / N) * totalAngle
      points[i] = { x: Math.sin(a * u + animDelta), y: Math.sin(b * u) }
    }
    return { kind: 'path', points }
  },
}

// ---------------------------------------------------------------------------
// Harmonograph — four damped pendulums (two per axis).
// ---------------------------------------------------------------------------

const harmonograph = {
  key: 'harmonograph',
  label: 'Harmonograph',
  description:
    'Damped pendulums tracing interfering curves — a Victorian-era kinetic ' +
    'instrument. Slow decay gives long fractal spirals; fast decay collapses ' +
    'to a small flower.',
  defaultParams: {
    f1: 2,    f2: 3,    f3: 2,    f4: 5,
    p1: 0,    p2: 1.5,  p3: 0.5,  p4: 2.4,
    d1: 0.002, d2: 0.0015, d3: 0.001, d4: 0.0025,
    damping: 0.002,        // global damping override — 0 disables override
    duration: 90,
    lineWidth: 0.7,
    n_components: 4,       // 2 / 3 / 4 — how many of the derived ratios to use
    max_denom: 6,          // round each freq to nearest p/q with denom ≤ this
  },
  paramSchema: [
    { key: 'n_components', label: 'Pendulums', type: 'int',    min: 2,   max: 4,    step: 1,
      format: (v) => v === 2 ? '2 (simple)' : v === 3 ? '3 (rich)' : '4 (full)' },
    { key: 'max_denom',    label: 'Simplify (max denom)', type: 'int', min: 0, max: 24, step: 1,
      format: (v) => v <= 0 ? 'off (raw decimals)' : `≤ ${v}` },
    { key: 'damping',   label: 'Damping',    type: 'slider', min: 0,   max: 0.02, step: 0.0001,
      format: (v) => v < 0.0001 ? 'off (per-pendulum)' : v.toFixed(4) },
    { key: 'duration',  label: 'Duration',   type: 'slider', min: 20,  max: 300, step: 5,
      format: (v) => `${v}s` },
    { key: 'lineWidth', label: 'Line width', type: 'slider', min: 0.3, max: 3,   step: 0.05 },
    // Per-pendulum frequencies are derived from analysis. Shown as read-only
    // in derived modes; editable when source = manual.
    { key: 'f1', label: 'X freq 1', type: 'slider', min: 0.1, max: 30, step: 0.01, derived: true,
      format: (v) => Number(v).toFixed(2) },
    { key: 'f2', label: 'X freq 2', type: 'slider', min: 0.1, max: 30, step: 0.01, derived: true,
      format: (v) => Number(v).toFixed(2) },
    { key: 'f3', label: 'Y freq 1', type: 'slider', min: 0.1, max: 30, step: 0.01, derived: true,
      format: (v) => Number(v).toFixed(2) },
    { key: 'f4', label: 'Y freq 2', type: 'slider', min: 0.1, max: 30, step: 0.01, derived: true,
      format: (v) => Number(v).toFixed(2) },
    { key: 'p1', label: 'X phase 1', type: 'slider', min: 0, max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π`, advanced: true },
    { key: 'p2', label: 'X phase 2', type: 'slider', min: 0, max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π`, advanced: true },
    { key: 'p3', label: 'Y phase 1', type: 'slider', min: 0, max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π`, advanced: true },
    { key: 'p4', label: 'Y phase 2', type: 'slider', min: 0, max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π`, advanced: true },
    { key: 'd1', label: 'Decay X1', type: 'slider', min: 0, max: 0.01, step: 0.00005,
      format: (v) => v.toFixed(4), advanced: true },
    { key: 'd2', label: 'Decay X2', type: 'slider', min: 0, max: 0.01, step: 0.00005,
      format: (v) => v.toFixed(4), advanced: true },
    { key: 'd3', label: 'Decay Y1', type: 'slider', min: 0, max: 0.01, step: 0.00005,
      format: (v) => v.toFixed(4), advanced: true },
    { key: 'd4', label: 'Decay Y2', type: 'slider', min: 0, max: 0.01, step: 0.00005,
      format: (v) => v.toFixed(4), advanced: true },
  ],
  fromRatios(ratios) {
    const picked = meaningfulRatios(ratios, 4, 30)
    return {
      f1: +(picked[0] * 2).toFixed(2),
      f2: +(picked[1] * 2).toFixed(2),
      f3: +(picked[2] * 2).toFixed(2),
      f4: +(picked[3] * 2).toFixed(2),
    }
  },
  slots: [
    { key: 'f1', scale: 2 },
    { key: 'f2', scale: 2 },
    { key: 'f3', scale: 2 },
    { key: 'f4', scale: 2 },
  ],
  fromDerivedRatios(derived, bindings = {}) {
    if (!derived?.length) return {}
    // Spread four derived ratios across the X/Y pendulum quartet, with a
    // 2× scaling so the visible frequency content lands in the 1–10 range.
    const pad = (key, defIdx, fallback) => {
      const i = Math.min(derived.length - 1, bindings[key] ?? defIdx)
      return +((derived[i] ?? fallback) * 2).toFixed(3)
    }
    return {
      f1: pad('f1', 0, 1.0),
      f2: pad('f2', 1, 1.5),
      f3: pad('f3', 2, 1.25),
      f4: pad('f4', 3, 1.66),
    }
  },
  // Snap a decimal frequency to the nearest p/q with denom ≤ maxDenom.
  // Exported so the renderer + UI use the same rounding.
  snap(f, maxDenom) {
    if (!maxDenom || maxDenom <= 0) return { f, n: null, d: null }
    const { n, d } = findFraction(f, Math.min(24, maxDenom))
    return { f: n / d, n, d }
  },
  render(params, t) {
    const { p1, p2, p3, p4, d1, d2, d3, d4, duration, damping, n_components = 4, max_denom = 0 } = params
    let { f1, f2, f3, f4 } = params
    // Optionally simplify each frequency to a small p/q ratio. Helps tame
    // messy patterns and produces musically-meaningful curves.
    const snap = (f) => {
      if (!max_denom || max_denom <= 0) return f
      const { n, d } = findFraction(f, Math.min(24, max_denom))
      return n / d
    }
    f1 = snap(f1); f2 = snap(f2); f3 = snap(f3); f4 = snap(f4)
    // Use the global damping when set (> tiny epsilon); otherwise the
    // per-pendulum decays in the Advanced panel.
    const useGlobal = damping != null && damping > 1e-5
    const dx1 = useGlobal ? damping : d1
    const dx2 = useGlobal ? damping : d2
    const dy1 = useGlobal ? damping : d3
    const dy2 = useGlobal ? damping : d4
    const nc = Math.max(2, Math.min(4, Math.round(n_components || 4)))
    const N = 6000
    const dt = duration / N
    const phaseDrift = (t || 0) * 0.1
    const points = new Array(N + 1)
    for (let i = 0; i <= N; i++) {
      const tt = i * dt
      // n_components = 2 → use just the first X (f1) and first Y (f3) pendulum
      // n_components = 3 → add f2 (X pair), keep single Y pendulum
      // n_components = 4 → full quartet
      let x = Math.exp(-dx1 * tt) * Math.sin(f1 * tt + p1 + phaseDrift)
      if (nc >= 3) {
        x += Math.exp(-dx2 * tt) * Math.sin(f2 * tt + p2)
      }
      let y = Math.exp(-dy1 * tt) * Math.sin(f3 * tt + p3)
      if (nc >= 4) {
        y += Math.exp(-dy2 * tt) * Math.sin(f4 * tt + p4 - phaseDrift)
      }
      points[i] = { x: x / 2, y: y / 2 }
    }
    return { kind: 'path', points }
  },
}

// ---------------------------------------------------------------------------
// Rose curve — r = cos(k θ), k = n/d derived from the harmonic ratio.
// ---------------------------------------------------------------------------

const rose = {
  key: 'rose',
  label: 'Rose',
  description:
    'Polar curve r = cos(k θ) where k = n/d is set by the chosen ratio. ' +
    'Even k gives 2k petals, odd k gives k petals, rational k gives ' +
    'self-intersecting flower-like patterns.',
  defaultParams: { n: 5, d: 4, cycles: 8, complexity: 1, max_denom: 12, d_mult: 1, lineWidth: 1.6 },
  paramSchema: [
    { key: 'n',         label: 'Numerator',   type: 'slider', min: 1,   max: 48, step: 1, derived: true },
    { key: 'd',         label: 'Denominator', type: 'slider', min: 1,   max: 48, step: 1, derived: true },
    { key: 'complexity',label: 'Complexity ×n', type: 'slider', min: 1, max: 12, step: 1 },
    { key: 'd_mult',    label: 'Denom ×',     type: 'slider', min: 1,   max: 12, step: 1 },
    { key: 'max_denom', label: 'Rounding (max denom)', type: 'int', min: 3, max: 24, step: 1,
      format: (v) => `≤ ${v}` },
    { key: 'cycles',    label: 'Cycles',      type: 'slider', min: 1,   max: 48, step: 1 },
    { key: 'lineWidth', label: 'Line width',  type: 'slider', min: 0.4, max: 4,  step: 0.1 },
  ],
  fromRatios(ratios) {
    const r = firstMeaningfulRatio(ratios)
    const { n, d } = findFraction(r, 16)
    return { n, d }
  },
  slots: [
    { key: 'n' },
    { key: 'd' },
  ],
  fromDerivedRatios(derived, bindings = {}, params = {}) {
    if (!derived?.length) return {}
    const ia = Math.min(derived.length - 1, bindings.n ?? 0)
    const ib = Math.min(derived.length - 1, bindings.d ?? 1)
    const md = Math.max(3, Math.min(24, params.max_denom ?? 12))
    return smallIntFractionForPair(derived, ia, ib, md)
  },
  render(params, t) {
    const { n, d, cycles, complexity = 1, d_mult = 1 } = params
    if (!Number.isFinite(n) || !Number.isFinite(d) || d <= 0) {
      return { kind: 'path', points: [] }
    }
    // Two independent multipliers: complexity multiplies n (adds petals),
    // d_mult multiplies d (rotates the petal symmetry / fractures it into
    // sub-petals). Together they unlock a much wider design space.
    const k = (n * Math.max(1, complexity)) / (d * Math.max(1, d_mult))
    // Always honour the user's cycles slider. Integer n/d would mathematically
    // close after d/gcd revolutions, but tracing extra cycles lays the same
    // curve back on itself — visually identical, and the slider stays
    // responsive when the user drags it.
    const periods = Math.max(1, cycles || 1)
    const N = Math.min(8000, Math.max(1500, periods * 400))
    const phase = (t || 0) * 0.4
    const points = new Array(N + 1)
    for (let i = 0; i <= N; i++) {
      const theta = (i / N) * periods * 2 * Math.PI
      const r = Math.cos(k * theta + phase)
      points[i] = { x: r * Math.cos(theta), y: r * Math.sin(theta) }
    }
    return { kind: 'path', points }
  },
}

// ---------------------------------------------------------------------------
// Spirograph (hypotrochoid) — small circle rolling inside a larger one.
// ---------------------------------------------------------------------------

const spirograph = {
  key: 'spirograph',
  label: 'Spirograph',
  description:
    'Hypotrochoid: the path traced by a point inside a small circle (radius ' +
    'r) rolling inside a large one (radius R). The R:r ratio sets the rose; ' +
    'the offset d sets how "deep" the petals reach.',
  defaultParams: { R: 21, r: 12, offset: 7, scale: 3, complexity: 1, lineWidth: 1.2 },
  paramSchema: [
    { key: 'R',         label: 'Outer R',     type: 'slider', min: 0.5, max: 60, step: 0.5, derived: true,
      format: (v) => Number(v).toFixed(1) },
    { key: 'r',         label: 'Inner r',     type: 'slider', min: 0.2, max: 40, step: 0.5, derived: true,
      format: (v) => Number(v).toFixed(1) },
    { key: 'offset',    label: 'Pen offset',  type: 'slider', min: 0.5, max: 30, step: 0.1 },
    { key: 'complexity',label: 'Complexity ×R', type: 'slider', min: 1, max: 8, step: 1 },
    { key: 'scale',     label: 'Pattern scale', type: 'slider', min: 1, max: 8, step: 0.5,
      format: (v) => `×${Number(v).toFixed(1)}` },
    { key: 'lineWidth', label: 'Line width',  type: 'slider', min: 0.3, max: 3,  step: 0.05 },
  ],
  fromRatios(ratios) {
    const r = firstMeaningfulRatio(ratios)
    const f = findFraction(r, 12)
    return { R: f.n * 3, r: f.d * 3, offset: Math.max(1, f.d * 2) }
  },
  slots: [
    { key: 'R' },
    { key: 'r' },
  ],
  fromDerivedRatios(derived, bindings = {}) {
    if (!derived?.length) return {}
    const iR = Math.min(derived.length - 1, bindings.R ?? 1)
    const ir = Math.min(derived.length - 1, bindings.r ?? 0)
    const { n, d } = smallIntFractionForPair(derived, iR, ir, 11)
    return {
      R: Math.max(n, d),
      r: Math.min(n, d),
    }
  },
  render(params, t) {
    const { offset, scale = 1, complexity = 1 } = params
    // Complexity multiplies the OUTER radius — increases (R-r)/r which is
    // the petal-density factor — without losing the data-driven R:r ratio.
    const R = (params.R || 1) * scale * Math.max(1, complexity)
    const r = (params.r || 1) * scale
    if (r <= 0 || R <= r) {
      return { kind: 'path', points: [] }
    }
    const ratio = (R - r) / r
    // Pattern closes after lcm(R, r)/R revolutions; approximate with the
    // integer denominator of R/r in reduced form.
    const Rs = Math.round(R * 100)
    const rs = Math.round(r * 100)
    const g = gcd(Rs, rs)
    const cycles = Math.min(20, Math.max(1, rs / g))
    const N = Math.min(5000, Math.max(400, cycles * 300))
    const drift = (t || 0) * 0.2
    const norm = R + offset
    const points = new Array(N + 1)
    for (let i = 0; i <= N; i++) {
      const tt = (i / N) * cycles * 2 * Math.PI
      const x = (R - r) * Math.cos(tt) + offset * Math.cos(ratio * tt + drift)
      const y = (R - r) * Math.sin(tt) - offset * Math.sin(ratio * tt + drift)
      points[i] = { x: x / norm, y: y / norm }
    }
    return { kind: 'path', points }
  },
}

// ---------------------------------------------------------------------------
// Chladni — analytical nodal pattern on a square plate, two-mode mixture.
//   ψ(x, y) = sin(m π x) sin(n π y) + sin(n π x) sin(m π y) cos(φ)
// ---------------------------------------------------------------------------

// Original JS-engine Chladni — kept here under a different key so the
// registry can still reference its render fn if needed. The user-facing
// Chladni now routes to the backend (see chladni_python below).
const chladniJs = {
  key: 'chladni_js',
  label: 'Chladni (legacy)',
  description: 'Legacy in-browser Chladni renderer.',
  defaultParams: {
    m: 5, n: 3, resolution: 256, contrast: 1.6, mix: Math.PI / 2,
    complexity: 0,        // 0 = vanilla 2-mode; >0 mixes higher-order terms
    rotation: 0,          // radians, rotates the (x,y) field
  },
  paramSchema: [
    { key: 'm',          label: 'Mode m',      type: 'int',    min: 1,  max: 14, step: 1, derived: true },
    { key: 'n',          label: 'Mode n',      type: 'int',    min: 1,  max: 14, step: 1, derived: true },
    { key: 'mix',        label: 'Mix angle',   type: 'slider', min: 0,  max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'complexity', label: 'Complexity',  type: 'slider', min: 0,  max: 1, step: 0.01,
      format: (v) => (v * 100).toFixed(0) + '%' },
    { key: 'rotation',   label: 'Rotation',    type: 'slider', min: 0,  max: Math.PI / 2, step: 0.01,
      format: (v) => `${((v / Math.PI) * 180).toFixed(0)}°` },
    { key: 'contrast',   label: 'Contrast',    type: 'slider', min: 0.4, max: 6, step: 0.05 },
    { key: 'resolution', label: 'Resolution',  type: 'slider', min: 64,  max: 512, step: 32, advanced: true },
  ],
  fromRatios(ratios) {
    const r = firstMeaningfulRatio(ratios)
    const f = findFraction(r, 10)
    return { m: f.n, n: f.d }
  },
  slots: [
    // Chladni modes are integers (standing-wave indices). The transform
    // maps a time-morphed decimal value into a valid mode integer.
    { key: 'm', transform: (v) => Math.max(2, Math.min(14, Math.round(v * 4))) },
    { key: 'n', transform: (v) => Math.max(2, Math.min(14, Math.round(v * 4))) },
  ],
  fromDerivedRatios(derived, bindings = {}) {
    if (!derived?.length) return {}
    const im = Math.min(derived.length - 1, bindings.m ?? 0)
    const inn = Math.min(derived.length - 1, bindings.n ?? 1)
    const { n, d } = smallIntFractionForPair(derived, im, inn, 7)
    return { m: n, n: d }
  },
  render(params, t) {
    const {
      m, n, mix, resolution,
      complexity = 0, rotation = 0,
    } = params
    const size = Math.max(32, Math.min(512, resolution))
    const data = new Float32Array(size * size)
    const phase = (mix || 0) + (t || 0) * 0.4
    const cosPhase = Math.cos(phase)
    // Higher-order companion modes — when complexity > 0 the field gains
    // sin((m+n) π x) sin(|m-n| π y) terms, breaking the regular tile grid.
    const p = Math.abs(m + n)
    const q = Math.max(1, Math.abs(m - n))
    const w = complexity
    const cosR = Math.cos(rotation)
    const sinR = Math.sin(rotation)
    for (let y = 0; y < size; y++) {
      const vy = y / (size - 1) - 0.5
      for (let x = 0; x < size; x++) {
        const vx = x / (size - 1) - 0.5
        // Rotate the sample point so axis-aligned tile patterns become
        // diagonal / diamond / circular at intermediate rotations.
        const ux = (vx * cosR - vy * sinR) + 0.5
        const uy = (vx * sinR + vy * cosR) + 0.5
        const sin_m_x = Math.sin(m * Math.PI * ux)
        const sin_n_x = Math.sin(n * Math.PI * ux)
        const sin_m_y = Math.sin(m * Math.PI * uy)
        const sin_n_y = Math.sin(n * Math.PI * uy)
        let v = sin_m_x * sin_n_y + cosPhase * sin_n_x * sin_m_y
        if (w > 0) {
          const sin_p_x = Math.sin(p * Math.PI * ux)
          const sin_q_y = Math.sin(q * Math.PI * uy)
          const sin_q_x = Math.sin(q * Math.PI * ux)
          const sin_p_y = Math.sin(p * Math.PI * uy)
          v = (1 - w) * v + w * (sin_p_x * sin_q_y - sin_q_x * sin_p_y)
        }
        data[y * size + x] = v
      }
    }
    return { kind: 'field', data, width: size, height: size }
  },
}

// ===========================================================================
// Python-engine (biotuner.harmonic_geometry) entries
// ===========================================================================
//
// These call the backend at /api/harmonic-geometry. They use the user's
// derived tuning directly (passed via the `tuning` field on the request),
// so the geometry literally encodes the harmonic structure of their signal.
//
// The shape is the same as the JS entries plus:
//   - `engine: 'python'`
//   - `style`: backend dispatch key
//   - `renderer`: '3d' | 'tree2d' (which client-side viewer to use)
//   - `paramKeys`: which params to forward to the backend
// ---------------------------------------------------------------------------

const harmonic_knot = {
  key: 'harmonic_knot',
  engine: 'python',
  style: 'harmonic_knot',
  renderer: '3d',
  label: 'Harmonic knot',
  description:
    'Torus knot T(p, q). Use "Knot preset" for the classic shapes ' +
    '(trefoil, cinquefoil, …) or set it to "From data" to drive winding ' +
    'numbers off the chosen derived ratio.',
  // The KNOT_PRESETS map (in GeometryTab) lets the user one-click a
  // recognisable T(p, q); "data" defers to the Dominant slot binding.
  slots: [{ key: 'dominant', label: 'Dominant' }],
  defaultParams: {
    knot_preset: 'data',     // 'data' | 'T_2_1' | 'T_3_2' | 'T_5_3' | 'T_5_4' | 'T_7_4'
    n_points: 500,
    tube_radius: 0.08,
    n_sides: 12,
    major_radius: 2.0,
    minor_radius: 0.7,
  },
  paramSchema: [
    { key: 'knot_preset', label: 'Knot preset', type: 'select',
      options: [
        { value: 'data',  label: 'From data (dominant slot)' },
        { value: 'T_2_1', label: 'T(2, 1) — Hopf link' },
        { value: 'T_3_2', label: 'T(3, 2) — Trefoil' },
        { value: 'T_5_3', label: 'T(5, 3)' },
        { value: 'T_5_4', label: 'T(5, 4) — Cinquefoil' },
        { value: 'T_7_4', label: 'T(7, 4)' },
        { value: 'T_8_3', label: 'T(8, 3)' },
      ] },
    { key: 'tube_radius',  label: 'Tube radius',    type: 'slider', min: 0.01, max: 0.3,  step: 0.005 },
    { key: 'major_radius', label: 'Major radius',   type: 'slider', min: 0.5,  max: 4,    step: 0.05 },
    { key: 'minor_radius', label: 'Minor radius',   type: 'slider', min: 0.1,  max: 2,    step: 0.05 },
    { key: 'n_points',     label: 'Detail',         type: 'slider', min: 100,  max: 1500, step: 50,  advanced: true },
    { key: 'n_sides',      label: 'Tube sides',     type: 'int',    min: 4,    max: 24,   step: 1,   advanced: true },
  ],
}

const lsystem_3d = {
  key: 'lsystem_3d',
  engine: 'python',
  style: 'lsystem_3d',
  renderer: '3d',
  label: 'L-system 3D',
  description:
    '3D turtle-graphics L-system whose branching angle = 360° / (p + q) is ' +
    'derived from the chosen "Dominant" ratio. Every binding choice picks a ' +
    'different angle → a different branching geometry.',
  // Same "single dominant ratio" approach as harmonic_knot.
  slots: [{ key: 'dominant', label: 'Dominant' }],
  defaultParams: {
    depth: 3,
    step_length: 1.0,
    axiom: 'F',
  },
  paramSchema: [
    { key: 'depth',       label: 'Depth',       type: 'int',    min: 1,    max: 5,   step: 1 },
    { key: 'step_length', label: 'Step length', type: 'slider', min: 0.1,  max: 3,   step: 0.05 },
  ],
}

const harmonic_point_cloud = {
  key: 'harmonic_point_cloud',
  engine: 'python',
  style: 'harmonic_point_cloud',
  renderer: '3d',
  label: 'Point cloud',
  description:
    'Fibonacci-distributed points on a chosen surface, density-modulated by ' +
    'the harmonic field of your tuning. Points where the field exceeds the ' +
    'median are retained — denser regions trace the harmonic resonance ' +
    'pattern of your signal on the surface.',
  defaultParams: {
    n_points: 3000,
    surface: 'sphere',
    ratio_scale: 1,
  },
  paramSchema: [
    { key: 'n_points', label: 'Points', type: 'slider', min: 400, max: 8000, step: 100 },
    { key: 'surface',  label: 'Surface', type: 'select',
      options: [
        { value: 'sphere',     label: 'Sphere' },
        { value: 'torus',      label: 'Torus' },
        { value: 'klein',      label: 'Klein bottle' },
        { value: 'hyperbolic', label: 'Hyperbolic' },
        { value: 'mos',        label: 'Möbius strip' },
      ] },
    // ratio_scale is a frontend-only param. GeometryTab multiplies every
    // tuning ratio by it before sending, spreading microtonal ratios
    // (which all cluster near 1) into a wider range that biotuner's
    // density field can actually distinguish.
    { key: 'ratio_scale', label: 'Ratio scale', type: 'slider', min: 1, max: 20, step: 0.5,
      format: (v) => `×${Number(v).toFixed(1)}` },
  ],
}

const recursive_polyhedron = {
  key: 'recursive_polyhedron',
  engine: 'python',
  style: 'recursive_polyhedron',
  renderer: '3d',
  label: 'Recursive polyhedron',
  description:
    'Koch-style recursively stellated Platonic solid. Every triangular face ' +
    'splits into four and a tetrahedral bump rises at the centre, scaled by ' +
    'the closest harmonic ratio — chord tones literally sculpt their own ' +
    'region of the surface.',
  defaultParams: {
    depth: 2,
    solid: 'icosahedron',
    per_face_bump: true,
    apex_twist: true,
  },
  paramSchema: [
    { key: 'depth', label: 'Depth', type: 'int', min: 0, max: 4, step: 1 },
    { key: 'solid', label: 'Base solid', type: 'select',
      options: [
        { value: 'tetrahedron', label: 'Tetrahedron' },
        { value: 'cube',        label: 'Cube' },
        { value: 'icosahedron', label: 'Icosahedron' },
      ] },
    { key: 'per_face_bump', label: 'Per-face bump', type: 'bool' },
    { key: 'apex_twist',    label: 'Apex twist',    type: 'bool' },
  ],
}

// JS-engine Chladni — pairwise cosine-product field on a square plate,
// matching biotuner.harmonic_geometry.media.eigenmode.rigid_plate.
// chladni_field_pairwise (no network round-trip; tweakable in real time).
//
// chord_to_int_modes equivalent: each ratio is rounded to its nearest p/q
// with denominator ≤ max_denom, then multiplied through by the LCM of
// the resulting denominators to get small integer "modes".
const chladni = {
  key: 'chladni',
  engine: 'js',
  label: 'Chladni',
  description:
    'Pairwise cosine-product field on a square plate. Each ratio is ' +
    'mapped to an integer mode; for every distinct pair (m, n), the ' +
    'antisymmetric mode cos(mπx)·cos(nπy)−cos(nπx)·cos(mπy) (or symmetric +) ' +
    'is summed. D4 symmetrisation produces the crystalline blooms shown in ' +
    "the biotuner docs.",
  defaultParams: {
    antisymmetric: true,
    symmetry: 'd4_max',
    max_denom: 6,
    n_modes: 3,
    resolution: 320,
    sigma: 0,
    line_sharpness: 1.0,
    animation: 'none',        // 'none' | 'phase' | 'breathe' | 'rotate'
    anim_speed: 0.3,
  },
  paramSchema: [
    { key: 'antisymmetric', label: 'Antisymmetric (− vs +)', type: 'bool' },
    { key: 'symmetry', label: 'Symmetry', type: 'select',
      options: [
        { value: 'none',   label: 'None (single tile)' },
        { value: 'd4_max', label: 'D4 (max blend)' },
        { value: 'd4_sum', label: 'D4 (sum blend)' },
      ] },
    { key: 'max_denom',  label: 'Simplify (max denom)', type: 'int', min: 2, max: 24, step: 1,
      format: (v) => `≤ ${v}` },
    { key: 'n_modes',    label: 'Number of modes',     type: 'int', min: 2, max: 6, step: 1 },
    { key: 'line_sharpness', label: 'Line sharpness',  type: 'slider', min: 0.3, max: 3, step: 0.05,
      format: (v) => `×${Number(v).toFixed(2)}` },
    { key: 'animation', label: 'Animation', type: 'select',
      options: [
        { value: 'none',    label: 'Static' },
        { value: 'phase',   label: 'Phase pulse (mix oscillates)' },
        { value: 'breathe', label: 'Breathe (sigma pulses)' },
        { value: 'rotate',  label: 'Rotate (slow spin)' },
      ] },
    { key: 'anim_speed', label: 'Animation speed',     type: 'slider', min: 0.05, max: 2, step: 0.05,
      format: (v) => `×${Number(v).toFixed(2)}` },
    { key: 'sigma',      label: 'σ override (0 = auto)', type: 'slider', min: 0, max: 0.2, step: 0.001,
      format: (v) => v <= 0 ? 'auto' : v.toFixed(3), advanced: true },
    { key: 'resolution', label: 'Resolution',          type: 'slider', min: 128, max: 512, step: 32, advanced: true },
  ],
  // Derive a small-integer mode set from the chosen analysis ratios.
  // Note: this geometry expects the analysis ratios to be passed inside
  // params.ratios (set by GeometryTab before calling render).
  render(params, t) {
    const {
      antisymmetric = true, symmetry = 'd4_max',
      max_denom = 6, n_modes = 3, resolution = 320,
      sigma: sigmaOverride = 0, line_sharpness = 1.0,
      animation = 'none', anim_speed = 0.3,
      ratios = [1, 5/4, 3/2],
    } = params
    const animT = (animation === 'none' || !t) ? 0 : (t * anim_speed)
    // Phase animation: smoothly cycle the antisymmetric ↔ symmetric mix
    // by interpolating the second term's sign via cos(animT).
    const mixSign = animation === 'phase'
      ? Math.cos(animT * Math.PI)         // -1 (antisym) ↔ +1 (sym)
      : (antisymmetric ? -1 : +1)

    // 1) Round each ratio to nearest p/q with denom ≤ max_denom.
    // 2) Compute LCM of the denominators.
    // 3) Multiply each ratio's p/q by LCM/d to get integer modes.
    const fractions = []
    for (const r of ratios) {
      if (!Number.isFinite(r) || r <= 0) continue
      const { n, d } = findFraction(r, Math.max(2, Math.min(24, max_denom)))
      fractions.push({ n, d })
    }
    if (!fractions.length) fractions.push({ n: 1, d: 1 }, { n: 5, d: 4 }, { n: 3, d: 2 })

    const lcmAll = fractions.reduce((acc, f) => lcm(acc, f.d), 1)
    let modes = fractions.map((f) => Math.round(f.n * lcmAll / f.d))
    // Reduce by gcd of all modes — keeps integers small and the field crisp.
    const g = modes.reduce((acc, m) => gcd(acc, m), modes[0] || 1)
    modes = modes.map((m) => Math.max(1, Math.round(m / g)))
    // De-duplicate and cap to n_modes.
    modes = Array.from(new Set(modes)).slice(0, Math.max(2, Math.min(6, n_modes)))
    while (modes.length < 2) modes.push((modes[modes.length - 1] || 2) + 1)

    const N = Math.max(64, Math.min(512, resolution))
    const data = new Float32Array(N * N)

    // Precompute cos(m π x / (N-1)) for each mode and each axis.
    const cosCache = new Array(modes.length)
    for (let mi = 0; mi < modes.length; mi++) {
      const m = modes[mi]
      const arr = new Float32Array(N)
      for (let i = 0; i < N; i++) arr[i] = Math.cos(m * Math.PI * i / (N - 1))
      cosCache[mi] = arr
    }

    // Pairwise sum. `mixSign` selects the antisymmetric (−) / symmetric (+)
    // form, or smoothly interpolates between them when animation === 'phase'.
    for (let i = 0; i < modes.length; i++) {
      const cxi = cosCache[i]
      const cyi = cosCache[i]
      for (let j = i + 1; j < modes.length; j++) {
        const cxj = cosCache[j]
        const cyj = cosCache[j]
        for (let y = 0; y < N; y++) {
          const cyi_y = cyi[y]
          const cyj_y = cyj[y]
          const off = y * N
          for (let x = 0; x < N; x++) {
            const a = cxi[x] * cyj_y
            const b = cxj[x] * cyi_y
            data[off + x] += a + mixSign * b
          }
        }
      }
    }

    // Rotation animation: re-sample the computed field at rotated (x, y)
    // coords. Cheap (nearest-neighbour, post-pairwise-sum) and gives the
    // characteristic "slow spin" without changing the modal structure.
    if (animation === 'rotate' && animT) {
      const theta = animT * Math.PI * 0.5  // slow spin
      const cosT = Math.cos(theta)
      const sinT = Math.sin(theta)
      const cx = (N - 1) / 2
      const cy = cx
      const orig = new Float32Array(data)
      for (let y = 0; y < N; y++) {
        const dy = y - cy
        for (let x = 0; x < N; x++) {
          const dx = x - cx
          const sx = Math.round(cx + dx * cosT - dy * sinT)
          const sy = Math.round(cy + dx * sinT + dy * cosT)
          data[y * N + x] = (sx >= 0 && sx < N && sy >= 0 && sy < N)
            ? orig[sy * N + sx]
            : 0
        }
      }
    }

    // D4 symmetrisation (4 rotations × 2 reflections = 8 transforms).
    if (symmetry === 'd4_max' || symmetry === 'd4_sum') {
      const orig = new Float32Array(data)
      const useMax = symmetry === 'd4_max'
      for (let y = 0; y < N; y++) {
        const yi = N - 1 - y
        for (let x = 0; x < N; x++) {
          const xi = N - 1 - x
          const v0 = orig[y  * N + x]                       // identity
          const v1 = orig[x  * N + yi]                      // rotate 90
          const v2 = orig[yi * N + xi]                      // rotate 180
          const v3 = orig[xi * N + y]                       // rotate 270
          const v4 = orig[y  * N + xi]                      // flip horizontal
          const v5 = orig[xi * N + yi]                      // flip diagonal
          const v6 = orig[yi * N + x]                       // flip vertical
          const v7 = orig[x  * N + y]                       // flip anti-diagonal
          data[y * N + x] = useMax
            ? Math.max(v0, v1, v2, v3, v4, v5, v6, v7)
            : (v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7) / 8
        }
      }
    }

    // Auto-sigma per biotuner.harmonic_geometry: σ = base / peak_mode,
    // scaled up by √(n_pairs/3) so high-pair-count chords don't collapse
    // to dotty zero-set patterns. Without this, high-mode fields render
    // as fuzz instead of intertwined nodal lines.
    const peakMode = Math.max(...modes.map((m) => Math.abs(m)), 1)
    const nPairs = (modes.length * (modes.length - 1)) / 2
    const pairFactor = nPairs <= 3 ? 1 : Math.sqrt(nPairs / 3)
    const autoSigma = Math.max(0.005, Math.min(0.18, 0.5 * pairFactor / peakMode))
    // Breathe: pulse the line width by ±50 % via a slow cosine envelope.
    const breatheMul = animation === 'breathe' && animT
      ? 1 + 0.5 * Math.cos(animT * Math.PI)
      : 1
    const sigma = (sigmaOverride > 0)
      ? sigmaOverride * breatheMul
      : autoSigma * (line_sharpness || 1) * breatheMul

    return { kind: 'field', data, width: N, height: N, modes, sigma }
  },
}

// Small GCD/LCM helpers used by Chladni mode derivation.
function lcm(a, b) {
  if (!a || !b) return Math.max(a, b) || 1
  return Math.abs(a * b) / gcd(a, b)
}

const subharmonic_tree = {
  key: 'subharmonic_tree',
  engine: 'python',
  style: 'subharmonic_tree',
  renderer: 'tree2d',
  label: 'Self-similar tuning',
  description:
    'Recursive subharmonic expansion: each peak f is the root of a sub-tree ' +
    'whose children are its first n subharmonics (f/2, f/3, …). Each child ' +
    'expands the same way, revealing the self-similar harmonic structure ' +
    'of the tuning.',
  defaultParams: {
    depth: 4,
    n_harmonics: 5,
    min_freq: 0.1,
    layout: 'polar',
  },
  paramSchema: [
    { key: 'depth',       label: 'Depth',          type: 'int',    min: 1,     max: 6,    step: 1 },
    { key: 'n_harmonics', label: 'Subharmonics',   type: 'int',    min: 2,     max: 9,    step: 1 },
    { key: 'min_freq',    label: 'Min freq (Hz)',  type: 'slider', min: 0.01,  max: 50,   step: 0.01 },
    { key: 'layout',      label: 'Layout',         type: 'select',
      options: [
        { value: 'polar', label: 'Radial (chord wheel)' },
        { value: 'depth', label: 'Horizontal levels' },
      ] },
  ],
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

export const GEOMETRY_TYPES = {
  // Real-time JS implementations
  lissajous,
  harmonograph,
  rose,
  spirograph,
  // biotuner-driven Python implementations
  chladni,
  harmonic_knot,
  lsystem_3d,
  harmonic_point_cloud,
  subharmonic_tree,
}

// Display order. L-system removed per user request — insufficient
// per-tuning variability to justify keeping it in the lineup.
export const GEOMETRY_ORDER = [
  'lissajous', 'harmonograph', 'rose', 'spirograph',
  'chladni', 'harmonic_knot',
  'harmonic_point_cloud', 'subharmonic_tree',
]
