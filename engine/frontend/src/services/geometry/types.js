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
    const ra = derived[ia] || 1
    const rb = derived[ib] || 1.5
    // Express the *ratio between* the two bindings as a small integer
    // fraction. This guarantees a recognisable Lissajous shape (3:2, 5:4,
    // 7:5, …) regardless of how close the underlying analysis ratios are.
    let k = rb / ra
    if (k < 1) k = 1 / k                // keep a ≤ b for nicer-looking ratios
    let { n, d } = findFraction(k, 12)
    if (n === d) {                       // ratio collapsed to 1/1 — fall back
      // Use the binding indices for variety
      n = Math.max(2, (ia + 2) % 12 + 1)
      d = Math.max(1, (ib + 1) % 12 + 1)
      if (n === d) n = (n % 12) + 1
    }
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
  },
  paramSchema: [
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
      return +((derived[i] ?? fallback) * 2).toFixed(2)
    }
    return {
      f1: pad('f1', 0, 1.0),
      f2: pad('f2', 1, 1.5),
      f3: pad('f3', 2, 1.25),
      f4: pad('f4', 3, 1.66),
    }
  },
  render(params, t) {
    const { f1, f2, f3, f4, p1, p2, p3, p4, d1, d2, d3, d4, duration, damping } = params
    // Use the global damping when set (> tiny epsilon); otherwise the
    // per-pendulum decays in the Advanced panel.
    const useGlobal = damping != null && damping > 1e-5
    const dx1 = useGlobal ? damping : d1
    const dx2 = useGlobal ? damping : d2
    const dy1 = useGlobal ? damping : d3
    const dy2 = useGlobal ? damping : d4
    const N = 6000
    const dt = duration / N
    const phaseDrift = (t || 0) * 0.1
    const points = new Array(N + 1)
    for (let i = 0; i <= N; i++) {
      const tt = i * dt
      const x =
        Math.exp(-dx1 * tt) * Math.sin(f1 * tt + p1 + phaseDrift) +
        Math.exp(-dx2 * tt) * Math.sin(f2 * tt + p2)
      const y =
        Math.exp(-dy1 * tt) * Math.sin(f3 * tt + p3) +
        Math.exp(-dy2 * tt) * Math.sin(f4 * tt + p4 - phaseDrift)
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
  defaultParams: { n: 1.125, d: 1, cycles: 8, lineWidth: 1.6 },
  paramSchema: [
    { key: 'n',         label: 'Numerator',   type: 'slider', min: 0.1, max: 24, step: 0.01, derived: true,
      format: (v) => Number(v).toFixed(3) },
    { key: 'd',         label: 'Denominator', type: 'slider', min: 0.1, max: 24, step: 0.01, derived: true,
      format: (v) => Number(v).toFixed(3) },
    { key: 'cycles',    label: 'Cycles',      type: 'slider', min: 1,   max: 32, step: 1 },
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
  fromDerivedRatios(derived, bindings = {}) {
    if (!derived?.length) return {}
    const ia = Math.min(derived.length - 1, bindings.n ?? 0)
    const ib = Math.min(derived.length - 1, bindings.d ?? 1)
    const ra = derived[ia] || 1
    const rb = derived[ib] || 1.5
    // Rose curve r = cos(k θ); k = n/d sets the petal structure. Pick the
    // ratio between the two slots, expressed as a small integer fraction
    // for crisp closed roses.
    let k = rb / ra
    if (k < 1) k = 1 / k
    let { n, d } = findFraction(k, 12)
    if (n === d) {
      n = (ia + 3) % 11 + 1
      d = (ib + 2) % 11 + 1
      if (n === d) n = (n % 11) + 1
    }
    return { n, d }
  },
  render(params, t) {
    const { n, d, cycles } = params
    if (!Number.isFinite(n) || !Number.isFinite(d) || d <= 0) {
      return { kind: 'path', points: [] }
    }
    const k = n / d
    // For integer n/d in lowest terms, the curve closes after d revolutions.
    // For decimal k it never closes — we sweep a user-controlled number of
    // cycles (default 8) which fills the rose densely enough to look complete.
    const isInteger =
      Math.abs(n - Math.round(n)) < 1e-3 && Math.abs(d - Math.round(d)) < 1e-3
    const periods = isInteger
      ? Math.round(d) / gcd(Math.round(n), Math.round(d))
      : (cycles || 8)
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
  defaultParams: { R: 7, r: 3, offset: 4, lineWidth: 1.2 },
  paramSchema: [
    { key: 'R',         label: 'Outer R',     type: 'slider', min: 0.5, max: 30, step: 0.05, derived: true,
      format: (v) => Number(v).toFixed(2) },
    { key: 'r',         label: 'Inner r',     type: 'slider', min: 0.2, max: 20, step: 0.05, derived: true,
      format: (v) => Number(v).toFixed(2) },
    { key: 'offset',    label: 'Pen offset',  type: 'slider', min: 0.5, max: 20, step: 0.1 },
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
    const ra = derived[ir] || 1.0
    const rb = derived[iR] || (ra * 1.5)
    // R:r ratio drives the rose count. Use integer fraction so spirograph
    // closes into recognisable petal shapes.
    let k = Math.max(ra, rb) / Math.min(ra, rb)
    let { n, d } = findFraction(k, 12)
    if (n === d) {
      n = ((iR + 3) % 11) + 2
      d = ((ir + 2) % 11) + 1
      if (n === d) d = (d % 11) + 1
    }
    return {
      R: n * 2,                              // outer radius
      r: d * 2,                              // inner radius
    }
  },
  render(params, t) {
    const { R, r, offset } = params
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

const chladni = {
  key: 'chladni',
  label: 'Chladni',
  description:
    'Nodal pattern from two-mode (m, n) interference on a rectangular ' +
    'plate. Bright lines mark where vibration is zero — where sand would ' +
    'collect on a real Chladni plate.',
  defaultParams: { m: 5, n: 3, resolution: 256, contrast: 1.6, mix: Math.PI / 2 },
  paramSchema: [
    { key: 'm',          label: 'Mode m',      type: 'int',    min: 1,  max: 14, step: 1, derived: true },
    { key: 'n',          label: 'Mode n',      type: 'int',    min: 1,  max: 14, step: 1, derived: true },
    { key: 'mix',        label: 'Mix angle',   type: 'slider', min: 0,  max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'contrast',   label: 'Contrast',    type: 'slider', min: 0.4, max: 6, step: 0.05 },
    { key: 'resolution', label: 'Resolution',  type: 'slider', min: 64,  max: 512, step: 32 },
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
    const ra = derived[im] || 1
    const rb = derived[inn] || 1.5
    // Get small integers from the ratio between bindings.
    let k = rb / ra
    if (k < 1) k = 1 / k
    let { n, d } = findFraction(k, 12)
    if (n === d) {
      // Fall back to index-based modes so different bindings always
      // produce visibly different Chladni patterns.
      n = ((im + 2) % 11) + 2
      d = ((inn + 3) % 11) + 2
      if (n === d) d = (d % 11) + 2
    }
    return { m: n, n: d }
  },
  render(params, t) {
    const { m, n, mix, resolution } = params
    const size = Math.max(32, Math.min(512, resolution))
    const data = new Float32Array(size * size)
    const phase = (mix || 0) + (t || 0) * 0.4
    const cosPhase = Math.cos(phase)
    for (let y = 0; y < size; y++) {
      const ny = y / (size - 1)
      const sin_m_y = Math.sin(m * Math.PI * ny)
      const sin_n_y = Math.sin(n * Math.PI * ny)
      for (let x = 0; x < size; x++) {
        const nx = x / (size - 1)
        const sin_m_x = Math.sin(m * Math.PI * nx)
        const sin_n_x = Math.sin(n * Math.PI * nx)
        data[y * size + x] =
          sin_m_x * sin_n_y + cosPhase * sin_n_x * sin_m_y
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
    'Torus knot T(p, q) where the dominant ratio sets the winding numbers. ' +
    '3/2 → trefoil knot; 5/4 → cinquefoil. Rendered as a tube — rotate by ' +
    'drag, zoom by scroll.',
  defaultParams: {
    p: 3,
    q: 2,
    n_points: 500,
    tube_radius: 0.08,
    n_sides: 12,
    major_radius: 2.0,
    minor_radius: 0.7,
  },
  paramSchema: [
    // p, q override the dominant ratio so you can dial any T(p, q) knot.
    // p=q=0 disables the override; biotuner picks from the tuning.
    { key: 'p',            label: 'p (winding)',    type: 'int',    min: 2,    max: 12,   step: 1 },
    { key: 'q',            label: 'q (winding)',    type: 'int',    min: 1,    max: 12,   step: 1 },
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
    '3D turtle-graphics L-system whose branching angle is derived from the ' +
    'dominant ratio (360° / (p + q)). Depth controls how many rewrite ' +
    'passes are applied before drawing.',
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
  chladni,
  // biotuner-driven Python implementations
  harmonic_knot,
  lsystem_3d,
  harmonic_point_cloud,
  recursive_polyhedron,
  subharmonic_tree,
}

// Display order: JS first (instant), then Python (compute-once).
export const GEOMETRY_ORDER = [
  'lissajous', 'harmonograph', 'rose', 'spirograph', 'chladni',
  'harmonic_knot', 'lsystem_3d', 'harmonic_point_cloud',
  'recursive_polyhedron', 'subharmonic_tree',
]
