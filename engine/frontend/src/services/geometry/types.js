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
    { key: 'a',         label: 'Frequency a', type: 'int',    min: 1,   max: 24, step: 1 },
    { key: 'b',         label: 'Frequency b', type: 'int',    min: 1,   max: 24, step: 1 },
    { key: 'delta',     label: 'Phase δ',     type: 'slider', min: 0,   max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'cycles',    label: 'Cycles',      type: 'slider', min: 0.5, max: 8,  step: 0.1 },
    { key: 'lineWidth', label: 'Line width',  type: 'slider', min: 0.4, max: 4,  step: 0.1 },
  ],
  fromRatios(ratios) {
    if (!ratios?.length) return {}
    const r = (ratios[1] || (ratios[0] * 3 / 2)) / (ratios[0] || 1)
    const { n, d } = findFraction(r, 12)
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
    duration: 90,
    lineWidth: 0.7,
  },
  paramSchema: [
    { key: 'f1', label: 'X freq 1', type: 'slider', min: 0.5, max: 20, step: 0.05 },
    { key: 'f2', label: 'X freq 2', type: 'slider', min: 0.5, max: 20, step: 0.05 },
    { key: 'f3', label: 'Y freq 1', type: 'slider', min: 0.5, max: 20, step: 0.05 },
    { key: 'f4', label: 'Y freq 2', type: 'slider', min: 0.5, max: 20, step: 0.05 },
    { key: 'p1', label: 'X phase 1', type: 'slider', min: 0, max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'p2', label: 'X phase 2', type: 'slider', min: 0, max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'p3', label: 'Y phase 1', type: 'slider', min: 0, max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'p4', label: 'Y phase 2', type: 'slider', min: 0, max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'd1', label: 'Decay X1', type: 'slider', min: 0, max: 0.01, step: 0.00005,
      format: (v) => v.toFixed(4) },
    { key: 'd2', label: 'Decay X2', type: 'slider', min: 0, max: 0.01, step: 0.00005,
      format: (v) => v.toFixed(4) },
    { key: 'd3', label: 'Decay Y1', type: 'slider', min: 0, max: 0.01, step: 0.00005,
      format: (v) => v.toFixed(4) },
    { key: 'd4', label: 'Decay Y2', type: 'slider', min: 0, max: 0.01, step: 0.00005,
      format: (v) => v.toFixed(4) },
    { key: 'duration', label: 'Duration', type: 'slider', min: 20, max: 300, step: 5,
      format: (v) => `${v}s` },
    { key: 'lineWidth', label: 'Line width', type: 'slider', min: 0.3, max: 3, step: 0.05 },
  ],
  fromRatios(ratios) {
    if (!ratios?.length) return {}
    const base = ratios[0] || 1
    return {
      f1: +(((ratios[0] || 1) / base) * 3).toFixed(2),
      f2: +(((ratios[1] || 1.5) / base) * 3).toFixed(2),
      f3: +(((ratios[2] || ratios[0] || 1.25) / base) * 3).toFixed(2),
      f4: +(((ratios[3] || ratios[1] || 1.66) / base) * 3).toFixed(2),
    }
  },
  render(params, t) {
    const { f1, f2, f3, f4, p1, p2, p3, p4, d1, d2, d3, d4, duration } = params
    const N = 6000
    const dt = duration / N
    const phaseDrift = (t || 0) * 0.1
    const points = new Array(N + 1)
    for (let i = 0; i <= N; i++) {
      const tt = i * dt
      const x =
        Math.exp(-d1 * tt) * Math.sin(f1 * tt + p1 + phaseDrift) +
        Math.exp(-d2 * tt) * Math.sin(f2 * tt + p2)
      const y =
        Math.exp(-d3 * tt) * Math.sin(f3 * tt + p3) +
        Math.exp(-d4 * tt) * Math.sin(f4 * tt + p4 - phaseDrift)
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
  defaultParams: { n: 5, d: 2, cycles: null, lineWidth: 1.6 },
  paramSchema: [
    { key: 'n',         label: 'Numerator',   type: 'int',    min: 1, max: 24, step: 1 },
    { key: 'd',         label: 'Denominator', type: 'int',    min: 1, max: 24, step: 1 },
    { key: 'lineWidth', label: 'Line width',  type: 'slider', min: 0.4, max: 4, step: 0.1 },
  ],
  fromRatios(ratios) {
    if (!ratios?.length) return {}
    const r = (ratios[1] || (ratios[0] * 3 / 2)) / (ratios[0] || 1)
    const { n, d } = findFraction(r, 16)
    return { n, d }
  },
  render(params, t) {
    const { n, d } = params
    const k = n / d
    // Closes after `d` full revolutions when n/d is in lowest terms.
    const periods = d / gcd(n, d)
    const N = 3000
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
    { key: 'R',         label: 'Outer R',     type: 'slider', min: 2,   max: 30, step: 0.1 },
    { key: 'r',         label: 'Inner r',     type: 'slider', min: 1,   max: 20, step: 0.1 },
    { key: 'offset',    label: 'Pen offset',  type: 'slider', min: 0.5, max: 20, step: 0.1 },
    { key: 'lineWidth', label: 'Line width',  type: 'slider', min: 0.3, max: 3,  step: 0.05 },
  ],
  fromRatios(ratios) {
    if (!ratios?.length) return {}
    const r = (ratios[1] || (ratios[0] * 3 / 2)) / (ratios[0] || 1)
    const f = findFraction(r, 12)
    return { R: f.n * 3, r: f.d * 3, offset: Math.max(1, f.d * 2) }
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
    { key: 'm',          label: 'Mode m',      type: 'int',    min: 1,  max: 14, step: 1 },
    { key: 'n',          label: 'Mode n',      type: 'int',    min: 1,  max: 14, step: 1 },
    { key: 'mix',        label: 'Mix angle',   type: 'slider', min: 0,  max: Math.PI * 2, step: 0.01,
      format: (v) => `${(v / Math.PI).toFixed(2)}π` },
    { key: 'contrast',   label: 'Contrast',    type: 'slider', min: 0.4, max: 6, step: 0.05 },
    { key: 'resolution', label: 'Resolution',  type: 'slider', min: 64,  max: 512, step: 32 },
  ],
  fromRatios(ratios) {
    if (!ratios?.length) return {}
    const r = (ratios[1] || (ratios[0] * 3 / 2)) / (ratios[0] || 1)
    const f = findFraction(r, 10)
    return { m: f.n, n: f.d }
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
    n_points: 800,
    tube_radius: 0.08,
    n_sides: 16,
    major_radius: 2.0,
    minor_radius: 0.7,
  },
  paramSchema: [
    { key: 'n_points',     label: 'Detail',         type: 'slider', min: 100,  max: 2000, step: 50 },
    { key: 'tube_radius',  label: 'Tube radius',    type: 'slider', min: 0.01, max: 0.3,  step: 0.005 },
    { key: 'n_sides',      label: 'Tube sides',     type: 'int',    min: 4,    max: 32,   step: 1 },
    { key: 'major_radius', label: 'Major radius',   type: 'slider', min: 0.5,  max: 4,    step: 0.05 },
    { key: 'minor_radius', label: 'Minor radius',   type: 'slider', min: 0.1,  max: 2,    step: 0.05 },
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
    layout: 'depth',
  },
  paramSchema: [
    { key: 'depth',       label: 'Depth',          type: 'int',    min: 1,     max: 6,    step: 1 },
    { key: 'n_harmonics', label: 'Subharmonics',   type: 'int',    min: 2,     max: 9,    step: 1 },
    { key: 'min_freq',    label: 'Min freq (Hz)',  type: 'slider', min: 0.01,  max: 50,   step: 0.01 },
    { key: 'layout',      label: 'Layout',         type: 'select',
      options: [
        { value: 'depth', label: 'Depth (radial)' },
        { value: 'log',   label: 'Log-frequency' },
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
