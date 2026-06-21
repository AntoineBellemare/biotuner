/**
 * Live in-browser ports of two biotuner wave-field generators, used by the
 * "Forbidden Symmetry" reel:
 *
 *   quasicrystalField — biotuner.harmonic_geometry.media.wave_field.interference
 *                       .quasicrystal_field_2d
 *       Ψ(x,y) = Σ_i Σ_{k<N} a_i · exp(i·(k_i·(x·cosα_k + y·sinα_k) + φ_i + k·ε))
 *       α_k = 2πk/n_fold,  k_i = 2π·r_i/L.  Non-crystallographic n_fold
 *       (5,7,12) → a quasicrystal: exact n-fold symmetry, no periodic tiling.
 *
 *   vortexField — .vortex_field_2d (radial_kind='propagating', the elementary
 *                 variant — no Bessel needed in the browser)
 *       Ψ(r,θ) = Σ_i a_i · radial_i(r) · exp(i·(l_i·θ + φ_i))
 *       l_i = numerator of the rationalised ratio (5/4 → 5, 3/2 → 3); the
 *       modulus shows spiral arms braided around dark phase singularities.
 *
 * Both return a Float32Array (row-major, length N·N) normalised to [0,1].
 * Per-component phases φ_i are animated by the scene → smooth flow; ratios
 * are morphed between chords → smooth chord-to-chord tweening.
 */

/** Best rational approximation p/q of x with q ≤ maxDen (continued fraction). */
export function limitDenominator(x: number, maxDen = 20): [number, number] {
  if (Math.abs(x - Math.round(x)) < 1e-9) return [Math.round(x), 1];
  let p0 = 0, q0 = 1, p1 = 1, q1 = 0;
  let frac = x;
  for (let i = 0; i < 64; i++) {
    const a = Math.floor(frac);
    const q2 = q0 + a * q1;
    if (q2 > maxDen) break;
    [p0, q0, p1, q1] = [p1, q1, p0 + a * p1, q2];
    const rem = frac - a;
    if (rem < 1e-9) break;
    frac = 1 / rem;
  }
  return [p1, q1];
}

function normAmps(ratios: number[], amps?: number[]): number[] {
  const a = amps ?? ratios.map(() => 1);
  const s = a.reduce((u, v) => u + v, 0) || 1;
  return a.map((v) => v / s);
}

/**
 * Quasicrystal field with exact discrete n-fold symmetry.
 * Returns Float32Array(N*N) in [0,1].
 */
export function quasicrystalField(
  ratios: number[],
  N: number,
  opts: {
    amps?: number[];
    phases?: number[];
    nFold?: number;
    basePeriod?: number;
    extent?: number;
    directionPhaseStep?: number;
    rotation?: number; // rigidly rotate the lattice (radians)
    power?: number;
  } = {}
): Float32Array {
  const nFold = opts.nFold ?? 5;
  const L = opts.basePeriod ?? 1.0;
  const extent = opts.extent ?? 1.5;
  const eps = opts.directionPhaseStep ?? 0;
  const rotation = opts.rotation ?? 0;
  const power = opts.power ?? 0.6;
  const amps = normAmps(ratios, opts.amps);
  const phases = opts.phases ?? ratios.map(() => 0);

  const re = new Float32Array(N * N);
  const im = new Float32Array(N * N);
  const d = (2 * extent) / (N - 1);
  const x0 = -extent;

  for (let i = 0; i < ratios.length; i++) {
    const ki = (2 * Math.PI * ratios[i]) / L;
    const ai = amps[i];
    for (let kk = 0; kk < nFold; kk++) {
      const alpha = (2 * Math.PI * kk) / nFold + rotation;
      const psi = phases[i] + kk * eps;
      const kx = ki * Math.cos(alpha);
      const ky = ki * Math.sin(alpha);
      const step = kx * d; // angle increment per column
      const cStep = Math.cos(step);
      const sStep = Math.sin(step);
      for (let r = 0; r < N; r++) {
        const Y = x0 + r * d;
        let ang = kx * x0 + ky * Y + psi; // angle at column 0
        let c = Math.cos(ang);
        let s = Math.sin(ang);
        const base = r * N;
        for (let col = 0; col < N; col++) {
          re[base + col] += ai * c;
          im[base + col] += ai * s;
          const cn = c * cStep - s * sStep; // advance angle by `step`
          s = s * cStep + c * sStep;
          c = cn;
        }
      }
    }
  }

  const norm = nFold; // max |field| ≈ nFold·Σa_i = nFold (Σa=1)
  const out = new Float32Array(N * N);
  const invNorm = 1 / norm;
  for (let k = 0; k < out.length; k++) {
    const mag = Math.hypot(re[k], im[k]) * invNorm;
    out[k] = Math.pow(mag > 1 ? 1 : mag, power);
  }
  return out;
}

/**
 * Vortex field: spiral arms + phase singularities (propagating radial).
 * Returns Float32Array(N*N) in [0,1].
 */
export function vortexField(
  ratios: number[],
  N: number,
  opts: {
    amps?: number[];
    phases?: number[];
    beamWaist?: number;
    extent?: number;
    chargeScale?: number;
    rotation?: number; // rigidly spin the whole spiral (radians)
    power?: number;
  } = {}
): Float32Array {
  const w = opts.beamWaist ?? 1.0;
  const extent = opts.extent ?? 2.0;
  const chargeScale = opts.chargeScale ?? 1.0;
  const rotation = opts.rotation ?? 0;
  const power = opts.power ?? 0.6;
  const amps = normAmps(ratios, opts.amps);
  const phases = opts.phases ?? ratios.map(() => 0);

  const d = (2 * extent) / (N - 1);
  const x0 = -extent;

  // Precompute r, θ, and a SHARED radial envelope per pixel. Sharing the
  // envelope across components (instead of the l-dependent rho^|l|) lets the
  // modes overlap and interfere → real angular structure; env = ρ·exp(-ρ²/2)
  // keeps a dark vortex core at the centre. A complex propagating phase
  // exp(i·k·r) (vs a real cos) makes the constant-phase lines true spirals.
  const Rr = new Float32Array(N * N);
  const Th = new Float32Array(N * N);
  const env = new Float32Array(N * N);
  for (let r = 0; r < N; r++) {
    const Y = x0 + r * d;
    const base = r * N;
    for (let col = 0; col < N; col++) {
      const X = x0 + col * d;
      const rr = Math.hypot(X, Y);
      Rr[base + col] = rr;
      Th[base + col] = Math.atan2(Y, X);
      const rho = rr / w;
      env[base + col] = rho * Math.exp(-(rho * rho) / 2);
    }
  }

  const re = new Float32Array(N * N);
  const im = new Float32Array(N * N);

  for (let i = 0; i < ratios.length; i++) {
    const ri = ratios[i];
    const [num] = limitDenominator(ri, 20);
    const l = Math.round(chargeScale * num);
    const ki = (2 * Math.PI * ri) / w;
    const ai = amps[i];
    const phi = phases[i];
    const lrot = l * rotation; // rigid rotation: θ → θ − rotation
    for (let k = 0; k < re.length; k++) {
      const e = env[k] * ai;
      const ang = l * Th[k] + ki * Rr[k] + phi - lrot; // spiral phase
      re[k] += e * Math.cos(ang);
      im[k] += e * Math.sin(ang);
    }
  }

  // max env ≈ e^-0.5 ≈ 0.607 (at ρ=1), so max |field| ≈ 0.607·Σa.
  const norm = 0.607 * (amps.reduce((u, v) => u + v, 0) || 1);
  const out = new Float32Array(N * N);
  const invNorm = 1 / norm;
  for (let k = 0; k < out.length; k++) {
    const mag = Math.hypot(re[k], im[k]) * invNorm;
    out[k] = Math.pow(mag > 1 ? 1 : mag, power);
  }
  return out;
}
