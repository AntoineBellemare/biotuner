/**
 * Live in-browser cymatics field computation.
 *
 * Mirrors `biotuner.harmonic_geometry.media.eigenmode.rigid_plate`'s
 * cymatics path so the reels can morph a chord into a Chladni nodal
 * pattern per-frame in canvas — no heavy pre-exported field data.
 *
 *   field(x,y) = Σ over ratio pairs (m,n):  cos(mπx)cos(nπy) − cos(nπx)cos(mπy)
 *   density    = exp(−field² / σ²)              (sand collects on the nodes)
 *   then D4-symmetrise the DENSITY (max over the 8-element dihedral orbit)
 *
 * σ is auto-derived from the chord's peak wavenumber and pair count, the
 * same formula as `_auto_sigma_for_modes`, so patterns stay legible across
 * very different chords.
 */

export type Chord = { name: string; label: string; ratios: number[] };

/** All distinct unordered index pairs of a ratio list. */
function pairs(n: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) out.push([i, j]);
  return out;
}

/** σ ∝ √(n_pairs / 3) / peak_wavenumber, clamped — matches Python auto-σ. */
export function autoSigma(ratios: number[]): number {
  const peak = Math.max(...ratios.map(Math.abs)) || 1;
  const nPairs = (ratios.length * (ratios.length - 1)) / 2;
  const pairFactor = Math.max(1, Math.sqrt(nPairs / 3));
  return Math.min(0.18, Math.max(0.005, (0.5 * pairFactor) / peak));
}

/**
 * Compute the D4-symmetrised nodal-density field for a chord.
 * Returns a Float32Array of length N*N in [0, 1] (row-major).
 */
export function cymaticsDensity(
  ratios: number[],
  N: number,
  opts: { symmetry?: "d4_max" | "d4_sum" | "none"; sigma?: number } = {}
): Float32Array {
  const symmetry = opts.symmetry ?? "d4_max";
  const sigma = opts.sigma ?? autoSigma(ratios);
  const inv2s2 = 1 / (sigma * sigma);
  const ps = pairs(ratios.length);

  // Precompute cos(k·π·t) for every grid line and every distinct wavenumber.
  // Grid coordinate t = i/(N-1) in [0,1]; argument is k·π·t.
  const wn = ratios;
  const cosTab: Float32Array[] = wn.map((k) => {
    const row = new Float32Array(N);
    for (let i = 0; i < N; i++) row[i] = Math.cos(k * Math.PI * (i / (N - 1)));
    return row;
  });

  // raw signed field. Each pair is weighted 1/n_pairs so the amplitudes
  // sum to 1, matching biotuner's _resolve_amps_phases default (without
  // this the field is scaled by n_pairs and the auto-σ density saturates).
  const wpair = 1 / ps.length;
  const field = new Float32Array(N * N);
  for (const [a, b] of ps) {
    const ca = cosTab[a];
    const cb = cosTab[b];
    for (let r = 0; r < N; r++) {
      const car = ca[r] * wpair;
      const cbr = cb[r] * wpair;
      const base = r * N;
      for (let c = 0; c < N; c++) {
        // (cos(mπx)cos(nπy) − cos(nπx)cos(mπy)) / n_pairs, x=col, y=row
        field[base + c] += car * cb[c] - cbr * ca[c];
      }
    }
  }

  // density = exp(−field²/σ²)
  const dens = new Float32Array(N * N);
  for (let k = 0; k < dens.length; k++) {
    const w = field[k];
    dens[k] = Math.exp(-w * w * inv2s2);
  }
  if (symmetry === "none") return dens;

  // D4 symmetrise the density. For output (i,j) gather the 8 orbit sources.
  const out = new Float32Array(N * N);
  const M = N - 1;
  const useMax = symmetry === "d4_max";
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const s = [
        dens[i * N + j],
        dens[j * N + (M - i)],
        dens[(M - i) * N + (M - j)],
        dens[(M - j) * N + i],
        dens[j * N + i],
        dens[(M - i) * N + j],
        dens[(M - j) * N + (M - i)],
        dens[i * N + (M - j)],
      ];
      if (useMax) {
        let m = s[0];
        for (let t = 1; t < 8; t++) if (s[t] > m) m = s[t];
        out[i * N + j] = m;
      } else {
        let acc = 0;
        for (let t = 0; t < 8; t++) acc += s[t];
        out[i * N + j] = acc / 8;
      }
    }
  }
  return out;
}

/** Cosine-eased component-wise interpolation between two chords' ratios. */
export function lerpRatios(a: number[], b: number[], f: number): number[] {
  const e = 0.5 * (1 - Math.cos(Math.PI * f)); // cosine ease
  const n = Math.max(a.length, b.length);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const ai = a[Math.min(i, a.length - 1)];
    const bi = b[Math.min(i, b.length - 1)];
    out.push((1 - e) * ai + e * bi);
  }
  return out;
}

/**
 * Warm "afmhot"-style ramp: black → deep red → orange → warm white.
 * v in [0,1], optional gamma to brighten midtones.
 */
export function afmhot(v: number, gamma = 0.8): [number, number, number] {
  const x = Math.pow(Math.max(0, Math.min(1, v)), gamma);
  // afmhot: r ramps fastest, then g, then b.
  const r = Math.min(1, x * 2.0);
  const g = Math.min(1, Math.max(0, x * 2.0 - 0.5));
  const b = Math.min(1, Math.max(0, x * 2.0 - 1.0));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/** Clean grayscale ramp (classic white-sand-on-black). */
export function grayscale(v: number, gamma = 0.7): [number, number, number] {
  const x = Math.round(Math.pow(Math.max(0, Math.min(1, v)), gamma) * 255);
  return [x, x, x];
}

/** Paint a density field into an ImageData-backed canvas, upscaled. */
export function paintDensity(
  ctx: CanvasRenderingContext2D,
  dens: Float32Array,
  N: number,
  ramp: (v: number) => [number, number, number]
): void {
  const img = ctx.createImageData(N, N);
  for (let k = 0; k < dens.length; k++) {
    const [r, g, b] = ramp(dens[k]);
    img.data[k * 4 + 0] = r;
    img.data[k * 4 + 1] = g;
    img.data[k * 4 + 2] = b;
    img.data[k * 4 + 3] = 255;
  }
  const off = document.createElement("canvas");
  off.width = N;
  off.height = N;
  off.getContext("2d")!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  ctx.drawImage(off, 0, 0, ctx.canvas.width, ctx.canvas.height);
}
