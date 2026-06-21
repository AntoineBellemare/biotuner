/**
 * Live in-canvas geometry renderers parameterised by a chord's ratios.
 *
 * These are line-art counterparts to the cymatics field — the same harmonic
 * input shown through different classical "harmonic geometry" generators:
 *
 *   - lissajous     : pairwise Lissajous figures (one per ratio pair)
 *   - harmonograph  : damped multi-pendulum pen path
 *   - interference  : travelling-wave interference field (animated cymatics)
 *
 * Each draws into a square canvas context; the caller sizes/clears it.
 */
import { cymaticsDensity, tidepool } from "./cymatics";

export type GeomKind =
  | "cymatics"
  | "lissajous"
  | "harmonograph"
  | "interference";

export const GEOM_LABEL: Record<GeomKind, string> = {
  cymatics: "Cymatics",
  lissajous: "Lissajous",
  harmonograph: "Harmonograph",
  interference: "Interference",
};

const TIDE = ["#1d5e6b", "#4aa89c", "#c8a04a", "#d9744a", "#f0d8b0"];

function pairs(n: number): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) out.push([i, j]);
  return out;
}

/** Pairwise Lissajous overlay — one glowing figure per ratio pair. */
export function drawLissajous(
  ctx: CanvasRenderingContext2D,
  ratios: number[],
  frame: number,
  fps: number
): void {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(W, H) * 0.42;
  const t = frame / fps;
  const N = 1400;
  const ps = pairs(ratios.length);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ps.forEach(([i, j], k) => {
    const a = ratios[i];
    const b = ratios[j];
    const phase = 0.35 * t + (k * Math.PI) / ps.length; // slow drift per pair
    const col = TIDE[k % TIDE.length];
    ctx.strokeStyle = col;
    ctx.shadowColor = col;
    ctx.shadowBlur = 16;
    ctx.lineWidth = 2.4;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    for (let s = 0; s <= N; s++) {
      const tau = (s / N) * 2 * Math.PI;
      const x = cx + R * Math.sin(a * tau + phase);
      const y = cy + R * Math.sin(b * tau);
      if (s === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

/**
 * Two-pendulum-per-axis harmonograph with detuned frequencies and slowly
 * drifting phases, so the figure CONTINUOUSLY MORPHS instead of settling
 * into a static rosette. Light damping keeps the whole knot visible; a soft
 * dual-tone gradient stroke gives it depth.
 */
export function drawHarmonograph(
  ctx: CanvasRenderingContext2D,
  ratios: number[],
  frame: number,
  fps: number
): void {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(W, H) * 0.4;
  const t = frame / fps;
  const cycles = 26; // longer path → richer knot
  const N = 6000;
  const damp = 0.16; // light: the figure stays full

  const n = ratios.length;
  const r = (i: number) => ratios[((i % n) + n) % n];
  // Assign distinct, slightly-detuned frequencies to the two x and two y
  // pendulums. Detuning is what makes the curve breathe/precess.
  const fx1 = r(0);
  const fx2 = r(2) * 1.003;
  const fy1 = r(1) * 1.002;
  const fy2 = r(3) * 0.997;
  // Phases drift with time → the whole figure morphs every frame.
  const d1 = 0.18 * t;
  const d2 = 0.13 * t;
  const d3 = 0.21 * t + Math.PI / 2;
  const d4 = 0.16 * t + Math.PI / 3;

  const pts: Array<[number, number]> = [];
  for (let s = 0; s <= N; s++) {
    const u = s / N;
    const tau = u * cycles * 2 * Math.PI;
    const env = Math.exp(-damp * u * cycles * 0.16);
    const x =
      0.6 * Math.sin(fx1 * tau + d1) + 0.4 * Math.sin(fx2 * tau + d2);
    const y =
      0.6 * Math.sin(fy1 * tau + d3) + 0.4 * Math.sin(fy2 * tau + d4);
    pts.push([cx + R * env * x, cy + R * env * y]);
  }

  const tracePath = () => {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let s = 1; s < pts.length; s++) ctx.lineTo(pts[s][0], pts[s][1]);
  };

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // Teal halo underlay (whole path) for depth, then gold core on top.
  ctx.strokeStyle = "#7ad6c1";
  ctx.shadowColor = "rgba(122,214,193,0.5)";
  ctx.shadowBlur = 16;
  ctx.lineWidth = 3.2;
  ctx.globalAlpha = 0.35;
  tracePath();
  ctx.stroke();

  ctx.strokeStyle = "#e8c98a";
  ctx.shadowColor = "rgba(232,201,138,0.6)";
  ctx.shadowBlur = 8;
  ctx.lineWidth = 1.5;
  ctx.globalAlpha = 0.95;
  tracePath();
  ctx.stroke();

  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

/** Travelling-wave interference field (animated). N×N, painted via tidepool. */
export function drawInterference(
  ctx: CanvasRenderingContext2D,
  ratios: number[],
  frame: number,
  fps: number,
  N = 200
): void {
  const t = frame / fps;
  const field = new Float32Array(N * N);
  // A small ring of sources, each emitting at one chord frequency.
  const srcs = ratios.map((r, i) => {
    const ang = (i / ratios.length) * 2 * Math.PI;
    return { x: 0.5 + 0.32 * Math.cos(ang), y: 0.5 + 0.32 * Math.sin(ang), k: r };
  });
  let mn = Infinity;
  let mx = -Infinity;
  for (let r = 0; r < N; r++) {
    const y = r / (N - 1);
    for (let c = 0; c < N; c++) {
      const x = c / (N - 1);
      let v = 0;
      for (const s of srcs) {
        const d = Math.hypot(x - s.x, y - s.y);
        v += Math.cos(2 * Math.PI * s.k * (d * 3 - 0.6 * t));
      }
      field[r * N + c] = v;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  const span = mx - mn || 1;
  const img = ctx.createImageData(N, N);
  for (let k = 0; k < field.length; k++) {
    const norm = (field[k] - mn) / span;
    const [rr, gg, bb] = tidepool(norm, 0.9);
    img.data[k * 4 + 0] = rr;
    img.data[k * 4 + 1] = gg;
    img.data[k * 4 + 2] = bb;
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

/** Paint the cymatics density (shared with the Reel-02 scene). */
export function drawCymatics(
  ctx: CanvasRenderingContext2D,
  ratios: number[],
  gamma: number,
  N = 320
): void {
  const dens = cymaticsDensity(ratios, N, { symmetry: "d4_max" });
  const img = ctx.createImageData(N, N);
  for (let k = 0; k < dens.length; k++) {
    const [r, g, b] = tidepool(dens[k], gamma);
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
