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

/**
 * Rotating 3-D Lissajous knot: x=sin(aτ), y=sin(bτ+φ), z=sin(cτ+ψ) for the
 * chord's first three ratios, rotated in 3-D and projected — a spinning knot
 * whose crossing structure is the chord's signature. Far more dynamic than a
 * flat 2-D figure, and clearly different per chord.
 */
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
  const R = Math.min(W, H) * 0.38;
  const t = frame / fps;
  const n = ratios.length;
  const a = ratios[0];
  const b = ratios[1 % n];
  const c = ratios[2 % n];
  const N = 2600;

  // Slowly drifting phases (morph) + a spin around two axes.
  const px = 0.0;
  const py = Math.PI / 3 + 0.12 * t;
  const pz = Math.PI / 5 + 0.08 * t;
  const ay = 0.35 * t;
  const ax = 0.45 + 0.12 * Math.sin(0.18 * t);
  const cay = Math.cos(ay), say = Math.sin(ay);
  const cax = Math.cos(ax), sax = Math.sin(ax);

  const pts: Array<[number, number, number]> = [];
  for (let s = 0; s <= N; s++) {
    const tau = (s / N) * 2 * Math.PI;
    const x = Math.sin(a * tau + px);
    const y = Math.sin(b * tau + py);
    const z = Math.sin(c * tau + pz);
    // rotate Y then X
    const x1 = x * cay + z * say;
    const z1 = -x * say + z * cay;
    const y1 = y * cax + z1 * sax;
    const z2 = -y * sax + z1 * cax;
    pts.push([cx + R * x1, cy + R * y1, z2]);
  }

  const trace = () => {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let s = 1; s < pts.length; s++) ctx.lineTo(pts[s][0], pts[s][1]);
  };
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // teal halo + gold core
  ctx.strokeStyle = "#4aa89c";
  ctx.shadowColor = "rgba(74,168,156,0.5)";
  ctx.shadowBlur = 18;
  ctx.lineWidth = 4;
  ctx.globalAlpha = 0.4;
  trace();
  ctx.stroke();
  ctx.strokeStyle = "#f0d8b0";
  ctx.shadowColor = "rgba(240,216,176,0.7)";
  ctx.shadowBlur = 8;
  ctx.lineWidth = 2.2;
  ctx.globalAlpha = 0.95;
  trace();
  ctx.stroke();
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
  const cycles = 42; // long path → full rosette
  const N = 6500;
  const damp = 0.26; // petals fade gently inward

  const n = ratios.length;
  const r = (i: number) => ratios[((i % n) + n) % n];
  // Ratio-driven flower: the chord's frequency ratios shape a distinct
  // rosette per chord. Low detune keeps the figure crisp (not a scribble);
  // gentle phase drift makes it breathe/rotate slowly.
  const fx1 = r(0);
  const fx2 = r(-1);
  const fy1 = r(1);
  const fy2 = r(2);
  const d1 = 0.07 * t;
  const d2 = 0.05 * t + 0.4;
  const d3 = 0.06 * t + 1.2;
  const d4 = 0.04 * t + 0.7;

  const pts: Array<[number, number]> = [];
  for (let s = 0; s <= N; s++) {
    const u = s / N;
    const tau = u * cycles * 2 * Math.PI;
    const env = Math.exp(-damp * u * cycles * 0.1);
    const x = Math.sin(fx1 * tau + d1) + 0.5 * Math.sin(fx2 * tau + d2);
    const y = Math.sin(fy1 * tau + d3) + 0.5 * Math.sin(fy2 * tau + d4);
    pts.push([cx + R * env * 0.62 * x, cy + R * env * 0.62 * y]);
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

/**
 * Vortex-interference field: each chord ratio contributes a spiral with that
 * many arms (cos(2π(αRk − kθ − ωt))). The spiral-arm count + density is the
 * chord's signature, and the rotating spirals look unmistakably different
 * from the lattice/curve geometries. Painted via the tidepool ramp.
 */
export function drawInterference(
  ctx: CanvasRenderingContext2D,
  ratios: number[],
  frame: number,
  fps: number,
  N = 200
): void {
  const t = frame / fps;
  const field = new Float32Array(N * N);
  let mn = Infinity;
  let mx = -Infinity;
  for (let r = 0; r < N; r++) {
    const y = (r / (N - 1)) * 2 - 1;
    for (let c = 0; c < N; c++) {
      const x = (c / (N - 1)) * 2 - 1;
      const R = Math.hypot(x, y);
      const th = Math.atan2(y, x);
      let v = 0;
      for (const k of ratios) {
        v += Math.cos(2 * Math.PI * (2.0 * R * k - k * th - 0.06 * t * k));
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
