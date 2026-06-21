/**
 * Multi-palette morphing for the meditative reel. A handful of earthy /
 * luminous palettes (RGB stops) and a `morphedStops(phase)` that smoothly
 * cross-fades between consecutive palettes as phase ∈ [0,1) sweeps the cycle.
 */

export type Stops = Array<[number, number, number]>;

const TIDEPOOL: Stops = [
  [10, 26, 31], [29, 94, 107], [74, 168, 156],
  [200, 160, 74], [217, 116, 74], [240, 216, 176],
];
const EMBER: Stops = [
  [0, 0, 0], [42, 10, 0], [140, 29, 4],
  [232, 89, 12], [255, 183, 0], [255, 243, 196],
];
const AURORA: Stops = [
  [0, 1, 13], [13, 35, 80], [14, 124, 107],
  [62, 214, 138], [182, 240, 106], [253, 255, 176],
];
const ULTRAVIOLET: Stops = [
  [4, 0, 13], [26, 5, 51], [91, 29, 140],
  [177, 74, 237], [231, 121, 240], [255, 214, 255],
];
const BIOLUM: Stops = [
  [0, 2, 8], [6, 48, 58], [14, 124, 134],
  [43, 212, 196], [155, 246, 224], [240, 255, 248],
];
const ROSE: Stops = [
  [10, 5, 8], [58, 13, 34], [156, 45, 82],
  [232, 99, 138], [244, 168, 156], [255, 232, 224],
];

export const MEDITATIVE_PALETTES: Stops[] = [
  TIDEPOOL, BIOLUM, AURORA, ULTRAVIOLET, ROSE, EMBER,
];

// Three more multi-hue ramps so each of the 9 gallery cells gets its own.
const SUNSET: Stops = [
  [8, 4, 16], [54, 12, 64], [142, 30, 86],
  [232, 92, 70], [255, 168, 64], [255, 240, 196],
];
const GLACIER: Stops = [
  [3, 8, 18], [16, 44, 92], [40, 110, 168],
  [96, 188, 214], [176, 232, 230], [240, 252, 255],
];
const VERDANT: Stops = [
  [6, 12, 6], [22, 56, 28], [60, 122, 52],
  [150, 196, 70], [226, 224, 120], [248, 252, 210],
];
const MAGMA: Stops = [
  [4, 2, 10], [44, 14, 56], [122, 32, 92],
  [206, 62, 80], [248, 142, 70], [253, 232, 168],
];

/** Nine distinct multi-hue palettes — one per gallery cell, so every brain
 *  and every heart in the 3×3 wall reads in its own colour world. */
export const GALLERY_PALETTES: Stops[] = [
  TIDEPOOL, SUNSET, AURORA, MAGMA, BIOLUM, ROSE, GLACIER, ULTRAVIOLET, VERDANT,
];

const lerp = (a: number, b: number, f: number) => a + (b - a) * f;

/** Interpolated 6-stop palette at cycle position `phase` ∈ [0,1). */
export function morphedStops(phase: number): Stops {
  const n = MEDITATIVE_PALETTES.length;
  const p = ((phase % 1) + 1) % 1 * n;
  const i = Math.floor(p) % n;
  const j = (i + 1) % n;
  const f = p - Math.floor(p);
  // cosine-eased crossfade so palette transitions are gentle
  const e = 0.5 * (1 - Math.cos(Math.PI * f));
  const A = MEDITATIVE_PALETTES[i];
  const B = MEDITATIVE_PALETTES[j];
  return A.map((s, k) => [
    Math.round(lerp(s[0], B[k][0], e)),
    Math.round(lerp(s[1], B[k][1], e)),
    Math.round(lerp(s[2], B[k][2], e)),
  ]) as Stops;
}

/** Sample evenly-spaced stops at v∈[0,1] with gamma. */
export function sampleStops(
  stops: Stops,
  v: number,
  gamma = 0.9
): [number, number, number] {
  const x = Math.pow(Math.max(0, Math.min(1, v)), gamma);
  const m = stops.length - 1;
  const t = x * m;
  const i = Math.min(Math.floor(t), m - 1);
  const f = t - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}
