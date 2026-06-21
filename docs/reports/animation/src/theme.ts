export const theme = {
  bg: "#0a0e1a",
  bgDeep: "#050811",
  ink: "#f0f4ff",
  accent: "#e8d68a", // warm gold for primary traces
  accentSoft: "#c8a85a",
  cool: "#6da3d8", // cool blue secondary
  coolSoft: "#3f6e9c",
  muted: "#4a5670",
  glow: "rgba(232, 214, 138, 0.35)",
};

export const fonts = {
  display: '"SF Pro Display", "Inter", -apple-system, system-ui, sans-serif',
  mono: '"SF Mono", "JetBrains Mono", "Menlo", monospace',
};

// ────────────────────────── CHORD-IDENTITY COLORS ───────────────────────────
//
// Each chord gets a signature hue applied consistently across every scene
// where the chord appears (Chladni titles, L-system bright stroke, point
// cloud high-end ramp, Pedagogy caption stripe). Lets the eye anchor "this
// is the same chord" across rotations and modalities.

export type ChordHue = {
  primary: string;   // bright stroke / title color
  soft: string;      // halo / accent-soft equivalent
  glow: string;      // rgba glow shadow
  rgb: [number, number, number]; // for color ramps that need to interpolate
};

const CHORD_HUE_TABLE: Record<string, ChordHue> = {
  major: {
    primary: "#e8d68a", soft: "#c8a85a",
    glow: "rgba(232,214,138,0.35)", rgb: [232, 214, 138],
  },
  minor: {
    primary: "#8a9be8", soft: "#5a6dc8",
    glow: "rgba(138,155,232,0.32)", rgb: [138, 155, 232],
  },
  sus4: {
    primary: "#7ad6c1", soft: "#4ea18d",
    glow: "rgba(122,214,193,0.30)", rgb: [122, 214, 193],
  },
  dom7: {
    primary: "#e8a26b", soft: "#c8814a",
    glow: "rgba(232,162,107,0.32)", rgb: [232, 162, 107],
  },
  maj7: {
    primary: "#f4d480", soft: "#c4a350",
    glow: "rgba(244,212,128,0.32)", rgb: [244, 212, 128],
  },
  aug: {
    primary: "#d27ed4", soft: "#a256a4",
    glow: "rgba(210,126,212,0.30)", rgb: [210, 126, 212],
  },
  augmented: {
    primary: "#d27ed4", soft: "#a256a4",
    glow: "rgba(210,126,212,0.30)", rgb: [210, 126, 212],
  },
  dim7: {
    primary: "#e87a7a", soft: "#c25555",
    glow: "rgba(232,122,122,0.30)", rgb: [232, 122, 122],
  },
  diminished: {
    primary: "#e87a7a", soft: "#c25555",
    glow: "rgba(232,122,122,0.30)", rgb: [232, 122, 122],
  },
};

/**
 * Normalise a chord label (which may carry trailing words like "7th",
 * embedded spaces, or surface suffixes like " / sphere") to a stable key
 * usable to look up its identity colour.
 */
function normalizeChordKey(name: string): string {
  const root = name.split("/")[0].trim().toLowerCase();
  // Strip whitespace, then remove trailing "th" so "dom 7th" → "dom7".
  return root.replace(/\s+/g, "").replace(/th$/, "");
}

/**
 * Get the chord-identity colour for a label, or fall back to the default
 * gold accent if the label doesn't match a known chord.
 */
export function getChordHue(name: string): ChordHue {
  const key = normalizeChordKey(name);
  return (
    CHORD_HUE_TABLE[key] ?? {
      primary: theme.accent,
      soft: theme.accentSoft,
      glow: theme.glow,
      rgb: [232, 214, 138],
    }
  );
}

/**
 * Linear interpolation between two RGB colours, returning an "rgb(...)"
 * string. Used by point cloud / field ramps to mix from a dark base into
 * the chord's identity colour.
 */
export function lerpRgb(
  a: [number, number, number],
  b: [number, number, number],
  t: number
): string {
  const tt = Math.max(0, Math.min(1, t));
  const r = Math.round(a[0] + (b[0] - a[0]) * tt);
  const g = Math.round(a[1] + (b[1] - a[1]) * tt);
  const bl = Math.round(a[2] + (b[2] - a[2]) * tt);
  return `rgb(${r},${g},${bl})`;
}

/**
 * Plasma-style sequential ramp: dark purple → red → orange → yellow → near-white.
 * Perceptually monotone in luminance so it reads cleanly under depth fading.
 */
export function plasma(t: number): [number, number, number] {
  const tt = Math.max(0, Math.min(1, t));
  // Hand-tuned 5-stop ramp approximating matplotlib's plasma.
  const stops: Array<[number, [number, number, number]]> = [
    [0.0,  [13,   8,  135]],   // deep purple
    [0.25, [126,  3,  167]],   // magenta
    [0.5,  [203, 71,  119]],   // pink-red
    [0.75, [248, 149, 64]],    // orange
    [1.0,  [240, 249, 33]],    // bright yellow
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (tt <= t1) {
      const u = (tt - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * u),
        Math.round(c0[1] + (c1[1] - c0[1]) * u),
        Math.round(c0[2] + (c1[2] - c0[2]) * u),
      ];
    }
  }
  return stops[stops.length - 1][1];
}
