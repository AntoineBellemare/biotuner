import React, { useMemo } from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { PedagogyCaption } from "../components/PedagogyCaption";
import { ChordNameOverlay } from "../components/ChordNameOverlay";
import { Stage } from "../components/Stage";
import { theme } from "../theme";
import { geometryData, Vec3, Lissajous3DKnotItem } from "../geometry";
import { tumbleProject, depthOpacity, tumble, project } from "../projection";

/** Each knot rotates 360° around the Y axis over its slot. */
export const FRAMES_PER_KNOT = 150; // 5 s

// ─────── Modern discrete palette per knot (one chord-tone family each) ───────
//
// Inspired by recent design-systems palettes (Tailwind / Radix). Each knot
// gets a single coherent hue family rather than a rainbow — three shades
// (deep, mid, bright) give us depth occlusion without colour clash.
//
// The palette rotates per knot so the four cases stay distinguishable:
//   3:4:5  → coral     (warm, the reference look)
//   2:3:5  → mint      (cool green)
//   3:5:7  → violet    (rich purple)
//   4:5:7  → amber     (warm gold)

type KnotPalette = {
  deep: string;    // back-of-tube ring, muted
  mid: string;     // tube body
  bright: string;  // front highlight
  glow: string;    // outer halo (rgba)
};

// ── Synthesize a "simpler" Lissajous knot for the IG / portrait version.
// The user wanted a clean intro knot resembling the one in the original
// pedagogical gif: low-order pairwise-coprime ratios (2 : 3 : 5), elegant
// phase offsets, smooth high point density. This gets PREPENDED to the
// items[] array in portrait so the IG composition opens with the simplest
// possible knot before stepping up to the existing 3 : 4 : 5 / etc.
function synthesizeSimpleKnot(): Lissajous3DKnotItem {
  const N = 480;
  const a = 2;
  const b = 3;
  const c = 5;
  const phix = 0;
  const phiy = Math.PI * 0.25;
  const phiz = Math.PI * 0.5;
  const verts: Vec3[] = [];
  for (let i = 0; i < N; i++) {
    const t = (i / N) * Math.PI * 2;
    verts.push([
      0.92 * Math.cos(a * t + phix),
      0.92 * Math.cos(b * t + phiy),
      0.92 * Math.cos(c * t + phiz),
    ]);
  }
  return {
    label: "2 : 3 : 5",
    subtitle:
      "The simplest 3-D Lissajous knot, three pairwise-coprime " +
      "frequencies with smooth phase offsets.",
    vertices: verts,
    is_knot: true,
  };
}

const SIMPLE_KNOT_IG = synthesizeSimpleKnot();

// Portrait / Reels pedagogy is scene-static: ONE title + ONE body for
// the whole 20-second knot slot. The four knots cycle visually so the
// viewer sees the variety, but the text stays put so they can finish
// reading.
const PEDAGOGY_IG_TITLE = "Lissajous knots";
const PEDAGOGY_IG_BODY =
  "Three frequencies in space. **Pairwise coprime** ratios form a " +
  "true knot that never closes.";

const KNOT_PALETTES: KnotPalette[] = [
  {
    deep: "#7a3a3f", mid: "#d97a82", bright: "#fb7185",
    glow: "rgba(251,113,133,0.30)",
  },
  {
    deep: "#2f6b5c", mid: "#5fb3a1", bright: "#7fdcc4",
    glow: "rgba(127,220,196,0.28)",
  },
  {
    deep: "#4b3673", mid: "#8b73c8", bright: "#a78bfa",
    glow: "rgba(167,139,250,0.30)",
  },
  {
    deep: "#7a5a2c", mid: "#d6a55a", bright: "#f5c97a",
    glow: "rgba(245,201,122,0.30)",
  },
];

// Cube vertices — used to draw a wireframe spatial reference behind the
// knot (Escher cabinet feel: the wireframe makes the knot read as living
// inside a 3-D enclosure rather than floating in flat space).
const CUBE_R = 0.92;
const CUBE_VERTS: Vec3[] = [
  [-CUBE_R, -CUBE_R, -CUBE_R],
  [+CUBE_R, -CUBE_R, -CUBE_R],
  [+CUBE_R, +CUBE_R, -CUBE_R],
  [-CUBE_R, +CUBE_R, -CUBE_R],
  [-CUBE_R, -CUBE_R, +CUBE_R],
  [+CUBE_R, -CUBE_R, +CUBE_R],
  [+CUBE_R, +CUBE_R, +CUBE_R],
  [-CUBE_R, +CUBE_R, +CUBE_R],
];
const CUBE_EDGES: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 0],   // bottom face (z=-)
  [4, 5], [5, 6], [6, 7], [7, 4],   // top face (z=+)
  [0, 4], [1, 5], [2, 6], [3, 7],   // vertical pillars
];

/**
 * 3-D Lissajous knots that rotate to reveal the weave. Each knot is
 * rendered as three concentric stroke passes (deep / mid / bright) that
 * are sorted back-to-front by depth so segments closer to the camera
 * physically over-paint segments behind them — yielding the "tube
 * crossing itself" Escher feel without an actual mesh.
 */
export const LissajousKnot3D: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const isPortrait = height > width;
  const rawItems = geometryData.scenes.lissajous_3d_knots.items;

  // Portrait playlist: [synthetic simple 2:3:5, original 0, original 1,
  // original 2] — same total count as landscape but starts on the
  // simplest knot and drops the last original (per user request). Slot
  // length stays at FRAMES_PER_KNOT so each knot still gets ~5 s and one
  // full rotation, like the landscape version.
  const items: Lissajous3DKnotItem[] = isPortrait
    ? [SIMPLE_KNOT_IG, ...rawItems.slice(0, rawItems.length - 1)]
    : rawItems;

  const itemIdx = Math.min(
    Math.floor(frame / FRAMES_PER_KNOT),
    items.length - 1
  );
  const localFrame = frame % FRAMES_PER_KNOT;
  const item = items[itemIdx];

  // Slow continuous rotation; small constant pitch so the knot never sits
  // edge-on. A tiny pitch oscillation keeps the depth read alive.
  const yaw = (localFrame / FRAMES_PER_KNOT) * Math.PI * 2;
  const pitch =
    0.32 +
    0.08 * Math.sin((localFrame / FRAMES_PER_KNOT) * Math.PI * 2);

  const projected = useMemo(
    () => tumbleProject(item.vertices as Vec3[], yaw, pitch, 1.7),
    [item.vertices, yaw, pitch]
  );

  // Build closed-curve segments with depth metadata. Smaller chunks =
  // smoother depth occlusion at curve crossings (back over-paint by
  // adjacent front).
  const segments = useMemo(() => {
    const chunk = 18;
    const total = projected.length;
    const segs: { d: string; zMid: number }[] = [];
    for (let i = 0; i < total; i += chunk) {
      const slice = projected.slice(i, Math.min(i + chunk + 1, total));
      if (slice.length < 2) continue;
      const zMid =
        slice.reduce((acc, p) => acc + p[2], 0) / slice.length;
      const d = slice
        .map((p, j) => `${j === 0 ? "M" : "L"} ${p[0]} ${p[1]}`)
        .join(" ");
      segs.push({ d, zMid });
    }
    // Close the loop
    const first = projected[0];
    const last = projected[projected.length - 1];
    if (first && last) {
      segs.push({
        d: `M ${last[0]} ${last[1]} L ${first[0]} ${first[1]}`,
        zMid: (first[2] + last[2]) * 0.5,
      });
    }
    return segs;
  }, [projected]);

  // Sort segments back-to-front so the painter's algorithm produces real
  // occlusion — front segments paint over back segments creating the
  // "impossible figure" tube-crossing effect.
  const sorted = useMemo(
    () => [...segments].sort((a, b) => a.zMid - b.zMid),
    [segments]
  );

  // Cube wireframe: project the 8 vertices each frame, draw 12 edges as
  // very thin low-opacity lines behind the knot.
  const cubeProjected = useMemo(
    () => CUBE_VERTS.map((v) => project(tumble(v, yaw, pitch), 1.7)),
    [yaw, pitch]
  );

  const palette = KNOT_PALETTES[itemIdx % KNOT_PALETTES.length];

  /** Modulate stroke colour by depth: back of tube = deep shade,
   *  middle = mid, front = bright. */
  function strokeForLayer(zMid: number, layer: "deep" | "mid" | "bright"): string {
    return palette[layer];
  }

  return (
    <AbsoluteFill>
      <Backdrop />
      {/* Centred watermark of the ratio (3 : 4 : 5 etc.) so the viewer
          knows which knot they're looking at without reading the bottom
          caption. Sits BEHIND the geometry; tube strokes paint over it. */}
      <ChordNameOverlay name={item.label} accent={palette.bright} />
      {/* Knot stays at landscape Stage size in portrait too — user
          requested everything bigger EXCEPT the knot, which already reads
          well at 760 px. */}
      <Stage portraitSize={760}>
        {/* Wireframe cube — Escher spatial reference */}
        {CUBE_EDGES.map(([a, b], k) => {
          const pa = cubeProjected[a];
          const pb = cubeProjected[b];
          const zMid = (pa[2] + pb[2]) * 0.5;
          const op = depthOpacity(zMid, 0.34, 0.10);
          return (
            <line
              key={`c${k}`}
              x1={pa[0]}
              y1={pa[1]}
              x2={pb[0]}
              y2={pb[1]}
              stroke={theme.muted}
              strokeWidth={0.0015}
              opacity={op}
            />
          );
        })}

        {/* Outer halo glow — same colour family, very wide & faint.
            Depth-sorted so it builds up most around front of knot. */}
        {sorted.map((s, k) => {
          const op = depthOpacity(s.zMid, 0.45, 0.14);
          return (
            <path
              key={`g${k}`}
              d={s.d}
              fill="none"
              stroke={palette.bright}
              strokeWidth={0.034}
              strokeLinecap="round"
              opacity={op}
            />
          );
        })}

        {/* Deep ring — under the tube, gives back-of-tube its shadow.
            Drawn back-to-front so we paint deep on the visible back, then
            over-paint with mid+bright on the front segments. */}
        {sorted.map((s, k) => {
          const op = depthOpacity(s.zMid, 1.0, 0.30);
          return (
            <path
              key={`d${k}`}
              d={s.d}
              fill="none"
              stroke={strokeForLayer(s.zMid, "deep")}
              strokeWidth={0.020}
              strokeLinecap="round"
              opacity={op}
            />
          );
        })}

        {/* Mid body — the tube proper */}
        {sorted.map((s, k) => {
          const op = depthOpacity(s.zMid, 1.0, 0.45);
          return (
            <path
              key={`m${k}`}
              d={s.d}
              fill="none"
              stroke={strokeForLayer(s.zMid, "mid")}
              strokeWidth={0.013}
              strokeLinecap="round"
              opacity={op}
            />
          );
        })}

        {/* Bright spine — only on the front-facing portion (depth > 0).
            Creates the highlight that sells the tube as 3-D. */}
        {sorted.map((s, k) => {
          if (s.zMid < -0.1) return null; // only mid/front
          const t = Math.max(0, Math.min(1, (s.zMid + 0.1) / 1.1));
          const op = 0.4 + 0.55 * t;
          return (
            <path
              key={`s${k}`}
              d={s.d}
              fill="none"
              stroke={strokeForLayer(s.zMid, "bright")}
              strokeWidth={0.0055}
              strokeLinecap="round"
              opacity={op}
            />
          );
        })}
      </Stage>
      <PedagogyCaption
        title={`3-D Lissajous knot · ${item.label}`}
        body={item.subtitle}
        meta={
          item.is_knot
            ? "frequencies pairwise coprime, this curve is a true knot"
            : "frequencies share a common factor, the curve closes onto itself"
        }
        accent={palette.bright}
        igTitle={PEDAGOGY_IG_TITLE}
        igBody={PEDAGOGY_IG_BODY}
      />
    </AbsoluteFill>
  );
};
