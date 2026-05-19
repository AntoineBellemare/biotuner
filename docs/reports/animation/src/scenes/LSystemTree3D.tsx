import React, { useMemo } from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { PedagogyCaption } from "../components/PedagogyCaption";
import { ChordNameOverlay } from "../components/ChordNameOverlay";
import { Stage } from "../components/Stage";
import { theme, getChordHue } from "../theme";
import { geometryData, Vec3 } from "../geometry";
import { tumbleProject, depthOpacity } from "../projection";

/** Each tree rotates ~270° over its slot — enough to reveal 3-D structure. */
export const FRAMES_PER_TREE = 150; // 5 s

// Portrait / Reels pedagogy is scene-static: ONE title + ONE body for
// the whole 20-second slot. The four chord trees cycle visually so the
// viewer sees how the structure shifts across chords, but the text stays
// put so they can finish reading.
const PEDAGOGY_IG_TITLE = "Fractal harmonic growth";
const PEDAGOGY_IG_BODY =
  "Each ratio becomes a **turning rule**. The chord grows into a " +
  "3D form.";

const PEDAGOGY: Record<string, string> = {
  Major:
    "Each chord-tone becomes a branching rule. The recursive turtle threads " +
    "the chord's intervals into a 3-D form, and consonant chords grow tidy, " +
    "balanced trees.",
  Dom7:
    "The added 7th doubles the branching factor. The tree splits more often, " +
    "producing a denser, more chaotic root system that mirrors the chord's " +
    "harmonic tension.",
  Sus4:
    "A suspended 4th replaces the 3rd. The branches lean toward perfect-fifth " +
    "symmetry, a near-mirror tree, structurally lighter than the major.",
  Dim7:
    "Stacked minor 3rds (≈ 6 : 5 ratios). The tree becomes a tightly-wound " +
    "spiral, and every branch repeats the same interval, giving a " +
    "fractal-like self-similarity.",
};

/**
 * Rotating 3-D L-system trees driven by chord ratios. Trunk and branches
 * rendered as line segments with depth-based opacity so the back of the
 * tree reads as further away.
 */
export const LSystemTree3D: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const isPortrait = height > width;
  const items = geometryData.scenes.lsystem_3d_variants.items;

  const itemIdx = Math.min(
    Math.floor(frame / FRAMES_PER_TREE),
    items.length - 1
  );
  const localFrame = frame % FRAMES_PER_TREE;
  const item = items[itemIdx];

  // Rotate ~270° (3π/2) over the slot, with a steady downward tilt so
  // we look at the tree from slightly above
  const yaw = (localFrame / FRAMES_PER_TREE) * Math.PI * 1.5;
  const pitch = -0.18;

  const projected = useMemo(
    () => tumbleProject(item.vertices as Vec3[], yaw, pitch, 1.8),
    [item.vertices, yaw, pitch]
  );

  // Growth animation. Reveal edges progressively over the first ~70 % of
  // the slot so the tree appears to *grow* before the rotation completes,
  // then hold on the full tree for the remainder. The L-system edges are
  // already in turtle-traversal order, so slicing from the start traces
  // the tree the way it was drawn.
  const growth = interpolate(
    localFrame,
    [4, FRAMES_PER_TREE * 0.72],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const visibleEdgeCount = Math.max(
    1,
    Math.floor(item.edges.length * growth)
  );
  const visibleEdges = item.edges.slice(0, visibleEdgeCount);

  // Chord-identity hue — bright stroke uses the chord's signature colour,
  // halo uses the soft variant. Major=gold, Sus4=teal, Dom7=amber,
  // Dim7=crimson — readable at a glance even before reading the caption.
  const hue = getChordHue(item.name);

  // Stroke widths in SVG-normalised units; bump them in portrait so the
  // tree silhouette stays bold on the bigger Stage.
  const haloWidth = isPortrait ? 0.014 : 0.011;
  const brightWidth = isPortrait ? 0.0044 : 0.0034;

  return (
    <AbsoluteFill>
      <Backdrop />
      {/* Centred watermark of the chord name. Sits behind the tree;
          branches paint over it as the tree grows. */}
      <ChordNameOverlay name={item.name} accent={hue.primary} />
      {/* Portrait: anchor a giant (1300 px) Stage to y=620, so the tree
          fills the lower 65 % of the frame. Root reaches close to the IG
          bottom UI safe zone, canopy starts well below the pedagogy card,
          horizontal overflow (110 px each side) is invisible because the
          tree silhouette is well within those bounds. */}
      <Stage portraitSize={1300} portraitTopOffset={620}>
        {/* Halo pass */}
        {visibleEdges.map(([a, b], k) => {
          const pa = projected[a];
          const pb = projected[b];
          if (!pa || !pb) return null;
          const op = depthOpacity((pa[2] + pb[2]) * 0.5, 1.0, 0.2);
          return (
            <line
              key={`h${k}`}
              x1={pa[0]}
              y1={pa[1]}
              x2={pb[0]}
              y2={pb[1]}
              stroke={hue.soft}
              strokeWidth={haloWidth}
              strokeLinecap="round"
              opacity={op * 0.32}
            />
          );
        })}
        {/* Bright pass */}
        {visibleEdges.map(([a, b], k) => {
          const pa = projected[a];
          const pb = projected[b];
          if (!pa || !pb) return null;
          const op = depthOpacity((pa[2] + pb[2]) * 0.5, 1.0, 0.2);
          return (
            <line
              key={`b${k}`}
              x1={pa[0]}
              y1={pa[1]}
              x2={pb[0]}
              y2={pb[1]}
              stroke={hue.primary}
              strokeWidth={brightWidth}
              strokeLinecap="round"
              opacity={op}
            />
          );
        })}
      </Stage>
      <PedagogyCaption
        title={`Fractal harmonic growth · ${item.name}`}
        body={
          PEDAGOGY[item.name] ??
          "A 3-D L-system grown from the chord's ratios. Each chord-tone " +
            "contributes a turning rule; depth-3 recursion produces the " +
            "branching above."
        }
        meta={`${item.n_segments} segments, depth = 3`}
        accent={hue.primary}
        igTitle={PEDAGOGY_IG_TITLE}
        igBody={PEDAGOGY_IG_BODY}
      />
    </AbsoluteFill>
  );
};
