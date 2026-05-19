import React, { useMemo } from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { Stage } from "../components/Stage";
import { PedagogyCaption } from "../components/PedagogyCaption";
import { ChordNameOverlay } from "../components/ChordNameOverlay";
import { theme, getChordHue } from "../theme";
import { geometryData, Vec2 } from "../geometry";

// 4 variants × 120 frames each (4 s) — slower than the v1 scene so the
// real-time pen-stroke effect actually feels like a pendulum drawing.
export const FRAMES_PER_VARIANT_RT = 120;

type Pedagogy = { title: string; body: string };

const PEDAGOGY: Pedagogy[] = [
  {
    title: "Harmonograph",
    body:
      "A pendulum drawing machine. Each axis swings at its own frequency and " +
      "decays exponentially. The pen, riding the sum, traces a slowly-shrinking " +
      "spiral whose shape encodes the relationship between the frequencies.",
  },
  {
    title: "A major chord",
    body:
      "Three frequencies in 4 : 5 : 6 ratio drive the same machine. Their " +
      "near-rational alignment produces a stable braided figure rather than a " +
      "smear, and the consonance of the chord becomes geometric stability.",
  },
  {
    title: "A dominant 7th",
    body:
      "Adding a 7th harmonic introduces a slow phase rotation. The figure " +
      "develops radial symmetry that the major chord lacked, and you can " +
      "see the extra tension as petals around the centre.",
  },
  {
    title: "A minor chord",
    body:
      "Heavy damping makes the pen settle quickly. Each successive cycle is " +
      "thinner than the last, producing a clean concentric spiral instead of " +
      "the dense overlapping smear of a long undamped trace.",
  },
];

// Portrait / Reels pedagogy is scene-static: ONE title + ONE explanation
// for the whole 16-second harmonograph slot. The variants below cycle
// visually but the text stays put so the viewer can read at their own
// pace. Wrap one keyword in **bold** to highlight in the accent colour.
// Short chord names for the centred watermark, indexed by variant.
// Variant 0 has no chord (just the harmonograph apparatus itself), so we
// show nothing — keeps the opening uncluttered.
const VARIANT_CHORD_NAME: (string | null)[] = [
  null,
  "Major",
  "Dom 7",
  "Minor",
];

const PEDAGOGY_IG_TITLE = "Harmonograph";
const PEDAGOGY_IG_BODY =
  "A pendulum draws the shape of a chord. The pen records each " +
  "note's **interplay**.";

/**
 * Real-time harmonograph reveal: each of four variants is drawn point-by-point
 * over its allotted 120 frames, like a physical pendulum recording. The first
 * 10 frames carry over a faded "ghost" of the previous trace.
 */
export const HarmonographRealtime: React.FC = () => {
  const frame = useCurrentFrame();
  const variants = geometryData.scenes.harmonograph_variants.variants;

  const variantIdx = Math.min(
    Math.floor(frame / FRAMES_PER_VARIANT_RT),
    variants.length - 1
  );
  const localFrame = frame % FRAMES_PER_VARIANT_RT;
  const variant = variants[variantIdx];
  const totalPoints = variant.points.length;

  // Reveal: 4 → 110 frames (longer than v1 so each cycle is visible).
  const progress = interpolate(localFrame, [4, 110], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const upTo = Math.max(2, Math.floor(progress * totalPoints));

  const d = useMemo(() => {
    const slice = variant.points.slice(0, upTo);
    return slice
      .map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`)
      .join(" ");
  }, [variant.points, upTo]);

  const head: Vec2 | undefined = variant.points[upTo - 1];

  // Ghost of previous variant during the first 12 frames
  const ghostOpacity =
    localFrame < 12 && variantIdx > 0
      ? interpolate(localFrame, [0, 12], [0.4, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        })
      : 0;
  const prevVariant =
    variantIdx > 0 ? variants[variantIdx - 1] : null;

  const pedagogy = PEDAGOGY[Math.min(variantIdx, PEDAGOGY.length - 1)];
  const chordName = VARIANT_CHORD_NAME[variantIdx] ?? null;
  const chordHue = chordName ? getChordHue(chordName) : null;

  return (
    <AbsoluteFill>
      <Backdrop />
      {chordName && (
        <ChordNameOverlay
          name={chordName}
          accent={chordHue?.primary ?? theme.accent}
        />
      )}
      <Stage>
        {prevVariant && ghostOpacity > 0.01 && (
          <FullTrace points={prevVariant.points} opacity={ghostOpacity} />
        )}
        <path
          d={d}
          fill="none"
          stroke={theme.coolSoft}
          strokeWidth={0.007}
          opacity={0.28}
        />
        <path
          d={d}
          fill="none"
          stroke={theme.accent}
          strokeWidth={0.0028}
          opacity={0.92}
          style={{ filter: `drop-shadow(0 0 5px ${theme.glow})` }}
        />
        {head ? (
          <>
            <circle
              cx={head[0]}
              cy={head[1]}
              r={0.026}
              fill={theme.accent}
              opacity={0.18}
            />
            <circle
              cx={head[0]}
              cy={head[1]}
              r={0.011}
              fill={theme.ink}
              style={{ filter: `drop-shadow(0 0 7px ${theme.accent})` }}
            />
          </>
        ) : null}
      </Stage>
      <PedagogyCaption
        title={pedagogy.title}
        body={pedagogy.body}
        meta={variant.subtitle}
        igTitle={PEDAGOGY_IG_TITLE}
        igBody={PEDAGOGY_IG_BODY}
      />
    </AbsoluteFill>
  );
};

const FullTrace: React.FC<{ points: Vec2[]; opacity: number }> = ({
  points,
  opacity,
}) => {
  const d = useMemo(
    () =>
      points
        .map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`)
        .join(" "),
    [points]
  );
  return (
    <path
      d={d}
      fill="none"
      stroke={theme.coolSoft}
      strokeWidth={0.003}
      opacity={opacity}
    />
  );
};
