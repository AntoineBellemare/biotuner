import React, { useMemo } from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { Caption } from "../components/Caption";
import { Stage } from "../components/Stage";
import { theme } from "../theme";
import { geometryData, Vec2 } from "../geometry";

// Each variant gets 75 frames (2.5 s): 60 s reveal + 15 s hold.
const FRAMES_PER_VARIANT = 75;

/**
 * Cycles through four harmonograph variants (harmonic drift, major chord,
 * dominant 7th, minor spiral). Each is revealed progressively like an actual
 * pendulum drawing on paper; the leading point is a glowing dot.
 */
export const Harmonograph: React.FC = () => {
  const frame = useCurrentFrame();
  const variants = geometryData.scenes.harmonograph_variants.variants;

  const variantIdx = Math.min(
    Math.floor(frame / FRAMES_PER_VARIANT),
    variants.length - 1
  );
  const localFrame = frame % FRAMES_PER_VARIANT;

  const variant = variants[variantIdx];
  const totalPoints = variant.points.length;

  // Reveal trace over the first 60 frames of each slot, then hold.
  const progress = interpolate(localFrame, [4, 62], [0, 1], {
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

  // Fade between variants: ghost of previous lingers during the first 10 frames.
  const ghostOpacity =
    localFrame < 10 && variantIdx > 0
      ? interpolate(localFrame, [0, 10], [0.45, 0], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        })
      : 0;

  const prevVariant =
    variantIdx > 0 ? variants[variantIdx - 1] : null;

  return (
    <AbsoluteFill>
      <Backdrop />
      <Stage>
        {/* Ghost of previous variant fading out */}
        {prevVariant && ghostOpacity > 0.01 && (
          <FullTrace points={prevVariant.points} opacity={ghostOpacity} />
        )}

        {/* Faint wide halo */}
        <path
          d={d}
          fill="none"
          stroke={theme.coolSoft}
          strokeWidth={0.007}
          opacity={0.3}
        />
        {/* Bright fine line */}
        <path
          d={d}
          fill="none"
          stroke={theme.accent}
          strokeWidth={0.003}
          opacity={0.9}
          style={{ filter: `drop-shadow(0 0 5px ${theme.glow})` }}
        />

        {/* Leading dot */}
        {head ? (
          <>
            <circle
              cx={head[0]}
              cy={head[1]}
              r={0.026}
              fill={theme.accent}
              opacity={0.2}
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

      <Caption
        title={variant.label}
        subtitle={variant.subtitle}
      />
    </AbsoluteFill>
  );
};

/** Render a full trace at a fixed opacity (used for ghost transitions). */
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
