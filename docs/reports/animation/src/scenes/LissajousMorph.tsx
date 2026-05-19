import React, { useMemo } from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { Caption } from "../components/Caption";
import { Stage } from "../components/Stage";
import { theme } from "../theme";
import { geometryData, Vec2 } from "../geometry";

/**
 * Smooth crossfade through Lissajous frames. Each frame traces in over
 * its own slice of the timeline, with the next one blooming on top.
 */
export const LissajousMorph: React.FC = () => {
  const frame = useCurrentFrame();
  const frames = geometryData.scenes.lissajous_morph.frames;
  const SECONDS_PER_FRAME = 1.4;
  const FPS = 30;
  const FRAMES_PER_FRAME = Math.round(SECONDS_PER_FRAME * FPS);

  // Determine which Lissajous index is currently active.
  const activeIdx = Math.min(
    Math.floor(frame / FRAMES_PER_FRAME),
    frames.length - 1
  );
  const local = (frame % FRAMES_PER_FRAME) / FRAMES_PER_FRAME;

  return (
    <AbsoluteFill>
      <Backdrop />
      <Stage>
        {frames.map((f, i) => {
          // Render the previous, current, and next as a soft cross-blend.
          const distance = i - activeIdx;
          if (Math.abs(distance) > 1) return null;
          let opacity = 0;
          if (distance === 0) {
            opacity = 1;
          } else if (distance === -1) {
            opacity = 1 - local;
          } else if (distance === 1) {
            opacity = local;
          }
          return <LissajousPath key={i} points={f.points} opacity={opacity} />;
        })}
      </Stage>
      <Caption
        title={frames[activeIdx]?.label}
        subtitle="lissajous_2d(ratio, phase)"
      />
    </AbsoluteFill>
  );
};

const LissajousPath: React.FC<{ points: Vec2[]; opacity: number }> = ({
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
    <>
      <path
        d={d}
        fill="none"
        stroke={theme.cool}
        strokeWidth={0.012}
        opacity={opacity * 0.45}
      />
      <path
        d={d}
        fill="none"
        stroke={theme.accent}
        strokeWidth={0.005}
        opacity={opacity}
        style={{ filter: `drop-shadow(0 0 8px ${theme.glow})` }}
      />
    </>
  );
};
