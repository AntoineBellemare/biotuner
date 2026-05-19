import React, { useMemo } from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import { geometryData, Vec2, TimesTableStep } from "../geometry";

export const FRAMES_PER_STEP = 30; // 1 s @ 30 fps

/**
 * Animated sweep through times-table circle multipliers 2–12.
 * Each step shows 200 equally-spaced points on a unit circle connected
 * by lines from point i to point (i * multiplier) % 200. The resulting
 * patterns range from simple chords to intricate mandalas.
 */
export const TimesTableSweep: React.FC = () => {
  const frame = useCurrentFrame();
  const steps = geometryData.scenes.times_table_sweep.steps;

  const activeIdx = Math.min(
    Math.floor(frame / FRAMES_PER_STEP),
    steps.length - 1
  );
  const local = (frame % FRAMES_PER_STEP) / FRAMES_PER_STEP;

  const labelOpacity = interpolate(frame % FRAMES_PER_STEP, [0, 14], [0, 1], {
    extrapolateRight: "clamp",
  });

  const current = steps[activeIdx];

  return (
    <AbsoluteFill>
      <Backdrop />

      {/* Main circle + edges */}
      <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
        <svg
          width={820}
          height={820}
          viewBox="-1.1 -1.1 2.2 2.2"
          style={{ overflow: "visible" }}
        >
          <g transform="scale(1,-1)">
            {/* Ghost of previous step fading out */}
            {activeIdx > 0 && (
              <StepEdges
                step={steps[activeIdx - 1]}
                opacity={(1 - local) * 0.35}
              />
            )}

            {/* Current step edges fading in */}
            <StepEdges step={current} opacity={Math.min(local * 3, 1) * 0.85} />

            {/* Points ring – always fully visible */}
            <CirclePoints points={current.points} />
          </g>
        </svg>
      </AbsoluteFill>

      {/* Multiplier label – large centred number */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: 0,
          right: 0,
          transform: "translateY(-50%)",
          textAlign: "center",
          pointerEvents: "none",
        }}
      >
        <div
          style={{
            fontSize: 120,
            fontWeight: 800,
            color: theme.accent,
            opacity: 0.07,
            fontFamily: fonts.display,
            lineHeight: 1,
            letterSpacing: -4,
          }}
        >
          ×{current.multiplier}
        </div>
      </div>

      {/* Caption */}
      <div
        style={{
          position: "absolute",
          bottom: 80,
          left: 0,
          right: 0,
          textAlign: "center",
          opacity: labelOpacity,
          fontFamily: fonts.display,
        }}
      >
        <div
          style={{
            fontSize: 46,
            fontWeight: 600,
            color: theme.ink,
            letterSpacing: -0.5,
            textShadow: `0 0 24px ${theme.glow}`,
          }}
        >
          multiplier × {current.multiplier}
        </div>
        <div
          style={{
            fontSize: 21,
            fontWeight: 300,
            color: theme.muted,
            marginTop: 6,
            fontFamily: fonts.mono,
          }}
        >
          times_table_circle(n=200, multiplier={current.multiplier})
        </div>
      </div>

      {/* Progress bar */}
      <StepBar total={steps.length} active={activeIdx} />
    </AbsoluteFill>
  );
};

const StepEdges: React.FC<{ step: TimesTableStep; opacity: number }> = ({
  step,
  opacity,
}) => {
  const d = useMemo(() => {
    const pts = step.points;
    return step.edges
      .map(([a, b]) => `M ${pts[a][0]} ${pts[a][1]} L ${pts[b][0]} ${pts[b][1]}`)
      .join(" ");
  }, [step]);

  return (
    <>
      <path
        d={d}
        fill="none"
        stroke={theme.cool}
        strokeWidth={0.008}
        opacity={opacity * 0.4}
      />
      <path
        d={d}
        fill="none"
        stroke={theme.accent}
        strokeWidth={0.003}
        opacity={opacity}
        style={{ filter: `drop-shadow(0 0 3px ${theme.glow})` }}
      />
    </>
  );
};

const CirclePoints: React.FC<{ points: Vec2[] }> = ({ points }) => (
  <>
    {points.map((p, i) => (
      <circle
        key={i}
        cx={p[0]}
        cy={p[1]}
        r={0.012}
        fill={theme.muted}
        opacity={0.7}
      />
    ))}
  </>
);

const StepBar: React.FC<{ total: number; active: number }> = ({
  total,
  active,
}) => (
  <div
    style={{
      position: "absolute",
      top: 36,
      left: "50%",
      transform: "translateX(-50%)",
      display: "flex",
      gap: 6,
    }}
  >
    {Array.from({ length: total }, (_, i) => (
      <div
        key={i}
        style={{
          width: i === active ? 28 : 8,
          height: 8,
          borderRadius: 4,
          background: i === active ? theme.accent : theme.muted,
          opacity: i === active ? 1 : 0.35,
        }}
      />
    ))}
  </div>
);
