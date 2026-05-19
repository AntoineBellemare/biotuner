import React from "react";
import { AbsoluteFill, interpolate, useCurrentFrame, spring, useVideoConfig } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { Caption } from "../components/Caption";
import { Stage } from "../components/Stage";
import { theme } from "../theme";
import { geometryData } from "../geometry";

export const TuningCircle: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const data = geometryData.scenes.tuning_circle;
  const points = data.points;
  const weights = data.weights;

  // Reference circle traces in over the first ~12 frames.
  const circleProgress = interpolate(frame, [0, 16], [0, 1], {
    extrapolateRight: "clamp",
  });
  const circumference = 2 * Math.PI;
  const dasharray = `${circleProgress * circumference} ${circumference}`;

  // Each point pops in sequentially.
  const popDelay = 8;
  const popDuration = 14;

  return (
    <AbsoluteFill>
      <Backdrop />
      <Stage>
        <circle
          cx={0}
          cy={0}
          r={0.85}
          fill="none"
          stroke={theme.muted}
          strokeWidth={0.005}
          strokeDasharray={dasharray}
        />
        {points.map((p, i) => {
          const local = frame - 18 - i * popDelay;
          const scale = spring({
            frame: local,
            fps,
            config: { damping: 12, stiffness: 100 },
          });
          if (local < 0) return null;
          const baseR = 0.05 + 0.04 * weights[i] / Math.max(...weights);
          return (
            <g key={i}>
              <circle
                cx={p[0]}
                cy={p[1]}
                r={baseR * 1.6 * scale}
                fill={theme.cool}
                opacity={0.25 * scale}
              />
              <circle
                cx={p[0]}
                cy={p[1]}
                r={baseR * scale}
                fill={theme.accent}
                style={{ filter: `drop-shadow(0 0 5px ${theme.glow})` }}
              />
              <text
                x={p[0] * 1.15}
                y={-p[1] * 1.15 + 0.025}
                fontSize={0.06}
                textAnchor="middle"
                fill={theme.ink}
                opacity={interpolate(local, [10, 22], [0, 1], {
                  extrapolateLeft: "clamp",
                  extrapolateRight: "clamp",
                })}
                style={{ fontFamily: '"SF Mono", monospace' }}
                transform="scale(1, -1)"
              >
                {data.labels[i]}
              </text>
            </g>
          );
        })}
      </Stage>
      <Caption
        title="tuning_circle"
        subtitle="just-intonation diatonic on the octave-equave"
      />
    </AbsoluteFill>
  );
};
