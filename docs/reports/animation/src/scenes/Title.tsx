import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { Stage } from "../components/Stage";
import { fonts, theme } from "../theme";
import { geometryData } from "../geometry";

export const Title: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const isPortrait = height > width;
  const titleSize = isPortrait ? 72 : 96;
  const subSize = isPortrait ? 28 : 28;
  const subLetterSpacing = isPortrait ? 6 : 6;

  // Animate a 1:1 phase π/2 Lissajous (unit circle) with phase drift.
  const phase = (frame / 60) * Math.PI * 2;
  const N = 360;
  const pts: [number, number][] = [];
  for (let i = 0; i < N; i++) {
    const t = (i / (N - 1)) * Math.PI * 2;
    pts.push([0.6 * Math.cos(t + phase * 0.05), 0.6 * Math.sin(t)]);
  }
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" ");

  const titleOpacity = spring({
    frame: frame - 12,
    fps,
    config: { damping: 200, stiffness: 80 },
  });
  const subOpacity = interpolate(frame, [40, 70], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <Backdrop />
      <Stage size={720}>
        <path
          d={path}
          fill="none"
          stroke={theme.cool}
          strokeWidth={0.005}
          opacity={0.5}
        />
        <path
          d={path}
          fill="none"
          stroke={theme.accent}
          strokeWidth={0.003}
          opacity={0.85}
          style={{ filter: `drop-shadow(0 0 6px ${theme.glow})` }}
        />
      </Stage>
      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            opacity: titleOpacity,
            transform: `translateY(${(1 - titleOpacity) * 20}px)`,
            color: theme.ink,
            fontFamily: fonts.display,
            fontSize: titleSize,
            fontWeight: 200,
            letterSpacing: isPortrait ? -1 : -2,
            textShadow: `0 0 36px ${theme.glow}`,
            textAlign: "center",
            padding: "0 32px",
          }}
        >
          {geometryData.title}
        </div>
        <div
          style={{
            opacity: subOpacity,
            color: theme.muted,
            fontFamily: fonts.mono,
            fontSize: subSize,
            letterSpacing: subLetterSpacing,
            textTransform: "uppercase",
            marginTop: 8,
          }}
        >
          {geometryData.subtitle}
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
