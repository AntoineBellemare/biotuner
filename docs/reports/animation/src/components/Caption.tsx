import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { fonts, theme } from "../theme";

type Props = {
  title?: string;
  subtitle?: string;
  fadeFrames?: number;
};

/**
 * Lower-thirds-style caption that fades in over `fadeFrames`.
 */
export const Caption: React.FC<Props> = ({
  title,
  subtitle,
  fadeFrames = 18,
}) => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, fadeFrames], [0, 1], {
    extrapolateRight: "clamp",
  });
  const lift = interpolate(frame, [0, fadeFrames], [12, 0], {
    extrapolateRight: "clamp",
  });
  return (
    <div
      style={{
        position: "absolute",
        bottom: 96,
        left: 0,
        right: 0,
        textAlign: "center",
        color: theme.ink,
        fontFamily: fonts.display,
        opacity,
        transform: `translateY(${lift}px)`,
      }}
    >
      {title ? (
        <div
          style={{
            fontSize: 44,
            fontWeight: 600,
            letterSpacing: -0.5,
            textShadow: `0 0 24px ${theme.glow}`,
          }}
        >
          {title}
        </div>
      ) : null}
      {subtitle ? (
        <div
          style={{
            fontSize: 22,
            fontWeight: 300,
            color: theme.muted,
            marginTop: 6,
            fontFamily: fonts.mono,
          }}
        >
          {subtitle}
        </div>
      ) : null}
    </div>
  );
};
