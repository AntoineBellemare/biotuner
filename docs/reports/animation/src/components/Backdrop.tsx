import React from "react";
import { AbsoluteFill, useCurrentFrame, interpolate } from "remotion";
import { theme } from "../theme";

/**
 * Subtle radial glow background, gently pulsing with frame.
 */
export const Backdrop: React.FC = () => {
  const frame = useCurrentFrame();
  const glow = interpolate(
    Math.sin(frame * 0.02),
    [-1, 1],
    [0.18, 0.32]
  );
  return (
    <AbsoluteFill
      style={{
        background: `radial-gradient(circle at 50% 55%, rgba(109, 163, 216, ${glow}) 0%, ${theme.bg} 55%, ${theme.bgDeep} 100%)`,
      }}
    />
  );
};
