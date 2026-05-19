import React from "react";
import { AbsoluteFill, useVideoConfig } from "remotion";

type Props = {
  /** Viewport size in pixels for the inner SVG. When omitted, picks a
   *  larger default in portrait (IG/Reels) so the geometry doesn't read
   *  as "tiny floating thing" on a phone screen. */
  size?: number;
  /** Override the portrait default specifically (used by the knot scene
   *  which the user wants to keep at landscape size). */
  portraitSize?: number;
  /** Pixels from the top of the canvas to anchor the SVG (portrait only).
   *  Default is centred. Use this to push the geometry below the IG
   *  pedagogy card instead of letting it overlap. */
  portraitTopOffset?: number;
  children: React.ReactNode;
};

/**
 * Centered square SVG canvas using normalized [-1, 1] coordinates.
 */
export const Stage: React.FC<Props> = ({
  size,
  portraitSize,
  portraitTopOffset,
  children,
}) => {
  const { width, height } = useVideoConfig();
  const isPortrait = height > width;
  const resolvedSize =
    size ?? (isPortrait ? portraitSize ?? 980 : 760);
  const useTopAnchor = isPortrait && portraitTopOffset !== undefined;
  return (
    <AbsoluteFill
      style={{
        justifyContent: useTopAnchor ? "flex-start" : "center",
        alignItems: "center",
        paddingTop: useTopAnchor ? portraitTopOffset : 0,
      }}
    >
      <svg
        width={resolvedSize}
        height={resolvedSize}
        viewBox="-1.05 -1.05 2.1 2.1"
        style={{ overflow: "visible" }}
      >
        {/* Y is flipped so positive y points up (math convention). */}
        <g transform="scale(1, -1)">{children}</g>
      </svg>
    </AbsoluteFill>
  );
};
