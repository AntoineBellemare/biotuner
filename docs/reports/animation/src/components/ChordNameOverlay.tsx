import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { fonts, theme } from "../theme";

type Props = {
  /** Text to show as the watermark (chord name, ratio, etc.). */
  name: string;
  /** Hue colour for the text. Defaults to theme.ink. Pass the chord's
   *  identity hue so the watermark colour-codes the scene. */
  accent?: string;
  /** Override the default font size (auto-picked by aspect ratio). */
  size?: number;
  /** Final opacity of the watermark. Default 0.42, low enough that
   *  geometry strokes painted on top stay legible. */
  opacity?: number;
  /** Frames over which the watermark fades in. */
  fadeFrames?: number;
  /** Pixels from the bottom of the canvas. Default 70. */
  bottom?: number;
};

/**
 * Big centred watermark of a chord name (or other identifier).
 *
 * Designed to sit BEHIND the geometric Stage so that geometry strokes
 * paint over the text. Thin display weight + wide letter-spacing +
 * accent-hue glow gives a "we know which chord this is" cue without
 * fighting the geometry for attention.
 *
 * Place between Backdrop and Stage in the scene tree:
 *   <Backdrop />
 *   <ChordNameOverlay name="Major" accent={hue.primary} />
 *   <Stage> ... geometry ... </Stage>
 *   <PedagogyCaption ... />
 */
export const ChordNameOverlay: React.FC<Props> = ({
  name,
  accent,
  size,
  opacity = 0.42,
  fadeFrames = 22,
  bottom = 70,
}) => {
  const { width, height } = useVideoConfig();
  const isPortrait = height > width;
  const frame = useCurrentFrame();
  const colour = accent ?? theme.ink;

  const baseSize = size ?? (isPortrait ? 200 : 240);

  const fade = interpolate(frame, [0, fadeFrames], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <div
      style={{
        position: "absolute",
        bottom,
        left: 0,
        right: 0,
        textAlign: "center",
        pointerEvents: "none",
        fontFamily: fonts.display,
        fontSize: baseSize,
        fontWeight: 200,
        lineHeight: 1,
        letterSpacing: -3,
        color: colour,
        opacity: opacity * fade,
        textShadow: `0 0 80px ${colour}55`,
        whiteSpace: "nowrap",
      }}
    >
      {name}
    </div>
  );
};
