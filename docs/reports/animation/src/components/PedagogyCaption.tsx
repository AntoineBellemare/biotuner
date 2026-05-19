import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { fonts, theme } from "../theme";
import { PedagogyCardIG } from "./PedagogyCardIG";

type Props = {
  /** Short bold lead-in (e.g. "Lissajous knot"). */
  title?: string;
  /** Pedagogical body — 1-2 sentences, plain language. */
  body?: string;
  /** Tiny mono caption underneath (optional, e.g. "3 : 4 : 5"). */
  meta?: string;
  /** Pixels from the bottom of the canvas. Default = 64. */
  bottom?: number;
  /** Frames over which the caption fades + lifts in. */
  fadeFrames?: number;
  /** When set, force a constant opacity instead of the fade-in. */
  fixedOpacity?: number;
  /** Accent color for the stripe + title — defaults to theme.accent. Pass
   *  the chord's identity hue to make captions colour-coded by chord. */
  accent?: string;
  // ── IG / Reels portrait overrides ──────────────────────────────────────
  // When the composition is portrait AND igTitle is provided, the
  // component routes to PedagogyCardIG. The IG card holds ONE title +
  // ONE body + optional meta for the entire scene, so the viewer reads
  // at their own pace while the geometry below cycles through variants.
  /** Scene-level title shown in portrait (e.g. "Lissajous knots"). */
  igTitle?: string;
  /** Scene-level explanation. Wrap one keyword in `**bold**` to highlight. */
  igBody?: string;
  /** Optional uppercase mono caption beneath body in portrait. */
  igMeta?: string;
};

/**
 * Lower-thirds caption tuned for pedagogical voice-over text:
 * a small bold title, a longer plain-language body paragraph constrained
 * to a comfortable measure, and an optional mono meta line.
 *
 * Lives in the bottom-third of the frame so it doesn't compete with the
 * geometric stage centred above it.
 */
export const PedagogyCaption: React.FC<Props> = ({
  title,
  body,
  meta,
  bottom,
  fadeFrames = 22,
  fixedOpacity,
  accent,
  igTitle,
  igBody,
  igMeta,
}) => {
  const accentColor = accent ?? theme.accent;
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  // Portrait (IG / Reels / TikTok) — bump up everything because the same
  // text would otherwise look tiny against 1920px of vertical canvas.
  const isPortrait = height > width;

  // Route to the IG card whenever portrait + the scene supplied an
  // igTitle. Keeps existing landscape look untouched. Once routed, the
  // card is fully scene-static (one title + one body for the whole slot).
  if (isPortrait && igTitle) {
    return (
      <PedagogyCardIG
        title={igTitle}
        body={igBody ?? body ?? ""}
        meta={igMeta ?? meta}
        accent={accent}
      />
    );
  }

  const titleSize = isPortrait ? 40 : 26;
  const bodySize = isPortrait ? 28 : 18;
  const metaSize = isPortrait ? 19 : 13;
  const stripeW = isPortrait ? 56 : 36;
  const stripeMb = isPortrait ? 22 : 14;
  const titleMb = body ? (isPortrait ? 16 : 10) : 0;
  const metaMt = isPortrait ? 18 : 12;
  const measureMax = isPortrait ? 940 : 760;
  const resolvedBottom = bottom ?? (isPortrait ? 110 : 64);
  const opacity =
    fixedOpacity !== undefined
      ? fixedOpacity
      : interpolate(frame, [0, fadeFrames], [0, 1], {
          extrapolateRight: "clamp",
        });
  const lift =
    fixedOpacity !== undefined
      ? 0
      : interpolate(frame, [0, fadeFrames], [16, 0], {
          extrapolateRight: "clamp",
        });
  return (
    <div
      style={{
        position: "absolute",
        bottom: resolvedBottom,
        left: 0,
        right: 0,
        textAlign: "center",
        color: theme.ink,
        fontFamily: fonts.display,
        opacity,
        transform: `translateY(${lift}px)`,
        pointerEvents: "none",
        padding: isPortrait ? "0 40px" : 0,
      }}
    >
      {/* accent stripe above the title — coloured by scene/chord */}
      {title && (
        <div
          style={{
            display: "inline-block",
            width: stripeW,
            height: 2,
            background: accentColor,
            opacity: 0.8,
            marginBottom: stripeMb,
          }}
        />
      )}
      {title && (
        <div
          style={{
            fontSize: titleSize,
            fontWeight: 600,
            letterSpacing: "0.02em",
            marginBottom: titleMb,
            color: accentColor,
          }}
        >
          {title}
        </div>
      )}
      {body && (
        <div
          style={{
            fontSize: bodySize,
            lineHeight: 1.45,
            maxWidth: measureMax,
            margin: "0 auto",
            color: theme.ink,
            opacity: 0.92,
            fontWeight: 300,
          }}
        >
          {body}
        </div>
      )}
      {meta && (
        <div
          style={{
            marginTop: metaMt,
            fontFamily: fonts.mono,
            fontSize: metaSize,
            color: theme.cool,
            opacity: 0.8,
            letterSpacing: "0.04em",
          }}
        >
          {meta}
        </div>
      )}
    </div>
  );
};
