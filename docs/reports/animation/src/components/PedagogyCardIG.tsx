import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { fonts, theme } from "../theme";

type Props = {
  /** Scene title, e.g. "Lissajous knots". Stays visible the whole scene. */
  title: string;
  /** One explanation paragraph for the WHOLE scene (not per-item).
   *  Wrap a single keyword in `**double-asterisks**` to highlight it
   *  in the accent colour. */
  body: string;
  /** Optional uppercase mono caption beneath the body. Scene-level too. */
  meta?: string;
  /** Hue colour. Picked up by the stripe, the title, and the keyword. */
  accent?: string;
};

/**
 * IG / Reels / Shorts pedagogy card.
 *
 * Top-anchored at y≈180 so it clears the IG feed's bottom UI overlay
 * (caption preview, profile pill, audio button, share/save chips).
 *
 * Static per scene. Fades in once at scene start, then holds for the
 * remainder of the slot. The viewer reads at their own pace and the
 * geometry below cycles through its variants without the text changing.
 *
 * Cadence (frames are scene-local because Series.Sequence resets the
 * frame counter at every scene boundary):
 *   • frames 0–6   : silent
 *   • frames 6–28  : title fades in (accent hue)
 *   • frames 22–48 : body fades in (one-sentence explanation)
 *   • frames 38–60 : meta fades in (optional ratios / call-out)
 *   • frames 60+   : everything held
 */
export const PedagogyCardIG: React.FC<Props> = ({
  title,
  body,
  meta,
  accent,
}) => {
  const accentColor = accent ?? theme.accent;
  const frame = useCurrentFrame();

  const titleOpacity = interpolate(frame, [6, 28], [0, 1], {
    extrapolateRight: "clamp",
    extrapolateLeft: "clamp",
  });
  const bodyOpacity = interpolate(frame, [22, 48], [0, 1], {
    extrapolateRight: "clamp",
    extrapolateLeft: "clamp",
  });
  const metaOpacity = interpolate(frame, [38, 60], [0, 1], {
    extrapolateRight: "clamp",
    extrapolateLeft: "clamp",
  });
  const lift = interpolate(frame, [0, 30], [12, 0], {
    extrapolateRight: "clamp",
  });

  // Parse `**bold**` segments. Odd-indexed slices are accent-coloured
  // keyword highlights, so prose can be written with one term wrapped.
  const bodySegments = body.split("**").map((seg, i) => ({
    text: seg,
    bold: i % 2 === 1,
  }));

  return (
    <div
      style={{
        position: "absolute",
        top: 180,
        left: 0,
        right: 0,
        padding: "0 56px",
        textAlign: "center",
        fontFamily: fonts.display,
        pointerEvents: "none",
        transform: `translateY(${lift}px)`,
      }}
    >
      {/* accent stripe */}
      <div
        style={{
          width: 100,
          height: 3,
          background: accentColor,
          opacity: 0.85 * titleOpacity,
          margin: "0 auto",
        }}
      />

      {/* Title */}
      <div
        style={{
          marginTop: 32,
          fontSize: 64,
          fontWeight: 500,
          lineHeight: 1.18,
          color: accentColor,
          letterSpacing: -0.6,
          textShadow: `0 0 22px ${accentColor}66`,
          opacity: titleOpacity,
        }}
      >
        {title}
      </div>

      {/* Body */}
      <div
        style={{
          marginTop: 28,
          fontSize: 46,
          fontWeight: 300,
          lineHeight: 1.32,
          color: theme.ink,
          opacity: bodyOpacity,
        }}
      >
        {bodySegments.map((seg, i) =>
          seg.bold ? (
            <span
              key={i}
              style={{
                color: accentColor,
                fontWeight: 600,
                textShadow: `0 0 14px ${accentColor}66`,
              }}
            >
              {seg.text}
            </span>
          ) : (
            <span key={i}>{seg.text}</span>
          )
        )}
      </div>

      {meta && (
        <div
          style={{
            marginTop: 28,
            fontFamily: fonts.mono,
            fontSize: 28,
            letterSpacing: 4,
            color: theme.cool,
            textTransform: "uppercase",
            opacity: metaOpacity * 0.78,
          }}
        >
          {meta}
        </div>
      )}
    </div>
  );
};
