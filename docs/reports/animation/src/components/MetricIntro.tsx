import React from "react";
import { interpolate, Easing } from "remotion";
import { theme, fonts } from "../theme";

/**
 * Series title card for the didactic metric reels. Eyebrow fades in, the metric
 * name scales up, the hook line rises — then the whole card fades out before the
 * reel proper begins. Driven by the raw composition frame; `dur` is the number
 * of frames the card occupies.
 */
export const MetricIntro: React.FC<{
  frame: number;
  dur: number;
  eyebrow: string;
  title: string;
  hook: string;
  accent: string;
}> = ({ frame, dur, eyebrow, title, hook, accent }) => {
  const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
  const eIn = interpolate(frame, [6, 24], [0, 1], clamp);
  const tIn = interpolate(frame, [10, 34], [0, 1], { ...clamp, easing: Easing.out(Easing.cubic) });
  const hIn = interpolate(frame, [22, 46], [0, 1], { ...clamp, easing: Easing.out(Easing.cubic) });
  const out = interpolate(frame, [dur - 16, dur], [1, 0], clamp);

  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center", opacity: out }}>
      <div style={{ fontFamily: fonts.mono, fontSize: 26, letterSpacing: 8,
        textTransform: "lowercase", color: theme.muted, opacity: eIn, marginBottom: 30 }}>
        {eyebrow}
      </div>
      <div style={{ fontFamily: fonts.display, fontSize: 80, fontWeight: 800, letterSpacing: 1,
        color: theme.ink, opacity: tIn, transform: `scale(${0.92 + 0.08 * tIn})` }}>
        {title}
      </div>
      <div style={{ fontFamily: fonts.display, fontSize: 34, fontWeight: 300, letterSpacing: 1,
        color: accent, opacity: hIn, transform: `translateY(${(1 - hIn) * 18}px)`, marginTop: 34,
        textShadow: `0 0 26px ${accent}55` }}>
        {hook}
      </div>
    </div>
  );
};
