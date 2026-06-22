import React from "react";
import { interpolate, Easing } from "remotion";
import { FlowerOfLife } from "../reels/motifs/FlowerOfLife";
import { theme, fonts } from "../theme";

/**
 * Series title card for the didactic metric reels — brand-consistent with the
 * channel's ReelIntro: the BIOTUNER wordmark above the Flower-of-Life motif,
 * the metric name framed in its heart, and the hook line below. Eyebrow/brand
 * fade in, the flower draws ring-by-ring, the name scales up, the hook rises,
 * then the whole card dissolves before the reel proper begins.
 *
 * Driven by the raw composition frame; `dur` = frames the card occupies.
 */
const BRAND_TEAL = "#7ad6c1";
const HOOK_GOLD = "#f2c14e";
const VIGNETTE = "#06070e";

function rgba(hex: string, a: number): string {
  const h = hex.replace("#", "");
  return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`;
}

export const MetricIntro: React.FC<{
  frame: number;
  dur: number;
  title: string;
  hook: string;
  /** flower + glow accent (defaults to the brand teal) */
  accent?: string;
}> = ({ frame, dur, title, hook, accent = BRAND_TEAL }) => {
  const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
  const cubic = { ...clamp, easing: Easing.out(Easing.cubic) };
  const brandIn = interpolate(frame, [6, 26], [0, 1], clamp);
  const flowerProg = interpolate(frame, [2, dur - 10], [0, 1], clamp);
  const titleIn = interpolate(frame, [18, 42], [0, 1], cubic);
  const hookIn = interpolate(frame, [30, 52], [0, 1], cubic);
  const out = interpolate(frame, [dur - 16, dur], [1, 0], clamp);

  const CY = 940; // vertical anchor of the flower / metric name

  return (
    <div style={{ position: "absolute", inset: 0, opacity: out }}>
      {/* Flower-of-Life hero */}
      <div style={{ position: "absolute", left: "50%", top: CY, transform: "translate(-50%,-50%)", opacity: 0.92 }}>
        <FlowerOfLife size={620} color={accent} glow={rgba(accent, 0.42)} progress={flowerProg} />
      </div>

      {/* BIOTUNER brand wordmark, above the flower */}
      <div style={{ position: "absolute", top: 532, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 50, fontWeight: 800, letterSpacing: 16,
        color: theme.ink, opacity: brandIn, transform: `translateY(${(1 - brandIn) * -14}px)`,
        textShadow: `0 0 28px ${rgba(accent, 0.45)}` }}>
        BIOTUNER
      </div>

      {/* readability vignette behind the metric name */}
      <div style={{ position: "absolute", left: "50%", top: CY, transform: "translate(-50%,-50%)",
        width: 820, height: 280, opacity: titleIn,
        background: `radial-gradient(ellipse at center, ${VIGNETTE} 0%, ${rgba(VIGNETTE, 0.92)} 40%, transparent 74%)` }} />

      {/* metric name, framed in the flower's heart */}
      <div style={{ position: "absolute", top: CY - 52, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 72, fontWeight: 800, letterSpacing: 1, color: theme.ink,
        opacity: titleIn, transform: `scale(${0.94 + 0.06 * titleIn})`,
        textShadow: `0 0 26px ${rgba(accent, 0.35)}` }}>
        {title}
      </div>

      {/* hook line, below the flower */}
      <div style={{ position: "absolute", top: 1318, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 35, fontWeight: 300, letterSpacing: 1, color: HOOK_GOLD,
        opacity: hookIn, transform: `translateY(${(1 - hookIn) * 18}px)`,
        textShadow: `0 0 24px ${rgba(HOOK_GOLD, 0.4)}` }}>
        {hook}
      </div>
    </div>
  );
};
