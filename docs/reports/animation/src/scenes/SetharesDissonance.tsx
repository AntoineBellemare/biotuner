import React from "react";
import {
  AbsoluteFill,
  Audio,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  interpolateColors,
  Easing,
} from "remotion";
import { evolvePath } from "@remotion/paths";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import data from "../../public/sethares.json";

/**
 * "Where does it get rough?" — the Sethares / Plomp-Levelt sensory-dissonance
 * curve. Two complex tones; the upper sweeps from unison to the octave. Total
 * roughness (summed over every pair of partials) peaks just above unison and
 * dips into VALLEYS at the simple ratios — the consonances a timbre prefers.
 * The curve is biotuner's real dissmeasure.
 */
const CURVE = data.curve as number[];
const VALLEYS = data.valleys as { alpha: number; label: string; diss_norm: number }[];
const NP = data.n_partials;
const TEAL = "#6fd6c4";
const HOT = "#e8746a";
const GOLD = "#f2c14e";

const INTRO = 30;
const SWEEP = 500;
const OUTRO = 46;
export const TOTAL_SETHARES = INTRO + SWEEP + OUTRO;

const dissAt = (alpha: number) =>
  CURVE[Math.max(0, Math.min(CURVE.length - 1, Math.round((alpha - 1) * (CURVE.length - 1))))];

export const SetharesDissonance: React.FC = () => {
  const frame = useCurrentFrame();
  const { width } = useVideoConfig();

  // upper tone sweeps 1 → 2, easing to a brief hold at the deep fifth + octave
  const swp = interpolate(frame, [INTRO, INTRO + SWEEP], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.sin),
  });
  const alpha = 1 + swp;
  const diss = dissAt(alpha);

  // ── partial combs (top) ──────────────────────────────────────────────────
  const FMAX = 6.6;
  const pPad = 110;
  const pW = width - 2 * pPad;
  const px = (f: number) => pPad + ((f - 1) / (FMAX - 1)) * pW;
  const axisY = 560;
  const lenOf = (k: number) => 150 * Math.pow(1 / k, 0.6);
  const lower = Array.from({ length: NP }, (_, i) => i + 1);
  const upper = Array.from({ length: NP }, (_, i) => i + 1)
    .map((k) => ({ k, f: k * alpha }))
    .filter((u) => u.f <= FMAX + 0.05);
  // a roughness pair: an upper partial within a critical band of a lower one
  const rough = (f: number) => {
    let r = 0;
    for (let m = 1; m <= NP; m++) {
      const d = Math.abs(f - m);
      if (d > 0.005 && d < 0.22) r = Math.max(r, 1 - d / 0.22);
    }
    return r;
  };

  // ── dissonance-curve plot ────────────────────────────────────────────────
  const plotPad = 96;
  const plotW = width - 2 * plotPad;
  const cBot = 1330, cTop = 800;
  const cx = (a: number) => plotPad + (a - 1) * plotW;
  const cy = (d: number) => cBot - d * (cBot - cTop);
  let path = "";
  for (let i = 0; i < CURVE.length; i++) {
    path += `${i === 0 ? "M" : "L"} ${cx(1 + i / (CURVE.length - 1)).toFixed(1)} ${cy(CURVE[i]).toFixed(1)} `;
  }
  const draw = evolvePath(swp, path);

  const introFade = interpolate(frame, [0, 16], [0, 1], { extrapolateRight: "clamp" });
  const curColor = interpolateColors(diss, [0, 0.4, 1], [TEAL, GOLD, HOT]);
  const peakAlpha = 1 + CURVE.indexOf(Math.max(...CURVE)) / (CURVE.length - 1);

  return (
    <AbsoluteFill style={{ opacity: introFade, backgroundColor: "#06070e" }}>
      <Audio src={staticFile("audio/sethares.wav")} />
      <Backdrop />

      <div style={{ position: "absolute", top: 110, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 48, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
        where does it get <b style={{ fontWeight: 800 }}>rough</b>?
      </div>
      <div style={{ position: "absolute", top: 186, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 26, letterSpacing: 2, color: theme.muted }}>
        two complex tones · Plomp-Levelt roughness
      </div>

      <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
        {/* partial combs */}
        <line x1={pPad} y1={axisY} x2={width - pPad} y2={axisY} stroke="rgba(180,200,230,0.16)" strokeWidth={2} />
        {lower.map((k) => (
          <line key={`l${k}`} x1={px(k)} y1={axisY} x2={px(k)} y2={axisY - lenOf(k)}
            stroke={TEAL} strokeWidth={k === 1 ? 6 : 4} strokeLinecap="round" opacity={0.4 + 0.4 / k} />
        ))}
        {upper.map((u) => {
          const ro = rough(u.f);
          const col = interpolateColors(ro, [0, 1], [GOLD, HOT]);
          return (
            <g key={`u${u.k}`}>
              {ro > 0.05 && (
                <circle cx={px(u.f)} cy={axisY} r={8 + 22 * ro} fill={HOT} opacity={0.28 * ro} />
              )}
              <line x1={px(u.f)} y1={axisY} x2={px(u.f)} y2={axisY + lenOf(u.k)}
                stroke={col} strokeWidth={u.k === 1 ? 6 : 4} strokeLinecap="round" opacity={0.45 + 0.4 / u.k} />
            </g>
          );
        })}
        <text x={px(1) + 14} y={axisY - lenOf(1) - 12} fill={TEAL} fontSize={24} fontFamily="monospace" opacity={0.8}>lower tone</text>
        <text x={px(1) + 14} y={axisY + lenOf(1) + 36} fill={GOLD} fontSize={24} fontFamily="monospace" opacity={0.8}>upper tone (sweeping)</text>

        {/* dissonance curve */}
        <text x={plotPad} y={cTop - 26} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>
          sensory dissonance · biotuner dissmeasure
        </text>
        <line x1={plotPad} y1={cBot} x2={width - plotPad} y2={cBot} stroke="rgba(180,200,230,0.18)" strokeWidth={2} />
        {/* peak marker */}
        <g opacity={0.6}>
          <line x1={cx(peakAlpha)} y1={cy(1)} x2={cx(peakAlpha)} y2={cBot} stroke={HOT} strokeWidth={1.5} strokeDasharray="3 6" />
          <text x={cx(peakAlpha) + 10} y={cy(1) + 6} fill={HOT} fontSize={22} fontFamily="monospace">roughest</text>
        </g>
        {/* valleys */}
        {VALLEYS.map((v) => {
          const near = Math.abs(alpha - v.alpha) < 0.02;
          return (
            <g key={v.label}>
              <circle cx={cx(v.alpha)} cy={cy(v.diss_norm)} r={near ? 10 : 5}
                fill={GOLD} opacity={near ? 1 : 0.55}
                style={near ? { filter: `drop-shadow(0 0 12px ${GOLD})` } : undefined} />
              <text x={cx(v.alpha)} y={cBot + 36} fill={near ? GOLD : theme.muted} fontSize={near ? 26 : 20}
                fontFamily="monospace" textAnchor="middle" opacity={near ? 1 : 0.6}>{v.label}</text>
            </g>
          );
        })}
        {/* the curve, traced as it sweeps */}
        <path d={path} fill="none" stroke="rgba(150,175,215,0.35)" strokeWidth={2} />
        <path d={path} fill="none" stroke={curColor} strokeWidth={4}
          strokeDasharray={draw.strokeDasharray} strokeDashoffset={draw.strokeDashoffset}
          style={{ filter: `drop-shadow(0 0 8px ${curColor})` }} />
        {/* sweeping marker */}
        <line x1={cx(alpha)} y1={cy(diss)} x2={cx(alpha)} y2={cBot} stroke={curColor} strokeWidth={1.5} opacity={0.4} />
        <circle cx={cx(alpha)} cy={cy(diss)} r={12} fill={curColor}
          style={{ filter: `drop-shadow(0 0 12px ${curColor})` }} />
      </svg>

      {/* roughness readout */}
      <div style={{ position: "absolute", bottom: 150, left: 96, right: 96, display: "flex",
        justifyContent: "space-between", alignItems: "baseline", fontFamily: fonts.mono,
        color: theme.muted, fontSize: 24, letterSpacing: 2 }}>
        <span>roughness</span>
        <span style={{ color: curColor, fontSize: 44, fontWeight: 700 }}>{Math.round(diss * 100)}</span>
      </div>

      <div style={{ position: "absolute", bottom: 80, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
        biotuner · sensory dissonance
      </div>
    </AbsoluteFill>
  );
};
