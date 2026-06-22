import React from "react";
import {
  AbsoluteFill,
  Audio,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  interpolateColors,
  Easing,
} from "remotion";
import { evolvePath } from "@remotion/paths";
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";
import data from "../../public/sethares.json";

/**
 * "Where does it get rough?" — the Sethares / Plomp-Levelt sensory-dissonance
 * curve, shown for THREE timbres against a fixed just-intonation grid. A simple
 * tone has few consonant valleys; a rich harmonic tone has many, exactly on the
 * grid; a STRETCHED (inharmonic) tone's valleys slide OFF the grid — the scale
 * follows the timbre. Curve = biotuner's real dissmeasure.
 */
const TIMBRES = data.timbres as {
  label: string; partials: number[]; amps: number[];
  curve: number[]; valleys: { alpha: number; label: string; on_grid: boolean; diss: number }[];
  peak_alpha: number;
}[];
const GRID = data.grid as { num: number; den: number; ratio: number }[];
const TEAL = "#6fd6c4";
const HOT = "#e8746a";
const GOLD = "#f2c14e";

const TITLE = 84;
const INTRO = 24;
const SWEEP = 150;
const HOLD = 70;
const BEAT = SWEEP + HOLD;
const OUTRO = 34;
export const TOTAL_SETHARES = TITLE + INTRO + TIMBRES.length * BEAT + OUTRO;

export const SetharesDissonance: React.FC = () => {
  const frame = useCurrentFrame();
  const { width } = useVideoConfig();

  const sf = frame - TITLE; // scene frame (content begins after the title card)
  const local = sf - INTRO;
  const ti = Math.max(0, Math.min(TIMBRES.length - 1, Math.floor(local / BEAT)));
  const beatLocal = local - ti * BEAT;
  const T = TIMBRES[ti];
  const CURVE = T.curve;
  const onGridCount = T.valleys.filter((v) => v.on_grid).length;

  const swp = interpolate(beatLocal, [0, SWEEP], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.sin),
  });
  const alpha = local < 0 ? 1 : 1 + swp;
  const dissAt = (a: number) =>
    CURVE[Math.max(0, Math.min(CURVE.length - 1, Math.round((a - 1) * (CURVE.length - 1))))];
  const diss = dissAt(alpha);
  const curColor = interpolateColors(diss, [0, 0.4, 1], [TEAL, GOLD, HOT]);

  // ── partial comb (top) — shows the timbre's spectrum (even vs stretched) ──
  const FMAX = 7.2;
  const pPad = 110, pW = width - 2 * pPad;
  const px = (f: number) => pPad + ((f - 1) / (FMAX - 1)) * pW;
  const axisY = 540;
  const lenOf = (i: number) => 140 * Math.pow(0.86, i);

  // ── dissonance plot ───────────────────────────────────────────────────────
  const plotPad = 96, plotW = width - 2 * plotPad;
  const cBot = 1330, cTop = 800;
  const cx = (a: number) => plotPad + (a - 1) * plotW;
  const cy = (d: number) => cBot - d * (cBot - cTop);
  let path = "";
  for (let i = 0; i < CURVE.length; i++)
    path += `${i === 0 ? "M" : "L"} ${cx(1 + i / (CURVE.length - 1)).toFixed(1)} ${cy(CURVE[i]).toFixed(1)} `;
  const draw = evolvePath(swp, path);

  const introFade = interpolate(sf, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const settled = beatLocal >= SWEEP;
  const verdict = ti === 0
    ? "few partials → few consonances"
    : T.label.startsWith("stretched")
    ? "valleys slide OFF the grid — the scale follows the timbre"
    : "valleys land ON the just grid";

  return (
    <AbsoluteFill style={{ backgroundColor: "#06070e" }}>
      <Sequence from={TITLE}>
        <Audio src={staticFile("audio/sethares.wav")} />
      </Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE}
        title="Dissonance Curve" hook="where do the smooth intervals fall?" />

      <AbsoluteFill style={{ opacity: introFade }}>
      <div style={{ position: "absolute", top: 108, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 46, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
        the consonant scale <b style={{ fontWeight: 800 }}>follows the timbre</b>
      </div>
      <div style={{ position: "absolute", top: 182, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 30, letterSpacing: 2, color: curColor }}>
        {T.label}
      </div>

      <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
        {/* partial comb — the timbre's spectrum */}
        <line x1={pPad} y1={axisY} x2={width - pPad} y2={axisY} stroke="rgba(180,200,230,0.16)" strokeWidth={2} />
        {T.partials.map((p, i) => p <= FMAX && (
          <line key={`l${i}`} x1={px(p)} y1={axisY} x2={px(p)} y2={axisY - lenOf(i)}
            stroke={TEAL} strokeWidth={i === 0 ? 6 : 4} strokeLinecap="round" opacity={0.4 + 0.35 * T.amps[i]} />
        ))}
        {T.partials.map((p, i) => p * alpha <= FMAX && (
          <line key={`u${i}`} x1={px(p * alpha)} y1={axisY} x2={px(p * alpha)} y2={axisY + lenOf(i)}
            stroke={GOLD} strokeWidth={i === 0 ? 6 : 4} strokeLinecap="round" opacity={0.4 + 0.35 * T.amps[i]} />
        ))}
        <text x={pPad} y={axisY - 150} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>
          {ti === 2 ? "partials stretched (inharmonic)" : "partials at 1 : 2 : 3 …  (harmonic)"}
        </text>

        {/* just-intonation reference grid (fixed across all timbres) */}
        {GRID.map((g) => (
          <g key={`${g.num}/${g.den}`}>
            <line x1={cx(g.ratio)} y1={cTop - 10} x2={cx(g.ratio)} y2={cBot}
              stroke="rgba(150,175,215,0.22)" strokeWidth={1.5} strokeDasharray="2 7" />
            <text x={cx(g.ratio)} y={cBot + 34} fill="rgba(160,185,225,0.6)" fontSize={20}
              fontFamily="monospace" textAnchor="middle">{g.num}/{g.den}</text>
          </g>
        ))}
        <text x={plotPad} y={cTop - 24} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>
          sensory dissonance · biotuner dissmeasure
        </text>
        <line x1={plotPad} y1={cBot} x2={width - plotPad} y2={cBot} stroke="rgba(180,200,230,0.18)" strokeWidth={2} />

        {/* the curve, traced as the upper tone sweeps */}
        <path d={path} fill="none" stroke="rgba(150,175,215,0.3)" strokeWidth={2} />
        <path d={path} fill="none" stroke={curColor} strokeWidth={4}
          strokeDasharray={draw.strokeDasharray} strokeDashoffset={draw.strokeDashoffset}
          style={{ filter: `drop-shadow(0 0 8px ${curColor})` }} />

        {/* valleys: gold ON grid, hot OFF grid */}
        {T.valleys.map((v, idx) => {
          const reached = settled || alpha >= v.alpha - 0.005;
          if (!reached) return null;
          const col = v.on_grid ? GOLD : HOT;
          return (
            <g key={idx}>
              <circle cx={cx(v.alpha)} cy={cy(v.diss)} r={v.on_grid ? 10 : 8} fill={col}
                opacity={0.95} style={{ filter: `drop-shadow(0 0 ${v.on_grid ? 14 : 8}px ${col})` }} />
              {!v.on_grid && (
                <line x1={cx(v.alpha)} y1={cy(v.diss)} x2={cx(v.alpha)} y2={cBot}
                  stroke={HOT} strokeWidth={1} opacity={0.35} />
              )}
            </g>
          );
        })}

        {/* sweeping marker */}
        {!settled && (
          <>
            <line x1={cx(alpha)} y1={cy(diss)} x2={cx(alpha)} y2={cBot} stroke={curColor} strokeWidth={1.5} opacity={0.4} />
            <circle cx={cx(alpha)} cy={cy(diss)} r={12} fill={curColor}
              style={{ filter: `drop-shadow(0 0 12px ${curColor})` }} />
          </>
        )}
      </svg>

      {/* verdict */}
      <div style={{ position: "absolute", bottom: 150, left: 60, right: 60, textAlign: "center",
        fontFamily: fonts.display, fontSize: 32, fontWeight: 300, letterSpacing: 0.5,
        color: ti === 2 ? HOT : GOLD, opacity: settled ? 1 : 0.3 }}>
        {verdict}
      </div>

      <div style={{ position: "absolute", bottom: 78, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
        biotuner · sensory dissonance · Sethares
      </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
