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
  spring,
  Easing,
} from "remotion";
import { noise2D } from "@remotion/noise";
import { evolvePath } from "@remotion/paths";
import { interpolateStyles } from "@remotion/animation-utils";
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";
import data from "../../public/harmonicity.json";

/**
 * "Do their harmonics agree?" — a didactic scene. A root tone and a sliding
 * tone each draw a harmonic comb (teeth at f, 2f, 3f…). As the upper tone
 * parks on the just-intonation intervals, its teeth align with the root's and
 * BLOOM; a live meter reads biotuner's real dyad_similarity at each stop.
 */
const STOPS = data.stops;
const CURVE = data.curve as number[];
const NH = data.n_harmonics; // teeth per comb
const FMAX = 6.6; // frequency-axis upper bound (× root)
const TEAL = "#6fd6c4";
const GOLD = "#f2c14e";

const TITLE = 84;
const INTRO = 36;
const APPROACH = 18;
const DWELL = 58;
const BEAT = APPROACH + DWELL;
export const TOTAL_HARMSIM = TITLE + INTRO + STOPS.length * BEAT + 56;

function simAt(r: number): number {
  const t = (r - 1) / (2 - 1);
  const i = Math.max(0, Math.min(CURVE.length - 1, Math.round(t * (CURVE.length - 1))));
  return CURVE[i];
}

export const HarmonicSimilarity: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();

  // ── ratio over time: glide to each stop, then dwell ──────────────────────
  const sf = frame - TITLE; // scene frame (content begins after the title card)
  const local = sf - INTRO;
  const beat = Math.max(0, Math.min(STOPS.length - 1, Math.floor(local / BEAT)));
  const beatLocal = local - beat * BEAT;
  const fromR = beat === 0 ? STOPS[0].ratio : STOPS[beat - 1].ratio;
  const toR = STOPS[beat].ratio;
  const move = interpolate(beatLocal, [0, APPROACH], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
    easing: Easing.bezier(0.5, 0, 0.15, 1),
  });
  const r = local < 0 ? STOPS[0].ratio : fromR + (toR - fromR) * move;
  const sim = simAt(r);
  const parked = beatLocal >= APPROACH && local >= 0;
  // spring that fires on each park (drives blooms + meter snap)
  const parkSpring = spring({
    frame: beatLocal - APPROACH, fps, config: { damping: 12, stiffness: 120 },
  });
  const stop = STOPS[Math.max(0, beat)];

  // ── geometry ─────────────────────────────────────────────────────────────
  const axisY = 940;
  const leftPad = 96;
  const axisW = width - 2 * leftPad;
  const xOf = (f: number) => leftPad + ((f - 1) / (FMAX - 1)) * axisW;
  const maxLen = 300;
  const lenOf = (k: number) => maxLen * Math.pow(1 / k, 0.7);

  const rootTeeth = Array.from({ length: NH }, (_, i) => i + 1);
  const slideTeeth = Array.from({ length: NH }, (_, i) => i + 1)
    .map((k) => ({ k, f: k * r, len: lenOf(k) }))
    .filter((s) => s.f <= FMAX + 0.05);
  // a slide tooth that lands on an integer ≤ NH coincides with a root tooth
  const alignOf = (f: number) => {
    const m = Math.round(f);
    return Math.abs(f - m) < 0.05 && m >= 1 && m <= NH ? m : null;
  };

  // ── consonance-curve backdrop (real dyad_similarity), drawn on at intro ──
  const sx = (rr: number) => leftPad + (rr - 1) * axisW;
  const sBot = 1430, sTop = 1230;
  const sy = (s: number) => sBot - (s / 100) * (sBot - sTop);
  const STEP = 4;
  let curvePath = "";
  for (let i = 0; i < CURVE.length; i += STEP) {
    const rr = 1 + i / (CURVE.length - 1);
    curvePath += `${i === 0 ? "M" : "L"} ${sx(rr).toFixed(1)} ${sy(CURVE[i]).toFixed(1)} `;
  }
  const draw = evolvePath(
    interpolate(sf, [4, 34], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }),
    curvePath
  );

  const introFade = interpolate(sf, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const meterFill = (parked ? sim : sim) / 100;

  return (
    <AbsoluteFill style={{ backgroundColor: "#05070e" }}>
      <Sequence from={TITLE}>
        <Audio src={staticFile("audio/harmonicity.wav")} />
      </Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE}
        title="Harmonic Similarity" hook="when do two tones share the same overtones?" />

      <AbsoluteFill style={{ opacity: introFade }}>
      {/* Title + live interval label */}
      <div style={{ position: "absolute", top: 116, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 50, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
        do their <b style={{ fontWeight: 800 }}>harmonics</b> agree?
      </div>
      <div style={{ position: "absolute", top: 196, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 40, letterSpacing: 4, color: GOLD,
        textShadow: `0 0 ${10 + 26 * parkSpring}px ${GOLD}`,
        transform: `scale(${1 + 0.06 * parkSpring})` }}>
        {stop.num} : {stop.den}
        <span style={{ color: theme.muted, fontSize: 26, marginLeft: 18, letterSpacing: 2 }}>
          {stop.label}
        </span>
      </div>

      <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`}
        style={{ position: "absolute", inset: 0 }}>
        {/* frequency axis */}
        <line x1={leftPad} y1={axisY} x2={width - leftPad} y2={axisY}
          stroke="rgba(180,200,230,0.18)" strokeWidth={2} />

        {/* ROOT comb — teeth up (teal) */}
        {rootTeeth.map((k) => {
          const x = xOf(k);
          const len = lenOf(k);
          return <line key={`r${k}`} x1={x} y1={axisY} x2={x} y2={axisY - len}
            stroke={TEAL} strokeWidth={k === 1 ? 6 : 4} strokeLinecap="round"
            opacity={0.34 + 0.4 / k} />;
        })}

        {/* SLIDE comb — teeth down (gold), shimmer via noise */}
        {slideTeeth.map((s) => {
          const x = xOf(s.f);
          const wob = noise2D("h", s.k * 1.7, sf * 0.03) * 5;
          const m = alignOf(s.f);
          return <line key={`s${s.k}`} x1={x} y1={axisY} x2={x} y2={axisY + s.len + wob}
            stroke={m ? GOLD : "#caa24a"} strokeWidth={s.k === 1 ? 6 : 4} strokeLinecap="round"
            opacity={m ? 0.95 : 0.36 + 0.34 / s.k} />;
        })}

        {/* ALIGNMENT blooms — where a slide tooth meets a root tooth */}
        {slideTeeth.map((s) => {
          const m = alignOf(s.f);
          if (!m) return null;
          const x = xOf(m);
          const len = lenOf(m);
          const st = interpolateStyles(parkSpring, [0, 1],
            [{ opacity: 0.2, r: 10 }, { opacity: 0.9, r: 26 }]) as { opacity: number; r: number };
          return (
            <g key={`a${s.k}`}>
              <line x1={x} y1={axisY - len} x2={x} y2={axisY + len}
                stroke="#fff3d6" strokeWidth={3} opacity={0.35 + 0.5 * parkSpring} />
              <circle cx={x} cy={axisY} r={st.r} fill={GOLD} opacity={st.opacity * 0.5} />
              <circle cx={x} cy={axisY} r={st.r * 0.45} fill="#fff7e6" opacity={st.opacity} />
            </g>
          );
        })}

        {/* labels for the two combs */}
        <text x={xOf(1) + 16} y={axisY - lenOf(1) - 16} fill={TEAL} fontSize={26}
          fontFamily="monospace" opacity={0.8}>root</text>
        <text x={xOf(1) + 16} y={axisY + lenOf(1) + 40} fill={GOLD} fontSize={26}
          fontFamily="monospace" opacity={0.8}>+ interval</text>

        {/* consonance landscape (real dyad_similarity), drawn on */}
        <path d={curvePath} fill="none" stroke="rgba(150,175,215,0.5)" strokeWidth={2.5}
          strokeDasharray={draw.strokeDasharray} strokeDashoffset={draw.strokeDashoffset} />
        {STOPS.map((s) => (
          <g key={s.label} opacity={0.7}>
            <circle cx={sx(s.ratio)} cy={sy(s.sim)} r={4} fill="rgba(180,200,230,0.6)" />
          </g>
        ))}
        {/* moving marker */}
        <circle cx={sx(r)} cy={sy(sim)} r={9 + 5 * parkSpring} fill={GOLD}
          opacity={0.9} style={{ filter: `drop-shadow(0 0 10px ${GOLD})` }} />
        <text x={leftPad} y={sTop - 22} fill={theme.muted} fontSize={22}
          fontFamily="monospace" letterSpacing={2}>consonance landscape · dyad_similarity</text>
      </svg>

      {/* Similarity meter */}
      <div style={{ position: "absolute", bottom: 250, left: 96, right: 96 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline",
          fontFamily: fonts.mono, color: theme.muted, fontSize: 24, letterSpacing: 2, marginBottom: 14 }}>
          <span>harmonic similarity</span>
          <span style={{ color: GOLD, fontSize: 44, fontWeight: 700 }}>{Math.round(sim)}</span>
        </div>
        <div style={{ height: 22, borderRadius: 11, background: "rgba(180,200,230,0.12)", overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${Math.max(0, Math.min(1, meterFill)) * 100}%`,
            borderRadius: 11,
            background: interpolateColors(sim, [0, 50, 100], ["#5a6b8a", "#caa24a", GOLD]),
            boxShadow: `0 0 ${10 + 30 * meterFill}px ${GOLD}` }} />
        </div>
      </div>

      <div style={{ position: "absolute", bottom: 80, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
        biotuner · harmonic geometry
      </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
