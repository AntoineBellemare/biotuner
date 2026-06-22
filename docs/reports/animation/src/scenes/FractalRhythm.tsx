import React from "react";
import {
  AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame, useVideoConfig,
  interpolate, spring,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";
import data from "../../public/fractal.json";

/**
 * "Fractal Rhythm" — biotuner's second_order_polyrhythm. The multi-voice
 * coincidences of a polyrhythm (gold) form a new rhythm; its IOI ratios seed a
 * fresh polyrhythm — drawn as an inner wheel. Rhythm nested inside rhythm.
 */
type Voice = { pulses: number; onsets: number[] };
type Order = { order: number; pulse_counts: number[]; poly_label: string;
  voices: Voice[]; coincidences: { t: number; n: number }[] };
const ORDERS = data.orders as Order[];
const O1 = ORDERS[0], O2 = ORDERS[1];
const TEAL = "#6fd6c4", GOLD = "#f2c14e", CORAL = "#e8746a";

const TITLE = 84;
const BUILD = 70;
const CYCLE = 96;
const NCYC = 5;
const TAIL = 44;
const PLAY_START = BUILD;
export const TOTAL_FRACTAL = TITLE + PLAY_START + NCYC * CYCLE + TAIL;

const CX = 540, CY = 1000;
const clampOpt = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
const rOf1 = (i: number) => 200 + i * (244 / (O1.voices.length - 1));
const rOf2 = (i: number) => 72 + i * (96 / Math.max(1, O2.voices.length - 1));
const RC = 472;

export const FractalRhythm: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();
  const sf = frame - TITLE;
  const introFade = interpolate(sf, [0, 16], [0, 1], clampOpt);

  const pf = sf - PLAY_START;
  const playing = pf >= 0;
  const cyc = playing ? pf / CYCLE : 0;
  const cyclePos = playing ? (pf % CYCLE) / CYCLE : 0;
  const o2op = interpolate(cyc, [1.8, 2.7], [0, 1], clampOpt);
  const o1dim = interpolate(cyc, [1.8, 2.7], [1, 0.45], clampOpt);

  const ang = (frac: number) => -Math.PI / 2 + 2 * Math.PI * frac;
  const pt = (r: number, frac: number): [number, number] => [CX + r * Math.cos(ang(frac)), CY + r * Math.sin(ang(frac))];
  const build = spring({ frame: sf, fps, config: { damping: 16, stiffness: 70 } });
  const fireOf = (onset: number) => playing ? Math.exp(-((((cyclePos - onset) % 1) + 1) % 1) * 8) : 0;

  const [phx, phy] = pt(RC + 22, cyclePos);

  return (
    <AbsoluteFill style={{ backgroundColor: "#06070e" }}>
      <Sequence from={TITLE}><Audio src={staticFile("audio/fractal.wav")} /></Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE} title="Fractal Rhythm" hook="rhythm inside rhythm inside rhythm" />

      <AbsoluteFill style={{ opacity: introFade }}>
        <div style={{ position: "absolute", top: 96, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 44, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
          a polyrhythm <b style={{ fontWeight: 800 }}>inside</b> a polyrhythm
        </div>
        <div style={{ position: "absolute", top: 168, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 2, color: theme.muted }}>
          order 1: {O1.poly_label}  →  coincidences  →  order 2: {O2.poly_label}
        </div>

        <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
          {/* ── ORDER 1 — outer clockwork ── */}
          <g opacity={o1dim}>
            {O1.voices.map((v, i) => {
              const r = rOf1(i), rev = Math.min(1, build * 1.2 - i * 0.06);
              if (rev <= 0) return null;
              return (
                <g key={i} opacity={rev}>
                  <circle cx={CX} cy={CY} r={r} fill="none" stroke={TEAL} strokeWidth={1.2} opacity={0.16} />
                  {v.onsets.map((o, j) => {
                    const [bx, by] = pt(r, o), fire = fireOf(o);
                    return <circle key={j} cx={bx} cy={by} r={3.5 + 5 * fire} fill={TEAL}
                      opacity={0.45 + 0.55 * fire}
                      style={fire > 0.3 ? { filter: `drop-shadow(0 0 ${7 * fire}px ${TEAL})` } : undefined} />;
                  })}
                </g>
              );
            })}
            {/* coincidence ring — the meta-rhythm */}
            <circle cx={CX} cy={CY} r={RC} fill="none" stroke={GOLD} strokeWidth={1} opacity={0.18 * build} />
            {O1.coincidences.map((c, i) => {
              const [bx, by] = pt(RC, c.t), fire = fireOf(c.t);
              return (
                <g key={`c${i}`} opacity={build}>
                  {fire > 0.3 && <circle cx={bx} cy={by} r={12 + 26 * (1 - fire)} fill="none" stroke={GOLD} strokeWidth={2} opacity={0.7 * fire} />}
                  {o2op > 0.05 && <line x1={bx} y1={by} x2={CX} y2={CY} stroke={GOLD} strokeWidth={1} opacity={0.18 * o2op} />}
                  <circle cx={bx} cy={by} r={6 + 7 * fire} fill={GOLD}
                    style={{ filter: `drop-shadow(0 0 ${8 + 12 * fire}px ${GOLD})` }} />
                </g>
              );
            })}
          </g>

          {/* ── ORDER 2 — inner wheel seeded by the coincidences ── */}
          {o2op > 0.02 && (
            <g opacity={o2op}>
              {O2.voices.map((v, i) => {
                const r = rOf2(i);
                return (
                  <g key={i}>
                    <circle cx={CX} cy={CY} r={r} fill="none" stroke={CORAL} strokeWidth={1.2} opacity={0.22} />
                    {v.onsets.map((o, j) => {
                      const [bx, by] = pt(r, o), fire = fireOf(o);
                      return <circle key={j} cx={bx} cy={by} r={4 + 6 * fire} fill={CORAL}
                        opacity={0.5 + 0.5 * fire}
                        style={fire > 0.3 ? { filter: `drop-shadow(0 0 ${9 * fire}px ${CORAL})` } : undefined} />;
                    })}
                  </g>
                );
              })}
            </g>
          )}

          {/* playhead */}
          {playing && <>
            {Array.from({ length: 8 }, (_, i) => i).map((i) => {
              const [tx, ty] = pt(RC + 22, cyclePos - i * 0.013);
              return <line key={i} x1={CX} y1={CY} x2={tx} y2={ty} stroke="#fff" strokeWidth={2.2 - i * 0.2} opacity={0.45 - i * 0.05} />;
            })}
            <circle cx={phx} cy={phy} r={6} fill="#fff" style={{ filter: "drop-shadow(0 0 8px #fff)" }} />
          </>}
          <circle cx={CX} cy={CY} r={6} fill="#fff" opacity={0.5} />
        </svg>

        <div style={{ position: "absolute", bottom: 150, left: 60, right: 60, textAlign: "center",
          fontFamily: fonts.display, fontSize: 28, fontWeight: 300, color: theme.muted }}>
          {cyc < 2
            ? <>the <b style={{ color: GOLD, fontWeight: 700 }}>coincidences</b> of a polyrhythm form a new rhythm</>
            : <>which seeds a <b style={{ color: CORAL, fontWeight: 700 }}>polyrhythm inside the polyrhythm</b></>}
        </div>
        <div style={{ position: "absolute", bottom: 78, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
          biotuner · rhythm · second_order_polyrhythm
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
