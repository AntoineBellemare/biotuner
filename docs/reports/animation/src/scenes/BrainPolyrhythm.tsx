import React from "react";
import {
  AbsoluteFill,
  Audio,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  Easing,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";
import data from "../../public/brain_polyrhythm.json";

/**
 * "Brain Polyrhythm" — biotuner's rhythm module on EEG. Spectral peaks → ratios
 * → scale2polyrhythm_continuous → a 5:7:11 polyrhythm. The reel moves through
 * three visual treatments of the same rhythm:
 *   ① necklace build — pulses fan out to even (Euclidean) positions
 *   ② phase wheel    — a radar playhead sweeps, beads fire, unison at the downbeat
 *   ③ ripple pond    — every pulse drops a colored ring; coincidences burst white
 */
type Voice = { pulses: number; onsets: number[]; ioi_ms: number };
const VOICES = data.voices as Voice[];
const PEAKS = data.peaks as number[];
const POLY = data.poly_label as string;
const COLORS = ["#6fd6c4", "#f2c14e", "#e8746a", "#9b8cff"];

const TITLE = 84;
const INTRO = 18;
const NECK = 160;          // necklace build
const CYCLE = 84;
const NCYC = 6;
const PLAY = NCYC * CYCLE;
const TAIL = 44;
const PLAY_START = INTRO + NECK;
export const TOTAL_BRAINPOLY = TITLE + PLAY_START + PLAY + TAIL;

const CX = 540, CY = 1030;
const R = [250, 340, 432];
const RIPPLE_LIFE = 78, RIPPLE_SPEED = 8.2;

export const BrainPolyrhythm: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();

  const sf = frame - TITLE;
  const introFade = interpolate(sf, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const pf = sf - PLAY_START;             // play frame (≥0 once the wheel plays)
  const playing = pf >= 0;
  const cyc = playing ? pf / CYCLE : 0;   // fractional cycle index
  const cyclePos = playing ? (pf % CYCLE) / CYCLE : 0;

  // movement blend: wheel → ripple around cycle 3
  const wheelOp = interpolate(cyc, [2.7, 3.35], [1, 0.14], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const rippleOp = interpolate(cyc, [2.6, 3.3], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const intensity = interpolate(cyc, [0, 5.5], [0.4, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const ang = (frac: number) => -Math.PI / 2 + 2 * Math.PI * frac;
  const pt = (r: number, frac: number): [number, number] => [CX + r * Math.cos(ang(frac)), CY + r * Math.sin(ang(frac))];

  // per-voice reveal + bead spread (clustered → even) during the necklace build
  const revealOf = (vi: number) => spring({ frame: sf - INTRO - vi * 44, fps, config: { damping: 14, stiffness: 80 } });

  const playheadFrac = cyclePos;
  const [hx, hy] = pt(R[VOICES.length - 1] + 26, playheadFrac);
  const downPulse = playing ? Math.exp(-((cyclePos + 1) % 1) * 7) + Math.exp(-cyclePos * 7) : 0;

  // collect live ripples (movement ③)
  const ripples: { age: number; vi: number; coincide: boolean }[] = [];
  if (playing) {
    const cNow = Math.floor(pf / CYCLE);
    for (let c = cNow - 2; c <= cNow; c++) {
      if (c < 0) continue;
      VOICES.forEach((v, vi) =>
        v.onsets.forEach((o) => {
          const age = pf - (c + o) * CYCLE;
          if (age >= 0 && age <= RIPPLE_LIFE) ripples.push({ age, vi, coincide: o === 0 });
        })
      );
    }
  }

  return (
    <AbsoluteFill style={{ backgroundColor: "#06070e" }}>
      <Sequence from={TITLE}>
        <Audio src={staticFile("audio/brain_polyrhythm.wav")} />
      </Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE}
        title="Brain Polyrhythm" hook="your brainwaves, as 5 : 7 : 11" />

      <AbsoluteFill style={{ opacity: introFade }}>
        <div style={{ position: "absolute", top: 92, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 46, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
          the rhythm in your <b style={{ fontWeight: 800 }}>brainwaves</b>
        </div>
        <div style={{ position: "absolute", top: 162, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 24, letterSpacing: 2, color: theme.muted }}>
          EEG peaks {PEAKS.join(" · ")} Hz  →  <b style={{ color: "#fff" }}>{POLY}</b>
        </div>

        <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
          {/* ── ring guides (present throughout) ── */}
          {VOICES.map((v, vi) => (
            <circle key={`ring${vi}`} cx={CX} cy={CY} r={R[vi]} fill="none" stroke={COLORS[vi]}
              strokeWidth={2} opacity={0.28 * Math.min(1, revealOf(vi))} />
          ))}

          {/* ── ③ ripple pond ── */}
          {rippleOp > 0.01 && ripples.map((rp, i) => {
            const rad = rp.age * RIPPLE_SPEED;
            const fade = (1 - rp.age / RIPPLE_LIFE);
            const col = rp.coincide ? "#fff" : COLORS[rp.vi];
            return (
              <circle key={`rip${i}`} cx={CX} cy={CY} r={rad} fill="none" stroke={col}
                strokeWidth={rp.coincide ? 3.5 : 2.2} opacity={rippleOp * fade * (rp.coincide ? 0.9 : 0.6)}
                style={rp.coincide ? { filter: "drop-shadow(0 0 10px #fff)" } : undefined} />
            );
          })}

          {/* ── ②+① wheel: beads + radar playhead ── */}
          <g opacity={wheelOp}>
            {VOICES.map((v, vi) => {
              const col = COLORS[vi % COLORS.length];
              const rev = Math.min(1, revealOf(vi));
              const r = R[vi];
              return (
                <g key={vi}>
                  {v.onsets.map((onset, j) => {
                    // spread: beads fan from the top (clustered) to even positions
                    const spread = Easing.out(Easing.cubic)(rev);
                    const frac = onset * spread;
                    const [bx, by] = pt(r, frac);
                    const rel = playing ? (cyclePos - frac + 1) % 1 : 1;
                    const fire = playing ? Math.exp(-rel * 7) : 0;
                    const beadIn = Math.min(1, Math.max(0, rev * 1.3 - j * 0.1));
                    const rad = (7 + (10 + 6 * intensity) * fire) * beadIn;
                    return (
                      <g key={j}>
                        {fire > 0.4 && (
                          <circle cx={bx} cy={by} r={14 + 26 * (1 - fire)} fill="none" stroke={col}
                            strokeWidth={2} opacity={0.5 * fire} />
                        )}
                        <circle cx={bx} cy={by} r={rad} fill={col} opacity={(0.55 + 0.45 * fire) * beadIn}
                          style={fire > 0.1 ? { filter: `drop-shadow(0 0 ${12 * fire}px ${col})` } : undefined} />
                      </g>
                    );
                  })}
                  <text x={CX} y={CY - r - 14} fill={col} fontSize={22} fontFamily="monospace"
                    textAnchor="middle" opacity={rev * 0.9}>{v.pulses}</text>
                </g>
              );
            })}

            {/* radar playhead with a fading sweep trail */}
            {playing && (
              <g>
                {Array.from({ length: 10 }, (_, i) => i).map((i) => {
                  const [tx, ty] = pt(R[2] + 26, playheadFrac - i * 0.014);
                  return <line key={i} x1={CX} y1={CY} x2={tx} y2={ty} stroke="#fff"
                    strokeWidth={2.5 - i * 0.18} opacity={(0.5 - i * 0.045) * (0.6 + 0.4 * intensity)} />;
                })}
                <circle cx={hx} cy={hy} r={7} fill="#fff" style={{ filter: "drop-shadow(0 0 10px #fff)" }} />
              </g>
            )}
          </g>

          {/* downbeat shockwave + core (both movements) */}
          {playing && downPulse > 0.04 && (
            <circle cx={CX} cy={CY} r={18 + 150 * (1 - downPulse)} fill="none" stroke="#fff"
              strokeWidth={2.5} opacity={0.45 * downPulse} />
          )}
          <circle cx={CX} cy={CY} r={8 + 8 * downPulse} fill="#fff" opacity={0.5 + 0.5 * downPulse}
            style={{ filter: `drop-shadow(0 0 ${8 + 24 * downPulse}px #fff)` }} />
        </svg>

        <div style={{ position: "absolute", bottom: 150, left: 60, right: 60, textAlign: "center",
          fontFamily: fonts.display, fontSize: 28, fontWeight: 300, color: theme.muted }}>
          {rippleOp > 0.5
            ? <>every pulse drops a ring · they <b style={{ color: "#fff", fontWeight: 700 }}>collide</b> at the downbeat</>
            : <>each peak → an evenly-pulsed voice · reuniting at the <b style={{ color: "#fff", fontWeight: 700 }}>downbeat</b></>}
        </div>
        <div style={{ position: "absolute", bottom: 78, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
          biotuner · rhythm · scale2polyrhythm
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
