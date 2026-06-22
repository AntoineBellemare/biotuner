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
import { evolvePath } from "@remotion/paths";
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";
import data from "../../public/brain_polyrhythm.json";

/**
 * "Brain Polyrhythm" — biotuner's rhythm module on EEG.
 *   ⓘ extract  — EEG signal → spectrum → spectral peaks → ratios → pulse counts
 *   ① necklace — pulses fan out to even (Euclidean) positions
 *   ② wheel    — a radar playhead sweeps, beads fire, unison at the downbeat
 *   ③ ripple   — every pulse drops a colored ring; coincidences burst white
 */
type Voice = { pulses: number; onsets: number[]; ioi_ms: number };
const VOICES = data.voices as Voice[];
const PEAKS = data.peaks as number[];
const AMPS = data.amps as number[];
const RATIOS = data.ratios as number[];
const POLY = data.poly_label as string;
const WAVE = data.wave as number[];
const SPEC_F = data.spec_f as number[];
const SPEC_M = data.spec_mag as number[];
const FMAX = data.fmax as number;
const COLORS = ["#6fd6c4", "#f2c14e", "#e8746a", "#9b8cff"];

const TITLE = 84;
const DIDACTIC = 200;       // EEG → peaks → ratios → pulses
const NECK = 150;
const CYCLE = 84;
const NCYC = 6;
const PLAY = NCYC * CYCLE;
const TAIL = 44;
const PLAY_START = DIDACTIC + NECK;
export const TOTAL_BRAINPOLY = TITLE + PLAY_START + PLAY + TAIL;

const CX = 540, CY = 1030;
const R = [250, 340, 432];
const RIPPLE_LIFE = 78, RIPPLE_SPEED = 8.2;
const PAD = 110, AXW = 1080 - 2 * PAD;

const clampOpt = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

export const BrainPolyrhythm: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();
  const sf = frame - TITLE;
  const introFade = interpolate(sf, [0, 16], [0, 1], clampOpt);

  // ── didactic vs wheel act blend ──
  const didOp = interpolate(sf, [6, 20], [0, 1], clampOpt) * interpolate(sf, [DIDACTIC - 26, DIDACTIC], [1, 0], clampOpt);
  const actOp = interpolate(sf, [DIDACTIC - 12, DIDACTIC + 16], [0, 1], clampOpt);

  // didactic sub-phases
  const waveP = interpolate(sf, [12, 84], [0, 1], { ...clampOpt, easing: Easing.out(Easing.cubic) });
  const specP = interpolate(sf, [74, 150], [0, 1], { ...clampOpt, easing: Easing.out(Easing.cubic) });
  const ratioP = interpolate(sf, [142, 196], [0, 1], { ...clampOpt, easing: Easing.out(Easing.cubic) });

  // ── play clock ──
  const pf = sf - PLAY_START;
  const playing = pf >= 0;
  const cyc = playing ? pf / CYCLE : 0;
  const cyclePos = playing ? (pf % CYCLE) / CYCLE : 0;
  const wheelOp = interpolate(cyc, [2.7, 3.35], [1, 0.14], clampOpt);
  const rippleOp = interpolate(cyc, [2.6, 3.3], [0, 1], clampOpt);
  const intensity = interpolate(cyc, [0, 5.5], [0.4, 1], clampOpt);

  const ang = (frac: number) => -Math.PI / 2 + 2 * Math.PI * frac;
  const pt = (r: number, frac: number): [number, number] => [CX + r * Math.cos(ang(frac)), CY + r * Math.sin(ang(frac))];
  const revealOf = (vi: number) => spring({ frame: sf - DIDACTIC - vi * 44, fps, config: { damping: 14, stiffness: 80 } });
  const playheadFrac = cyclePos;
  const [hx, hy] = pt(R[2] + 26, playheadFrac);
  const downPulse = playing ? Math.exp(-((cyclePos + 1) % 1) * 7) + Math.exp(-cyclePos * 7) : 0;

  // didactic geometry
  const wfx = (i: number) => PAD + (i / (WAVE.length - 1)) * AXW;
  const waveCY = 600, waveAmp = 62;
  const wavePath = WAVE.map((v, i) => `${i === 0 ? "M" : "L"} ${wfx(i).toFixed(1)} ${(waveCY - v * waveAmp).toFixed(1)}`).join(" ");
  const wd = evolvePath(waveP, wavePath);
  const sBase = 1040, sH = 190;
  const sfx = (f: number) => PAD + (f / FMAX) * AXW;
  const sfy = (m: number) => sBase - m * sH;
  const specPath = `M ${PAD} ${sBase} ` + SPEC_F.map((f, i) => `L ${sfx(f).toFixed(1)} ${sfy(SPEC_M[i] * specP).toFixed(1)}`).join(" ") + ` L ${width - PAD} ${sBase} Z`;

  // ripples (movement ③)
  const ripples: { age: number; vi: number; coincide: boolean }[] = [];
  if (playing) {
    const cNow = Math.floor(pf / CYCLE);
    for (let c = cNow - 2; c <= cNow; c++) {
      if (c < 0) continue;
      VOICES.forEach((v, vi) => v.onsets.forEach((o) => {
        const age = pf - (c + o) * CYCLE;
        if (age >= 0 && age <= RIPPLE_LIFE) ripples.push({ age, vi, coincide: o === 0 });
      }));
    }
  }

  return (
    <AbsoluteFill style={{ backgroundColor: "#06070e" }}>
      <Sequence from={TITLE}>
        <Audio src={staticFile("audio/brain_polyrhythm.wav")} />
      </Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE} title="Brain Polyrhythm" hook="your brainwaves, as 5 : 7 : 11" />

      <AbsoluteFill style={{ opacity: introFade }}>
        {/* headers */}
        <div style={{ position: "absolute", top: 96, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 44, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
          {sf < DIDACTIC - 10
            ? <>how we read the brain's <b style={{ fontWeight: 800 }}>rhythm</b></>
            : <>the rhythm in your <b style={{ fontWeight: 800 }}>brainwaves</b></>}
        </div>

        {/* ── ⓘ DIDACTIC: EEG → spectrum → peaks → ratios → pulses ── */}
        <div style={{ opacity: didOp }}>
          <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
            <text x={PAD} y={waveCY - 120} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>1 · EEG signal</text>
            <path d={wavePath} fill="none" stroke={COLORS[0]} strokeWidth={2.5}
              strokeDasharray={wd.strokeDasharray} strokeDashoffset={wd.strokeDashoffset} opacity={0.9} />

            <g opacity={specP}>
              <text x={PAD} y={sBase - sH - 16} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>2 · spectrum → peaks</text>
              <path d={specPath} fill="rgba(120,150,210,0.16)" stroke="rgba(150,180,220,0.5)" strokeWidth={2} />
              <line x1={PAD} y1={sBase} x2={width - PAD} y2={sBase} stroke="rgba(180,200,230,0.18)" strokeWidth={2} />
            </g>
            {PEAKS.map((pkf, i) => {
              const x = sfx(pkf), pop = Math.max(0, Math.min(1, specP * 1.5 - i * 0.12));
              return (
                <g key={i} opacity={pop}>
                  <line x1={x} y1={sBase} x2={x} y2={sfy(AMPS[i])} stroke={COLORS[1]} strokeWidth={3} />
                  <circle cx={x} cy={sfy(AMPS[i])} r={6} fill={COLORS[1]} style={{ filter: `drop-shadow(0 0 7px ${COLORS[1]})` }} />
                  <text x={x} y={sBase + 28} fill={COLORS[1]} fontSize={18} fontFamily="monospace" textAnchor="middle">{pkf}Hz</text>
                </g>
              );
            })}
          </svg>

          {/* ratio → pulse derivation */}
          <div style={{ position: "absolute", top: 1190, left: 0, right: 0, textAlign: "center",
            fontFamily: fonts.mono, fontSize: 30, letterSpacing: 1, color: theme.ink, opacity: interpolate(ratioP, [0, 0.3], [0, 1], clampOpt) }}>
            peaks ÷ lowest →&nbsp;
            <b style={{ color: COLORS[0] }}>{RATIOS.map((r) => r.toFixed(2)).join("  :  ")}</b>
          </div>
          <div style={{ position: "absolute", top: 1270, left: 0, right: 0, textAlign: "center",
            fontFamily: fonts.display, fontSize: 40, fontWeight: 300, color: theme.muted, opacity: interpolate(ratioP, [0.4, 0.7], [0, 1], clampOpt) }}>
            nearest whole pulses →
          </div>
          <div style={{ position: "absolute", top: 1338, left: 0, right: 0, textAlign: "center",
            fontFamily: fonts.display, fontSize: 72, fontWeight: 800, letterSpacing: 4, color: "#fff",
            opacity: interpolate(ratioP, [0.6, 1], [0, 1], clampOpt),
            textShadow: `0 0 26px ${COLORS[1]}66` }}>
            {POLY}
          </div>
        </div>

        {/* ── ① ② ③ the polyrhythm ── */}
        <div style={{ opacity: actOp }}>
          <div style={{ position: "absolute", top: 162, left: 0, right: 0, textAlign: "center",
            fontFamily: fonts.mono, fontSize: 24, letterSpacing: 2, color: theme.muted }}>
            EEG peaks {PEAKS.join(" · ")} Hz → <b style={{ color: "#fff" }}>{POLY}</b>
          </div>
          <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
            {VOICES.map((v, vi) => (
              <circle key={`ring${vi}`} cx={CX} cy={CY} r={R[vi]} fill="none" stroke={COLORS[vi]}
                strokeWidth={2} opacity={0.28 * Math.min(1, revealOf(vi))} />
            ))}

            {rippleOp > 0.01 && ripples.map((rp, i) => {
              const rad = rp.age * RIPPLE_SPEED, fade = 1 - rp.age / RIPPLE_LIFE;
              const col = rp.coincide ? "#fff" : COLORS[rp.vi];
              return <circle key={`rip${i}`} cx={CX} cy={CY} r={rad} fill="none" stroke={col}
                strokeWidth={rp.coincide ? 3.5 : 2.2} opacity={rippleOp * fade * (rp.coincide ? 0.9 : 0.6)}
                style={rp.coincide ? { filter: "drop-shadow(0 0 10px #fff)" } : undefined} />;
            })}

            <g opacity={wheelOp}>
              {VOICES.map((v, vi) => {
                const col = COLORS[vi % COLORS.length], rev = Math.min(1, revealOf(vi)), r = R[vi];
                return (
                  <g key={vi}>
                    {v.onsets.map((onset, j) => {
                      const spread = Easing.out(Easing.cubic)(rev);
                      const frac = onset * spread;
                      const [bx, by] = pt(r, frac);
                      const rel = playing ? (cyclePos - frac + 1) % 1 : 1;
                      const fire = playing ? Math.exp(-rel * 7) : 0;
                      const beadIn = Math.min(1, Math.max(0, rev * 1.3 - j * 0.1));
                      const rad = (7 + (10 + 6 * intensity) * fire) * beadIn;
                      return (
                        <g key={j}>
                          {fire > 0.4 && <circle cx={bx} cy={by} r={14 + 26 * (1 - fire)} fill="none" stroke={col} strokeWidth={2} opacity={0.5 * fire} />}
                          <circle cx={bx} cy={by} r={rad} fill={col} opacity={(0.55 + 0.45 * fire) * beadIn}
                            style={fire > 0.1 ? { filter: `drop-shadow(0 0 ${12 * fire}px ${col})` } : undefined} />
                        </g>
                      );
                    })}
                    <text x={CX} y={CY - r - 14} fill={col} fontSize={22} fontFamily="monospace" textAnchor="middle" opacity={rev * 0.9}>{v.pulses}</text>
                  </g>
                );
              })}
              {playing && (
                <g>
                  {Array.from({ length: 10 }, (_, i) => i).map((i) => {
                    const [tx, ty] = pt(R[2] + 26, playheadFrac - i * 0.014);
                    return <line key={i} x1={CX} y1={CY} x2={tx} y2={ty} stroke="#fff" strokeWidth={2.5 - i * 0.18} opacity={(0.5 - i * 0.045) * (0.6 + 0.4 * intensity)} />;
                  })}
                  <circle cx={hx} cy={hy} r={7} fill="#fff" style={{ filter: "drop-shadow(0 0 10px #fff)" }} />
                </g>
              )}
            </g>

            {playing && downPulse > 0.04 && (
              <circle cx={CX} cy={CY} r={18 + 150 * (1 - downPulse)} fill="none" stroke="#fff" strokeWidth={2.5} opacity={0.45 * downPulse} />
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
        </div>

        <div style={{ position: "absolute", bottom: 78, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
          biotuner · rhythm · scale2polyrhythm
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
