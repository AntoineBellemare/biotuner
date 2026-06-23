import React from "react";
import {
  AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame, useVideoConfig,
  interpolate, spring, Easing,
} from "remotion";
import { evolvePath } from "@remotion/paths";
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";
import data from "../../public/brain_grooves.json";

/**
 * "Brain Grooves" — the world's rhythms hidden in EEG. Each brain peak-ratio
 * becomes a Euclidean rhythm (Bjorklund); biotuner names it from Toussaint's
 * catalog. A short didactic front-end, then the named rhythms LAYER one by one
 * onto concentric rings — each on its own hand-drum — into a world polygroove.
 */
type Rhythm = { label: string; pulses: number; steps: number; pattern: number[];
  ivec: number[]; name: string; region: string; blurb: string };
const RH = data.rhythms as Rhythm[];
const F = data.front as { wave: number[]; spec_f: number[]; spec_mag: number[]; fmax: number; peaks: number[]; amps: number[] };
const COLORS = ["#f2c14e", "#6fd6c4", "#e8746a", "#9b8cff", "#7ad6a1"];

const TITLE = 84;
const DIDACTIC = 116;
const ADD = 120;       // frames between each rhythm joining (≈1.5 bars to digest)
const BAR = 80;        // playhead revolution
const HOLD = 150;
const OUTRO = 44;
const NR = RH.length;
const MIX_START = DIDACTIC;
const MIX_LEN = (NR - 1) * ADD + HOLD;
export const TOTAL_BRAINGROOVES = TITLE + DIDACTIC + MIX_LEN + OUTRO;

const clampOpt = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
const PAD = 110, AXW = 1080 - 2 * PAD;
const CX = 540, CY = 790;
const RAD = [132, 206, 280, 354, 424];

export const BrainGrooves: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();
  const sf = frame - TITLE;
  const introFade = interpolate(sf, [0, 16], [0, 1], clampOpt);

  const didOp = interpolate(sf, [6, 20], [0, 1], clampOpt) * interpolate(sf, [DIDACTIC - 22, DIDACTIC], [1, 0], clampOpt);
  const mixOp = interpolate(sf, [DIDACTIC - 10, DIDACTIC + 16], [0, 1], clampOpt);
  const waveP = interpolate(sf, [10, 58], [0, 1], { ...clampOpt, easing: Easing.out(Easing.cubic) });
  const specP = interpolate(sf, [48, 100], [0, 1], { ...clampOpt, easing: Easing.out(Easing.cubic) });

  const mf = sf - MIX_START;
  const playing = mf >= 0;
  const cyclePos = playing ? (mf % BAR) / BAR : 0;
  const ang = (frac: number) => -Math.PI / 2 + 2 * Math.PI * frac;
  const pt = (r: number, frac: number): [number, number] => [CX + r * Math.cos(ang(frac)), CY + r * Math.sin(ang(frac))];
  const fireOf = (onset: number) => playing ? Math.exp(-((((cyclePos - onset) % 1) + 1) % 1) * 8) : 0;
  const activeAt = (i: number) => spring({ frame: mf - i * ADD, fps, config: { damping: 15, stiffness: 90 } });
  const [phx, phy] = pt(RAD[NR - 1] + 22, cyclePos);

  // which rhythm is "entering" right now (for the announce banner)
  const ai = Math.floor(mf / ADD);
  const announce = playing && ai >= 0 && ai < NR && (mf - ai * ADD) < 42 ? ai : -1;

  // didactic waveform + spectrum
  const wfx = (i: number) => PAD + (i / (F.wave.length - 1)) * AXW;
  const waveCY = 560, waveAmp = 56;
  const wavePath = F.wave.map((v, i) => `${i === 0 ? "M" : "L"} ${wfx(i).toFixed(1)} ${(waveCY - v * waveAmp).toFixed(1)}`).join(" ");
  const wd = evolvePath(waveP, wavePath);
  const sBase = 960, sH = 170;
  const sfx = (f: number) => PAD + (f / F.fmax) * AXW;
  const sfy = (m: number) => sBase - m * sH;
  const specPath = `M ${PAD} ${sBase} ` + F.spec_f.map((f, i) => `L ${sfx(f).toFixed(1)} ${sfy(F.spec_mag[i] * specP).toFixed(1)}`).join(" ") + ` L ${width - PAD} ${sBase} Z`;

  return (
    <AbsoluteFill style={{ backgroundColor: "#06070e" }}>
      <Sequence from={TITLE}><Audio src={staticFile("audio/brain_grooves.wav")} /></Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE} title="Brain Grooves" hook="the world's rhythms, hidden in your brainwaves" />

      <AbsoluteFill style={{ opacity: introFade }}>
        <div style={{ position: "absolute", top: 96, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 44, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
          {sf < DIDACTIC - 10
            ? <>brain peaks → <b style={{ fontWeight: 800 }}>Euclidean</b> rhythms</>
            : <>your brain plays the <b style={{ fontWeight: 800 }}>world</b></>}
        </div>

        {/* ── DIDACTIC ── */}
        <div style={{ opacity: didOp }}>
          <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
            <text x={PAD} y={waveCY - 110} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>EEG signal</text>
            <path d={wavePath} fill="none" stroke={COLORS[1]} strokeWidth={2.5} strokeDasharray={wd.strokeDasharray} strokeDashoffset={wd.strokeDashoffset} opacity={0.9} />
            <g opacity={specP}>
              <text x={PAD} y={sBase - sH - 16} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>spectral peaks → Euclidean rhythms</text>
              <path d={specPath} fill="rgba(120,150,210,0.16)" stroke="rgba(150,180,220,0.5)" strokeWidth={2} />
              <line x1={PAD} y1={sBase} x2={width - PAD} y2={sBase} stroke="rgba(180,200,230,0.18)" strokeWidth={2} />
            </g>
            {F.peaks.map((pkf, i) => {
              const x = sfx(pkf), pop = Math.max(0, Math.min(1, specP * 1.5 - i * 0.12));
              return <g key={i} opacity={pop}>
                <line x1={x} y1={sBase} x2={x} y2={sfy(F.amps[i])} stroke={COLORS[0]} strokeWidth={3} />
                <circle cx={x} cy={sfy(F.amps[i])} r={6} fill={COLORS[0]} style={{ filter: `drop-shadow(0 0 7px ${COLORS[0]})` }} />
              </g>;
            })}
          </svg>
          <div style={{ position: "absolute", top: 1120, left: 0, right: 0, textAlign: "center",
            fontFamily: fonts.display, fontSize: 30, fontWeight: 300, color: theme.muted,
            opacity: interpolate(sf, [80, 106], [0, 1], clampOpt) }}>
            and these are real rhythms from <b style={{ color: "#fff", fontWeight: 700 }}>around the world</b>
          </div>
        </div>

        {/* ── MIX: layered rings ── */}
        <div style={{ opacity: mixOp }}>
          <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
            {RH.map((R, i) => {
              const act = Math.min(1, activeAt(i));
              if (act <= 0.01) return null;
              const r = RAD[i], col = COLORS[i % COLORS.length], n = R.steps;
              return (
                <g key={i} opacity={act}>
                  <circle cx={CX} cy={CY} r={r} fill="none" stroke={col} strokeWidth={1.4} opacity={0.2} />
                  {Array.from({ length: n }, (_, j) => j).map((j) => {
                    if (!R.pattern[j]) return null;
                    const [bx, by] = pt(r, j / n), fire = fireOf(j / n);
                    return (
                      <g key={j}>
                        {fire > 0.35 && <circle cx={bx} cy={by} r={11 + 22 * (1 - fire)} fill="none" stroke={col} strokeWidth={2} opacity={0.6 * fire} />}
                        <circle cx={bx} cy={by} r={(6.5 + 8 * fire) * act} fill={col} opacity={0.5 + 0.5 * fire}
                          style={fire > 0.2 ? { filter: `drop-shadow(0 0 ${11 * fire}px ${col})` } : undefined} />
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {/* shared playhead */}
            {playing && <>
              {Array.from({ length: 8 }, (_, i) => i).map((i) => {
                const [tx, ty] = pt(RAD[NR - 1] + 22, cyclePos - i * 0.013);
                return <line key={i} x1={CX} y1={CY} x2={tx} y2={ty} stroke="#fff" strokeWidth={2.2 - i * 0.2} opacity={0.4 - i * 0.045} />;
              })}
              <circle cx={phx} cy={phy} r={6} fill="#fff" style={{ filter: "drop-shadow(0 0 8px #fff)" }} />
            </>}
            <circle cx={CX} cy={CY} r={6} fill="#fff" opacity={0.5} />
          </svg>

          {/* announce banner */}
          {announce >= 0 && (
            <div style={{ position: "absolute", top: CY - 30, left: 0, right: 0, textAlign: "center",
              opacity: interpolate((mf - announce * ADD), [0, 6, 34, 42], [0, 1, 1, 0], clampOpt) }}>
              <div style={{ fontFamily: fonts.display, fontSize: 56, fontWeight: 800, color: "#fff",
                textShadow: `0 0 26px ${COLORS[announce % COLORS.length]}` }}>+ {RH[announce].name}</div>
            </div>
          )}

          {/* legend */}
          <div style={{ position: "absolute", top: 1290, left: 130, right: 130, display: "flex", flexDirection: "column", gap: 10 }}>
            {RH.map((R, i) => {
              const act = Math.min(1, activeAt(i));
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 16, opacity: act,
                  transform: `translateX(${(1 - act) * -20}px)`, fontFamily: fonts.display }}>
                  <div style={{ width: 12, height: 12, borderRadius: 6, background: COLORS[i % COLORS.length],
                    boxShadow: `0 0 ${8 + 10 * fireOf(0)}px ${COLORS[i % COLORS.length]}` }} />
                  <div style={{ fontSize: 30, fontWeight: 700, color: "#fff" }}>{R.name}</div>
                  <div style={{ fontSize: 26, fontWeight: 300, color: COLORS[i % COLORS.length] }}>{R.region}</div>
                  <div style={{ fontSize: 20, color: theme.muted, fontFamily: fonts.mono, marginLeft: "auto" }}>{R.label}</div>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ position: "absolute", bottom: 62, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 21, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
          biotuner · rhythm · scale2euclid · dict_rhythms
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
