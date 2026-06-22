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
 * catalog. The same EEG that plays a 5:7:11 polyrhythm also reproduces the
 * Cuban tresillo, the Arabic Nawakhat, India's Savari tala, Bulgaria's
 * Ruchenitza — even a Frank Zappa meter. A didactic front-end, then a gallery.
 */
type Rhythm = { label: string; pulses: number; steps: number; pattern: number[];
  ivec: number[]; name: string; region: string; blurb: string };
const RH = data.rhythms as Rhythm[];
const F = data.front as { wave: number[]; spec_f: number[]; spec_mag: number[]; fmax: number; peaks: number[]; amps: number[] };
const COLORS = ["#f2c14e", "#6fd6c4", "#e8746a", "#9b8cff", "#7ad6a1"];

const TITLE = 84;
const DIDACTIC = 168;
const RHYTHM = 156;
const NR = RH.length;
const TAIL = 46;
const STEP_FRAMES = 8;
export const TOTAL_BRAINGROOVES = TITLE + DIDACTIC + NR * RHYTHM + TAIL;

const clampOpt = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
const PAD = 110, AXW = 1080 - 2 * PAD;
const NCX = 540, NCY = 900, NR_RAD = 300;

export const BrainGrooves: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();
  const sf = frame - TITLE;
  const introFade = interpolate(sf, [0, 16], [0, 1], clampOpt);

  const didOp = interpolate(sf, [6, 20], [0, 1], clampOpt) * interpolate(sf, [DIDACTIC - 26, DIDACTIC], [1, 0], clampOpt);
  const galOp = interpolate(sf, [DIDACTIC - 10, DIDACTIC + 16], [0, 1], clampOpt);

  // didactic sub-phases
  const waveP = interpolate(sf, [12, 80], [0, 1], { ...clampOpt, easing: Easing.out(Easing.cubic) });
  const specP = interpolate(sf, [70, 140], [0, 1], { ...clampOpt, easing: Easing.out(Easing.cubic) });

  // gallery: which rhythm + local frame
  const gf = sf - DIDACTIC;
  const ri = Math.max(0, Math.min(NR - 1, Math.floor(gf / RHYTHM)));
  const rl = gf - ri * RHYTHM;
  const R = RH[ri];
  const col = COLORS[ri % COLORS.length];
  const n = R.steps;
  const reveal = spring({ frame: rl - 4, fps, config: { damping: 15, stiffness: 90 } });
  const nameIn = spring({ frame: rl - 18, fps, config: { damping: 14, stiffness: 80 } });
  const stepPos = rl > 24 ? ((rl - 24) / STEP_FRAMES) % n : -1; // continuous sweep

  const ang = (i: number) => -Math.PI / 2 + 2 * Math.PI * (i / n);
  const pt = (rad: number, i: number): [number, number] => [NCX + rad * Math.cos(ang(i)), NCY + rad * Math.sin(ang(i))];
  const [phx, phy] = stepPos >= 0 ? pt(NR_RAD + 24, stepPos) : [NCX, NCY];

  // didactic waveform + spectrum
  const wfx = (i: number) => PAD + (i / (F.wave.length - 1)) * AXW;
  const waveCY = 620, waveAmp = 64;
  const wavePath = F.wave.map((v, i) => `${i === 0 ? "M" : "L"} ${wfx(i).toFixed(1)} ${(waveCY - v * waveAmp).toFixed(1)}`).join(" ");
  const wd = evolvePath(waveP, wavePath);
  const sBase = 1060, sH = 190;
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

        {/* ── DIDACTIC: EEG → spectrum → peaks → Euclidean ── */}
        <div style={{ opacity: didOp }}>
          <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
            <text x={PAD} y={waveCY - 120} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>EEG signal</text>
            <path d={wavePath} fill="none" stroke={COLORS[1]} strokeWidth={2.5} strokeDasharray={wd.strokeDasharray} strokeDashoffset={wd.strokeDashoffset} opacity={0.9} />
            <g opacity={specP}>
              <text x={PAD} y={sBase - sH - 16} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>spectral peaks → ratios</text>
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
          <div style={{ position: "absolute", top: 1230, left: 0, right: 0, textAlign: "center",
            fontFamily: fonts.display, fontSize: 32, fontWeight: 300, color: theme.muted,
            opacity: interpolate(sf, [120, 160], [0, 1], clampOpt) }}>
            Bjorklund spreads each ratio's pulses <b style={{ color: "#fff", fontWeight: 700 }}>evenly</b> — a Euclidean rhythm
          </div>
        </div>

        {/* ── GALLERY: necklace + name/region ── */}
        <div style={{ opacity: galOp }}>
          <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
            <circle cx={NCX} cy={NCY} r={NR_RAD} fill="none" stroke={col} strokeWidth={1.5} opacity={0.2 * reveal} />
            {/* step ticks + pulse beads */}
            {Array.from({ length: n }, (_, i) => i).map((i) => {
              const isPulse = R.pattern[i] === 1;
              const [tx, ty] = pt(NR_RAD, i);
              const rel = stepPos >= 0 ? (((stepPos - i) % n) + n) % n : 99;
              const fire = isPulse && rel < 1.4 ? Math.exp(-rel * 2.2) : 0;
              const bIn = Math.min(1, Math.max(0, reveal * 1.3 - i / n));
              if (!isPulse) return <circle key={i} cx={tx} cy={ty} r={3 * bIn} fill="rgba(180,200,230,0.4)" opacity={bIn} />;
              return (
                <g key={i}>
                  {fire > 0.3 && <circle cx={tx} cy={ty} r={14 + 30 * (1 - fire)} fill="none" stroke={col} strokeWidth={2} opacity={0.6 * fire} />}
                  <circle cx={tx} cy={ty} r={(9 + 9 * fire) * bIn} fill={col} opacity={(0.55 + 0.45 * fire) * bIn}
                    style={fire > 0.1 ? { filter: `drop-shadow(0 0 ${12 * fire}px ${col})` } : undefined} />
                </g>
              );
            })}
            {/* playhead */}
            {stepPos >= 0 && <>
              <line x1={NCX} y1={NCY} x2={phx} y2={phy} stroke="#fff" strokeWidth={2} opacity={0.45} />
              <circle cx={phx} cy={phy} r={6} fill="#fff" style={{ filter: "drop-shadow(0 0 8px #fff)" }} />
            </>}
            <circle cx={NCX} cy={NCY} r={6} fill="#fff" opacity={0.5} />
          </svg>

          {/* name / region / blurb */}
          <div style={{ position: "absolute", top: 1280, left: 0, right: 0, textAlign: "center", opacity: nameIn,
            transform: `translateY(${(1 - nameIn) * 20}px)` }}>
            <div style={{ fontFamily: fonts.mono, fontSize: 26, letterSpacing: 4, color: col }}>{R.label} · {R.ivec.join("·")}</div>
            <div style={{ fontFamily: fonts.display, fontSize: 74, fontWeight: 800, color: "#fff", letterSpacing: 1, marginTop: 6,
              textShadow: `0 0 26px ${col}66` }}>{R.name}</div>
            <div style={{ fontFamily: fonts.display, fontSize: 38, fontWeight: 300, color: col, marginTop: 4 }}>{R.region}</div>
            <div style={{ fontFamily: fonts.display, fontSize: 26, fontWeight: 300, color: theme.muted, marginTop: 14 }}>{R.blurb}</div>
          </div>
          {/* progress dots */}
          <div style={{ position: "absolute", bottom: 120, left: 0, right: 0, display: "flex", justifyContent: "center", gap: 14 }}>
            {RH.map((_, i) => <div key={i} style={{ width: 10, height: 10, borderRadius: 5,
              background: i === ri ? COLORS[i % COLORS.length] : "rgba(180,200,230,0.25)" }} />)}
          </div>
        </div>

        <div style={{ position: "absolute", bottom: 70, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
          biotuner · rhythm · scale2euclid · dict_rhythms
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
