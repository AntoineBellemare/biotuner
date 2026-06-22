import React from "react";
import {
  AbsoluteFill,
  Audio,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  interpolateColors,
  spring,
  Easing,
} from "remotion";
import { noise2D } from "@remotion/noise";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import data from "../../public/subharmonicity.json";

/**
 * "Is there one home note?" — subharmonic tension. Each chord note grows a
 * subharmonic ladder (f, f/2, f/3 …); where ladders meet, a glowing row marks
 * a shared subharmonic — a candidate common fundamental. Harmonic chords
 * converge on one home (calm); inharmonic chords never do (the notes jitter).
 * The gauge reads biotuner's real compute_subharmonic_tension.
 */
const BASE = data.base_freq;
const NSUB = data.n_subharm;
const CHORDS = data.chords;
const TEAL = "#6fd6c4";
const HOT = "#e8746a";
const GOLD = "#f2c14e";

const INTRO = 30;
const DRAW = 46;
const DWELL = 88;
const BEAT = DRAW + DWELL;
const TAIL = 46;
export const TOTAL_SUBHARM = INTRO + CHORDS.length * BEAT + TAIL;

const FMIN = 36, FMAX = 470;

export const SubharmonicTension: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();

  const local = frame - INTRO;
  const ci = Math.max(0, Math.min(CHORDS.length - 1, Math.floor(local / BEAT)));
  const beatLocal = local - ci * BEAT;
  const chord = CHORDS[ci];
  const freqs = chord.ratios.map((r) => r * BASE);
  const nNotes = freqs.length;
  const tNorm = chord.tension_norm as number;

  // reveal of the ladders, settle of the convergence
  const reveal = interpolate(beatLocal, [0, DRAW], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });
  const settle = spring({ frame: beatLocal - DRAW, fps, config: { damping: 14, stiffness: 90 } });

  // layout: column per note (x), frequency on y (log, high freq at top)
  const leftPad = 150, rightPad = 150;
  const axisW = width - leftPad - rightPad;
  const xCol = (i: number) => leftPad + ((i + 0.5) / nNotes) * axisW;
  const yTop = 470, yBot = 1500;
  const lf = (f: number) => Math.log(f);
  const yOf = (f: number) =>
    interpolate(lf(f), [lf(FMIN), lf(FMAX)], [yBot, yTop], {
      extrapolateLeft: "clamp", extrapolateRight: "clamp",
    });

  // subharmonic ticks per note + convergence bins (shared subharmonics)
  type Tick = { i: number; k: number; f: number };
  const ticks: Tick[] = [];
  for (let i = 0; i < nNotes; i++)
    for (let k = 1; k <= NSUB; k++) {
      const f = freqs[i] / k;
      if (f >= FMIN) ticks.push({ i, k, f });
    }
  // bin by rounded frequency (≈ delta tolerance) → which columns share it
  const TOL = 4;
  const bins = new Map<number, Set<number>>();
  for (const t of ticks) {
    const key = Math.round(t.f / TOL);
    if (!bins.has(key)) bins.set(key, new Set());
    bins.get(key)!.add(t.i);
  }
  const convergences = [...bins.entries()]
    .filter(([, cols]) => cols.size >= 2)
    .map(([key, cols]) => ({ f: key * TOL, cols: cols.size }));
  // the home: lowest shared-by-all subharmonic
  const home = convergences.filter((c) => c.cols >= nNotes).sort((a, b) => a.f - b.f)[0];
  const isAligned = (t: Tick) => (bins.get(Math.round(t.f / TOL))?.size ?? 0) >= 2;

  const introFade = interpolate(frame, [0, 16], [0, 1], { extrapolateRight: "clamp" });
  const orbColor = interpolateColors(tNorm, [0, 0.45, 1], [TEAL, GOLD, HOT]);

  return (
    <AbsoluteFill style={{ opacity: introFade, backgroundColor: "#06070e" }}>
      <Audio src={staticFile("audio/subharmonicity.wav")} />
      <Backdrop />

      <div style={{ position: "absolute", top: 116, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 50, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
        is there one <b style={{ fontWeight: 800 }}>home</b> note?
      </div>
      <div style={{ position: "absolute", top: 196, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 32, letterSpacing: 3, color: orbColor }}>
        {chord.label}
      </div>

      <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
        {/* convergence rows (shared subharmonics) */}
        {convergences.map((c, idx) => {
          const y = yOf(c.f);
          const full = c.cols >= nNotes;
          const op = (full ? 0.5 + 0.5 * settle : 0.18 + 0.12 * c.cols) * reveal;
          return (
            <line key={`c${idx}`} x1={leftPad - 30} y1={y} x2={width - rightPad + 30} y2={y}
              stroke={full ? GOLD : "rgba(150,180,220,0.5)"} strokeWidth={full ? 3 : 1.5}
              opacity={op} />
          );
        })}

        {/* per-note columns: subharmonic ladders */}
        {Array.from({ length: nNotes }, (_, i) => i).map((i) => {
          const x = xCol(i);
          const jitter = noise2D("o", i * 2.1, frame * 0.12) * 26 * tNorm * settle;
          const orbY = yOf(freqs[i]);
          const colTicks = ticks.filter((t) => t.i === i);
          return (
            <g key={`col${i}`}>
              {/* vertical guide */}
              <line x1={x} y1={orbY} x2={x} y2={yOf(freqs[i] / NSUB)}
                stroke="rgba(160,185,225,0.12)" strokeWidth={2} />
              {/* subharmonic ticks */}
              {colTicks.map((t) => {
                const ty = yOf(t.f);
                const shown = reveal > (t.k - 1) / NSUB;
                if (!shown) return null;
                const al = isAligned(t);
                return (
                  <g key={`t${t.k}`}>
                    <line x1={x - (al ? 26 : 16)} y1={ty} x2={x + (al ? 26 : 16)} y2={ty}
                      stroke={al ? GOLD : "rgba(180,205,240,0.55)"}
                      strokeWidth={al ? 5 : 3} strokeLinecap="round" />
                    {al && <circle cx={x} cy={ty} r={6 + 5 * settle} fill={GOLD} opacity={0.85} />}
                  </g>
                );
              })}
              {/* the note orb (jitters with tension) */}
              <circle cx={x + jitter} cy={orbY} r={26}
                fill={orbColor} opacity={0.95}
                style={{ filter: `drop-shadow(0 0 ${16 + 18 * tNorm}px ${orbColor})` }} />
              <circle cx={x + jitter} cy={orbY} r={11} fill="#fff" opacity={0.85} />
            </g>
          );
        })}

        {/* HOME marker / verdict */}
        {home ? (
          <g opacity={settle}>
            <circle cx={width / 2} cy={yOf(home.f)} r={20 + 10 * settle} fill="none"
              stroke={GOLD} strokeWidth={3} opacity={0.8} />
            <text x={width / 2} y={yOf(home.f) + 64} fill={GOLD} fontSize={28}
              fontFamily="monospace" textAnchor="middle" letterSpacing={2} opacity={0.9}>
              one shared fundamental
            </text>
          </g>
        ) : (
          <text x={width / 2} y={yBot - 30} fill={HOT} fontSize={28} fontFamily="monospace"
            textAnchor="middle" letterSpacing={2} opacity={settle * 0.9}>
            no shared home — tension
          </text>
        )}
      </svg>

      {/* tension gauge */}
      <div style={{ position: "absolute", bottom: 170, left: 150, right: 150 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline",
          fontFamily: fonts.mono, color: theme.muted, fontSize: 24, letterSpacing: 2, marginBottom: 12 }}>
          <span>subharmonic tension</span>
          <span style={{ color: orbColor, fontSize: 40, fontWeight: 700 }}>
            {(chord.tension as number).toFixed(3)}
          </span>
        </div>
        <div style={{ height: 20, borderRadius: 10, background: "rgba(180,200,230,0.12)", overflow: "hidden" }}>
          <div style={{ height: "100%", width: `${tNorm * 100}%`, borderRadius: 10,
            background: orbColor, boxShadow: `0 0 ${10 + 26 * tNorm}px ${orbColor}` }} />
        </div>
      </div>

      <div style={{ position: "absolute", bottom: 80, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
        biotuner · subharmonic tension
      </div>
    </AbsoluteFill>
  );
};
