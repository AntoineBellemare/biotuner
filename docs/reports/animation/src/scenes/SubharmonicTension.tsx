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
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";
import data from "../../public/subharmonicity.json";

/**
 * "Is there one home note?" — subharmonic tension. Each chord note grows a
 * subharmonic ladder (f, f/2, f/3 …). Where ladders from different notes nearly
 * meet (within biotuner's delta_lim), a SHARED SUBHARMONIC lights up — brighter
 * the tighter the alignment. A perfectly tight alignment touched by every note
 * is a common fundamental ("home"): harmonic chords lock onto one (calm);
 * inharmonic chords only ever graze loose, faint ones (the notes jitter).
 * The gauge reads biotuner's real compute_subharmonic_tension.
 */
type Align = {
  freq: number;
  members: { i: number; k: number }[];
  n_notes: number;
  spread: number;
  tight: number;
  full: boolean;
};

const BASE = data.base_freq;
const NSUB = data.n_subharm;
const CHORDS = data.chords;
const TEAL = "#6fd6c4";
const HOT = "#e8746a";
const GOLD = "#f2c14e";

const TITLE = 84;
const INTRO = 30;
const DRAW = 50;
const DWELL = 92;
const BEAT = DRAW + DWELL;
const TAIL = 46;
export const TOTAL_SUBHARM = TITLE + INTRO + CHORDS.length * BEAT + TAIL;

const FMIN = 13, FMAX = 440;

export const SubharmonicTension: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();

  const sf = frame - TITLE; // scene frame (content begins after the title card)
  const local = sf - INTRO;
  const ci = Math.max(0, Math.min(CHORDS.length - 1, Math.floor(local / BEAT)));
  const beatLocal = local - ci * BEAT;
  const chord = CHORDS[ci];
  const freqs = chord.ratios.map((r) => r * BASE);
  const nNotes = freqs.length;
  const tNorm = chord.tension_norm as number;
  const aligns = chord.alignments as Align[];

  const reveal = interpolate(beatLocal, [0, DRAW], [0, 1], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic),
  });
  const settle = spring({ frame: beatLocal - DRAW, fps, config: { damping: 15, stiffness: 80 } });

  // layout: column per note (x), frequency on y (log, high freq at top)
  const leftPad = 150, rightPad = 150;
  const axisW = width - leftPad - rightPad;
  const xCol = (i: number) => leftPad + ((i + 0.5) / nNotes) * axisW;
  const yTop = 430, yBot = 1530;
  const lf = (f: number) => Math.log(f);
  const yOf = (f: number) =>
    interpolate(lf(f), [lf(FMIN), lf(FMAX)], [yBot, yTop], {
      extrapolateLeft: "clamp", extrapolateRight: "clamp",
    });

  // which (note,k) rungs participate in an alignment, and how tightly
  const memTight = new Map<string, number>();
  aligns.forEach((a) =>
    a.members.forEach((m) => {
      const key = `${m.i}-${m.k}`;
      const q = a.tight * (a.full ? 1 : 0.7);
      memTight.set(key, Math.max(memTight.get(key) ?? 0, q));
    })
  );

  // the home: lowest-frequency full alignment; "locked" if tight enough
  const fulls = aligns.filter((a) => a.full);
  const maxFullTight = fulls.length ? Math.max(...fulls.map((a) => a.tight)) : 0;
  const home = fulls.slice().sort((a, b) => a.freq - b.freq)[0];
  const locked = maxFullTight >= 0.8;

  const introFade = interpolate(sf, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const orbColor = interpolateColors(tNorm, [0, 0.45, 1], [TEAL, GOLD, HOT]);

  return (
    <AbsoluteFill style={{ backgroundColor: "#06070e" }}>
      <Sequence from={TITLE}>
        <Audio src={staticFile("audio/subharmonicity.wav")} />
      </Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE}
        title="Subharmonic Tension" hook="does a chord agree on one home note?" />

      <AbsoluteFill style={{ opacity: introFade }}>
      <div style={{ position: "absolute", top: 110, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 50, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
        is there one <b style={{ fontWeight: 800 }}>home</b> note?
      </div>
      <div style={{ position: "absolute", top: 190, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 32, letterSpacing: 3, color: orbColor }}>
        {chord.label}
      </div>

      <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
        {/* convergence: faint lines folding every note down to the home */}
        {locked && home &&
          Array.from({ length: nNotes }, (_, i) => i).map((i) => (
            <line key={`fold${i}`} x1={xCol(i)} y1={yOf(freqs[i])} x2={width / 2} y2={yOf(home.freq)}
              stroke={GOLD} strokeWidth={1.4} opacity={0.22 * settle} />
          ))}

        {/* shared subharmonics — brightness ∝ tightness, gold (all notes) / teal */}
        {aligns.map((a, idx) => {
          const y = yOf(a.freq);
          const q = a.tight;
          const op = reveal * settle * (0.08 + 0.92 * q * q) * (a.full ? 1 : 0.55);
          const col = a.full ? GOLD : TEAL;
          return (
            <line key={`al${idx}`} x1={leftPad - 36} y1={y} x2={width - rightPad + 36} y2={y}
              stroke={col} strokeWidth={1.4 + 4.4 * q} opacity={op}
              strokeDasharray={q < 0.5 ? "5 9" : undefined}
              style={q > 0.55 ? { filter: `drop-shadow(0 0 ${5 + 10 * q}px ${col})` } : undefined} />
          );
        })}

        {/* per-note columns: subharmonic ladders */}
        {Array.from({ length: nNotes }, (_, i) => i).map((i) => {
          const x = xCol(i);
          const jitter = noise2D("o", i * 2.1, sf * 0.12) * 24 * tNorm * settle;
          const orbY = yOf(freqs[i]);
          return (
            <g key={`col${i}`}>
              {/* vertical guide down the ladder */}
              <line x1={x} y1={orbY} x2={x} y2={yOf(Math.max(FMIN, freqs[i] / NSUB))}
                stroke="rgba(160,185,225,0.1)" strokeWidth={2} />
              {/* subharmonic rungs */}
              {Array.from({ length: NSUB }, (_, kk) => kk + 1).map((k) => {
                const f = freqs[i] / k;
                if (f < FMIN) return null;
                if (reveal <= (k - 1) / NSUB) return null;
                const ty = yOf(f);
                const q = memTight.get(`${i}-${k}`) ?? 0;
                const lit = q > 0.05 ? settle : 1;
                const half = 14 + 16 * q;
                return (
                  <g key={`t${k}`}>
                    <line x1={x - half} y1={ty} x2={x + half} y2={ty}
                      stroke={q > 0.05 ? GOLD : "rgba(180,205,240,0.5)"}
                      strokeWidth={2.5 + 4 * q} strokeLinecap="round"
                      opacity={(q > 0.05 ? 0.4 + 0.6 * q : 0.5) * lit}
                      style={q > 0.4 ? { filter: `drop-shadow(0 0 ${6 * q}px ${GOLD})` } : undefined} />
                    {q > 0.4 && <circle cx={x} cy={ty} r={4 + 5 * q * settle} fill={GOLD} opacity={0.85} />}
                  </g>
                );
              })}
              {/* the note orb (jitters with tension) */}
              <circle cx={x + jitter} cy={orbY} r={26} fill={orbColor} opacity={0.95}
                style={{ filter: `drop-shadow(0 0 ${16 + 18 * tNorm}px ${orbColor})` }} />
              <circle cx={x + jitter} cy={orbY} r={11} fill="#fff" opacity={0.85} />
            </g>
          );
        })}

        {/* HOME marker / verdict */}
        {locked && home ? (
          <g opacity={settle}>
            <circle cx={width / 2} cy={yOf(home.freq)} r={18 + 12 * settle} fill="none"
              stroke={GOLD} strokeWidth={3} opacity={0.85}
              style={{ filter: `drop-shadow(0 0 14px ${GOLD})` }} />
            <circle cx={width / 2} cy={yOf(home.freq)} r={7} fill={GOLD} />
            <text x={width / 2} y={yOf(home.freq) + 60} fill={GOLD} fontSize={28}
              fontFamily="monospace" textAnchor="middle" letterSpacing={2} opacity={0.92}>
              {aligns.length} shared subharmonics → one home
            </text>
          </g>
        ) : (
          <text x={width / 2} y={yBot - 18} fill={HOT} fontSize={28} fontFamily="monospace"
            textAnchor="middle" letterSpacing={2} opacity={settle * 0.9}>
            {aligns.length} shared subharmonics, all loose — tension
          </text>
        )}
      </svg>

      {/* tension gauge */}
      <div style={{ position: "absolute", bottom: 168, left: 150, right: 150 }}>
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
        biotuner · subharmonic tension · δ {data.delta_lim}ms
      </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
