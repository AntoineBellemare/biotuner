import React from "react";
import {
  AbsoluteFill,
  Audio,
  Sequence,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";
import data from "../../public/heart_brain.json";

/**
 * "Heart × Brain" — a duet. The ECG heartbeat (real R-peaks, real HRV) is the
 * master clock scrolling along the bottom; the brain's 5:7:11 polyrhythm
 * (biotuner's rhythm module) weaves across it in three lanes above. Where a
 * brain pulse lands on a heartbeat, a beam lights up — the groove accent.
 * Intuition: independent rhythms phasing against a living tactus; the groove
 * lives in the coincidences.
 */
type Voice = { pulses: number; onsets: number[]; ioi_ms: number };
const VOICES = data.brain_voices as Voice[];
const ECG = data.ecg as number[];
const R = data.r_times as number[];
const DUR = data.dur_s as number;
const HR = data.hr_bpm as number;
const HRV = data.hrv_ms as number;
const POLY = data.poly_label as string;
const COLORS = ["#6fd6c4", "#f2c14e", "#9b8cff"];
const CORAL = "#e8746a";

const TITLE = 84;
const INTRO = 18;
const LEADIN = 1.2;          // seconds of timeline before t=0 reaches playhead
const CYCLE = 2.0;           // brain polyrhythm period (≈ 2 heartbeats)
const COINC_TOL = 0.07;      // s — brain pulse "lands on" a heartbeat
const PX_PER_S = 200;
const PLAYS = Math.ceil((DUR + 1.0) * 30) + INTRO;
export const TOTAL_HEARTBRAIN = TITLE + PLAYS + 36;

const ECG_SR = ECG.length / DUR;     // samples per second
const LANE_Y = [560, 700, 840];      // brain voices 5,7,11
const ECG_BASE = 1240, ECG_AMP = 150;

// brain onsets (metronomic) + which land on a real heartbeat
const BRAIN: { vi: number; t: number; coincide: boolean }[] = [];
VOICES.forEach((v, vi) => {
  const nC = Math.ceil(DUR / CYCLE) + 1;
  for (let c = 0; c < nC; c++)
    for (const o of v.onsets) {
      const t = (c + o) * CYCLE;
      if (t > DUR + 0.5) continue;
      const coincide = R.some((r) => Math.abs(r - t) < COINC_TOL);
      BRAIN.push({ vi, t, coincide });
    }
});

export const HeartBrainDuet: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();

  const sf = frame - TITLE;
  const introFade = interpolate(sf, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const now = (sf - INTRO) / fps - LEADIN;   // current timeline position (s)
  const playheadX = width * 0.33;
  const xOf = (t: number) => playheadX + (t - now) * PX_PER_S;
  const visible = (t: number) => xOf(t) >= -40 && xOf(t) <= width + 40;

  // ECG waveform path (visible window)
  let ecgPath = "";
  let started = false;
  for (let i = 0; i < ECG.length; i++) {
    const t = i / ECG_SR;
    const x = xOf(t);
    if (x < -20 || x > width + 20) { started = false; continue; }
    const y = ECG_BASE - ECG[i] * ECG_AMP;
    ecgPath += `${started ? "L" : "M"} ${x.toFixed(1)} ${y.toFixed(1)} `;
    started = true;
  }

  // instantaneous heart rate at `now`
  let ibi = 60 / HR;
  for (let i = 1; i < R.length; i++) if (R[i] >= now) { ibi = R[i] - R[i - 1]; break; }
  const instHr = Math.round(60 / ibi);
  const beatFlash = Math.max(0, ...R.map((r) => Math.exp(-Math.abs(r - now) * 22)));

  return (
    <AbsoluteFill style={{ backgroundColor: "#06070e" }}>
      <Sequence from={TITLE}>
        <Audio src={staticFile("audio/heart_brain.wav")} />
      </Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE}
        title="Heart × Brain" hook="your heart keeps time, your brain plays across it" />

      <AbsoluteFill style={{ opacity: introFade }}>
        <div style={{ position: "absolute", top: 92, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 46, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
          one keeps <b style={{ fontWeight: 800, color: CORAL }}>time</b>, the other <b style={{ fontWeight: 800, color: COLORS[0] }}>plays</b>
        </div>
        <div style={{ position: "absolute", top: 162, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 24, letterSpacing: 2, color: theme.muted }}>
          ECG {instHr} bpm · HRV {HRV}ms  ×  brain {POLY}
        </div>

        <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
          {/* lane guides + labels */}
          {VOICES.map((v, vi) => (
            <g key={`lane${vi}`}>
              <line x1={0} y1={LANE_Y[vi]} x2={width} y2={LANE_Y[vi]}
                stroke={COLORS[vi]} strokeWidth={1} opacity={0.12} />
              <text x={36} y={LANE_Y[vi] - 14} fill={COLORS[vi]} fontSize={22} fontFamily="monospace" opacity={0.8}>
                {v.pulses}
              </text>
            </g>
          ))}

          {/* coincidence beams (brain pulse on a heartbeat) */}
          {BRAIN.filter((b) => b.coincide && visible(b.t)).map((b, i) => {
            const x = xOf(b.t);
            const flash = Math.exp(-Math.abs(b.t - now) * 16);
            return (
              <line key={`beam${i}`} x1={x} y1={LANE_Y[b.vi]} x2={x} y2={ECG_BASE}
                stroke="#fff" strokeWidth={1 + 2 * flash} opacity={0.12 + 0.55 * flash} />
            );
          })}

          {/* brain pulses */}
          {BRAIN.filter((b) => visible(b.t)).map((b, i) => {
            const x = xOf(b.t);
            const fire = Math.exp(-Math.abs(b.t - now) * 14);
            const col = b.coincide ? "#fff" : COLORS[b.vi];
            return (
              <circle key={`b${i}`} cx={x} cy={LANE_Y[b.vi]} r={5.5 + 9 * fire + (b.coincide ? 2 : 0)}
                fill={col} opacity={0.55 + 0.45 * fire}
                style={fire > 0.2 ? { filter: `drop-shadow(0 0 ${11 * fire}px ${col})` } : undefined} />
            );
          })}

          {/* ECG waveform (the master pulse) */}
          <path d={ecgPath} fill="none" stroke={CORAL} strokeWidth={2.5} opacity={0.85}
            style={{ filter: `drop-shadow(0 0 6px ${CORAL}88)` }} />
          {/* R-peak markers */}
          {R.filter(visible).map((r, i) => {
            const x = xOf(r);
            const fire = Math.exp(-Math.abs(r - now) * 18);
            return (
              <g key={`r${i}`}>
                <line x1={x} y1={ECG_BASE - ECG_AMP - 30} x2={x} y2={ECG_BASE + 40}
                  stroke={CORAL} strokeWidth={1 + 2 * fire} opacity={0.2 + 0.5 * fire} />
                <circle cx={x} cy={ECG_BASE - ECG_AMP - 30} r={5 + 8 * fire} fill={CORAL}
                  opacity={0.6 + 0.4 * fire}
                  style={fire > 0.2 ? { filter: `drop-shadow(0 0 ${12 * fire}px ${CORAL})` } : undefined} />
              </g>
            );
          })}

          {/* NOW playhead */}
          <line x1={playheadX} y1={480} x2={playheadX} y2={ECG_BASE + 70}
            stroke="#fff" strokeWidth={2} opacity={0.55} />
          <text x={playheadX} y={462} fill="#fff" fontSize={20} fontFamily="monospace"
            textAnchor="middle" opacity={0.6}>now</text>

          {/* beating heart, pulsing on each R-peak */}
          <g transform={`translate(${width / 2}, ${ECG_BASE + 210}) scale(${4.4 + 1.6 * beatFlash}) translate(-12,-12)`}>
            <path d="M12 21s-9-5.5-9-12a5 5 0 0 1 9-3 5 5 0 0 1 9 3c0 6.5-9 12-9 12z"
              fill={CORAL} opacity={0.7 + 0.3 * beatFlash}
              style={{ filter: `drop-shadow(0 0 ${6 + 22 * beatFlash}px ${CORAL})` }} />
          </g>
        </svg>

        <div style={{ position: "absolute", bottom: 150, left: 60, right: 60, textAlign: "center",
          fontFamily: fonts.display, fontSize: 28, fontWeight: 300, color: theme.muted }}>
          where a brain pulse lands on a beat → a <b style={{ color: "#fff", fontWeight: 700 }}>groove accent</b>
        </div>
        <div style={{ position: "absolute", bottom: 78, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
          biotuner · rhythm · ECG × EEG coincidence
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
