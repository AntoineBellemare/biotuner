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
 * "Brain Polyrhythm" — biotuner's rhythm module on EEG. The brain's spectral
 * peaks become frequency ratios, and scale2polyrhythm_continuous turns each
 * ratio into an evenly-pulsed voice: here 5 : 7 : 11. Three concentric rings
 * carry those pulses; a playhead sweeps once per cycle, firing each bead it
 * crosses; the voices only reunite at the downbeat — the satisfying unison.
 */
type Voice = { pulses: number; onsets: number[]; ioi_ms: number };
const VOICES = data.voices as Voice[];
const PEAKS = data.peaks as number[];
const POLY = data.poly_label as string;
const COLORS = ["#6fd6c4", "#f2c14e", "#e8746a", "#9b8cff"];

const TITLE = 84;
const INTRO = 18;
const BUILD = 170;          // rings snap in (one voice at a time)
const CYCLE = 80;           // frames per polyrhythm cycle
const NCYC = 6;
const PLAY = NCYC * CYCLE;
const TAIL = 40;
export const TOTAL_BRAINPOLY = TITLE + INTRO + BUILD + PLAY + TAIL;

const CX = 540, CY = 1040;
const R = [250, 340, 432];  // ring radii for voices 0,1,2

export const BrainPolyrhythm: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();

  const sf = frame - TITLE;
  const introFade = interpolate(sf, [0, 16], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });

  const playFrame = sf - INTRO - BUILD;          // ≥0 once the wheel plays
  const playing = playFrame >= 0;
  const cyclePos = playing ? (playFrame % CYCLE) / CYCLE : 0;

  // angle helper: start at top (12 o'clock), sweep clockwise
  const ang = (frac: number) => -Math.PI / 2 + 2 * Math.PI * frac;
  const pt = (r: number, frac: number) => [CX + r * Math.cos(ang(frac)), CY + r * Math.sin(ang(frac))];

  // per-voice reveal during BUILD (one ring at a time)
  const revealOf = (vi: number) =>
    spring({ frame: sf - INTRO - vi * 48, fps, config: { damping: 14, stiffness: 80 } });

  const playheadFrac = cyclePos;
  const [hx, hy] = pt(R[VOICES.length - 1] + 26, playheadFrac);

  // downbeat pulse (all voices fire at frac 0)
  const downPulse = playing ? Math.exp(-((cyclePos + 1) % 1) * 7) + Math.exp(-cyclePos * 7) : 0;

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
          {VOICES.map((v, vi) => {
            const col = COLORS[vi % COLORS.length];
            const rev = revealOf(vi);
            const r = R[vi];
            return (
              <g key={vi} opacity={Math.min(1, rev)}>
                {/* ring */}
                <circle cx={CX} cy={CY} r={r} fill="none" stroke={col} strokeWidth={2} opacity={0.3} />
                {/* beads */}
                {v.onsets.map((onset, j) => {
                  const [bx, by] = pt(r, onset);
                  const rel = playing ? (cyclePos - onset + 1) % 1 : 1;
                  const fire = playing ? Math.exp(-rel * 7) : 0;
                  const beadIn = Math.min(1, Math.max(0, rev * 1.3 - j * 0.12));
                  const rad = (7 + 12 * fire) * beadIn;
                  return (
                    <g key={j}>
                      {fire > 0.15 && (
                        <line x1={CX} y1={CY} x2={bx} y2={by} stroke={col}
                          strokeWidth={1.5} opacity={0.25 * fire} />
                      )}
                      <circle cx={bx} cy={by} r={rad} fill={col}
                        opacity={(0.6 + 0.4 * fire) * beadIn}
                        style={fire > 0.1 ? { filter: `drop-shadow(0 0 ${10 * fire}px ${col})` } : undefined} />
                    </g>
                  );
                })}
                {/* voice label at the top bead */}
                <text x={CX} y={CY - r - 14} fill={col} fontSize={22} fontFamily="monospace"
                  textAnchor="middle" opacity={Math.min(1, rev) * 0.9}>
                  {v.pulses}
                </text>
              </g>
            );
          })}

          {/* playhead */}
          {playing && (
            <g>
              <line x1={CX} y1={CY} x2={hx} y2={hy} stroke="#fff" strokeWidth={2.5} opacity={0.5} />
              <circle cx={hx} cy={hy} r={7} fill="#fff" style={{ filter: "drop-shadow(0 0 10px #fff)" }} />
            </g>
          )}

          {/* downbeat shockwave — expands out from the centre on the unison */}
          {playing && downPulse > 0.04 && (
            <circle cx={CX} cy={CY} r={18 + 150 * (1 - downPulse)} fill="none" stroke="#fff"
              strokeWidth={2.5} opacity={0.45 * downPulse} />
          )}
          {/* center core — pulses on the downbeat unison */}
          <circle cx={CX} cy={CY} r={14 + 26 * downPulse} fill="none" stroke="#fff"
            strokeWidth={2} opacity={0.2 + 0.6 * downPulse} />
          <circle cx={CX} cy={CY} r={8 + 6 * downPulse} fill="#fff" opacity={0.5 + 0.5 * downPulse}
            style={{ filter: `drop-shadow(0 0 ${8 + 22 * downPulse}px #fff)` }} />
        </svg>

        <div style={{ position: "absolute", bottom: 150, left: 60, right: 60, textAlign: "center",
          fontFamily: fonts.display, fontSize: 28, fontWeight: 300, color: theme.muted }}>
          each peak → an evenly-pulsed voice · they reunite at the <b style={{ color: "#fff", fontWeight: 700 }}>downbeat</b>
        </div>
        <div style={{ position: "absolute", bottom: 78, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
          biotuner · rhythm · scale2polyrhythm
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
