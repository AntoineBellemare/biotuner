import React from "react";
import {
  AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame, useVideoConfig,
  interpolate, interpolateColors,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { MetricIntro } from "../components/MetricIntro";
import { theme, fonts } from "../theme";

/**
 * "Pitch = Rhythm" — biotuner's beat_envelope. Two tones Δf apart beat Δf times
 * a second. Sweep Δf up: the beat envelope ripples faster and faster, and as it
 * crosses the ear's ~20 Hz threshold a rhythm fuses into a pitch. One slider
 * spans the whole continuum of musical time (Stockhausen).
 */
const TEAL = "#6fd6c4", GOLD = "#f2c14e", CORAL = "#e8746a";
const TITLE = 84;
const INTRO = 18;
const PLAY = 470;
const TAIL = 44;
export const TOTAL_PITCHRHYTHM = TITLE + INTRO + PLAY + TAIL;

const D_LO = 1.5, D_HI = 38, THRESH = 20;
const TW = 0.7, FC_VIS = 46;       // window seconds, visual carrier (slowed for display)
const PAD = 80;
const clampOpt = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

const smooth = (x: number) => { const c = Math.max(0, Math.min(1, x)); return c * c * (3 - 2 * c); };
export const deltaAt = (u: number) => D_LO * Math.pow(D_HI / D_LO, smooth((u - 0.12) / 0.76));

export const PitchRhythm: React.FC = () => {
  const frame = useCurrentFrame();
  const { width } = useVideoConfig();
  const sf = frame - TITLE;
  const introFade = interpolate(sf, [0, 16], [0, 1], clampOpt);

  const u = interpolate(sf, [INTRO, INTRO + PLAY], [0, 1], clampOpt);
  const D = deltaAt(u);
  const beats = D;                                  // beats per second = Δf
  const isPitch = D >= THRESH;
  const mix = interpolate(D, [THRESH - 5, THRESH + 5], [0, 1], clampOpt);
  const col = interpolateColors(mix, [0, 1], [TEAL, CORAL]);

  // beating signal across a 0.7 s window
  const cy = 780, amp = 230, W = width - 2 * PAD, N = 720;
  let env = "", car = "";
  for (let i = 0; i <= N; i++) {
    const x = PAD + (i / N) * W, t = (i / N) * TW;
    const e = Math.abs(Math.cos(Math.PI * D * t));
    env += `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${(cy - e * amp).toFixed(1)} `;
  }
  for (let i = N; i >= 0; i--) {
    const x = PAD + (i / N) * W, t = (i / N) * TW;
    env += `L ${x.toFixed(1)} ${(cy + Math.abs(Math.cos(Math.PI * D * t)) * amp).toFixed(1)} `;
  }
  for (let i = 0; i <= N; i++) {
    const x = PAD + (i / N) * W, t = (i / N) * TW;
    const e = Math.abs(Math.cos(Math.PI * D * t));
    car += `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${(cy - e * Math.sin(2 * Math.PI * FC_VIS * t) * amp).toFixed(1)} `;
  }

  // continuum bar (log scale)
  const barY = 1300, barL = PAD + 20, barR = width - PAD - 20;
  const xOfD = (d: number) => barL + (Math.log(d) - Math.log(D_LO)) / (Math.log(D_HI) - Math.log(D_LO)) * (barR - barL);

  return (
    <AbsoluteFill style={{ backgroundColor: "#06070e" }}>
      <Sequence from={TITLE}><Audio src={staticFile("audio/pitch_rhythm.wav")} /></Sequence>
      <Backdrop />
      <MetricIntro frame={frame} dur={TITLE} title="Pitch = Rhythm" hook="speed up a rhythm and it becomes a pitch" />

      <AbsoluteFill style={{ opacity: introFade }}>
        <div style={{ position: "absolute", top: 100, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 46, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
          two tones, <b style={{ fontWeight: 800, color: col }}>Δf</b> apart
        </div>
        <div style={{ position: "absolute", top: 174, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 26, letterSpacing: 2, color: theme.muted }}>
          beat {beats < 10 ? beats.toFixed(1) : Math.round(beats)} times / second
        </div>

        <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
          <line x1={PAD} y1={cy} x2={width - PAD} y2={cy} stroke="rgba(180,200,230,0.14)" strokeWidth={1.5} />
          <path d={env} fill={col} opacity={0.16} />
          <path d={car} fill="none" stroke={col} strokeWidth={2.2}
            style={{ filter: `drop-shadow(0 0 6px ${col}88)` }} />

          {/* continuum bar */}
          <line x1={barL} y1={barY} x2={barR} y2={barY} stroke="rgba(180,200,230,0.25)" strokeWidth={3} strokeLinecap="round" />
          <line x1={xOfD(THRESH)} y1={barY - 26} x2={xOfD(THRESH)} y2={barY + 26} stroke="#fff" strokeWidth={2} opacity={0.6} strokeDasharray="3 5" />
          <text x={xOfD(THRESH)} y={barY - 38} fill="#fff" fontSize={18} fontFamily="monospace" textAnchor="middle" opacity={0.7}>~20 Hz · ear's threshold</text>
          <text x={barL} y={barY + 52} fill={TEAL} fontSize={22} fontFamily="monospace">rhythm</text>
          <text x={barR} y={barY + 52} fill={CORAL} fontSize={22} fontFamily="monospace" textAnchor="end">pitch</text>
          <circle cx={xOfD(D)} cy={barY} r={13} fill={col} style={{ filter: `drop-shadow(0 0 12px ${col})` }} />
        </svg>

        {/* big state label + Δf readout */}
        <div style={{ position: "absolute", top: 980, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 96, fontWeight: 800, letterSpacing: 6,
          color: col, textShadow: `0 0 34px ${col}66` }}>
          {isPitch ? "PITCH" : "RHYTHM"}
        </div>
        <div style={{ position: "absolute", top: 1110, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 30, letterSpacing: 2, color: theme.muted }}>
          Δf = {D < 10 ? D.toFixed(1) : Math.round(D)} Hz
        </div>

        <div style={{ position: "absolute", bottom: 150, left: 60, right: 60, textAlign: "center",
          fontFamily: fonts.display, fontSize: 28, fontWeight: 300, color: theme.muted }}>
          slow → your ear hears a <b style={{ color: TEAL, fontWeight: 700 }}>rhythm</b> · fast → it fuses into a <b style={{ color: CORAL, fontWeight: 700 }}>pitch</b>
        </div>
        <div style={{ position: "absolute", bottom: 78, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
          biotuner · rhythm · beat_envelope
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
