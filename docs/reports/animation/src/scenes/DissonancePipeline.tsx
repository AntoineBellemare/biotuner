import React from "react";
import {
  AbsoluteFill,
  Audio,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  Easing,
} from "remotion";
import { evolvePath } from "@remotion/paths";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import data from "../../public/diss_pipeline.json";

/**
 * "From signal to scale" — the biotuner Sethares pipeline, twice:
 *   waveform → spectrum → spectral PEAKS → the dissonance COMB (sweep a copy of
 *   the peaks unison→octave, summing Plomp-Levelt roughness = biotuner
 *   dissmeasure) → the curve's valleys = the scale that signal implies.
 * Shown for a synthetic signal, then for real EEG — same math on brain rhythms.
 */
type Sig = typeof data.signal;
const SIGNALS: Sig[] = [data.signal as Sig, data.eeg as Sig];
const GRID = data.grid as { num: number; den: number; ratio: number }[];
const TEAL = "#6fd6c4";
const HOT = "#e8746a";
const GOLD = "#f2c14e";

const INTRO = 24;
const BEAT = 408;
const OUTRO = 30;
export const TOTAL_DISSPIPE = INTRO + SIGNALS.length * BEAT + OUTRO;

const FMAX_COMB = 6.6;

export const DissonancePipeline: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();

  const local = frame - INTRO;
  const ti = Math.max(0, Math.min(SIGNALS.length - 1, Math.floor(local / BEAT)));
  const bl = local - ti * BEAT;
  const d = SIGNALS[ti];
  const isEeg = ti === 1;

  // phase reveals
  const waveP = interpolate(bl, [0, 90], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const specP = interpolate(bl, [80, 170], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const peaksP = spring({ frame: bl - 160, fps, config: { damping: 12, stiffness: 110 } });
  const combP = interpolate(bl, [205, 235], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const sweepRaw = interpolate(bl, [235, 372], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.sin) });
  const settled = bl >= 372;
  const alpha = 1 + sweepRaw;

  const leftPad = 96, axisW = width - 2 * leftPad;

  // ── waveform ──────────────────────────────────────────────────────────────
  const wave = d.wave as number[];
  const waveCY = 320, waveAmp = 66;
  const wfx = (i: number) => leftPad + (i / (wave.length - 1)) * axisW;
  let wavePath = wave.map((v, i) => `${i === 0 ? "M" : "L"} ${wfx(i).toFixed(1)} ${(waveCY - v * waveAmp).toFixed(1)}`).join(" ");
  const wd = evolvePath(waveP, wavePath);

  // ── spectrum + peaks ─────────────────────────────────────────────────────
  const sf = d.spec_f as number[], sm = d.spec_mag as number[];
  const specBase = 690, specH = 210;
  const sfx = (f: number) => leftPad + (f / d.fmax) * axisW;
  const sfy = (m: number) => specBase - m * specH;
  let specPath = `M ${leftPad} ${specBase} ` + sf.map((f, i) => `L ${sfx(f).toFixed(1)} ${sfy(sm[i] * specP).toFixed(1)}`).join(" ") + ` L ${width - leftPad} ${specBase} Z`;

  // ── comb (peaks as ratios) + dissonance curve ────────────────────────────
  const peaks = d.peaks as { f: number; a: number }[];
  const minf = Math.min(...peaks.map((p) => p.f));
  const ratios = peaks.map((p) => ({ r: p.f / minf, a: p.a }));
  const combY = 880;
  const cmbx = (r: number) => leftPad + ((r - 1) / (FMAX_COMB - 1)) * axisW;
  const tlen = (a: number) => 70 + 90 * a;

  const curve = d.curve as number[];
  const valleys = d.valleys as { alpha: number; label: string; on_grid: boolean; diss: number }[];
  const plotPad = 96, plotW = width - 2 * plotPad;
  const cBot = 1430, cTop = 1080;
  const ccx = (a: number) => plotPad + (a - 1) * plotW;
  const ccy = (v: number) => cBot - v * (cBot - cTop);
  let curvePath = curve.map((v, i) => `${i === 0 ? "M" : "L"} ${ccx(1 + i / (curve.length - 1)).toFixed(1)} ${ccy(v).toFixed(1)}`).join(" ");
  const cd = evolvePath(sweepRaw, curvePath);
  const dissAt = (a: number) => curve[Math.max(0, Math.min(curve.length - 1, Math.round((a - 1) * (curve.length - 1))))];

  const introFade = interpolate(frame, [0, 16], [0, 1], { extrapolateRight: "clamp" });

  const stageLabel =
    bl < 160 ? "1 · spectral peaks" : bl < 235 ? "2 · build the comb" : "3 · dissonance → scale";

  return (
    <AbsoluteFill style={{ opacity: introFade, backgroundColor: "#06070e" }}>
      <Audio src={staticFile("audio/diss_pipeline.wav")} />
      <Backdrop />

      <div style={{ position: "absolute", top: 64, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.display, fontSize: 44, fontWeight: 300, letterSpacing: 1, color: theme.ink }}>
        from <b style={{ fontWeight: 800 }}>{d.label}</b> to a scale
      </div>
      <div style={{ position: "absolute", top: 130, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 24, letterSpacing: 3, color: isEeg ? HOT : TEAL }}>
        {d.sublabel} · {stageLabel}
      </div>

      <svg width="100%" height="100%" viewBox={`0 0 ${width} 1920`} style={{ position: "absolute", inset: 0 }}>
        {/* waveform */}
        <path d={wavePath} fill="none" stroke={isEeg ? HOT : TEAL} strokeWidth={2.5}
          strokeDasharray={wd.strokeDasharray} strokeDashoffset={wd.strokeDashoffset} opacity={0.9} />
        <text x={leftPad} y={waveCY - 110} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2}>waveform</text>

        {/* spectrum */}
        <path d={specPath} fill="rgba(120,150,210,0.16)" stroke="rgba(150,180,220,0.55)" strokeWidth={2} opacity={specP} />
        <line x1={leftPad} y1={specBase} x2={width - leftPad} y2={specBase} stroke="rgba(180,200,230,0.18)" strokeWidth={2} opacity={specP} />
        <text x={leftPad} y={specBase - specH - 16} fill={theme.muted} fontSize={22} fontFamily="monospace" letterSpacing={2} opacity={specP}>spectrum → peaks</text>
        {/* peak markers */}
        {peaks.map((p, i) => {
          const x = sfx(p.f);
          const pop = Math.max(0, Math.min(1, peaksP));
          return (
            <g key={`pk${i}`} opacity={pop}>
              <line x1={x} y1={specBase} x2={x} y2={sfy(p.a * p.a)} stroke={GOLD} strokeWidth={3} />
              <circle cx={x} cy={sfy(p.a * p.a)} r={6 + 3 * pop} fill={GOLD} style={{ filter: `drop-shadow(0 0 8px ${GOLD})` }} />
              <text x={x} y={specBase + 28} fill={GOLD} fontSize={18} fontFamily="monospace" textAnchor="middle">{p.f}Hz</text>
            </g>
          );
        })}

        {/* comb — peaks (teal, up) + swept copy (gold, down) */}
        {combP > 0 && (
          <g opacity={combP}>
            <line x1={leftPad} y1={combY} x2={width - leftPad} y2={combY} stroke="rgba(180,200,230,0.16)" strokeWidth={2} />
            {ratios.map((rt, i) => (
              <line key={`cl${i}`} x1={cmbx(rt.r)} y1={combY} x2={cmbx(rt.r)} y2={combY - tlen(rt.a)}
                stroke={TEAL} strokeWidth={4} strokeLinecap="round" opacity={0.5 + 0.4 * rt.a} />
            ))}
            {ratios.map((rt, i) => cmbx(rt.r * alpha) <= width - leftPad && (
              <line key={`cu${i}`} x1={cmbx(rt.r * alpha)} y1={combY} x2={cmbx(rt.r * alpha)} y2={combY + tlen(rt.a)}
                stroke={GOLD} strokeWidth={4} strokeLinecap="round" opacity={0.5 + 0.4 * rt.a} />
            ))}
            <text x={leftPad} y={combY - 150} fill={theme.muted} fontSize={20} fontFamily="monospace" letterSpacing={2}>peak comb · sweep a copy →</text>
          </g>
        )}

        {/* dissonance curve + just grid */}
        {combP > 0 && (
          <g opacity={combP}>
            {GRID.map((g) => (
              <line key={`${g.num}/${g.den}`} x1={ccx(g.ratio)} y1={cTop - 6} x2={ccx(g.ratio)} y2={cBot}
                stroke="rgba(150,175,215,0.2)" strokeWidth={1.4} strokeDasharray="2 7" />
            ))}
            {GRID.map((g) => (
              <text key={`t${g.num}`} x={ccx(g.ratio)} y={cBot + 30} fill="rgba(160,185,225,0.55)" fontSize={18}
                fontFamily="monospace" textAnchor="middle">{g.num}/{g.den}</text>
            ))}
            <line x1={plotPad} y1={cBot} x2={width - plotPad} y2={cBot} stroke="rgba(180,200,230,0.18)" strokeWidth={2} />
            <path d={curvePath} fill="none" stroke="rgba(150,175,215,0.28)" strokeWidth={2} />
            <path d={curvePath} fill="none" stroke={isEeg ? HOT : GOLD} strokeWidth={4}
              strokeDasharray={cd.strokeDasharray} strokeDashoffset={cd.strokeDashoffset}
              style={{ filter: `drop-shadow(0 0 8px ${isEeg ? HOT : GOLD})` }} />
            {/* valleys (the derived scale) */}
            {valleys.map((v, i) => (settled || alpha >= v.alpha - 0.005) && (
              <g key={`v${i}`}>
                <circle cx={ccx(v.alpha)} cy={ccy(v.diss)} r={v.on_grid ? 9 : 7}
                  fill={v.on_grid ? GOLD : HOT} opacity={0.95}
                  style={{ filter: `drop-shadow(0 0 10px ${v.on_grid ? GOLD : HOT})` }} />
              </g>
            ))}
            {/* sweep marker */}
            {!settled && (
              <circle cx={ccx(alpha)} cy={ccy(dissAt(alpha))} r={11} fill={isEeg ? HOT : GOLD}
                style={{ filter: `drop-shadow(0 0 12px ${isEeg ? HOT : GOLD})` }} />
            )}
            <text x={plotPad} y={cTop - 18} fill={theme.muted} fontSize={20} fontFamily="monospace" letterSpacing={2}>sensory dissonance · biotuner dissmeasure</text>
          </g>
        )}
      </svg>

      {/* verdict */}
      <div style={{ position: "absolute", bottom: 120, left: 60, right: 60, textAlign: "center",
        fontFamily: fonts.display, fontSize: 30, fontWeight: 300, color: isEeg ? HOT : GOLD,
        opacity: settled ? 1 : 0.25 }}>
        {valleys.length} consonant intervals — {isEeg ? "your brain's own scale" : "this signal's scale"}
      </div>
      <div style={{ position: "absolute", bottom: 64, left: 0, right: 0, textAlign: "center",
        fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3, color: theme.muted, opacity: 0.7 }}>
        biotuner · peaks → dissonance → tuning
      </div>
    </AbsoluteFill>
  );
};
