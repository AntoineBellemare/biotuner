import React, { useEffect, useRef } from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
  delayRender,
  continueRender,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import { quasicrystalField, vortexField, limitDenominator } from "../reels/interference";
import { lerpRatios, type Chord } from "../reels/cymatics";
import { sampleStops, type Stops } from "../reels/palettes";
import type { ReelData } from "../reels/reelData";

const FIELD_N = 300;

// Didactic acts: each is a visual mode spanning `chords` of the morph
// sequence, with a title and a sequence of teaching lines that fade through.
type Act = {
  mode: "quasi" | "vortex";
  nFold?: number;
  chords: number;
  title: string;
  lines: string[];
  stops: Stops;
};

const QUASI_STOPS: Stops = [
  [8, 10, 18], [46, 34, 78], [40, 96, 112],
  [150, 156, 150], [206, 200, 196], [228, 222, 214],
];
const QUASI7_STOPS: Stops = [
  [8, 14, 14], [30, 56, 50], [56, 110, 92],
  [134, 168, 120], [200, 200, 150], [228, 226, 196],
];
const QUASI12_STOPS: Stops = [
  [10, 8, 14], [52, 30, 44], [120, 70, 70],
  [186, 132, 96], [214, 184, 140], [232, 218, 196],
];

const ACTS: Act[] = [
  {
    mode: "quasi", nFold: 5, chords: 3, title: "FORBIDDEN SYMMETRY",
    stops: QUASI_STOPS,
    lines: [
      "a crystal can only repeat 2· 3· 4· or 6·fold",
      "5·fold is impossible in any periodic crystal",
      "yet a chord paints it effortlessly",
    ],
  },
  {
    mode: "quasi", nFold: 7, chords: 2, title: "SEVEN·FOLD",
    stops: QUASI7_STOPS,
    lines: [
      "7·fold — also forbidden to crystals",
      "the harmonics simply don't care",
    ],
  },
  {
    mode: "quasi", nFold: 12, chords: 2, title: "QUASICRYSTALS",
    stops: QUASI12_STOPS,
    lines: [
      "12·fold — ordered, but never repeating",
      "found in nature in 2011 — a Nobel Prize",
    ],
  },
];

// Cumulative chord boundaries: [0, 3, 5, 8].
const ACT_BOUNDS = ACTS.reduce<number[]>(
  (acc, a) => [...acc, acc[acc.length - 1] + a.chords],
  [0]
);
const TOTAL_CHORDS = ACT_BOUNDS[ACT_BOUNDS.length - 1];

/**
 * "Forbidden Symmetry" — a didactic reel. A chord's harmonics paint
 * quasicrystals (5- and 12-fold symmetries impossible in real crystals) and
 * optical-vortex spirals, with teaching captions. The field flows (animated
 * phases) and morphs between chords; acts dissolve into one another.
 */
export const ForbiddenSymmetry: React.FC<{ data: ReelData }> = ({ data }) => {
  const frame = useCurrentFrame();
  const { width, fps } = useVideoConfig();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const chords: Chord[] = data.chords;
  const seg = data.frames_per_segment;
  const outroFrames = data.outro_frames ?? 0;
  const chordFrames = TOTAL_CHORDS * seg;
  const inOutro = frame >= chordFrames;
  const outroLocal = inOutro
    ? Math.min(1, (frame - chordFrames) / Math.max(1, outroFrames))
    : 0;

  const rawSeg = Math.floor(frame / seg);
  const segIdx = inOutro ? TOTAL_CHORDS - 1 : rawSeg % TOTAL_CHORDS;
  const local = (frame % seg) / seg;
  const hold = data.hold_fraction ?? 0.3;
  const morphLocal = inOutro
    ? 0
    : hold > 0 ? Math.max(0, Math.min(1, (local - hold) / (1 - hold))) : local;

  // Which act owns this chord? (the outro holds the last act.)
  let actIdx = ACTS.length - 1;
  if (!inOutro) {
    for (let a = 0; a < ACTS.length; a++) {
      if (segIdx >= ACT_BOUNDS[a] && segIdx < ACT_BOUNDS[a + 1]) { actIdx = a; break; }
    }
  }
  const act = ACTS[actIdx];
  const actStartChord = ACT_BOUNDS[actIdx];
  const chordInAct = segIdx - actStartChord; // 0-based within act
  const isLastAct = actIdx === ACTS.length - 1;

  const fromC = chords[segIdx];
  const toC = !inOutro && chordInAct + 1 < act.chords ? chords[segIdx + 1] : fromC;

  // Eased time: motion decelerates to a full rest across the outro.
  const tOutroStart = chordFrames / fps;
  const tau = outroFrames / fps;
  const t = inOutro
    ? tOutroStart + tau * (1 - (1 - outroLocal) * (1 - outroLocal))
    : frame / fps;
  const rot = (act.mode === "quasi" ? 0.42 : 0.28) * t;
  const qRatios = lerpRatios(fromC.ratios, toC.ratios, morphLocal);

  // Chord-onset bloom — a quick brightness + scale pulse so each chord change
  // is felt as a beat (the patterns within an act are otherwise close).
  const bloom = inOutro ? 0 : Math.exp(-local * 6.5);

  // Act-local time for caption sequencing + act dissolves.
  const actStartFrame = actStartChord * seg;
  const actFrames = act.chords * seg;
  const actLocal = (frame - actStartFrame) / actFrames;
  // The last act does NOT fade out at its end — the outro continues it.
  const actFade = inOutro
    ? 1
    : interpolate(
        actLocal,
        [0, 0.06, isLastAct ? 1.1 : 0.94, isLastAct ? 1.2 : 1],
        [0, 1, 1, isLastAct ? 1 : 0],
        { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
      );
  // Plate opacity dissolves between acts, then fades out gracefully in the outro.
  const plateFade = inOutro
    ? interpolate(outroLocal, [0.72, 1], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
    : actFade;
  // Didactic captions hide during the outro; the closing line takes over.
  const contentFade = inOutro ? 0 : actFade;
  const closeOp = inOutro
    ? interpolate(outroLocal, [0.12, 0.34, 0.78, 1], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
    : 0;

  const gamma = (0.78 + 0.14 * Math.sin(Math.PI * morphLocal)) * (1 - 0.3 * bloom);

  useEffect(() => {
    const handle = delayRender("forbidden");
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (canvas && ctx) {
      let dens: Float32Array;
      if (act.mode === "quasi") {
        // Motion = rigid spin (rot) + breathing zoom (rings pulse in/out) +
        // a chiral wind (directionPhaseStep advances → the lattice swirls).
        const ph = qRatios.map((_, i) => t * (0.12 + 0.03 * i));
        dens = quasicrystalField(qRatios, FIELD_N, {
          nFold: act.nFold, phases: ph, extent: 1.5,
          basePeriod: 0.6 + 0.14 * Math.sin(t * 0.55),
          rotation: rot, directionPhaseStep: 0.16 * t,
        });
      } else {
        // Vortex: charges come from each DISCRETE chord (stable, no jitter);
        // spin via rotation and crossfade between chords during the morph.
        const vfield = (rr: number[]) =>
          vortexField(rr, FIELD_N, {
            phases: rr.map((_, i) => t * (0.18 + 0.05 * i)),
            extent: 2.0, beamWaist: 1.15, rotation: rot,
          });
        const dA = vfield(fromC.ratios);
        if (morphLocal > 0 && toC !== fromC) {
          const dB = vfield(toC.ratios);
          const e = 0.5 * (1 - Math.cos(Math.PI * morphLocal));
          dens = new Float32Array(dA.length);
          for (let k = 0; k < dA.length; k++) dens[k] = dA[k] * (1 - e) + dB[k] * e;
        } else {
          dens = dA;
        }
      }
      const img = ctx.createImageData(FIELD_N, FIELD_N);
      for (let k = 0; k < dens.length; k++) {
        const [r, g, b] = sampleStops(act.stops, dens[k], gamma);
        img.data[k * 4 + 0] = r;
        img.data[k * 4 + 1] = g;
        img.data[k * 4 + 2] = b;
        img.data[k * 4 + 3] = 255;
      }
      const off = document.createElement("canvas");
      off.width = FIELD_N; off.height = FIELD_N;
      off.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    }
    continueRender(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  // Didactic line: cycle through this act's lines over its duration.
  const lineSpan = 1 / act.lines.length;
  const lineIdx = Math.min(act.lines.length - 1, Math.floor(actLocal / lineSpan));
  const lineLocal = (actLocal - lineIdx * lineSpan) / lineSpan;
  const lineOp = interpolate(lineLocal, [0, 0.14, 0.86, 1], [0, 1, 1, 0], {
    extrapolateLeft: "clamp", extrapolateRight: "clamp",
  });

  // Mechanism line: connect chord → geometry.
  const mechanism =
    act.mode === "quasi"
      ? `${act.nFold}·fold · ${qRatios.length} harmonics`
      : `spiral arms  ${fromC.ratios.map((r) => limitDenominator(r, 20)[0]).join(" · ")}`;

  const plate = Math.min(width * 0.9, 1000);
  const introFade = interpolate(frame, [0, 14], [0, 1], { extrapolateRight: "clamp" });

  return (
    <AbsoluteFill style={{ opacity: introFade, backgroundColor: "#05060d" }}>
      <Backdrop />

      {/* Plate — a tiny scale punch on each chord onset (bloom). */}
      <div
        style={{
          position: "absolute", top: "47%", left: "50%",
          transform: `translate(-50%, -50%) scale(${1 + 0.03 * bloom})`,
          width: plate, height: plate, borderRadius: 12, overflow: "hidden",
          opacity: plateFade,
          boxShadow: `0 0 ${110 + 70 * bloom}px rgba(120,150,220,${0.22 + 0.18 * bloom})`,
        }}
      >
        <canvas
          ref={canvasRef}
          width={FIELD_N * 3}
          height={FIELD_N * 3}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
      </div>

      {/* Act title */}
      <div
        style={{
          position: "absolute", top: 116, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 56, fontWeight: 800,
          letterSpacing: 2, color: theme.ink,
          textShadow: "0 0 26px rgba(120,150,220,0.35)", opacity: contentFade,
        }}
      >
        {act.title}
      </div>

      {/* Chord name — changes (and pops) on every chord, so the change is felt. */}
      <div
        style={{
          position: "absolute", top: 188, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 28, fontWeight: 500, letterSpacing: 4,
          color: theme.accent, opacity: contentFade,
          transform: `scale(${1 + 0.08 * bloom})`,
          textShadow: `0 0 ${10 + 26 * bloom}px ${theme.accent}`,
        }}
      >
        {fromC.name.toUpperCase()}
      </div>

      {/* Didactic line (cycles through the act's teaching lines) */}
      <div
        style={{
          position: "absolute", bottom: 188, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.display, fontSize: 34, fontWeight: 300,
          letterSpacing: 0.5, color: theme.ink, padding: "0 70px",
          opacity: lineOp * contentFade,
        }}
      >
        {act.lines[lineIdx]}
      </div>

      {/* Mechanism line: chord → geometry */}
      <div
        style={{
          position: "absolute", bottom: 132, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 24, letterSpacing: 2,
          color: theme.accent, opacity: 0.85 * contentFade,
        }}
      >
        {mechanism}
      </div>

      {/* Closing card (outro) */}
      <div
        style={{
          position: "absolute", top: "50%", left: 0, right: 0, textAlign: "center",
          transform: "translateY(-50%)", fontFamily: fonts.display,
          fontSize: 48, fontWeight: 300, letterSpacing: 1, color: theme.ink,
          opacity: closeOp, padding: "0 60px",
          textShadow: "0 0 30px rgba(120,150,220,0.4)",
        }}
      >
        every chord — a crystal
      </div>

      {/* Footer */}
      <div
        style={{
          position: "absolute", bottom: 70, left: 0, right: 0, textAlign: "center",
          fontFamily: fonts.mono, fontSize: 22, letterSpacing: 3,
          color: theme.muted, opacity: 0.7,
        }}
      >
        biotuner · harmonic geometry
      </div>
    </AbsoluteFill>
  );
};
