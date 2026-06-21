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
import { theme, fonts, getChordHue } from "../theme";
import { cymaticsDensity, lerpRatios, tidepool, type Chord } from "../reels/cymatics";
import type { ReelData } from "../reels/reelData";

// Four CYMATICS variants of the same plate — different mode / symmetry /
// density so the four panels are four readings of one Chladni field.
type Variant = {
  label: string;
  opts: Parameters<typeof cymaticsDensity>[2];
};
// Four distinct cymatics READINGS of the same chord: the classic D4-folded
// nodal lattice, the symmetric (+) plate mode, the raw unfolded plate, and
// the open root-pairs-only lattice.
const VARIANTS: Variant[] = [
  { label: "nodal · D4", opts: { symmetry: "d4_max", antisymmetric: true } },
  { label: "symmetric", opts: { symmetry: "d4_max", antisymmetric: false } },
  { label: "raw plate", opts: { symmetry: "none", antisymmetric: true } },
  { label: "root modes", opts: { symmetry: "d4_max", antisymmetric: true, pairSubset: "root" } },
];

/**
 * 2×2 grid showing the SAME chord through four cymatics readings
 * (nodal / symmetric-mode / raw-asymmetric / antinodal) while the chord
 * morphs through a long sequence.
 */
export const QuadGeometry: React.FC<{ data: ReelData }> = ({ data }) => {
  const frame = useCurrentFrame();
  const { width } = useVideoConfig();
  const refs = [
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
  ];

  const chords: Chord[] = data.chords;
  const n = chords.length;
  const seg = data.frames_per_segment;
  const rawSeg = Math.floor(frame / seg);
  const segIdx = data.loop ? rawSeg % n : Math.min(rawSeg, n - 1);
  const local = (frame % seg) / seg;
  const fromChord = chords[segIdx];
  const toChord = chords[(segIdx + 1) % n];
  const hold = data.hold_fraction ?? 0.5;
  const morphLocal =
    hold > 0 ? Math.max(0, Math.min(1, (local - hold) / (1 - hold))) : local;
  const ratios = lerpRatios(fromChord.ratios, toChord.ratios, morphLocal);
  const gamma = 0.72 + 0.4 * Math.sin(Math.PI * morphLocal);

  useEffect(() => {
    const handle = delayRender("quad cymatics");
    const N = 190;
    VARIANTS.forEach((v, q) => {
      const canvas = refs[q].current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dens = cymaticsDensity(ratios, N, v.opts);
      const img = ctx.createImageData(N, N);
      for (let k = 0; k < dens.length; k++) {
        const [r, gg, b] = tidepool(dens[k], gamma);
        img.data[k * 4 + 0] = r;
        img.data[k * 4 + 1] = gg;
        img.data[k * 4 + 2] = b;
        img.data[k * 4 + 3] = 255;
      }
      const off = document.createElement("canvas");
      off.width = N;
      off.height = N;
      off.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    });
    continueRender(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  const cell = Math.min(width * 0.46, 500);
  const labelHue = getChordHue(fromChord.label);
  const toHue = getChordHue(toChord.label);
  const toOp = interpolate(local, [hold + 0.04, 0.96], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const introFade = interpolate(frame, [0, 12], [0, 1], { extrapolateRight: "clamp" });

  return (
    <AbsoluteFill style={{ opacity: introFade }}>
      <Backdrop />
      <div
        style={{
          position: "absolute",
          top: 130,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          fontSize: 38,
          fontWeight: 300,
          letterSpacing: 1,
          color: theme.ink,
          opacity: 0.85,
          padding: "0 60px",
        }}
        dangerouslySetInnerHTML={{ __html: data.hook ?? "one chord, four cymatics" }}
      />
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          display: "grid",
          gridTemplateColumns: `repeat(2, ${cell}px)`,
          gridTemplateRows: `repeat(2, ${cell}px)`,
          gap: 18,
        }}
      >
        {VARIANTS.map((v, q) => (
          <div
            key={v.label}
            style={{
              position: "relative",
              width: cell,
              height: cell,
              borderRadius: 8,
              overflow: "hidden",
              boxShadow: `0 0 36px ${labelHue.glow}, inset 0 0 0 1px rgba(255,255,255,0.05)`,
            }}
          >
            <canvas
              ref={refs[q]}
              width={760}
              height={760}
              style={{ width: "100%", height: "100%", display: "block" }}
            />
            <div
              style={{
                position: "absolute",
                left: 12,
                bottom: 10,
                fontFamily: fonts.mono,
                fontSize: 20,
                letterSpacing: 1,
                color: "rgba(240,244,255,0.7)",
              }}
            >
              {v.label}
            </div>
          </div>
        ))}
      </div>
      <div
        style={{
          position: "absolute",
          bottom: 130,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          height: 90,
        }}
      >
        <div style={{ position: "absolute", left: 0, right: 0, opacity: 1 - toOp }}>
          <div style={{ fontSize: 64, fontWeight: 700, color: labelHue.primary,
            textShadow: `0 0 26px ${labelHue.glow}`, letterSpacing: -1 }}>
            {fromChord.name}
          </div>
        </div>
        <div style={{ position: "absolute", left: 0, right: 0, opacity: toOp }}>
          <div style={{ fontSize: 64, fontWeight: 700, color: toHue.primary,
            textShadow: `0 0 26px ${toHue.glow}`, letterSpacing: -1 }}>
            {toChord.name}
          </div>
        </div>
      </div>
      <div
        style={{
          position: "absolute",
          bottom: 64,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.mono,
          fontSize: 22,
          letterSpacing: 3,
          color: theme.muted,
          opacity: 0.8,
        }}
      >
        biotuner · harmonic geometry
      </div>
    </AbsoluteFill>
  );
};
