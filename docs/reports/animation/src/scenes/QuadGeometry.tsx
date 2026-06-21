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
import { lerpRatios, type Chord } from "../reels/cymatics";
import type { ReelData } from "../reels/reelData";
import {
  drawCymatics,
  drawLissajous,
  drawHarmonograph,
  drawInterference,
  GEOM_LABEL,
  type GeomKind,
} from "../reels/geometries";

const QUAD: GeomKind[] = [
  "cymatics",
  "lissajous",
  "harmonograph",
  "interference",
];

/**
 * 2×2 grid showing the SAME chord through all four geometries at once,
 * while the chord morphs through a long sequence — so you can compare how
 * each geometry responds, chord after chord.
 */
export const QuadGeometry: React.FC<{ data: ReelData }> = ({ data }) => {
  const frame = useCurrentFrame();
  const { width, fps } = useVideoConfig();
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
    const handle = delayRender("quad geometry");
    QUAD.forEach((kind, q) => {
      const canvas = refs[q].current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      if (kind === "cymatics") drawCymatics(ctx, ratios, gamma, 200);
      else if (kind === "lissajous") drawLissajous(ctx, ratios, frame, fps);
      else if (kind === "harmonograph")
        drawHarmonograph(ctx, ratios, frame, fps);
      else if (kind === "interference")
        drawInterference(ctx, ratios, frame, fps, 150);
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
  const introFade = interpolate(frame, [0, 12], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ opacity: introFade }}>
      <Backdrop />

      {/* Hook */}
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
        dangerouslySetInnerHTML={{ __html: data.hook ?? "four geometries, one chord" }}
      />

      {/* 2×2 grid */}
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
        {QUAD.map((kind, q) => {
          const isField = kind === "cymatics" || kind === "interference";
          return (
            <div
              key={kind}
              style={{
                position: "relative",
                width: cell,
                height: cell,
                borderRadius: 8,
                overflow: "hidden",
                boxShadow: isField
                  ? `0 0 36px ${labelHue.glow}, inset 0 0 0 1px rgba(255,255,255,0.05)`
                  : "inset 0 0 0 1px rgba(255,255,255,0.05)",
              }}
            >
              <canvas
                ref={refs[q]}
                width={isField ? 760 : 680}
                height={isField ? 760 : 680}
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
                {GEOM_LABEL[kind]}
              </div>
            </div>
          );
        })}
      </div>

      {/* Cycling chord name (the music) */}
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

      {/* Footer */}
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
