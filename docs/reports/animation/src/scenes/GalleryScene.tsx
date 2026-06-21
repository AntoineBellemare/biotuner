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
import { cymaticsDensity } from "../reels/cymatics";
import { GALLERY_PALETTES, sampleStops } from "../reels/palettes";
import type { ReelData } from "../reels/reelData";

const FIELD_N = 140;

/**
 * Gallery scene: a 3×3 grid of cymatics patterns per "phase". Used for the
 * Brain vs Heart galleries — a wall of intricate brains, then a wall of
 * ordered hearts, so the population-level difference is unmistakable.
 */
export const GalleryScene: React.FC<{ data: ReelData }> = ({ data }) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const phases = data.gallery_phases ?? [];
  const nPhase = phases.length || 1;
  const seg = data.frames_per_segment;

  const rawSeg = Math.floor(frame / seg);
  const phaseIdx = data.loop ? rawSeg % nPhase : Math.min(rawSeg, nPhase - 1);
  const local = (frame % seg) / seg;
  const phase = phases[phaseIdx] ?? { title: "", subtitle: "", accent: theme.accent, cells: [] };
  const cells = phase.cells.slice(0, 9);

  const refs = Array.from({ length: 9 }, () => useRef<HTMLCanvasElement>(null));
  const time = frame / data.fps;

  useEffect(() => {
    const handle = delayRender("gallery");
    cells.forEach((cell, i) => {
      const canvas = refs[i].current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      // Each cell is a living, vibrating plate (time-animated) with its own
      // phase offset (animSeed) and its own multi-hue palette.
      const dens = cymaticsDensity(cell.ratios, FIELD_N, {
        symmetry: "d4_max",
        time,
        animSeed: i + phaseIdx * 4,
      });
      const pal = GALLERY_PALETTES[i % GALLERY_PALETTES.length];
      const img = ctx.createImageData(FIELD_N, FIELD_N);
      for (let k = 0; k < dens.length; k++) {
        const [r, g, b] = sampleStops(pal, dens[k], 0.85);
        img.data[k * 4 + 0] = r;
        img.data[k * 4 + 1] = g;
        img.data[k * 4 + 2] = b;
        img.data[k * 4 + 3] = 255;
      }
      const off = document.createElement("canvas");
      off.width = FIELD_N;
      off.height = FIELD_N;
      off.getContext("2d")!.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
    });
    continueRender(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  // Phase fade in/out so the two walls swap cleanly.
  const phaseFade = interpolate(local, [0, 0.08, 0.9, 1], [0, 1, 1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const cellPx = Math.min(width * 0.29, 320);
  const introFade = interpolate(frame, [0, 12], [0, 1], { extrapolateRight: "clamp" });

  return (
    <AbsoluteFill style={{ opacity: introFade }}>
      <Backdrop />

      {/* Big title */}
      <div
        style={{
          position: "absolute",
          top: 150,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          opacity: phaseFade,
        }}
      >
        <div style={{ fontSize: 96, fontWeight: 800, letterSpacing: 4,
          color: phase.accent, textShadow: `0 0 30px ${phase.accent}66` }}>
          {phase.title}
        </div>
        <div style={{ marginTop: 6, fontSize: 30, fontWeight: 300,
          letterSpacing: 1, color: theme.muted }}>
          {phase.subtitle}
        </div>
      </div>

      {/* 3×3 grid */}
      <div
        style={{
          position: "absolute",
          top: "52%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          display: "grid",
          gridTemplateColumns: `repeat(3, ${cellPx}px)`,
          gridTemplateRows: `repeat(3, ${cellPx}px)`,
          gap: 14,
          opacity: phaseFade,
        }}
      >
        {cells.map((_, i) => {
          // Staggered cell reveal at the phase start.
          const cellIn = interpolate(
            local,
            [0.05 + i * 0.02, 0.12 + i * 0.02],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={i}
              style={{
                width: cellPx,
                height: cellPx,
                borderRadius: 6,
                overflow: "hidden",
                opacity: cellIn,
                transform: `scale(${0.92 + 0.08 * cellIn})`,
                boxShadow: `0 0 22px ${phase.accent}33, inset 0 0 0 1px rgba(255,255,255,0.05)`,
              }}
            >
              <canvas
                ref={refs[i]}
                width={FIELD_N * 3}
                height={FIELD_N * 3}
                style={{ width: "100%", height: "100%", display: "block" }}
              />
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div
        style={{
          position: "absolute",
          bottom: 80,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.mono,
          fontSize: 24,
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
