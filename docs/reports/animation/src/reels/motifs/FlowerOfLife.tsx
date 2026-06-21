import React from "react";
import { interpolate, useCurrentFrame, spring, useVideoConfig } from "remotion";

/**
 * Flower of Life intro motif. The classic 19-circle hexagonal arrangement
 * draws in ring by ring, slowly rotating; a second, fainter counter-rotating
 * copy creates a moiré "illusion" shimmer where the two lattices overlap.
 *
 * Reused by ReelIntro; the stroke colour is themeable so each reel's motif
 * can echo its palette.
 */

type Circle = { x: number; y: number; ring: number };

// Build the 19 unit-radius circle centres (radius = 1).
function flowerCenters(): Circle[] {
  const c: Circle[] = [{ x: 0, y: 0, ring: 0 }];
  const deg = (d: number) => (d * Math.PI) / 180;
  // Ring 1 — 6 circles at distance 1.
  for (let k = 0; k < 6; k++) {
    const a = deg(k * 60);
    c.push({ x: Math.cos(a), y: Math.sin(a), ring: 1 });
  }
  // Ring 2 — 6 at distance √3 (offset 30°) + 6 at distance 2.
  for (let k = 0; k < 6; k++) {
    const a = deg(30 + k * 60);
    c.push({ x: Math.SQRT2 * 1.2247 * Math.cos(a), y: Math.SQRT2 * 1.2247 * Math.sin(a), ring: 2 });
  }
  for (let k = 0; k < 6; k++) {
    const a = deg(k * 60);
    c.push({ x: 2 * Math.cos(a), y: 2 * Math.sin(a), ring: 2 });
  }
  return c;
}

const CENTERS = flowerCenters();

export const FlowerOfLife: React.FC<{
  size: number; // px diameter of the motif
  color?: string;
  glow?: string;
  /** 0→1 overall reveal progress (drives ring-staggered draw-in). */
  progress: number;
}> = ({ size, color = "#7ad6c1", glow = "rgba(122,214,193,0.5)", progress }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Map unit coords (circle r = 1, flower spans ±3) into the SVG box.
  const VB = 3.4; // half-extent
  const unit = size / (2 * VB); // px per unit
  const cx = size / 2;
  const cy = size / 2;
  const r = unit; // circle radius in px

  const rot = frame * 0.12; // slow base rotation (deg)
  const breathe = 1 + 0.015 * Math.sin(frame * 0.06);

  const renderRing = (rotDeg: number, opacityScale: number, strokeW: number) => (
    <g
      transform={`rotate(${rotDeg} ${cx} ${cy}) scale(${breathe}) translate(${
        cx - cx * breathe
      } ${cy - cy * breathe})`}
    >
      {CENTERS.map((c, i) => {
        // Ring-staggered reveal: ring 0 first, then 1, then 2.
        const start = c.ring * 0.22;
        const local = interpolate(progress, [start, start + 0.5], [0, 1], {
          extrapolateLeft: "clamp",
          extrapolateRight: "clamp",
        });
        const s = spring({
          frame: Math.max(0, frame - c.ring * 6),
          fps,
          config: { damping: 14, mass: 0.6 },
          durationInFrames: 30,
        });
        const appear = Math.min(local, s);
        return (
          <circle
            key={i}
            cx={cx + c.x * unit}
            cy={cy + c.y * unit}
            r={r * (0.6 + 0.4 * appear)}
            fill="none"
            stroke={color}
            strokeWidth={strokeW}
            opacity={appear * opacityScale}
          />
        );
      })}
      {/* enclosing rings */}
      <circle cx={cx} cy={cy} r={3 * unit} fill="none" stroke={color}
        strokeWidth={strokeW * 1.1} opacity={progress * 0.55 * opacityScale} />
      <circle cx={cx} cy={cy} r={3 * unit + r * 0.18} fill="none" stroke={color}
        strokeWidth={strokeW * 0.7} opacity={progress * 0.35 * opacityScale} />
    </g>
  );

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{ filter: `drop-shadow(0 0 14px ${glow})`, overflow: "visible" }}
    >
      {/* Base lattice */}
      {renderRing(rot, 1.0, 2.2)}
      {/* Counter-rotating faint copy → moiré shimmer illusion */}
      {renderRing(-rot * 1.6, 0.32, 1.4)}
    </svg>
  );
};
