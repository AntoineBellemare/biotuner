import React, { useEffect, useRef } from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  delayRender,
  continueRender,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { Caption } from "../components/Caption";
import { theme, fonts } from "../theme";
import { geometryData } from "../geometry";

const FRAMES_PER_MODE = 35; // ~1.17 s @ 30 fps

/**
 * Chladni rectangular + circular fields rendered to a canvas heatmap.
 * Cross-blends between 12 modes (8 rect + 4 circ).
 * A small pill label shows "rect" vs "circ" so viewers know the plate type.
 */
export const ChladniMorph: React.FC = () => {
  const frame = useCurrentFrame();
  const items = geometryData.scenes.chladni_morph.items;
  const activeIdx = Math.min(
    Math.floor(frame / FRAMES_PER_MODE),
    items.length - 1
  );
  const local = (frame % FRAMES_PER_MODE) / FRAMES_PER_MODE;

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handle = delayRender("Painting Chladni field");
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      continueRender(handle);
      return;
    }
    const item = items[activeIdx];
    const next = items[Math.min(activeIdx + 1, items.length - 1)];
    paintField(ctx, item.field, item.resolution, 1 - local);
    if (next !== item) {
      paintField(ctx, next.field, next.resolution, local);
    }
    continueRender(handle);
  }, [frame, activeIdx, local, items]);

  const item = items[activeIdx];
  const isCirc = item.plate === "circ";

  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill
        style={{ justifyContent: "center", alignItems: "center" }}
      >
        <div
          style={{
            position: "relative",
            width: 700,
            height: 700,
            borderRadius: isCirc ? "50%" : 8,
            overflow: "hidden",
            boxShadow: `0 0 60px ${theme.glow}, inset 0 0 0 1px ${theme.muted}`,
            transition: "border-radius 0.4s ease",
          }}
        >
          <canvas
            ref={canvasRef}
            width={item.resolution * 4}
            height={item.resolution * 4}
            style={{ width: "100%", height: "100%", display: "block" }}
          />
        </div>
      </AbsoluteFill>

      {/* Plate-type pill */}
      <div
        style={{
          position: "absolute",
          top: 44,
          left: "50%",
          transform: "translateX(-50%)",
          padding: "6px 18px",
          borderRadius: 20,
          background: isCirc ? "rgba(109,163,216,0.18)" : "rgba(232,214,138,0.14)",
          border: `1px solid ${isCirc ? theme.cool : theme.accentSoft}`,
          color: isCirc ? theme.cool : theme.accentSoft,
          fontSize: 15,
          fontFamily: fonts.mono,
          letterSpacing: 1,
        }}
      >
        {isCirc ? "circular plate" : "rectangular plate"}
      </div>

      {/* Progress dots */}
      <ModeProgressDots total={items.length} active={activeIdx} />

      <Caption
        title={`chladni mode ${item.label}`}
        subtitle={isCirc ? "chladni_field_circular()" : "chladni_field_rectangular()"}
      />
    </AbsoluteFill>
  );
};

function paintField(
  ctx: CanvasRenderingContext2D,
  field: number[],
  resolution: number,
  alpha: number
) {
  const W = ctx.canvas.width;
  const H = ctx.canvas.height;
  const small = ctx.createImageData(resolution, resolution);
  for (let i = 0; i < field.length; i++) {
    const v = field[i];
    const [r, g, b] = colorFor(v);
    small.data[i * 4 + 0] = r;
    small.data[i * 4 + 1] = g;
    small.data[i * 4 + 2] = b;
    small.data[i * 4 + 3] = Math.round(255 * alpha);
  }
  const off = document.createElement("canvas");
  off.width = resolution;
  off.height = resolution;
  off.getContext("2d")!.putImageData(small, 0, 0);
  ctx.imageSmoothingEnabled = true;
  if (alpha >= 1) ctx.clearRect(0, 0, W, H);
  ctx.globalAlpha = 1.0;
  ctx.drawImage(off, 0, 0, W, H);
}

/** Divergent palette: cool blue (negative) → near-black (zero) → warm gold (positive). */
function colorFor(v: number): [number, number, number] {
  const x = Math.max(-1, Math.min(1, v));
  const ax = Math.abs(x);
  if (x >= 0) {
    return [
      Math.round(8 + ax * (232 - 8)),
      Math.round(12 + ax * (214 - 12)),
      Math.round(24 + ax * (138 - 24)),
    ];
  }
  return [
    Math.round(8 + ax * (109 - 8)),
    Math.round(12 + ax * (163 - 12)),
    Math.round(24 + ax * (216 - 24)),
  ];
}

const ModeProgressDots: React.FC<{ total: number; active: number }> = ({
  total,
  active,
}) => (
  <div
    style={{
      position: "absolute",
      bottom: 44,
      left: "50%",
      transform: "translateX(-50%)",
      display: "flex",
      gap: 6,
    }}
  >
    {Array.from({ length: total }, (_, i) => (
      <div
        key={i}
        style={{
          width: i === active ? 20 : 6,
          height: 6,
          borderRadius: 3,
          background: i === active ? theme.accent : theme.muted,
          opacity: i === active ? 1 : 0.35,
        }}
      />
    ))}
  </div>
);
