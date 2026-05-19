import React, { useEffect, useRef } from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  delayRender,
  continueRender,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import { geometryData } from "../geometry";

export const FRAMES_PER_CHORD_CHLADNI = 60; // 2 s @ 30 fps

/**
 * Shows the same HarmonicInput (chord) visualized through three plate
 * geometries simultaneously: rectangular, circular, and pentagon.
 * Cross-fades through all seven chords.
 */
export const ChladniFromInput: React.FC = () => {
  const frame = useCurrentFrame();
  const items = geometryData.scenes.chladni_from_input.items;

  const activeIdx = Math.min(
    Math.floor(frame / FRAMES_PER_CHORD_CHLADNI),
    items.length - 1
  );
  const local = (frame % FRAMES_PER_CHORD_CHLADNI) / FRAMES_PER_CHORD_CHLADNI;

  const rectRef = useRef<HTMLCanvasElement>(null);
  const circRef = useRef<HTMLCanvasElement>(null);
  const polyRef = useRef<HTMLCanvasElement>(null);
  const canvasRefs = [rectRef, circRef, polyRef];

  useEffect(() => {
    const handle = delayRender("Painting Chladni from-input plates");
    const item = items[activeIdx];
    const next = items[Math.min(activeIdx + 1, items.length - 1)];

    canvasRefs.forEach((ref, pi) => {
      const canvas = ref.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      // Always clear so previous frame content doesn't bleed through.
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      paintLayer(ctx, item.plates[pi].field, item.plates[pi].resolution, 1);
      if (next !== item && local > 0.01) {
        paintLayer(ctx, next.plates[pi].field, next.plates[pi].resolution, local);
      }
    });

    continueRender(handle);
  }, [frame, activeIdx, local, items]);

  const item = items[activeIdx];
  const labelOpacity = interpolate(frame % FRAMES_PER_CHORD_CHLADNI, [0, 18], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <Backdrop />

      {/* ── Three plates side-by-side ── */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "center",
          gap: 36,
          paddingBottom: 130,
          paddingTop: 60,
        }}
      >
        {item.plates.map((plate, pi) => {
          const isCirc = plate.type === "circ";
          const isPoly = plate.type === "poly5";
          const borderRadius = isCirc ? "50%" : isPoly ? "12%" : 6;
          return (
            <div
              key={pi}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 14,
              }}
            >
              {/* Plate-type label */}
              <div
                style={{
                  fontSize: 17,
                  fontFamily: fonts.mono,
                  color: isCirc
                    ? theme.cool
                    : isPoly
                    ? theme.accentSoft
                    : theme.muted,
                  letterSpacing: 1,
                  opacity: labelOpacity,
                }}
              >
                {plate.label}
              </div>

              {/* Canvas container */}
              <div
                style={{
                  width: 380,
                  height: 380,
                  borderRadius,
                  overflow: "hidden",
                  boxShadow: isCirc
                    ? `0 0 50px rgba(109,163,216,0.3), inset 0 0 0 1px ${theme.cool}33`
                    : isPoly
                    ? `0 0 50px rgba(232,214,138,0.25), inset 0 0 0 1px ${theme.accentSoft}33`
                    : `0 0 50px ${theme.glow}, inset 0 0 0 1px ${theme.muted}`,
                  transition: "border-radius 0.35s ease",
                }}
              >
                <canvas
                  ref={canvasRefs[pi]}
                  width={plate.resolution * 4}
                  height={plate.resolution * 4}
                  style={{ width: "100%", height: "100%", display: "block" }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Chord label ── */}
      <div
        style={{
          position: "absolute",
          bottom: 64,
          left: 0,
          right: 0,
          textAlign: "center",
          opacity: labelOpacity,
          fontFamily: fonts.display,
        }}
      >
        <div
          style={{
            fontSize: 48,
            fontWeight: 700,
            color: theme.ink,
            letterSpacing: -1,
            textShadow: `0 0 28px ${theme.glow}`,
          }}
        >
          {item.chord_name}
        </div>
        <div
          style={{
            fontSize: 24,
            fontWeight: 300,
            color: theme.accentSoft,
            marginTop: 4,
            fontFamily: fonts.mono,
            letterSpacing: 3,
          }}
        >
          {item.ratios_str}
        </div>
        <div
          style={{
            fontSize: 16,
            fontWeight: 300,
            color: theme.muted,
            marginTop: 4,
            fontFamily: fonts.mono,
          }}
        >
          chladni_from_input(inp, plate=…) · same HarmonicInput, three geometries
        </div>
      </div>

      {/* ── Chord progress dots ── */}
      <ChordDots total={items.length} active={activeIdx} />
    </AbsoluteFill>
  );
};

/**
 * Paint a single field layer onto the canvas at the given opacity.
 * Does NOT clear the canvas — call ctx.clearRect before the first layer.
 */
function paintLayer(
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
  ctx.globalAlpha = 1.0;
  ctx.drawImage(off, 0, 0, W, H);
}

/** Divergent palette matching the report figures: cool blue ↔ near-black ↔ warm gold. */
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

const ChordDots: React.FC<{ total: number; active: number }> = ({
  total,
  active,
}) => (
  <div
    style={{
      position: "absolute",
      top: 30,
      left: "50%",
      transform: "translateX(-50%)",
      display: "flex",
      gap: 8,
    }}
  >
    {Array.from({ length: total }, (_, i) => (
      <div
        key={i}
        style={{
          width: i === active ? 26 : 8,
          height: 8,
          borderRadius: 4,
          background: i === active ? theme.accent : theme.muted,
          opacity: i === active ? 1 : 0.35,
        }}
      />
    ))}
  </div>
);
