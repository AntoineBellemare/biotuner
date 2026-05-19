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
import { PedagogyCaption } from "../components/PedagogyCaption";
import { theme, fonts, getChordHue } from "../theme";
import { geometryData } from "../geometry";

export const FRAMES_PER_CHORD_EXPANDED = 75; // 2.5 s @ 30 fps

// Portrait / Reels pedagogy is scene-static: ONE title + ONE body for the
// whole 17.5-second slot. Seven chords cycle through visually but the
// text stays put so the viewer can finish reading.
const PEDAGOGY_IG_TITLE = "Chladni patterns";
const PEDAGOGY_IG_BODY =
  "A chord drives a plate. The **boundary geometry** picks which modes " +
  "resonate.";

const PEDAGOGY: Record<string, string> = {
  Major:
    "Real plates resonate at specific modes; a chord's harmonics excite " +
    "several modes simultaneously. The four shapes are the SAME chord on " +
    "FOUR different boundary geometries.",
  Minor:
    "The same chord's modes are constrained by the plate boundary. A " +
    "rectangle's modes are sin·sin standing waves; a circle's are Bessel " +
    "modes; a polygon's come from a numerical eigen-solver.",
  "Dom 7th":
    "Adding the 7th adds a fourth (m, n) mode pair. On the box, the field " +
    "extends into the third dimension, so we show a mid-z slice through the " +
    "volume.",
  "Maj 7th":
    "Maj 7 (8 : 10 : 12 : 15) packs four close-spaced modes. Plates with " +
    "more boundary symmetry show stronger constructive interference.",
  "Sus 4":
    "Sus 4 replaces the 3rd with a 4th. The mode pair is now (4, 3), and " +
    "the rectangular plate's nodal pattern shifts perceptibly.",
  Augmented:
    "Augmented stacks two major thirds. The 8/5 sub-octave introduces a " +
    "near-fifth interaction that breaks the chord's symmetry on the " +
    "circular plate.",
  "Dim 7th":
    "Dim 7 stacks minor 3rds. All four ratios contribute to the field, " +
    "producing a denser nodal pattern than the major chord above.",
};

/**
 * Same HarmonicInput rendered through FOUR plate geometries simultaneously
 * (rectangular, circular, pentagon, 3-D box mid-z slice). Cross-fades
 * across all 7 chords with pedagogical caption.
 */
export const ChladniExpanded: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const isPortrait = height > width;
  const chordNameSize = isPortrait ? 42 : 30;
  const ratiosSize = isPortrait ? 24 : 18;
  const plateLabelSize = isPortrait ? 24 : 13;
  const items = geometryData.scenes.chladni_expanded.items;

  const activeIdx = Math.min(
    Math.floor(frame / FRAMES_PER_CHORD_EXPANDED),
    items.length - 1
  );
  const local =
    (frame % FRAMES_PER_CHORD_EXPANDED) / FRAMES_PER_CHORD_EXPANDED;

  const refs = [
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
  ];

  // ── Portrait mode shows 2 plates only (rectangle + circle) — the user
  // wants pentagon and 3-D box dropped, and the remaining two stacked
  // vertically and much larger so they read on a phone screen.
  const visiblePlates = isPortrait
    ? items[activeIdx].plates.filter(
        (p) => p.type === "rect" || p.type === "circ"
      )
    : items[activeIdx].plates;

  useEffect(() => {
    const handle = delayRender("Painting Chladni expanded plates");
    const item = items[activeIdx];
    const next = items[Math.min(activeIdx + 1, items.length - 1)];

    const platesToPaint = isPortrait
      ? item.plates.filter(
          (p) => p.type === "rect" || p.type === "circ"
        )
      : item.plates;

    platesToPaint.forEach((plate, pi) => {
      const canvas = refs[pi]?.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      paintLayer(ctx, plate.field, plate.resolution, 1);
      if (next !== item && local > 0.01) {
        // Match next chord's plate by type so the cross-fade is between
        // the SAME boundary geometry across chords.
        const nextPlate =
          next.plates.find((p) => p.type === plate.type) ?? plate;
        paintLayer(ctx, nextPlate.field, nextPlate.resolution, local);
      }
    });

    continueRender(handle);
  }, [frame, activeIdx, local, items, isPortrait]);

  const item = items[activeIdx];
  const labelOpacity = interpolate(
    frame % FRAMES_PER_CHORD_EXPANDED,
    [0, 18],
    [0, 1],
    { extrapolateRight: "clamp" }
  );
  const hue = getChordHue(item.chord_name);

  return (
    <AbsoluteFill>
      <Backdrop />

      {/* Plate grid.
          - Landscape: 2×2 grid at 290 px each.
          - Portrait : single column, two BIG plates (rect over circ),
                       sized to fill the central reading zone of the phone
                       between the IG card (~y=460) and IG bottom UI safe
                       (~y=1700). */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "grid",
          gridTemplateColumns: isPortrait ? "1fr" : "repeat(2, auto)",
          gridTemplateRows: isPortrait ? "repeat(2, auto)" : "repeat(2, auto)",
          gap: isPortrait ? 48 : 28,
          alignContent: "center",
          justifyContent: "center",
          paddingTop: isPortrait ? 620 : 80,
          paddingBottom: isPortrait ? 180 : 220,
        }}
      >
        {visiblePlates.map((plate, pi) => {
          const isCirc = plate.type === "circ";
          const isPoly = plate.type === "poly5";
          const borderRadius = isCirc ? "50%" : isPoly ? "12%" : 6;
          const accentColor = isCirc
            ? theme.cool
            : isPoly
            ? theme.accentSoft
            : plate.type === "box3d"
            ? "#9bb89e"
            : theme.muted;
          const plateSize = isPortrait ? 480 : 290;
          return (
            <div
              key={pi}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: isPortrait ? 14 : 8,
              }}
            >
              <div
                style={{
                  fontSize: plateLabelSize,
                  fontFamily: fonts.mono,
                  color: accentColor,
                  letterSpacing: 1,
                  opacity: labelOpacity,
                }}
              >
                {plate.label}
              </div>
              <div
                style={{
                  width: plateSize,
                  height: plateSize,
                  borderRadius,
                  overflow: "hidden",
                  // Chord-identity glow on the OUTER shadow; plate-specific
                  // accent on the INNER border so each plate stays
                  // recognisable while the whole frame breathes the chord's
                  // signature colour.
                  boxShadow: `0 0 ${
                    isPortrait ? 48 : 30
                  }px ${hue.glow}, inset 0 0 0 1px ${accentColor}33`,
                }}
              >
                <canvas
                  ref={refs[pi]}
                  width={plate.resolution * 4}
                  height={plate.resolution * 4}
                  style={{ width: "100%", height: "100%", display: "block" }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Chord label — chord name takes the chord's signature hue. Hidden
          in portrait because the IG pedagogy card carries the chord
          identity via its hook + meta line and we don't want competing
          headers. */}
      <div
        style={{
          position: "absolute",
          top: 26,
          left: 0,
          right: 0,
          textAlign: "center",
          opacity: labelOpacity,
          fontFamily: fonts.display,
          display: isPortrait ? "none" : "block",
        }}
      >
        <div
          style={{
            fontSize: chordNameSize,
            fontWeight: 700,
            color: hue.primary,
            letterSpacing: -0.5,
            textShadow: `0 0 18px ${hue.glow}`,
          }}
        >
          {item.chord_name}
          <span
            style={{
              marginLeft: 18,
              fontSize: ratiosSize,
              fontFamily: fonts.mono,
              color: hue.soft,
              letterSpacing: 2,
              fontWeight: 300,
            }}
          >
            {item.ratios_str}
          </span>
        </div>
      </div>

      <PedagogyCaption
        title="Chladni, same chord, four plates"
        body={PEDAGOGY[item.chord_name] ?? PEDAGOGY["Major"]}
        meta="chladni_from_input(inp, plate='rectangular' | 'circular' | 'polygon' | 'box_3d')"
        bottom={48}
        accent={hue.primary}
        igTitle={PEDAGOGY_IG_TITLE}
        igBody={PEDAGOGY_IG_BODY}
      />

      {/* Progress dots — hidden in portrait (would collide with IG card
          at top:180; chord changes already announce themselves via the
          IG card's per-slot body line). */}
      <div
        style={{
          position: "absolute",
          top: 86,
          left: "50%",
          transform: "translateX(-50%)",
          display: isPortrait ? "none" : "flex",
          gap: 8,
        }}
      >
        {items.map((_, i) => (
          <div
            key={i}
            style={{
              width: i === activeIdx ? 26 : 8,
              height: 8,
              borderRadius: 4,
              background: i === activeIdx ? theme.accent : theme.muted,
              opacity: i === activeIdx ? 1 : 0.35,
            }}
          />
        ))}
      </div>
    </AbsoluteFill>
  );
};

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
