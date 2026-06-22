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
import { cymaticsDensity, lerpRatios, tidepool } from "../reels/cymatics";

export const FRAMES_PER_CHORD_EXPANDED = 75; // 2.5 s @ 30 fps

// Square: the reels' live cymatics (D4-symmetric), morphed via the chord's
// interpolated wavenumbers. Circle: the module's real Bessel field, morphed by
// interpolating the field itself between chords, then derived to sand (σ).
const SQ_N = 200;
const SIGMA = 0.11;

/** Parse integer plate wavenumbers from a "4 : 5 : 6"-style ratio string. */
function parseWavenumbers(s: string): number[] {
  const wn = s.split(/[^0-9.]+/).map(Number).filter((n) => n > 0);
  return wn.length ? wn : [4, 5, 6];
}

const PEDAGOGY_IG_TITLE = "Chladni patterns";
const PEDAGOGY_IG_BODY =
  "Sand on a vibrating plate gathers on the still lines. Watch it flow " +
  "as one chord **morphs** into the next.";

const PEDAGOGY: Record<string, string> = {
  Major:
    "Real plates resonate at specific modes; a chord's harmonics excite " +
    "several modes simultaneously. The two shapes are the SAME chord on " +
    "two different boundary geometries.",
  Minor:
    "The same chord's modes are constrained by the plate boundary. A " +
    "rectangle's modes are sin·sin standing waves; a circle's are Bessel " +
    "modes — the sand finds the nodal lines of each.",
  "Dom 7th":
    "Adding the 7th adds a fourth (m, n) mode pair, so the sand pattern " +
    "grows an extra family of nodal lines.",
  "Maj 7th":
    "Maj 7 (8 : 10 : 12 : 15) packs four close-spaced modes. More boundary " +
    "symmetry → stronger constructive interference, sharper sand lines.",
  "Sus 4":
    "Sus 4 replaces the 3rd with a 4th. The mode pair is now (4, 3), and " +
    "the rectangular plate's nodal pattern shifts perceptibly.",
  Augmented:
    "Augmented stacks two major thirds. The near-fifth interaction breaks " +
    "the chord's symmetry on the circular plate.",
  "Dim 7th":
    "Dim 7 stacks minor 3rds. All four ratios contribute, producing a " +
    "denser nodal pattern than the major chord above.",
};

/**
 * Same HarmonicInput rendered through real plate geometries (rectangular +
 * circular), painted as nodal-density "sand" in the tidepool palette — the
 * cymatics look used by the short-form reels. Cross-fades across all 7 chords.
 */
export const ChladniNice: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height, fps } = useVideoConfig();
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
  // Settle on each chord, then morph its wavenumbers into the next chord's.
  const HOLD = 0.45;
  const morphLocal = Math.max(0, Math.min(1, (local - HOLD) / (1 - HOLD)));
  const curWN = parseWavenumbers(items[activeIdx].ratios_str);
  const nextWN = parseWavenumbers(
    items[Math.min(activeIdx + 1, items.length - 1)].ratios_str
  );
  const wn = lerpRatios(curWN, nextWN, morphLocal);

  const refs = [
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
    useRef<HTMLCanvasElement>(null),
  ];

  // Portrait shows the two readable plates (rectangle + circle), big.
  const visiblePlates = isPortrait
    ? items[activeIdx].plates.filter(
        (p) => p.type === "rect" || p.type === "circ"
      )
    : items[activeIdx].plates;

  useEffect(() => {
    const handle = delayRender("Painting Chladni morph");
    const platesToPaint = isPortrait
      ? items[activeIdx].plates.filter(
          (p) => p.type === "rect" || p.type === "circ"
        )
      : items[activeIdx].plates;

    platesToPaint.forEach((plate, pi) => {
      const canvas = refs[pi]?.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (plate.type === "rect") {
        // Square: live symmetric cymatics on the interpolated wavenumbers.
        const dens = cymaticsDensity(wn, SQ_N, { symmetry: "d4_max" });
        paintDensity(ctx, dens, SQ_N, 1);
      } else {
        // Circle: the real Bessel field, kept organic + non-symmetric. It
        // morphs (interpolate this chord's field into the next) AND is resampled
        // through a slow, evolving sinusoidal domain WARP, so the sand pattern
        // undulates/flows like a rippling plate — continuous movement with no
        // net rotation, even while the square holds.
        const curField = plate.field;
        const nextItem = items[Math.min(activeIdx + 1, items.length - 1)];
        const nextField =
          nextItem.plates.find((p) => p.type === "circ")?.field ?? curField;
        const N = plate.resolution;
        const inv2s2 = 1 / (SIGMA * SIGMA);
        const t = frame / fps;
        const A = 5.5; // warp amplitude in grid cells
        const TAU = 2 * Math.PI;
        const fieldAt = (idx: number) =>
          (1 - morphLocal) * curField[idx] +
          morphLocal * (nextField[idx] ?? curField[idx]);
        const dens = new Float32Array(N * N);
        for (let oi = 0; oi < N; oi++) {
          const v = oi / (N - 1);
          for (let oj = 0; oj < N; oj++) {
            const u = oj / (N - 1);
            // two-octave travelling ripple → organic, swirling-but-not-spinning
            const sx =
              oj +
              A * Math.sin(TAU * 1.3 * v + 0.9 * t) +
              0.5 * A * Math.sin(TAU * 2.1 * u - 0.6 * t);
            const sy =
              oi +
              A * Math.sin(TAU * 1.1 * u + 0.7 * t + 1.7) +
              0.5 * A * Math.sin(TAU * 1.9 * v - 0.5 * t + 0.4);
            let f = 0;
            if (sx >= 0 && sx <= N - 1 && sy >= 0 && sy <= N - 1) {
              const x0 = Math.floor(sx);
              const y0 = Math.floor(sy);
              const x1 = Math.min(x0 + 1, N - 1);
              const y1 = Math.min(y0 + 1, N - 1);
              const fx = sx - x0;
              const fy = sy - y0;
              const f00 = fieldAt(y0 * N + x0);
              const f10 = fieldAt(y0 * N + x1);
              const f01 = fieldAt(y1 * N + x0);
              const f11 = fieldAt(y1 * N + x1);
              f =
                (f00 * (1 - fx) + f10 * fx) * (1 - fy) +
                (f01 * (1 - fx) + f11 * fx) * fy;
            }
            dens[oi * N + oj] = Math.exp(-f * f * inv2s2);
          }
        }
        paintDensity(ctx, dens, N, 1);
      }
    });

    continueRender(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

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
                  boxShadow: `0 0 ${
                    isPortrait ? 48 : 30
                  }px ${hue.glow}, inset 0 0 0 1px ${accentColor}33`,
                }}
              >
                <canvas
                  ref={refs[pi]}
                  width={560}
                  height={560}
                  style={{ width: "100%", height: "100%", display: "block" }}
                />
              </div>
            </div>
          );
        })}
      </div>

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
        title="Chladni — sand on a vibrating plate"
        body={PEDAGOGY[item.chord_name] ?? PEDAGOGY["Major"]}
        meta="chladni_from_input(inp, plate='rectangular' | 'circular')"
        bottom={48}
        accent={hue.primary}
        igTitle={PEDAGOGY_IG_TITLE}
        igBody={PEDAGOGY_IG_BODY}
      />

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

/** Paint a [0,1] density array with the tidepool ramp (upscaled, alpha). */
function paintDensity(
  ctx: CanvasRenderingContext2D,
  dens: Float32Array,
  N: number,
  alpha: number
) {
  const small = ctx.createImageData(N, N);
  for (let i = 0; i < dens.length; i++) {
    const [r, g, b] = tidepool(dens[i], 0.85);
    small.data[i * 4 + 0] = r;
    small.data[i * 4 + 1] = g;
    small.data[i * 4 + 2] = b;
    small.data[i * 4 + 3] = Math.round(255 * alpha);
  }
  const off = document.createElement("canvas");
  off.width = N;
  off.height = N;
  off.getContext("2d")!.putImageData(small, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.globalAlpha = 1.0;
  ctx.drawImage(off, 0, 0, ctx.canvas.width, ctx.canvas.height);
}
