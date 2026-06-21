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
import {
  cymaticsDensity,
  lerpRatios,
  tidepool,
  type Chord,
} from "../reels/cymatics";
import type { ReelData } from "../reels/reelData";

// Field resolution computed live per frame. Higher = crisper when upscaled
// to the ~880 px plate (120 was visibly pixelated). 384 gives ~2 source
// pixels per displayed pixel and renders comfortably offline.
const FIELD_N = 384;

// Gamma "sweep": each chord blooms BOLD (low gamma → curves push into the
// warm coral/sand end of the Tidepool ramp) as it lands, then cools/thins
// (high gamma → more teal mid-tones) through the morph to the next chord.
// Pulses both brightness AND hue in time with the chord rhythm.
const GAMMA_BOLD = 0.72;
const GAMMA_THIN = 1.15;

/**
 * "What does a chord look like?" — morphs a sequence of chords through their
 * Chladni cymatics nodal patterns, computed live in canvas per frame. Each
 * chord's name + just-intonation ratio float over the plate in the chord's
 * signature hue.
 */
export const CymaticsChordMorph: React.FC<{ data: ReelData }> = ({ data }) => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const chords: Chord[] = data.chords;
  const n = chords.length;
  const seg = data.frames_per_segment;

  // Which morph segment + local eased progress. Looping: chord i → i+1 (wrap).
  const rawSeg = Math.floor(frame / seg);
  const segIdx = data.loop ? rawSeg % n : Math.min(rawSeg, n - 1);
  const local = (frame % seg) / seg; // 0..1
  const fromChord = chords[segIdx];
  const toChord = chords[(segIdx + 1) % n];

  // hold_fraction > 0 keeps each chord settled for that fraction of the
  // segment, then morphs over the remainder (so each pattern reads clearly
  // before it changes). 0 = continuous morph.
  const hold = data.hold_fraction ?? 0;
  const morphLocal =
    hold > 0
      ? Math.max(0, Math.min(1, (local - hold) / (1 - hold)))
      : local;
  const ratios = lerpRatios(fromChord.ratios, toChord.ratios, morphLocal);

  // Gamma bloom: bold (low γ) while settled, thin (high γ) mid-morph.
  const gamma =
    GAMMA_BOLD + (GAMMA_THIN - GAMMA_BOLD) * Math.sin(Math.PI * morphLocal);

  // Paint the field whenever the frame changes.
  useEffect(() => {
    const handle = delayRender("cymatics field");
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      if (ctx) {
        const dens = cymaticsDensity(ratios, FIELD_N, {
          symmetry: data.symmetry,
        });
        // paint inline (Tidepool earthy multicolor ramp, animated gamma)
        const img = ctx.createImageData(FIELD_N, FIELD_N);
        for (let k = 0; k < dens.length; k++) {
          const [r, g, b] = tidepool(dens[k], gamma);
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
      }
    }
    continueRender(handle);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [frame]);

  // Label crossfade aligned with the morph window: from-chord holds, then
  // fades to the to-chord as the pattern morphs.
  const labelStart = hold > 0 ? hold + 0.04 : 0.65;
  const toOpacity = interpolate(local, [labelStart, Math.min(labelStart + 0.3, 0.97)], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const fromOpacity = 1 - toOpacity;
  const fromHue = hueOf(fromChord);
  const toHue = hueOf(toChord);

  const plate = Math.min(width * 0.82, 900);
  const ratioStr = fromChord.ratio_str ?? fromChord.ratios.join(" : ");
  const toRatioStr = toChord.ratio_str ?? toChord.ratios.join(" : ");
  const hook = data.hook ?? "every chord has a shape";

  // Whole-frame entrance fade (first 12 frames only, once).
  const introFade = interpolate(frame, [0, 12], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill style={{ opacity: introFade }}>
      <Backdrop />

      {/* Hook line */}
      <div
        style={{
          position: "absolute",
          top: 150,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          fontSize: 40,
          fontWeight: 300,
          letterSpacing: 1,
          color: theme.ink,
          opacity: 0.85,
          padding: "0 60px",
        }}
        dangerouslySetInnerHTML={{ __html: hook }}
      />

      {/* Cymatics plate, centered */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          width: plate,
          height: plate,
          borderRadius: 10,
          overflow: "hidden",
          boxShadow: `0 0 80px ${fromHue.glow}, inset 0 0 0 1px rgba(255,255,255,0.06)`,
        }}
      >
        <canvas
          ref={canvasRef}
          width={FIELD_N * 4}
          height={FIELD_N * 4}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
      </div>

      {/* Chord name + ratios, crossfading between from/to chords */}
      <div
        style={{
          position: "absolute",
          top: `calc(50% + ${plate / 2 + 70}px)`,
          left: 0,
          right: 0,
          textAlign: "center",
          fontFamily: fonts.display,
          height: 120,
        }}
      >
        <div style={{ position: "absolute", left: 0, right: 0, opacity: fromOpacity }}>
          <div
            style={{
              fontSize: 78,
              fontWeight: 700,
              color: fromHue.primary,
              textShadow: `0 0 28px ${fromHue.glow}`,
              letterSpacing: -1,
            }}
          >
            {fromChord.name}
          </div>
          <div
            style={{
              fontSize: 34,
              fontFamily: fonts.mono,
              color: fromHue.soft,
              letterSpacing: 3,
              marginTop: 4,
            }}
          >
            {ratioStr}
          </div>
          {fromChord.tag ? (
            <div style={tagStyle(fromHue.primary)}>{fromChord.tag}</div>
          ) : null}
        </div>
        <div style={{ position: "absolute", left: 0, right: 0, opacity: toOpacity }}>
          <div
            style={{
              fontSize: 78,
              fontWeight: 700,
              color: toHue.primary,
              textShadow: `0 0 28px ${toHue.glow}`,
              letterSpacing: -1,
            }}
          >
            {toChord.name}
          </div>
          <div
            style={{
              fontSize: 34,
              fontFamily: fonts.mono,
              color: toHue.soft,
              letterSpacing: 3,
              marginTop: 4,
            }}
          >
            {toRatioStr}
          </div>
          {toChord.tag ? (
            <div style={tagStyle(toHue.primary)}>{toChord.tag}</div>
          ) : null}
        </div>
      </div>

      {/* Branding footer */}
      <div
        style={{
          position: "absolute",
          bottom: 90,
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

/** Chord label hue: an explicit per-chord accent overrides the chord-name hue. */
function hueOf(c: Chord): { primary: string; soft: string; glow: string } {
  if (c.accent) {
    return {
      primary: c.accent,
      soft: c.accent,
      glow: hexToRgba(c.accent, 0.35),
    };
  }
  const h = getChordHue(c.label);
  return { primary: h.primary, soft: h.soft, glow: h.glow };
}

function tagStyle(color: string): React.CSSProperties {
  return {
    marginTop: 12,
    fontSize: 26,
    fontFamily: fonts.display,
    fontWeight: 500,
    letterSpacing: 1,
    color,
    opacity: 0.85,
  };
}

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
