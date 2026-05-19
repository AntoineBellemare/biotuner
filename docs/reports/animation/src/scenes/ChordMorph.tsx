import React, { useMemo } from "react";
import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { Backdrop } from "../components/Backdrop";
import { theme, fonts } from "../theme";
import { geometryData, Vec2, ChordFrame } from "../geometry";

export const FRAMES_PER_CHORD = 60; // 2 s @ 30 fps

/**
 * Side-by-side compound Lissajous (left) and harmonograph (right) for each
 * musical chord in just intonation. Chords cross-fade every 2 seconds.
 */
export const ChordMorph: React.FC = () => {
  const frame = useCurrentFrame();
  const chords = geometryData.scenes.chord_morph.chords;

  const activeIdx = Math.min(
    Math.floor(frame / FRAMES_PER_CHORD),
    chords.length - 1
  );
  const local = (frame % FRAMES_PER_CHORD) / FRAMES_PER_CHORD;
  const chord = chords[activeIdx];

  // Fade in scene title during first 24 frames.
  const labelOpacity = interpolate(frame % FRAMES_PER_CHORD, [0, 18], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <Backdrop />

      {/* ─── Two SVG panels side-by-side ─── */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          flexDirection: "row",
          alignItems: "center",
          justifyContent: "center",
          gap: 48,
          paddingBottom: 140,
        }}
      >
        {/* Left: compound Lissajous */}
        <PanelSvg size={640} label="lissajous_compound">
          {chords.map((c, i) => {
            const op = blendOpacity(i, activeIdx, local);
            if (op < 0.01) return null;
            return (
              <TracePath
                key={i}
                points={c.lissajous}
                opacity={op}
                glow
              />
            );
          })}
        </PanelSvg>

        {/* Thin vertical rule */}
        <div
          style={{
            width: 1,
            height: 500,
            background: theme.muted,
            opacity: 0.25,
            flexShrink: 0,
          }}
        />

        {/* Right: harmonograph trace */}
        <PanelSvg size={640} label="harmonograph_lateral">
          {chords.map((c, i) => {
            const op = blendOpacity(i, activeIdx, local);
            if (op < 0.01) return null;
            return (
              <TracePath
                key={i}
                points={c.harmonograph}
                opacity={op}
                glow={false}
                strokeColor={theme.cool}
                glowColor={`rgba(109,163,216,0.35)`}
              />
            );
          })}
        </PanelSvg>
      </div>

      {/* ─── Chord label ─── */}
      <div
        style={{
          position: "absolute",
          bottom: 80,
          left: 0,
          right: 0,
          textAlign: "center",
          opacity: labelOpacity,
          fontFamily: fonts.display,
        }}
      >
        <div
          style={{
            fontSize: 52,
            fontWeight: 700,
            color: theme.ink,
            letterSpacing: -1,
            textShadow: `0 0 28px ${theme.glow}`,
          }}
        >
          {chord.name}
        </div>
        <div
          style={{
            fontSize: 26,
            fontWeight: 300,
            color: theme.accentSoft,
            marginTop: 6,
            fontFamily: fonts.mono,
            letterSpacing: 3,
          }}
        >
          {chord.ratios_str}
        </div>
        <div
          style={{
            fontSize: 18,
            fontWeight: 300,
            color: theme.muted,
            marginTop: 4,
            fontFamily: fonts.mono,
          }}
        >
          lissajous_compound · harmonograph_lateral
        </div>
      </div>

      {/* ─── Progress dots ─── */}
      <ChordDots total={chords.length} active={activeIdx} />
    </AbsoluteFill>
  );
};

/** Crossfade opacity helper: 0 = invisible, 1 = full. */
function blendOpacity(idx: number, active: number, local: number): number {
  const dist = idx - active;
  if (dist === 0) return local < 0.85 ? 1 : 1 - (local - 0.85) / 0.15;
  if (dist === 1) return local > 0.85 ? (local - 0.85) / 0.15 : 0;
  return 0;
}

/** Small SVG container with a subtle function name label above it. */
const PanelSvg: React.FC<{
  size: number;
  label: string;
  children: React.ReactNode;
}> = ({ size, label, children }) => (
  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
    <div
      style={{
        fontSize: 15,
        color: theme.muted,
        fontFamily: fonts.mono,
        letterSpacing: 0.5,
      }}
    >
      {label}
    </div>
    <svg
      width={size}
      height={size}
      viewBox="-1.05 -1.05 2.1 2.1"
      style={{ overflow: "visible" }}
    >
      <g transform="scale(1,-1)">{children}</g>
    </svg>
  </div>
);

/** Dual-layer path: soft wide glow + sharp fine line. */
const TracePath: React.FC<{
  points: Vec2[];
  opacity: number;
  glow?: boolean;
  strokeColor?: string;
  glowColor?: string;
}> = ({
  points,
  opacity,
  glow = true,
  strokeColor = theme.accent,
  glowColor = theme.glow,
}) => {
  const d = useMemo(
    () => points.map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`).join(" "),
    [points]
  );
  return (
    <>
      <path
        d={d}
        fill="none"
        stroke={glow ? theme.cool : theme.coolSoft}
        strokeWidth={0.012}
        opacity={opacity * 0.35}
      />
      <path
        d={d}
        fill="none"
        stroke={strokeColor}
        strokeWidth={0.005}
        opacity={opacity}
        style={glow ? { filter: `drop-shadow(0 0 6px ${glowColor})` } : undefined}
      />
    </>
  );
};

/** Progress indicator row at the top. */
const ChordDots: React.FC<{ total: number; active: number }> = ({
  total,
  active,
}) => (
  <div
    style={{
      position: "absolute",
      top: 36,
      left: 0,
      right: 0,
      display: "flex",
      justifyContent: "center",
      gap: 10,
    }}
  >
    {Array.from({ length: total }, (_, i) => (
      <div
        key={i}
        style={{
          width: i === active ? 24 : 8,
          height: 8,
          borderRadius: 4,
          background: i === active ? theme.accent : theme.muted,
          opacity: i === active ? 1 : 0.4,
          transition: "width 0.3s",
        }}
      />
    ))}
  </div>
);
