import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Backdrop } from "../components/Backdrop";
import { fonts, theme } from "../theme";

export const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const isPortrait = height > width;
  const headlineSize = isPortrait ? 80 : 56;
  const bodySize = isPortrait ? 40 : 22;
  const tagSize = isPortrait ? 26 : 18;
  const opacity = interpolate(frame, [4, 22], [0, 1], {
    extrapolateRight: "clamp",
  });
  const lineWidth = interpolate(frame, [16, 60], [0, 1], {
    extrapolateRight: "clamp",
  });
  // Portrait-only CTA pill fades in last so it lands as the call-to-action.
  const ctaOpacity = interpolate(frame, [40, 70], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill>
      <Backdrop />
      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          flexDirection: "column",
          color: theme.ink,
          fontFamily: fonts.display,
          opacity,
          padding: "0 80px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontSize: headlineSize,
            fontWeight: 200,
            letterSpacing: -1,
            textShadow: `0 0 36px ${theme.glow}`,
            lineHeight: 1.1,
          }}
        >
          from spectrum to space
        </div>
        <div
          style={{
            width: `${lineWidth * (isPortrait ? 320 : 420)}px`,
            height: 2,
            marginTop: 22,
            background: `linear-gradient(90deg, transparent, ${theme.accent}, transparent)`,
            boxShadow: `0 0 12px ${theme.glow}`,
          }}
        />
        <div
          style={{
            marginTop: 30,
            fontSize: bodySize,
            fontWeight: 300,
            maxWidth: isPortrait ? 940 : 880,
            lineHeight: 1.5,
            color: theme.ink,
            opacity: 0.88,
          }}
        >
          biotuner now extends harmonic analysis into geometry.
          Turn every chord, peak set, and ratio into a spatial
          object you can rotate, listen to, and resonate.
        </div>
        <div
          style={{
            marginTop: 32,
            fontFamily: fonts.mono,
            fontSize: tagSize,
            color: theme.muted,
            letterSpacing: 4,
            textTransform: "uppercase",
          }}
        >
          biotuner.harmonic_geometry
        </div>
        {/* IG / Reels CTA pill — only in portrait. Verbal call-to-action
            since IG can't link from inside the video; viewers tap into
            the caption to find the actual repo URL. */}
        {isPortrait && (
          <div
            style={{
              marginTop: 44,
              opacity: ctaOpacity,
              transform: `translateY(${(1 - ctaOpacity) * 14}px)`,
              fontFamily: fonts.mono,
              fontSize: 38,
              color: theme.accent,
              letterSpacing: 0.5,
              border: `1px solid ${theme.accent}55`,
              padding: "22px 46px",
              borderRadius: 999,
              background: `${theme.accent}10`,
              boxShadow: `0 0 22px ${theme.glow}`,
            }}
          >
            pip install biotuner
          </div>
        )}
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
