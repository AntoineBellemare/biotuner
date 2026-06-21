import React from "react";
import { AbsoluteFill, Audio, staticFile } from "remotion";

/**
 * Generic reel frame: full-bleed dark stage + optional soundtrack. Every
 * reel composition renders its scene content as children of this so audio
 * wiring and framing stay consistent across reels.
 */
export const Reel: React.FC<{
  audio?: string;
  children: React.ReactNode;
}> = ({ audio, children }) => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#050811" }}>
      {audio ? <Audio src={staticFile(audio)} /> : null}
      {children}
    </AbsoluteFill>
  );
};
