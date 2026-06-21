import React from "react";
import { Composition } from "remotion";
import { MainV2, TOTAL_FRAMES_V2 } from "./MainV2";
import { REEL_SPECS } from "./reels/specs";

// GeometryV2 is the flagship showcase; GeometryV2-IG is its Instagram
// portrait crop (Reel 01). The v1 "MainVideo" composition has been retired.
// Short-form reels are registered from src/reels/specs.ts.

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="GeometryV2"
        component={MainV2}
        durationInFrames={TOTAL_FRAMES_V2}
        fps={30}
        width={1920}
        height={1080}
      />
      {/* Instagram Reels / TikTok / Shorts — 9:16 portrait at 1080×1920. */}
      <Composition
        id="GeometryV2-IG"
        component={MainV2}
        durationInFrames={TOTAL_FRAMES_V2}
        fps={30}
        width={1080}
        height={1920}
      />

      {/* Short-form reels — one <Composition> per spec. */}
      {REEL_SPECS.map((spec) => (
        <Composition
          key={spec.id}
          id={spec.id}
          component={spec.Component}
          durationInFrames={spec.durationInFrames}
          fps={spec.fps}
          width={spec.width}
          height={spec.height}
        />
      ))}
    </>
  );
};
