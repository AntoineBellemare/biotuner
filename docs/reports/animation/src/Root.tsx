import React from "react";
import { Composition } from "remotion";
import { MainV2, TOTAL_FRAMES_V2 } from "./MainV2";

// Note: the legacy "MainVideo" / Phase 1-3 complete composition has been
// retired. GeometryV2 is the active animation; the v1 scene files remain
// on disk in case any are reused.

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
      {/* Instagram Reels / TikTok / Shorts — 9:16 portrait at 1080×1920.
          Same scene timeline; layouts use AbsoluteFill + centered Stage so
          the portrait crop reads cleanly. */}
      <Composition
        id="GeometryV2-IG"
        component={MainV2}
        durationInFrames={TOTAL_FRAMES_V2}
        fps={30}
        width={1080}
        height={1920}
      />
    </>
  );
};
