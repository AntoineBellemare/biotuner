import React from "react";
import { AbsoluteFill, Audio, Series, staticFile } from "remotion";
import { SCENE_FRAMES_V2, TOTAL_FRAMES_V2 } from "./MainV2";
import { Title } from "./scenes/Title";
import { Outro } from "./scenes/Outro";
import { HarmonographRealtime } from "./scenes/HarmonographRealtime";
import { LissajousKnot3D } from "./scenes/LissajousKnot3D";
import { LSystemTree3D } from "./scenes/LSystemTree3D";
import { PointCloud3D } from "./scenes/PointCloud3D";
import { ChladniNice } from "./scenes/ChladniNice";

// GeometryV3: identical to V2 but the Chladni section is rendered in the
// short-form reels' nodal-density "sand" style (ChladniNice) instead of the
// signed-field diverging-colour plates. Same timeline, same soundtrack.
export const TOTAL_FRAMES_V3 = TOTAL_FRAMES_V2;

export const MainV3: React.FC = () => {
  return (
    <AbsoluteFill>
      <Audio src={staticFile("audio/score.wav")} />
      <Series>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.title}>
          <Title />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.harmonograph}>
          <HarmonographRealtime />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.knots}>
          <LissajousKnot3D />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.chladni}>
          <ChladniNice />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.trees}>
          <LSystemTree3D />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.clouds}>
          <PointCloud3D />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES_V2.outro}>
          <Outro />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};
