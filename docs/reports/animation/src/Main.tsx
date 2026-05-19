import React from "react";
import { AbsoluteFill, Series } from "remotion";
import { Title } from "./scenes/Title";
import { ChordMorph, FRAMES_PER_CHORD } from "./scenes/ChordMorph";
import { LissajousMorph } from "./scenes/LissajousMorph";
import { Harmonograph } from "./scenes/Harmonograph";
import { StarPolygons } from "./scenes/StarPolygons";
import { TuningCircle } from "./scenes/TuningCircle";
import { TimesTableSweep, FRAMES_PER_STEP } from "./scenes/TimesTableSweep";
import { ChladniMorph } from "./scenes/ChladniMorph";
import { ChladniFromInput, FRAMES_PER_CHORD_CHLADNI } from "./scenes/ChladniFromInput";
import { Outro } from "./scenes/Outro";

// Chord morph: 7 chords × 60 frames each
const CHORD_MORPH_FRAMES = 7 * FRAMES_PER_CHORD; // 420

// Times-table sweep: multipliers 2–12 = 11 steps × 30 frames
const TIMES_TABLE_FRAMES = 11 * FRAMES_PER_STEP; // 330

// Chladni from-input: 7 chords × 60 frames (rect + circ + pentagon simultaneously)
const CHLADNI_FROM_INPUT_FRAMES = 7 * FRAMES_PER_CHORD_CHLADNI; // 420

// Per-scene durations in frames (30 fps).
export const SCENE_FRAMES = {
  title: 90,                        //  3.0 s
  chordMorph: CHORD_MORPH_FRAMES,   // 14.0 s
  lissajous: 252,                   //  8.4 s  (6 frames × 42 f)
  harmonograph: 300,                // 10.0 s  (4 variants × 75 f)
  starPolygons: 160,                //  5.3 s
  tuningCircle: 150,                //  5.0 s
  timesTable: TIMES_TABLE_FRAMES,   // 11.0 s
  chladni: 420,                     // 14.0 s  (12 modes × 35 f)
  chladniFromInput: CHLADNI_FROM_INPUT_FRAMES, // 14.0 s  (7 chords × 60 f)
  outro: 90,                        //  3.0 s
};

export const TOTAL_FRAMES =
  Object.values(SCENE_FRAMES).reduce((a, b) => a + b, 0);

export const Main: React.FC = () => {
  return (
    <AbsoluteFill>
      <Series>
        <Series.Sequence durationInFrames={SCENE_FRAMES.title}>
          <Title />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.chordMorph}>
          <ChordMorph />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.lissajous}>
          <LissajousMorph />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.harmonograph}>
          <Harmonograph />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.starPolygons}>
          <StarPolygons />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.tuningCircle}>
          <TuningCircle />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.timesTable}>
          <TimesTableSweep />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.chladni}>
          <ChladniMorph />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.chladniFromInput}>
          <ChladniFromInput />
        </Series.Sequence>
        <Series.Sequence durationInFrames={SCENE_FRAMES.outro}>
          <Outro />
        </Series.Sequence>
      </Series>
    </AbsoluteFill>
  );
};
