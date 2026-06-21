/**
 * Typed accessors for the per-reel data written by export_reels.py into
 * public/reels/<id>.json. Imported at build time (the files are tiny —
 * just chord sequences + timing).
 */
import reel02 from "../../public/reels/Reel02-Cymatics.json";
import type { Chord } from "./cymatics";
import type { IntroConfig } from "./ReelIntro";

export type ReelData = {
  id: string;
  fps: number;
  frames_per_segment: number;
  intro_frames: number;
  morph_frames: number;
  total_frames: number;
  symmetry: "d4_max" | "d4_sum" | "none";
  loop: boolean;
  intro: IntroConfig | null;
  chords: Chord[];
  audio: string; // staticFile-relative path, e.g. "audio/Reel02-Cymatics.wav"
};

export const REEL_DATA: Record<string, ReelData> = {
  "Reel02-Cymatics": reel02 as unknown as ReelData,
};
