import data from "../public/geometry.json";

export type Vec2 = [number, number];

export type LissajousFrame = {
  label: string;
  ratio_p: number;
  ratio_q: number;
  points: Vec2[];
  metadata: { lobes_x: number; lobes_y: number };
};

export type ChordFrame = {
  name: string;
  ratios_str: string;
  lissajous: Vec2[];
  harmonograph: Vec2[];
};

export type HarmonographVariant = {
  label: string;
  subtitle: string;
  points: Vec2[];
  duration_s: number;
};

export type StarPolygonItem = {
  label: string;
  n: number;
  k: number;
  points: Vec2[];
};

export type TimesTableStep = {
  multiplier: number;
  points: Vec2[];
  edges: [number, number][];
};

export type ChladniItem = {
  label: string;
  plate: "rect" | "circ";
  resolution: number;
  field: number[];
};

export type ChladniPlate = {
  type: "rect" | "circ" | "poly5";
  label: string;
  resolution: number;
  field: number[];
};

export type ChladniFromInputItem = {
  chord_name: string;
  ratios_str: string;
  plates: [ChladniPlate, ChladniPlate, ChladniPlate];
};

export type RoseCycloidsScene = {
  name: "rose_and_cycloids";
  items: { label: string; points: Vec2[] }[];
};

// ── v2 scene types (rotating 3-D + expanded chladni) ─────────────────────────
export type Vec3 = [number, number, number];

export type Lissajous3DKnotItem = {
  label: string;
  subtitle: string;
  vertices: Vec3[];
  is_knot: boolean;
};

export type Lsystem3DItem = {
  name: string;
  vertices: Vec3[];
  edges: [number, number][];
  n_segments: number;
};

export type PointCloudItem = {
  name: string;
  surface: "sphere" | "torus" | "klein";
  subtitle: string;
  vertices: Vec3[];
  weights: number[];
};

export type ChladniExpandedPlate = {
  type: "rect" | "circ" | "poly5" | "box3d";
  label: string;
  resolution: number;
  field: number[];
};

export type ChladniExpandedItem = {
  chord_name: string;
  ratios_str: string;
  plates: ChladniExpandedPlate[];
};

type GeometryData = {
  title: string;
  subtitle: string;
  scenes: {
    chord_morph: { name: "chord_morph"; chords: ChordFrame[] };
    lissajous_morph: { name: "lissajous_morph"; frames: LissajousFrame[] };
    harmonograph_variants: {
      name: "harmonograph_variants";
      variants: HarmonographVariant[];
    };
    star_polygons: { name: "star_polygons"; items: StarPolygonItem[] };
    times_table_sweep: { name: "times_table_sweep"; steps: TimesTableStep[] };
    tuning_circle: {
      name: "tuning_circle";
      points: Vec2[];
      weights: number[];
      labels: string[];
    };
    rose_and_cycloids: RoseCycloidsScene;
    chladni_morph: { name: "chladni_morph"; items: ChladniItem[] };
    chladni_from_input: { name: "chladni_from_input"; items: ChladniFromInputItem[] };
    // v2 additions
    lissajous_3d_knots: { name: "lissajous_3d_knots"; items: Lissajous3DKnotItem[] };
    lsystem_3d_variants: { name: "lsystem_3d_variants"; items: Lsystem3DItem[] };
    harmonic_point_clouds: { name: "harmonic_point_clouds"; items: PointCloudItem[] };
    chladni_expanded: { name: "chladni_expanded"; items: ChladniExpandedItem[] };
  };
};

export const geometryData = data as unknown as GeometryData;
