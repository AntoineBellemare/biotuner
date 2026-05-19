// 3-D rotation + orthographic projection helper used by the v2 scenes
// (LissajousKnot3D, LSystem3D, PointCloud3D). Pure-numeric, no React.

export type Vec3 = [number, number, number];

/** Rotate a point around the Y axis by angle (radians). */
export function rotateY(p: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]];
}

/** Rotate around X axis. */
export function rotateX(p: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [p[0], c * p[1] - s * p[2], s * p[1] + c * p[2]];
}

/**
 * Apply a Y-then-X rotation (typical "tumble" used by every 3-D scene),
 * returning the rotated point. Inputs / outputs are plain triples.
 */
export function tumble(p: Vec3, yawY: number, pitchX: number): Vec3 {
  return rotateX(rotateY(p, yawY), pitchX);
}

/**
 * Project a rotated point to a 2-D screen coordinate in the [-1, 1] stage
 * using a small perspective bias for depth feel. Returns [x_screen,
 * y_screen, z_camera] — the third component is the camera-space depth
 * used for back-to-front sorting and depth-based opacity.
 */
export function project(
  p: Vec3,
  perspective: number = 1.6,
): [number, number, number] {
  const z = p[2];
  // Mild perspective: closer points appear larger (nearer to viewer ⇒ z > 0)
  const k = perspective / (perspective - z * 0.45);
  return [p[0] * k, p[1] * k, z];
}

/**
 * Convert a depth value (in [-1, 1] approximately) to an opacity in
 * [near_op, far_op]. Used so the back of a rotating mesh dims naturally.
 */
export function depthOpacity(
  z: number,
  nearOp: number = 1.0,
  farOp: number = 0.18,
): number {
  // z ≈ +1 is closest, -1 is farthest. Map to [near, far].
  const t = Math.max(0, Math.min(1, (1 - z) * 0.5));
  return nearOp + (farOp - nearOp) * t;
}

/** Convenience: full pipeline (tumble + project) on a list of Vec3. */
export function tumbleProject(
  pts: Vec3[],
  yawY: number,
  pitchX: number,
  perspective: number = 1.6,
): [number, number, number][] {
  return pts.map((p) => project(tumble(p, yawY, pitchX), perspective));
}
