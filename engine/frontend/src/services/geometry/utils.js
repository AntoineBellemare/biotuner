/**
 * Shared utilities for the harmonic-geometry renderers.
 *
 *   - Math:  gcd, lcm, findFraction (rational approximation), clamp.
 *   - Drawing:  drawPath / drawField onto a Canvas2D context.
 *   - Export:  pointsToSvg / fieldToPng (SVG for line geometry, PNG-in-SVG
 *     for field geometry like Chladni).
 *
 * Geometry modules return one of two shapes:
 *   { kind: 'path', points: [{x, y}, ...] }              // x,y in [-1, 1]
 *   { kind: 'field', data: Float32Array, width, height } // values in [-1, 1]
 */

export function gcd(a, b) {
  a = Math.abs(Math.round(a))
  b = Math.abs(Math.round(b))
  while (b) { [a, b] = [b, a % b] }
  return a || 1
}

export function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v))
}

/**
 * Best rational approximation a/b to `value`, with a, b ≤ maxDenom and
 * gcd(a, b) = 1. Brute force; maxDenom typically 12–20.
 */
export function findFraction(value, maxDenom = 12) {
  if (!Number.isFinite(value) || value <= 0) return { n: 1, d: 1 }
  let best = { n: 1, d: 1, err: Infinity }
  for (let d = 1; d <= maxDenom; d++) {
    for (let n = 1; n <= maxDenom; n++) {
      if (gcd(n, d) !== 1) continue
      const err = Math.abs(value - n / d)
      if (err < best.err) best = { n, d, err }
    }
  }
  return { n: best.n, d: best.d }
}

// ---------------------------------------------------------------------------
// Canvas drawing
// ---------------------------------------------------------------------------

export function drawPath(ctx, points, opts = {}) {
  const {
    width, height,
    lineWidth = 1.5,
    color = '#06b6d4',
    background = '#0a0a0a',
    glow = true,
  } = opts
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = background
  ctx.fillRect(0, 0, width, height)
  if (!points || points.length === 0) return

  const cx = width / 2
  const cy = height / 2
  const scale = Math.min(width, height) * 0.46

  ctx.beginPath()
  for (let i = 0; i < points.length; i++) {
    const p = points[i]
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue
    const sx = cx + p.x * scale
    const sy = cy + p.y * scale
    if (i === 0) ctx.moveTo(sx, sy)
    else ctx.lineTo(sx, sy)
  }
  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  if (glow) {
    ctx.shadowColor = color
    ctx.shadowBlur = lineWidth * 4
  }
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.stroke()
  ctx.shadowBlur = 0
}

/**
 * Render a 2D field to canvas. We map field values (∈ [-1, 1]) to the
 * "Chladni" look: zero-crossings = bright, anti-nodes = dark. The user can
 * shift colors via the `palette` option.
 */
export function drawField(ctx, field, opts = {}) {
  const {
    width: cw,
    height: ch,
    contrast = 1.5,
    background = '#0a0a0a',
    palette = 'mono',
    accentColor = [6, 182, 212],  // biotuner-primary
  } = opts
  const { data, width, height } = field

  // Offscreen at field resolution; we'll upscale onto the canvas.
  const off = document.createElement('canvas')
  off.width = width
  off.height = height
  const offCtx = off.getContext('2d')
  const img = offCtx.createImageData(width, height)

  // Pre-multiplied palette: emphasize nodal lines (|v| ≈ 0).
  for (let i = 0; i < data.length; i++) {
    const v = data[i] * contrast
    // Gaussian "ridge" centered on zero — classic Chladni sand look.
    const intensity = Math.exp(-v * v * 4)
    const c = Math.round(intensity * 255)
    let r, g, b
    if (palette === 'accent') {
      r = Math.round(intensity * accentColor[0])
      g = Math.round(intensity * accentColor[1])
      b = Math.round(intensity * accentColor[2])
    } else {
      r = g = b = c
    }
    const k = i * 4
    img.data[k]     = r
    img.data[k + 1] = g
    img.data[k + 2] = b
    img.data[k + 3] = 255
  }
  offCtx.putImageData(img, 0, 0)

  // Paint the dark background underneath, then crisp-nearest upscale.
  ctx.fillStyle = background
  ctx.fillRect(0, 0, cw, ch)
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(off, 0, 0, cw, ch)
}

// ---------------------------------------------------------------------------
// SVG / PNG export
// ---------------------------------------------------------------------------

export function pointsToSvg(points, opts = {}) {
  const {
    width = 800,
    height = 800,
    lineWidth = 1.5,
    color = '#06b6d4',
    background = '#0a0a0a',
  } = opts
  if (!points?.length) return ''
  const cx = width / 2
  const cy = height / 2
  const scale = Math.min(width, height) * 0.46

  let d = ''
  for (let i = 0; i < points.length; i++) {
    const p = points[i]
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue
    const sx = cx + p.x * scale
    const sy = cy + p.y * scale
    d += `${i === 0 ? 'M' : 'L'} ${sx.toFixed(2)} ${sy.toFixed(2)} `
  }

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="100%" height="100%" fill="${background}"/>
  <path d="${d.trim()}" fill="none" stroke="${color}" stroke-width="${lineWidth}"
        stroke-linejoin="round" stroke-linecap="round"/>
</svg>`
}

export function canvasToPngDataUrl(canvas) {
  return canvas.toDataURL('image/png')
}

/**
 * Field geometry doesn't have a path; embed a PNG inside the SVG.
 */
export function fieldCanvasToSvg(canvas, opts = {}) {
  const { width = canvas.width, height = canvas.height, background = '#0a0a0a' } = opts
  const dataUrl = canvas.toDataURL('image/png')
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="100%" height="100%" fill="${background}"/>
  <image x="0" y="0" width="${width}" height="${height}" preserveAspectRatio="none"
         href="${dataUrl}"/>
</svg>`
}

export function downloadFile(dataOrUrl, filename, mimeType) {
  let url
  let revoke = false
  if (typeof dataOrUrl === 'string' && dataOrUrl.startsWith('data:')) {
    url = dataOrUrl
  } else {
    const blob = typeof dataOrUrl === 'string'
      ? new Blob([dataOrUrl], { type: mimeType })
      : dataOrUrl
    url = URL.createObjectURL(blob)
    revoke = true
  }
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  if (revoke) setTimeout(() => URL.revokeObjectURL(url), 0)
}
