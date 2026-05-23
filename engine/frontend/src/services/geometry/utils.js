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
    colorEnd = null,            // when set + colorMode==='gradient', interpolate
    colorMode = 'solid',        // 'solid' | 'gradient' | 'rainbow'
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

  ctx.lineJoin = 'round'
  ctx.lineCap = 'round'
  ctx.lineWidth = lineWidth

  // Solid path: single beginPath/stroke is far cheaper.
  if (colorMode === 'solid' || (colorMode === 'gradient' && !colorEnd)) {
    ctx.beginPath()
    for (let i = 0; i < points.length; i++) {
      const p = points[i]
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue
      const sx = cx + p.x * scale
      const sy = cy + p.y * scale
      if (i === 0) ctx.moveTo(sx, sy)
      else ctx.lineTo(sx, sy)
    }
    if (glow) {
      ctx.shadowColor = color
      ctx.shadowBlur = lineWidth * 4
    }
    ctx.strokeStyle = color
    ctx.stroke()
    ctx.shadowBlur = 0
    return
  }

  // Gradient / rainbow: draw segment-by-segment with interpolated colors.
  // Skip glow (shadowBlur per-segment kills perf).
  const isRainbow = colorMode === 'rainbow'
  const baseHsl = isRainbow ? hexToHsl(color) : null
  const cStart = !isRainbow ? hexToRgb(color) : null
  const cFinal = !isRainbow ? hexToRgb(colorEnd) : null
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1]
    const b = points[i]
    if (!Number.isFinite(a.x) || !Number.isFinite(b.x)) continue
    const tFrac = i / (points.length - 1)
    if (isRainbow) {
      const h = (baseHsl.h + tFrac * 360) % 360
      ctx.strokeStyle = `hsl(${h}, ${Math.round(baseHsl.s * 100)}%, ${Math.round(baseHsl.l * 100)}%)`
    } else {
      ctx.strokeStyle = rgbToCss(lerpRgb(cStart, cFinal, tFrac))
    }
    ctx.beginPath()
    ctx.moveTo(cx + a.x * scale, cy + a.y * scale)
    ctx.lineTo(cx + b.x * scale, cy + b.y * scale)
    ctx.stroke()
  }
}

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

export function hexToRgb(hex) {
  let h = (hex || '#000000').replace('#', '')
  if (h.length === 3) h = h.split('').map((c) => c + c).join('')
  const n = parseInt(h, 16)
  return [(n >> 16) & 0xff, (n >> 8) & 0xff, n & 0xff]
}

export function rgbToCss([r, g, b]) {
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`
}

export function lerpRgb(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/** Convert a hex color to HSL with h in [0, 360], s, l in [0, 1]. */
export function hexToHsl(hex) {
  const [r, g, b] = hexToRgb(hex)
  const rn = r / 255, gn = g / 255, bn = b / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const l = (max + min) / 2
  let h, s
  if (max === min) { h = 0; s = 0 }
  else {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    switch (max) {
      case rn: h = ((gn - bn) / d + (gn < bn ? 6 : 0)); break
      case gn: h = ((bn - rn) / d + 2); break
      default: h = ((rn - gn) / d + 4); break
    }
    h *= 60
  }
  return { h, s, l }
}

/**
 * Default gradient companion: rotate the starting color's hue by 90°
 * for a complementary-ish end color.
 */
export function rotateHue(hex, degrees = 90) {
  const [r, g, b] = hexToRgb(hex)
  // RGB → HSL
  const rn = r / 255, gn = g / 255, bn = b / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const l = (max + min) / 2
  let h, s
  if (max === min) {
    h = 0; s = 0
  } else {
    const d = max - min
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min)
    switch (max) {
      case rn: h = ((gn - bn) / d + (gn < bn ? 6 : 0)); break
      case gn: h = ((bn - rn) / d + 2); break
      default: h = ((rn - gn) / d + 4); break
    }
    h /= 6
  }
  h = (h + degrees / 360) % 1
  // HSL → RGB
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1
    if (t > 1) t -= 1
    if (t < 1 / 6) return p + (q - p) * 6 * t
    if (t < 1 / 2) return q
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
    return p
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  const rOut = hue2rgb(p, q, h + 1 / 3)
  const gOut = hue2rgb(p, q, h)
  const bOut = hue2rgb(p, q, h - 1 / 3)
  const toHex = (v) => Math.round(v * 255).toString(16).padStart(2, '0')
  return `#${toHex(rOut)}${toHex(gOut)}${toHex(bOut)}`
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
  const { data, width, height, sigma } = field

  // Offscreen at field resolution; we'll upscale onto the canvas.
  const off = document.createElement('canvas')
  off.width = width
  off.height = height
  const offCtx = off.getContext('2d')
  const img = offCtx.createImageData(width, height)

  // Find peak |v| for normalisation so contrast is consistent across modes.
  let maxAbs = 0
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i])
    if (a > maxAbs) maxAbs = a
  }
  maxAbs = maxAbs || 1
  // σ controls nodal-line width. Use the biotuner formula when supplied;
  // otherwise fall back to a fixed-width Gaussian (legacy behaviour).
  // exp(-w²/σ²) where w = data[i] / maxAbs. Smaller σ = thinner sharper lines.
  const useSigma = Number.isFinite(sigma) && sigma > 0
  const sigSq = useSigma ? (sigma * sigma) : (1 / (4 * Math.max(0.1, contrast)))
  for (let i = 0; i < data.length; i++) {
    const w = data[i] / maxAbs
    const intensity = Math.exp(-(w * w) / sigSq)
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
