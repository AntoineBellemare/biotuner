/**
 * Canvas renderer for biotuner GeometryData of `field_2d` shape (Chladni
 * fields produced by biotuner.harmonic_geometry.media.eigenmode.rigid_plate).
 * The field array is in `coordinates` as a 2-D nested list.
 *
 * Renders the iconic Chladni "sand on nodal lines" look — bright where the
 * field crosses zero, dark elsewhere — with optional rainbow / gradient
 * coloring.
 */

import { useEffect, useRef } from 'react'
import { hexToRgb, hexToHsl } from '../../services/geometry/utils'

export default function FieldViewer({
  geometry,
  color = '#ffffff',
  colorEnd = null,
  gradient = false,
  colorMode = null,           // 'solid' | 'gradient' | 'rainbow'
  background = '#0a0a0a',
  contrast = 1.8,
}) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return
    const resize = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return
      const cssSize = Math.min(container.clientWidth, 720)
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.style.width = `${cssSize}px`
      canvas.style.height = `${cssSize}px`
      canvas.width = Math.round(cssSize * dpr)
      canvas.height = Math.round(cssSize * dpr)
      draw()
    }
    const ro = new ResizeObserver(resize)
    ro.observe(containerRef.current)
    resize()
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    draw()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometry, color, colorEnd, gradient, colorMode, background, contrast])

  function draw() {
    const canvas = canvasRef.current
    if (!canvas || !geometry?.coordinates?.length) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    // field_2d: coordinates is a 2-D nested array of field values.
    const field = geometry.coordinates
    const rows = field.length
    const cols = field[0]?.length || 0
    if (!rows || !cols) return

    // Render to an offscreen image at field resolution, then upscale.
    const off = document.createElement('canvas')
    off.width = cols
    off.height = rows
    const offCtx = off.getContext('2d')
    const img = offCtx.createImageData(cols, rows)

    const mode = colorMode || (gradient ? 'gradient' : 'solid')
    const baseHsl = (mode === 'rainbow') ? hexToHsl(color) : null
    const [r0, g0, b0] = hexToRgb(color)
    const [r1, g1, b1] = colorEnd ? hexToRgb(colorEnd) : [r0, g0, b0]

    // Find max |v| for normalisation
    let maxAbs = 0
    for (let y = 0; y < rows; y++) {
      const row = field[y]
      for (let x = 0; x < cols; x++) {
        const v = Math.abs(row[x])
        if (v > maxAbs) maxAbs = v
      }
    }
    maxAbs = maxAbs || 1

    for (let y = 0; y < rows; y++) {
      const row = field[y]
      for (let x = 0; x < cols; x++) {
        const v = row[x] / maxAbs
        // Gaussian ridge centred on zero — classic Chladni sand look.
        const intensity = Math.exp(-v * v * 4 * contrast)
        const k = (y * cols + x) * 4
        let R, G, B
        if (mode === 'rainbow') {
          // Hue cycles with intensity, saturated at high intensity
          const hue = (baseHsl.h + (1 - intensity) * 200) % 360
          const sat = 0.9 * intensity
          const lit = 0.5 * intensity
          const [rr, gg, bb] = hslToRgb(hue / 360, sat, lit)
          R = rr; G = gg; B = bb
        } else if (mode === 'gradient') {
          // Interpolate start→end by intensity
          const t = intensity
          R = Math.round(r0 + (r1 - r0) * t) * intensity
          G = Math.round(g0 + (g1 - g0) * t) * intensity
          B = Math.round(b0 + (b1 - b0) * t) * intensity
          R = Math.round(R); G = Math.round(G); B = Math.round(B)
        } else {
          // Tint the user's chosen color by intensity
          R = Math.round(r0 * intensity)
          G = Math.round(g0 * intensity)
          B = Math.round(b0 * intensity)
        }
        img.data[k]     = R
        img.data[k + 1] = G
        img.data[k + 2] = B
        img.data[k + 3] = 255
      }
    }
    offCtx.putImageData(img, 0, 0)

    ctx.fillStyle = background
    ctx.fillRect(0, 0, W, H)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(off, 0, 0, W, H)
  }

  return (
    <div
      ref={containerRef}
      className="w-full h-full flex items-center justify-center bg-biotuner-dark-900 rounded-xl border border-biotuner-dark-600 overflow-hidden"
      style={{ aspectRatio: '1 / 1' }}
    >
      <canvas ref={canvasRef} className="block max-w-full max-h-full" />
    </div>
  )
}

function hslToRgb(h, s, l) {
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1
    if (t > 1) t -= 1
    if (t < 1 / 6) return p + (q - p) * 6 * t
    if (t < 1 / 2) return q
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
    return p
  }
  if (s === 0) return [l * 255, l * 255, l * 255]
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s
  const p = 2 * l - q
  return [
    Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
    Math.round(hue2rgb(p, q, h) * 255),
    Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
  ]
}
