/**
 * 2D canvas renderer for biotuner GeometryData of `tree` or `graph` shape
 * with 2-D coordinates (subharmonic_tree, farey_sequence_layout, etc).
 *
 * Auto-fits to the bounding box, draws edges with a soft glow, then nodes
 * sized by amplitude / weight when available.
 */

import { useEffect, useRef } from 'react'
import { hexToRgb, rgbToCss, lerpRgb } from '../../services/geometry/utils'

export default function TreeViewer({
  geometry,
  color = '#06b6d4',
  colorEnd = null,
  gradient = false,
  background = '#0a0a0a',
  nodeSize = 3,
  edgeWidth = 1,
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
  }, [geometry, color, colorEnd, gradient, background, nodeSize, edgeWidth])

  function draw() {
    const canvas = canvasRef.current
    if (!canvas || !geometry?.coordinates?.length) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    ctx.fillStyle = background
    ctx.fillRect(0, 0, W, H)

    const coords = geometry.coordinates
    const edges = geometry.edges || []
    const weights = geometry.weights || null
    const params = geometry.parameters || {}
    const meta = geometry.metadata || {}

    // Auto-fit to bounding box. Use first two dimensions; for higher-D coords
    // we project onto x/y.
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
    for (const c of coords) {
      const x = c[0] ?? 0
      const y = c[1] ?? 0
      if (x < xMin) xMin = x
      if (x > xMax) xMax = x
      if (y < yMin) yMin = y
      if (y > yMax) yMax = y
    }
    const dx = (xMax - xMin) || 1
    const dy = (yMax - yMin) || 1
    const margin = 0.08 * Math.min(W, H)
    const sx = (W - 2 * margin) / dx
    const sy = (H - 2 * margin) / dy
    const s = Math.min(sx, sy)
    const ox = (W - s * dx) / 2 - s * xMin
    const oy = (H - s * dy) / 2 - s * yMin
    const toCanvas = (c) => ({ x: ox + s * (c[0] ?? 0), y: oy + s * (c[1] ?? 0) })

    // Concentric guide circles for radial/polar layouts. We detect a
    // "centered" tree by checking whether coords cluster around the origin
    // (root at center) or whether parameters.layout is 'depth' or 'polar'.
    const isRadial =
      params.layout === 'depth' || params.layout === 'polar' ||
      (Math.abs(xMin + xMax) < 0.3 * dx && Math.abs(yMin + yMax) < 0.3 * dy)
    if (isRadial) {
      // Use depth metadata if available; otherwise infer rings from the
      // max radius in the data, with up to 6 evenly spaced rings.
      const center = toCanvas([0, 0])
      let maxR = 0
      for (const c of coords) {
        const r = Math.hypot(c[0] ?? 0, c[1] ?? 0)
        if (r > maxR) maxR = r
      }
      const maxRpx = s * maxR
      const nRings = Math.min(6, Math.max(2, Math.round(params.depth || 4)))
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)'
      ctx.lineWidth = 1 * Math.max(1, W / 800)
      for (let i = 1; i <= nRings; i++) {
        const r = (i / nRings) * maxRpx
        ctx.beginPath()
        ctx.arc(center.x, center.y, r, 0, Math.PI * 2)
        ctx.stroke()
      }
    }

    // Precompute per-vertex t (radial distance 0..1) for gradient coloring
    const cStart = hexToRgb(color)
    const cFinal = colorEnd ? hexToRgb(colorEnd) : null
    const useGradient = gradient && cFinal
    let maxR = 0
    for (const c of coords) {
      const r = Math.hypot(c[0] ?? 0, c[1] ?? 0)
      if (r > maxR) maxR = r
    }
    maxR = maxR || 1
    const vertexColor = (idx) => {
      if (!useGradient) return color
      const c = coords[idx]
      const r = Math.hypot(c[0] ?? 0, c[1] ?? 0) / maxR
      return rgbToCss(lerpRgb(cStart, cFinal, r))
    }

    // Edges
    ctx.globalAlpha = 0.55
    ctx.lineCap = 'round'
    ctx.lineWidth = edgeWidth * Math.max(1, W / 800)
    if (!useGradient) {
      ctx.shadowColor = color
      ctx.shadowBlur = edgeWidth * 3
    }
    for (const [a, b] of edges) {
      const pa = coords[a]
      const pb = coords[b]
      if (!pa || !pb) continue
      const ca = toCanvas(pa)
      const cb = toCanvas(pb)
      // Average the two endpoint colors for the edge
      if (useGradient) {
        const colA = vertexColor(a)
        const colB = vertexColor(b)
        const grad = ctx.createLinearGradient(ca.x, ca.y, cb.x, cb.y)
        grad.addColorStop(0, colA)
        grad.addColorStop(1, colB)
        ctx.strokeStyle = grad
      } else {
        ctx.strokeStyle = color
      }
      ctx.beginPath()
      ctx.moveTo(ca.x, ca.y)
      ctx.lineTo(cb.x, cb.y)
      ctx.stroke()
    }
    ctx.shadowBlur = 0
    ctx.globalAlpha = 1

    // Nodes — size by weight if provided
    let wMin = 0, wMax = 1
    if (weights?.length === coords.length) {
      wMin = Math.min(...weights)
      wMax = Math.max(...weights) || 1
    }
    for (let i = 0; i < coords.length; i++) {
      const p = toCanvas(coords[i])
      const w = weights?.[i] ?? null
      const norm = w != null
        ? Math.max(0, (w - wMin) / (wMax - wMin || 1))
        : 1
      const r = nodeSize * (0.6 + norm * 1.6) * Math.max(1, W / 800)
      ctx.beginPath()
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
      ctx.fillStyle = vertexColor(i)
      ctx.fill()
    }
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
