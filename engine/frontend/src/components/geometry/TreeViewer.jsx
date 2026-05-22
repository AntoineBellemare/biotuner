/**
 * 2D canvas renderer for biotuner GeometryData of `tree` or `graph` shape
 * with 2-D coordinates (subharmonic_tree, farey_sequence_layout, etc).
 *
 * Auto-fits to the bounding box, draws edges with a soft glow, then nodes
 * sized by amplitude / weight when available.
 */

import { useEffect, useRef } from 'react'

export default function TreeViewer({
  geometry,
  color = '#06b6d4',
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
  }, [geometry, color, background, nodeSize, edgeWidth])

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
    const margin = 0.06 * Math.min(W, H)
    const sx = (W - 2 * margin) / dx
    const sy = (H - 2 * margin) / dy
    const s = Math.min(sx, sy)
    const ox = (W - s * dx) / 2 - s * xMin
    const oy = (H - s * dy) / 2 - s * yMin
    const toCanvas = (c) => ({ x: ox + s * (c[0] ?? 0), y: oy + s * (c[1] ?? 0) })

    // Edges
    ctx.strokeStyle = color
    ctx.globalAlpha = 0.55
    ctx.lineCap = 'round'
    ctx.lineWidth = edgeWidth * Math.max(1, W / 800)
    ctx.shadowColor = color
    ctx.shadowBlur = edgeWidth * 3
    for (const [a, b] of edges) {
      const pa = coords[a]
      const pb = coords[b]
      if (!pa || !pb) continue
      const ca = toCanvas(pa)
      const cb = toCanvas(pb)
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
      const r = nodeSize * (0.6 + norm * 1.4) * Math.max(1, W / 800)
      ctx.beginPath()
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
      ctx.fillStyle = color
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
