import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Play, Pause, Sparkles, Download, RotateCcw, Shuffle, Image as ImageIcon, Cpu, Loader2 } from 'lucide-react'
import {
  GEOMETRY_TYPES,
  GEOMETRY_ORDER,
} from '../../services/geometry/types'
import {
  drawPath,
  drawField,
  pointsToSvg,
  fieldCanvasToSvg,
  canvasToPngDataUrl,
  downloadFile,
} from '../../services/geometry/utils'
import apiClient from '../../services/api'

const ThreeViewer = lazy(() => import('../geometry/ThreeViewer'))
const TreeViewer = lazy(() => import('../geometry/TreeViewer'))

const CANVAS_DPR_CAP = 2

// One-shot helpers --------------------------------------------------------

function paramId(geomKey, paramKey) {
  return `${geomKey}::${paramKey}`
}

function randomizeParams(geom) {
  const out = { ...geom.defaultParams }
  for (const p of geom.paramSchema) {
    if (p.type === 'slider' || p.type === 'int') {
      const range = p.max - p.min
      let v = p.min + Math.random() * range
      if (p.type === 'int' || (p.step && p.step >= 1)) v = Math.round(v)
      else v = Math.round(v / p.step) * p.step
      out[p.key] = v
    } else if (p.type === 'bool') {
      out[p.key] = Math.random() < 0.5
    } else if (p.type === 'select') {
      out[p.key] = p.options[Math.floor(Math.random() * p.options.length)].value
    }
  }
  return out
}

// Debounce helper for Python-engine geometries.
function useDebounced(value, delay = 250) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), delay)
    return () => clearTimeout(t)
  }, [value, delay])
  return v
}

// =============================================================================

export default function GeometryTab({ analysisResult }) {
  const [type, setType] = useState('lissajous')
  // Seed each geometry's params with its defaults *merged with* the analysis
  // mapping, so the user sees a pattern derived from their signal as soon as
  // they land here rather than the abstract defaults.
  const [paramsByType, setParamsByType] = useState(() => {
    const ratios = analysisResult?.tuning || []
    const out = {}
    for (const g of Object.values(GEOMETRY_TYPES)) {
      const base = { ...g.defaultParams }
      if (g.engine !== 'python' && typeof g.fromRatios === 'function' && ratios.length) {
        Object.assign(base, g.fromRatios(ratios) || {})
      }
      out[g.key] = base
    }
    return out
  })
  const [animate, setAnimate] = useState(true)
  const [color, setColor] = useState('#06b6d4')
  const [palette, setPalette] = useState('mono')  // 'mono' | 'accent'
  const [showAdvanced, setShowAdvanced] = useState(false)

  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const lastOutputRef = useRef(null)  // for SVG export of the current frame
  const tStartRef = useRef(performance.now())

  const geom = GEOMETRY_TYPES[type]
  const params = paramsByType[type]
  const isPython = geom.engine === 'python'

  // ---- Python-engine state: geometry payload from backend ----
  const [pythonGeom, setPythonGeom] = useState(null)
  const [pythonLoading, setPythonLoading] = useState(false)
  const [pythonError, setPythonError] = useState(null)
  const [autoRotate, setAutoRotate] = useState(true)
  const [wireframe, setWireframe] = useState(false)
  const debouncedParams = useDebounced(params, 250)
  const ratiosKey = useMemo(
    () => (analysisResult?.tuning || []).join(','),
    [analysisResult]
  )

  // Trigger backend compute when a Python style is active and its
  // (debounced) params change. The same effect also fires when the user
  // switches tuning (e.g. after a new analysis).
  useEffect(() => {
    if (!isPython) {
      setPythonError(null)
      return
    }
    let cancelled = false
    setPythonLoading(true)
    apiClient
      .computeHarmonicGeometry({
        style: geom.style,
        params: debouncedParams,
        tuning: analysisResult?.tuning || null,
        peaks: analysisResult?.peaks || null,
      })
      .then((data) => {
        if (cancelled) return
        setPythonGeom(data)
        setPythonError(null)
      })
      .catch((err) => {
        if (cancelled) return
        setPythonError(err?.response?.data?.detail || err?.message || 'Geometry compute failed')
        setPythonGeom(null)
      })
      .finally(() => {
        if (!cancelled) setPythonLoading(false)
      })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPython, geom.style, debouncedParams, ratiosKey])

  const setParam = (key, value) => {
    setParamsByType((prev) => ({
      ...prev,
      [type]: { ...prev[type], [key]: value },
    }))
  }

  // -------------------------------------------------------------------------
  // Apply derived tuning to current geometry's params
  // -------------------------------------------------------------------------
  const ratios = analysisResult?.tuning || []
  const ratiosKeyForJs = useMemo(() => ratios.join(','), [ratios])

  // Auto-snap every JS geometry to the new analysis whenever the ratios
  // change. This keeps the analysis "live": pick a pattern, see it derived
  // from your signal; switch tabs and come back, still derived. The user
  // can override any individual slider afterwards.
  useEffect(() => {
    if (!ratios.length) return
    setParamsByType((prev) => {
      const next = { ...prev }
      for (const g of Object.values(GEOMETRY_TYPES)) {
        if (g.engine === 'python') continue
        if (typeof g.fromRatios !== 'function') continue
        const updates = g.fromRatios(ratios) || {}
        if (Object.keys(updates).length === 0) continue
        next[g.key] = { ...prev[g.key], ...updates }
      }
      return next
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ratiosKeyForJs])

  const applyFromAnalysis = () => {
    // Python-engine geometries already consume the tuning automatically
    // (sent on every backend request); the button is mainly for JS engines.
    if (geom.engine === 'python' || !geom.fromRatios) {
      // Just bump the params object reference so the backend re-fires.
      setParamsByType((prev) => ({ ...prev, [type]: { ...prev[type] } }))
      return
    }
    const updates = geom.fromRatios(ratios)
    setParamsByType((prev) => ({
      ...prev,
      [type]: { ...prev[type], ...updates },
    }))
  }

  const resetParams = () => {
    setParamsByType((prev) => ({ ...prev, [type]: { ...geom.defaultParams } }))
  }

  const randomize = () => {
    setParamsByType((prev) => ({ ...prev, [type]: randomizeParams(geom) }))
    tStartRef.current = performance.now()
  }

  // -------------------------------------------------------------------------
  // Canvas sizing (responsive + DPR-aware) — only for JS engine
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (isPython) return
    if (!canvasRef.current || !containerRef.current) return
    const resize = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return
      const cssSize = Math.min(container.clientWidth, 720)
      const dpr = Math.min(window.devicePixelRatio || 1, CANVAS_DPR_CAP)
      canvas.style.width = `${cssSize}px`
      canvas.style.height = `${cssSize}px`
      canvas.width = Math.round(cssSize * dpr)
      canvas.height = Math.round(cssSize * dpr)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [isPython])

  // -------------------------------------------------------------------------
  // Render loop — only for JS engine
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (isPython) return
    let rafId = null

    const draw = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      const t = animate ? (performance.now() - tStartRef.current) / 1000 : 0
      const out = geom.render(params, t)
      lastOutputRef.current = out

      const opts = {
        width: canvas.width,
        height: canvas.height,
        lineWidth: (params.lineWidth || 1.5) * (canvas.width / 600),
        color,
        background: '#0a0a0a',
        contrast: params.contrast,
        palette,
        glow: true,
      }
      if (out.kind === 'path') drawPath(ctx, out.points, opts)
      else if (out.kind === 'field') drawField(ctx, out, opts)

      if (animate) rafId = requestAnimationFrame(draw)
    }
    draw()
    return () => { if (rafId) cancelAnimationFrame(rafId) }
  }, [isPython, geom, params, animate, color, palette])

  // -------------------------------------------------------------------------
  // Exports
  // -------------------------------------------------------------------------
  const exportSvg = () => {
    const filename = `harmonic_${geom.key}.svg`
    if (isPython) {
      // For Python geometries, dump the raw GeometryData as JSON — the user
      // gets the exact backend payload, perfect for downstream tools.
      if (!pythonGeom) return
      downloadFile(
        JSON.stringify(pythonGeom, null, 2),
        `harmonic_${geom.key}.json`,
        'application/json',
      )
      return
    }
    const out = lastOutputRef.current
    if (!out) return
    if (out.kind === 'path') {
      const svg = pointsToSvg(out.points, {
        width: 1024, height: 1024,
        lineWidth: params.lineWidth || 1.5,
        color,
        background: '#0a0a0a',
      })
      downloadFile(svg, filename, 'image/svg+xml')
    } else if (out.kind === 'field') {
      const svg = fieldCanvasToSvg(canvasRef.current, { background: '#0a0a0a' })
      downloadFile(svg, filename, 'image/svg+xml')
    }
  }

  const exportPng = () => {
    // JS canvas → toDataURL; Python 3D viewer → query its canvas via DOM.
    if (isPython) {
      const c = containerRef.current?.querySelector('canvas')
      if (c) downloadFile(c.toDataURL('image/png'), `harmonic_${geom.key}.png`, 'image/png')
      return
    }
    if (!canvasRef.current) return
    downloadFile(canvasToPngDataUrl(canvasRef.current), `harmonic_${geom.key}.png`, 'image/png')
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="space-y-5">
      {/* Geometry type picker */}
      <div className="flex flex-wrap gap-2">
        {GEOMETRY_ORDER.map((k) => {
          const g = GEOMETRY_TYPES[k]
          const active = k === type
          return (
            <button
              key={k}
              onClick={() => setType(k)}
              aria-pressed={active}
              className={`
                min-h-[40px] px-4 py-2 rounded-lg text-sm font-medium border transition-all
                ${active
                  ? 'bg-biotuner-primary text-biotuner-dark-900 border-biotuner-primary shadow'
                  : 'bg-biotuner-dark-800 text-biotuner-light/80 border-biotuner-dark-600 hover:border-biotuner-primary/50'}
              `}
            >
              {g.label}
            </button>
          )
        })}
      </div>

      {/* Description + engine badge */}
      <div className="flex items-start gap-3">
        <p className="flex-1 text-sm text-biotuner-light/70 leading-relaxed">{geom.description}</p>
        <span className={`flex-shrink-0 inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] uppercase tracking-wider font-medium
          ${isPython
            ? 'bg-biotuner-accent/15 text-biotuner-accent border border-biotuner-accent/30'
            : 'bg-biotuner-primary/15 text-biotuner-primary border border-biotuner-primary/30'}`}>
          <Cpu className="w-3 h-3" />
          {isPython ? 'biotuner core' : 'real-time'}
        </span>
      </div>

      {/* Canvas + side panel */}
      <div className="flex flex-col lg:flex-row gap-4">
        {/* Canvas / 3D viewer column */}
        <div className="flex-1 min-w-0">
          <div
            ref={containerRef}
            className="bg-biotuner-dark-900 rounded-xl border border-biotuner-dark-600 overflow-hidden flex items-center justify-center relative"
            style={{ aspectRatio: '1 / 1' }}
          >
            {/* JS engine: 2D canvas */}
            {!isPython && (
              <canvas ref={canvasRef} className="block max-w-full max-h-full" />
            )}
            {/* Python engine, 3D renderer */}
            {isPython && geom.renderer === '3d' && (
              <Suspense fallback={<ViewerFallback message="Loading 3D engine…" />}>
                {pythonGeom ? (
                  <ThreeViewer
                    geometry={pythonGeom}
                    color={color}
                    background="#0a0a0a"
                    autoRotate={autoRotate}
                    wireframe={wireframe}
                  />
                ) : (
                  <ViewerFallback message={pythonError || (pythonLoading ? 'Computing geometry…' : '—')} error={!!pythonError} />
                )}
              </Suspense>
            )}
            {/* Python engine, 2D tree renderer */}
            {isPython && geom.renderer === 'tree2d' && (
              <Suspense fallback={<ViewerFallback message="Loading…" />}>
                {pythonGeom ? (
                  <TreeViewer
                    geometry={pythonGeom}
                    color={color}
                    background="#0a0a0a"
                  />
                ) : (
                  <ViewerFallback message={pythonError || (pythonLoading ? 'Computing geometry…' : '—')} error={!!pythonError} />
                )}
              </Suspense>
            )}
            {/* Loading overlay during recompute */}
            {isPython && pythonLoading && pythonGeom && (
              <div className="absolute top-2 right-2 flex items-center gap-1.5 px-2 py-1 rounded-md bg-biotuner-dark-900/80 border border-biotuner-primary/30 text-xs text-biotuner-primary">
                <Loader2 className="w-3 h-3 animate-spin" />
                computing…
              </div>
            )}
          </div>

          {/* Toolbar under canvas */}
          <div className="mt-3 flex flex-wrap gap-2 items-center">
            {!isPython && (
              <button
                onClick={() => {
                  setAnimate((a) => !a)
                  if (!animate) tStartRef.current = performance.now()
                }}
                aria-pressed={animate}
                className={`
                  min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border
                  ${animate
                    ? 'bg-biotuner-primary/15 border-biotuner-primary/50 text-biotuner-primary'
                    : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/80'}
                `}
              >
                {animate ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                {animate ? 'Pause' : 'Animate'}
              </button>
            )}
            {isPython && geom.renderer === '3d' && (
              <>
                <button
                  onClick={() => setAutoRotate((a) => !a)}
                  aria-pressed={autoRotate}
                  className={`
                    min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border
                    ${autoRotate
                      ? 'bg-biotuner-primary/15 border-biotuner-primary/50 text-biotuner-primary'
                      : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/80'}
                  `}
                >
                  {autoRotate ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  Auto-rotate
                </button>
                <button
                  onClick={() => setWireframe((w) => !w)}
                  aria-pressed={wireframe}
                  className={`
                    min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium border
                    ${wireframe
                      ? 'bg-biotuner-primary/15 border-biotuner-primary/50 text-biotuner-primary'
                      : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/80'}
                  `}
                >
                  Wireframe
                </button>
              </>
            )}

            <button
              onClick={applyFromAnalysis}
              disabled={!ratios.length}
              title={ratios.length
                ? 'Map current tuning ratios into this pattern'
                : 'Run an analysis first'}
              className="min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg bg-biotuner-accent/15 border border-biotuner-accent/40 text-biotuner-accent text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Sparkles className="w-4 h-4" /> From analysis
            </button>

            <button
              onClick={randomize}
              className="min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/80 text-sm hover:border-biotuner-primary/50"
            >
              <Shuffle className="w-4 h-4" /> Randomize
            </button>

            <button
              onClick={resetParams}
              className="min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/80 text-sm hover:border-biotuner-primary/50"
            >
              <RotateCcw className="w-4 h-4" /> Reset
            </button>

            <div className="flex-1" />

            <button
              onClick={exportSvg}
              className="min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/80 text-sm hover:border-biotuner-primary/50"
            >
              <Download className="w-4 h-4" /> {isPython ? 'JSON' : 'SVG'}
            </button>
            <button
              onClick={exportPng}
              className="min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/80 text-sm hover:border-biotuner-primary/50"
            >
              <ImageIcon className="w-4 h-4" /> PNG
            </button>
          </div>
        </div>

        {/* Parameter panel */}
        <div className="lg:w-80 lg:flex-shrink-0 bg-biotuner-dark-900/70 border border-biotuner-dark-600 rounded-xl p-4 space-y-3">
          <h3 className="text-xs font-bold text-biotuner-primary/80 uppercase tracking-widest border-b border-biotuner-dark-600 pb-2">
            Parameters
          </h3>

          {/* Color + palette */}
          <div className="flex items-center gap-3">
            <label className="text-xs text-biotuner-light/60 uppercase tracking-wider">
              Color
            </label>
            <input
              type="color"
              value={color}
              onChange={(e) => setColor(e.target.value)}
              className="w-10 h-9 rounded border border-biotuner-dark-600 bg-transparent cursor-pointer"
            />
            {geom.key === 'chladni' && (
              <select
                value={palette}
                onChange={(e) => setPalette(e.target.value)}
                className="flex-1 bg-biotuner-dark-800 text-biotuner-light/90 border border-biotuner-dark-600 rounded-md p-1.5 text-sm"
              >
                <option value="mono">Mono (sand)</option>
                <option value="accent">Accent color</option>
              </select>
            )}
          </div>

          {/* Sliders for each schema entry — basic first, advanced behind a toggle */}
          {(() => {
            const basic = geom.paramSchema.filter((p) => !p.advanced)
            const advanced = geom.paramSchema.filter((p) => p.advanced)
            return (
              <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
                {basic.map((p) => (
                  <ParamControl
                    key={paramId(geom.key, p.key)}
                    schema={p}
                    value={params[p.key]}
                    onChange={(v) => setParam(p.key, v)}
                  />
                ))}
                {advanced.length > 0 && (
                  <>
                    <button
                      onClick={() => setShowAdvanced((s) => !s)}
                      className="w-full text-left text-[10px] uppercase tracking-wider text-biotuner-light/40 hover:text-biotuner-light/70 border-t border-biotuner-dark-600 pt-3 mt-1"
                    >
                      {showAdvanced ? '▼' : '▶'} Advanced ({advanced.length})
                    </button>
                    {showAdvanced && advanced.map((p) => (
                      <ParamControl
                        key={paramId(geom.key, p.key)}
                        schema={p}
                        value={params[p.key]}
                        onChange={(v) => setParam(p.key, v)}
                      />
                    ))}
                  </>
                )}
              </div>
            )
          })()}
        </div>
      </div>
    </div>
  )
}

// =============================================================================

function ParamControl({ schema, value, onChange }) {
  const { key, label, type, min, max, step, format, options } = schema
  const display = format
    ? format(value)
    : type === 'slider' || type === 'int'
      ? (typeof value === 'number' ? +value.toFixed(step < 1 ? 3 : 0) : value)
      : ''
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1">
        <label
          htmlFor={key}
          className="text-xs font-medium text-biotuner-light/60 uppercase tracking-wider"
        >
          {label}
        </label>
        {display !== '' && (
          <span className="text-xs font-mono text-biotuner-primary/90 tabular-nums">
            {display}
          </span>
        )}
      </div>
      {(type === 'slider' || type === 'int') && (
        <input
          id={key}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => {
            let v = parseFloat(e.target.value)
            if (type === 'int') v = Math.round(v)
            onChange(v)
          }}
          className="w-full accent-biotuner-primary cursor-pointer"
        />
      )}
      {type === 'select' && (
        <select
          id={key}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full bg-biotuner-dark-800 text-biotuner-light/90 border border-biotuner-dark-600 rounded-md p-1.5 text-sm"
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      )}
      {type === 'bool' && (
        <label htmlFor={key} className="flex items-center gap-2 text-sm text-biotuner-light/80 cursor-pointer min-h-[28px]">
          <input
            id={key}
            type="checkbox"
            checked={!!value}
            onChange={(e) => onChange(e.target.checked)}
            className="w-4 h-4 accent-biotuner-primary"
          />
          <span>{value ? 'On' : 'Off'}</span>
        </label>
      )}
    </div>
  )
}

function ViewerFallback({ message, error = false }) {
  return (
    <div className={`flex flex-col items-center justify-center gap-2 text-sm ${
      error ? 'text-red-400' : 'text-biotuner-light/60'
    }`}>
      {!error && <Loader2 className="w-5 h-5 animate-spin opacity-70" />}
      <span className="px-3 text-center">{message}</span>
    </div>
  )
}
