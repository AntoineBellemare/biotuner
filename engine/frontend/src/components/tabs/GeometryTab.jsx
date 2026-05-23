import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react'
import { Play, Pause, Sparkles, Download, RotateCcw, Shuffle, Image as ImageIcon, Cpu, Loader2, Lock } from 'lucide-react'
import {
  GEOMETRY_TYPES,
  GEOMETRY_ORDER,
} from '../../services/geometry/types'
import { SOURCE_MODES, SOURCE_MODE_ORDER, deriveFromMode } from '../../services/geometry/sources'
import {
  drawPath,
  drawField,
  pointsToSvg,
  fieldCanvasToSvg,
  canvasToPngDataUrl,
  downloadFile,
  rotateHue,
} from '../../services/geometry/utils'
import apiClient from '../../services/api'

const ThreeViewer = lazy(() => import('../geometry/ThreeViewer'))
const TreeViewer = lazy(() => import('../geometry/TreeViewer'))
const FieldViewer = lazy(() => import('../geometry/FieldViewer'))

const CANVAS_DPR_CAP = 2

// One-shot helpers --------------------------------------------------------

function paramId(geomKey, paramKey) {
  return `${geomKey}::${paramKey}`
}

function randomizeParams(geom, current = {}) {
  // Keep derived params (those tied to slot bindings — a, b for Lissajous,
  // m, n for Chladni, etc.) and advanced params (hidden by default) so
  // Randomize only re-rolls the controls the user can actually see and
  // expects to vary. Resolution / per-pendulum decays / mode counts stay
  // at their current values.
  const out = { ...current }
  for (const p of geom.paramSchema) {
    if (p.derived || p.advanced) continue
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
  // Color mode: solid / gradient (2-color blend) / rainbow (full hue cycle)
  const [colorMode, setColorMode] = useState('solid')
  const gradient = colorMode !== 'solid'
  const colorEnd = useMemo(() => rotateHue(color, 90), [color])
  // Source mode per geometry — controls how the JS-engine geometries
  // derive their frequency-defining params from the analysis tuning.
  const [sourceModeByType, setSourceModeByType] = useState(() =>
    Object.fromEntries(GEOMETRY_ORDER.map((k) => [k, 'direct']))
  )
  // Per-geometry, which indices of the analysis tuning to keep when sending
  // to the backend (Python geoms). Default = all. Mirrors selected ratios
  // shown as chips. A small "selectAllVersion" bumps when the user clicks
  // "All" so we force-resync after a new analysis lands.
  const [selectedRatiosByType, setSelectedRatiosByType] = useState({})
  // Per-slot ratio bindings for JS geometries: bindingsByType[geom][slotKey] = derivedIndex
  const [bindingsByType, setBindingsByType] = useState({})
  // Auto-morph for JS bindings — cycles each slot through the derived list.
  const [morph, setMorph] = useState(false)
  // Seconds per full sweep through the derived list. Bigger = slower.
  const [morphPeriod, setMorphPeriod] = useState(30)

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
  // Strip frontend-only flags from the request.
  const requestParams = useMemo(() => {
    if (!params) return {}
    const out = { ...params }
    delete out.use_override
    delete out.knot_preset
    delete out.ratio_scale          // applied to tuning, not a backend param
    return out
  }, [params])
  // No debouncing — the previous useDebounced(250) caused a 250 ms window
  // where the effect fired with stale params from the previous geometry,
  // and on first-visit to a Python tab the user often saw nothing because
  // the stale request was discarded by the cancel guard. Instead we use a
  // request-id ref so rapid param changes still coalesce — only the
  // most-recently-fired request's result is applied.
  const requestParamsKey = useMemo(() => JSON.stringify(requestParams), [requestParams])
  // Tuning the backend actually receives for the active Python geometry:
  // 1) chip selection filters the raw analysis ratios → "the ratios I want
  //    biotuner to consider";
  // 2) source mode (direct / harmonics / subharmonics / intermod) then
  //    expands those into the set of values actually sent. Lets the user
  //    explore variations of the same underlying tuning without forcing
  //    explicit p/q overrides on the backend.
  const selectedRatios = useMemo(() => {
    const tuning = analysisResult?.tuning || []
    if (!tuning.length) return []
    // 1) chip filter — which raw analysis ratios participate
    const sel = selectedRatiosByType[type]
    let filtered = tuning
    if (sel && sel.length === tuning.length) {
      filtered = tuning.filter((_, i) => sel[i])
    }
    if (!filtered.length) filtered = tuning
    // 2) source-mode expansion — direct / harmonics / subharmonics / IM
    const mode = sourceModeByType[type] || 'direct'
    let expanded = (mode === 'manual' || mode === 'direct')
      ? filtered
      : (deriveFromMode(filtered, mode) || filtered)
    if (!expanded?.length) expanded = filtered
    // 3) For Python geoms with slot bindings, narrow further to JUST the
    // selected ratios — so e.g. harmonic_knot sees exactly one ratio and
    // biotuner's "pick simplest" lands unambiguously on it. This makes
    // slot bindings the actual control surface for T(p, q) variation.
    // Knot presets short-circuit everything: send a single synthetic ratio
    // so biotuner makes exactly T(p, q). 'data' falls through to the slot
    // picker so the geometry stays tuning-driven.
    if (type === 'harmonic_knot') {
      const preset = params?.knot_preset || 'data'
      const KNOT_PRESET_RATIOS = {
        T_2_1: 2 / 1, T_3_2: 3 / 2, T_5_3: 5 / 3,
        T_5_4: 5 / 4, T_7_4: 7 / 4, T_8_3: 8 / 3,
      }
      if (preset !== 'data' && KNOT_PRESET_RATIOS[preset]) {
        return [KNOT_PRESET_RATIOS[preset]]
      }
    }
    const g = GEOMETRY_TYPES[type]
    if (g?.engine === 'python' && g.slots?.length > 0) {
      const b = bindingsByType[type] || {}
      const picked = []
      const seen = new Set()
      for (const slot of g.slots) {
        const idx = Math.max(
          0,
          Math.min(expanded.length - 1, b[slot.key] ?? 0)
        )
        const r = expanded[idx]
        const key = r?.toFixed(6)
        if (r != null && !seen.has(key)) {
          seen.add(key)
          picked.push(r)
        }
      }
      if (picked.length) return picked
    }
    // Point cloud: apply user-controlled ratio_scale to spread microtonal
    // ratios across a wider range. Biotuner's harmonic_point_cloud's
    // density field reads these as frequencies; ratios that cluster near
    // 1 produce visually-identical patterns no matter which subset is
    // sent.  Multiplying by ratio_scale (× 1–20) changes that.
    if (type === 'harmonic_point_cloud') {
      const scale = Math.max(0.1, Number(params?.ratio_scale ?? 1))
      if (scale !== 1) return expanded.map((r) => r * scale)
    }
    return expanded
  }, [analysisResult, selectedRatiosByType, sourceModeByType, type, bindingsByType, params])
  const ratiosKey = useMemo(() => selectedRatios.join(','), [selectedRatios])

  // Trigger backend compute when a Python style is active. Fires
  // immediately on style change, ratio change, or param change — no
  // debouncing so the first visit to a Python tab renders without a
  // hidden 250 ms window. Request-id ref guards against rapid re-fires.
  useEffect(() => {
    if (!isPython) {
      setPythonError(null)
      setPythonGeom(null)        // wipe prior python geometry on switch to JS
      return
    }
    const reqId = ++requestIdRef.current
    setPythonLoading(true)
    apiClient
      .computeHarmonicGeometry({
        style: geom.style,
        params: requestParams,
        tuning: selectedRatios.length ? selectedRatios : null,
        peaks: analysisResult?.peaks || null,
      })
      .then((data) => {
        if (reqId !== requestIdRef.current) return
        setPythonGeom(data)
        setPythonError(null)
      })
      .catch((err) => {
        if (reqId !== requestIdRef.current) return
        setPythonError(err?.response?.data?.detail || err?.message || 'Geometry compute failed')
        setPythonGeom(null)
      })
      .finally(() => {
        if (reqId === requestIdRef.current) setPythonLoading(false)
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPython, geom.style, requestParamsKey, ratiosKey])

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

  // Re-derive every JS geometry's params from the (ratios, sourceMode,
  // bindings) triple. Fires on analysis-tuning changes, source-mode flips,
  // or per-slot binding changes.
  const sourceMode = sourceModeByType[type] || 'direct'
  const bindings = bindingsByType[type] || {}
  const bindingsKey = JSON.stringify(bindings)
  // What appears in the slot-picker dropdown — exactly the post-pipeline
  // list of ratios after chip filter + source-mode expansion. Keeps the
  // UI honest: what you see is what the backend gets (further narrowed
  // by slot bindings for Python geoms with slots).
  const derivedRatios = useMemo(() => {
    if (sourceMode === 'manual') return []
    if (!ratios.length) return []
    const sel = selectedRatiosByType[type]
    let filtered = ratios
    if (sel && sel.length === ratios.length) {
      filtered = ratios.filter((_, i) => sel[i])
    }
    if (!filtered.length) filtered = ratios
    if (sourceMode === 'direct') return filtered
    return deriveFromMode(filtered, sourceMode) || filtered
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ratiosKeyForJs, sourceMode, selectedRatiosByType, type])

  // Track derivation-relevant params (currently only Rose's max_denom)
  // so the auto-derive refires when the user changes them.
  const activeMaxDenom = paramsByType[type]?.max_denom ?? null

  useEffect(() => {
    if (!ratios.length) return
    setParamsByType((prev) => {
      const next = { ...prev }
      let changed = false
      for (const g of Object.values(GEOMETRY_TYPES)) {
        if (g.engine === 'python') continue
        const mode = sourceModeByType[g.key] || 'direct'
        if (mode === 'manual') continue
        const derived = deriveFromMode(ratios, mode)
        const b = bindingsByType[g.key] || {}
        const updates = (typeof g.fromDerivedRatios === 'function'
          ? g.fromDerivedRatios(derived, b, prev[g.key] || {})
          : (typeof g.fromRatios === 'function' ? g.fromRatios(ratios) : {})) || {}
        if (Object.keys(updates).length === 0) continue
        // No-op guard: skip if every updated key already matches prev — avoids
        // the auto-derive ↔ setState loop when max_denom (a derivation input)
        // is also tracked as a dep.
        let differs = false
        for (const k of Object.keys(updates)) {
          if (prev[g.key]?.[k] !== updates[k]) { differs = true; break }
        }
        if (!differs) continue
        next[g.key] = { ...prev[g.key], ...updates }
        changed = true
      }
      return changed ? next : prev
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ratiosKeyForJs, sourceMode, bindingsKey, activeMaxDenom])

  // -------------------------------------------------------------------------
  // Auto-morph: time-modulate each derived slot through the derived list.
  // Returns the morphed-overlay params for the JS render loop to consume.
  // -------------------------------------------------------------------------
  function morphParams(baseParams, t) {
    if (!morph || sourceMode === 'manual' || isPython) return baseParams
    if (!geom.slots?.length || !derivedRatios.length) return baseParams
    const N = derivedRatios.length
    const cycleSec = Math.max(2, morphPeriod || 30)
    const next = { ...baseParams }
    for (let i = 0; i < geom.slots.length; i++) {
      const slot = geom.slots[i]
      const phase = ((t / cycleSec) * N + i * (N / geom.slots.length)) % N
      const idxA = Math.floor(phase)
      const idxB = (idxA + 1) % N
      let val
      if (slot.transform) {
        // Integer-quantised slot (Chladni m, n etc.): hold each value for
        // the whole step instead of interpolating, otherwise the rounded
        // integer "snaps" mid-step and the pattern looks jumpy. The cycle
        // now feels like a discrete tour rather than a stuttering blur.
        val = derivedRatios[idxA]
        if (slot.scale) val *= slot.scale
        val = slot.transform(val)
      } else {
        // Continuous slot — cosine-ease for smooth glide.
        const linear = phase - idxA
        const fract = 0.5 - 0.5 * Math.cos(linear * Math.PI)
        val = derivedRatios[idxA] * (1 - fract) + derivedRatios[idxB] * fract
        if (slot.scale) val *= slot.scale
      }
      next[slot.key] = val
    }
    return next
  }

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
    setParamsByType((prev) => ({
      ...prev,
      [type]: randomizeParams(geom, prev[type] || geom.defaultParams),
    }))
    tStartRef.current = performance.now()
  }

  // "Shuffle ratios" — keep the visual params, only re-roll slot bindings.
  // Lets the user explore which ratios drive a working pattern without
  // losing tuned phase / line width / scale settings.
  const shuffleRatios = () => {
    if (!derivedRatios.length || !geom.slots?.length) return
    const next = {}
    for (const slot of geom.slots) {
      next[slot.key] = Math.floor(Math.random() * derivedRatios.length)
    }
    setBindingsByType((prev) => ({ ...prev, [type]: next }))
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
      // Geometry-specific animation (e.g. Chladni phase/rotate/breathe)
      // needs a continuously-advancing t even when the global Animate
      // toggle is off.
      const hasGeomAnim = params.animation && params.animation !== 'none'
      const t = (animate || morph || hasGeomAnim)
        ? (performance.now() - tStartRef.current) / 1000
        : 0
      // Chladni's render needs the analysis ratios to compute its integer
      // mode set; merge them into the params just for the render call.
      const effective = { ...morphParams(params, t), ratios: derivedRatios.length ? derivedRatios : ratios }
      const out = geom.render(effective, t)
      lastOutputRef.current = out

      const opts = {
        width: canvas.width,
        height: canvas.height,
        lineWidth: (params.lineWidth || 1.5) * (canvas.width / 600),
        color,
        colorEnd,
        colorMode,
        background: '#0a0a0a',
        contrast: params.contrast,
        palette,
        glow: colorMode === 'solid',
      }
      if (out.kind === 'path') drawPath(ctx, out.points, opts)
      else if (out.kind === 'field') drawField(ctx, out, opts)

      if (animate || morph || hasGeomAnim) rafId = requestAnimationFrame(draw)
    }
    draw()
    return () => { if (rafId) cancelAnimationFrame(rafId) }
  }, [isPython, geom, params, animate, morph, derivedRatios, color, colorEnd, colorMode, palette])

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
                    colorEnd={colorEnd}
                    gradient={gradient}
                    colorMode={colorMode}
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
                    colorEnd={colorEnd}
                    gradient={gradient}
                    colorMode={colorMode}
                    background="#0a0a0a"
                  />
                ) : (
                  <ViewerFallback message={pythonError || (pythonLoading ? 'Computing geometry…' : '—')} error={!!pythonError} />
                )}
              </Suspense>
            )}
            {/* Python engine, 2D field renderer (Chladni) */}
            {isPython && geom.renderer === 'field2d' && (
              <Suspense fallback={<ViewerFallback message="Loading…" />}>
                {pythonGeom ? (
                  <FieldViewer
                    geometry={pythonGeom}
                    color={color}
                    colorEnd={colorEnd}
                    gradient={gradient}
                    colorMode={colorMode}
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
              title="Randomize all visual params (keeps ratio bindings)"
              className="min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/80 text-sm hover:border-biotuner-primary/50"
            >
              <Shuffle className="w-4 h-4" /> Randomize
            </button>
            {!isPython && geom.slots?.length > 0 && derivedRatios.length > 1 && (
              <button
                onClick={shuffleRatios}
                title="Re-roll only the ratio slot bindings — keep everything else"
                className="min-h-[40px] flex items-center gap-2 px-3 py-2 rounded-lg bg-biotuner-accent/10 border border-biotuner-accent/40 text-biotuner-accent text-sm hover:bg-biotuner-accent/20"
              >
                <Shuffle className="w-4 h-4" /> Shuffle ratios
              </button>
            )}

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
          {/* Ratio source — JS geoms apply this to derive their params,
              Python geoms apply it before sending ratios to the backend.
              Manual mode is only meaningful for JS (Python always uses
              the analysis tuning). */}
          {ratios.length > 0 && (
            <div className="pb-3 border-b border-biotuner-dark-600 space-y-2">
              <label className="block text-xs font-bold text-biotuner-accent/80 uppercase tracking-widest">
                Ratio source
              </label>
              <select
                value={sourceMode === 'manual' && isPython ? 'direct' : sourceMode}
                onChange={(e) =>
                  setSourceModeByType((prev) => ({ ...prev, [type]: e.target.value }))
                }
                className="w-full bg-biotuner-dark-800 text-biotuner-light/90 border border-biotuner-dark-600 rounded-md p-2 text-sm"
              >
                {SOURCE_MODE_ORDER
                  .filter((k) => !(isPython && k === 'manual'))
                  .map((k) => (
                    <option key={k} value={k}>{SOURCE_MODES[k].label}</option>
                  ))}
              </select>
              <p className="text-[10px] text-biotuner-light/40 leading-snug">
                {SOURCE_MODES[sourceMode].description}
              </p>

              {/* Per-slot index picker — choose which derived ratio feeds each
                  geometry slot. Hidden in manual mode and during morph. */}
              {sourceMode !== 'manual' && geom.slots?.length > 0 && derivedRatios.length > 0 && !morph && (
                <div className="space-y-1.5 pt-1">
                  <div className="text-[10px] uppercase tracking-wider text-biotuner-light/40">
                    Slot bindings ({derivedRatios.length} ratios available)
                  </div>
                  {geom.slots.map((slot, slotIdx) => {
                    const cur = bindings[slot.key] ?? slotIdx
                    return (
                      <div key={slot.key} className="flex items-center gap-2">
                        <span className="text-xs text-biotuner-light/70 w-16 font-mono">
                          {slot.label || slot.key}
                        </span>
                        <select
                          value={cur}
                          onChange={(e) => {
                            const idx = parseInt(e.target.value, 10)
                            setBindingsByType((prev) => ({
                              ...prev,
                              [type]: { ...(prev[type] || {}), [slot.key]: idx },
                            }))
                          }}
                          className="flex-1 bg-biotuner-dark-800 text-biotuner-light/90 border border-biotuner-dark-600 rounded-md p-1 text-xs"
                        >
                          {derivedRatios.map((r, i) => (
                            <option key={i} value={i}>
                              #{i + 1} → {r.toFixed(3)}
                            </option>
                          ))}
                        </select>
                      </div>
                    )
                  })}
                </div>
              )}

              {/* Auto-morph toggle + speed */}
              {sourceMode !== 'manual' && geom.slots?.length > 0 && derivedRatios.length > 1 && (
                <div className="mt-1 space-y-1.5">
                  <button
                    onClick={() => {
                      setMorph((m) => !m)
                      if (!morph) tStartRef.current = performance.now()
                    }}
                    aria-pressed={morph}
                    className={`w-full min-h-[36px] flex items-center justify-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium border transition-all
                      ${morph
                        ? 'bg-biotuner-accent/15 border-biotuner-accent/50 text-biotuner-accent'
                        : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/70 hover:border-biotuner-accent/50'}`}
                  >
                    {morph ? '⟳ Morphing ratios…' : '⟳ Auto-morph through ratios'}
                  </button>
                  {morph && (
                    <div>
                      <div className="flex items-baseline justify-between text-[10px] uppercase tracking-wider">
                        <span className="text-biotuner-light/50">Morph speed</span>
                        <span className="text-biotuner-accent/80 font-mono">{morphPeriod}s / cycle</span>
                      </div>
                      <input
                        type="range"
                        min={5}
                        max={120}
                        step={1}
                        value={morphPeriod}
                        onChange={(e) => setMorphPeriod(parseInt(e.target.value, 10))}
                        className="w-full accent-biotuner-accent cursor-pointer"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Ratio selection (Python geometries) — toggleable chips so the
              user picks WHICH analysis ratios go into the backend call.
              Now applied to all Python geoms including point_cloud: combined
              with the source-mode picker above this gives genuine variation. */}
          {isPython && ratios.length > 0 && (
            <div className="pb-3 border-b border-biotuner-dark-600 space-y-2">
              <div className="flex items-center justify-between">
                <label className="block text-xs font-bold text-biotuner-accent/80 uppercase tracking-widest">
                  Ratios used ({(selectedRatiosByType[type] || ratios.map(() => true)).filter(Boolean).length}/{ratios.length})
                </label>
                <div className="flex gap-1">
                  <button
                    onClick={() => setSelectedRatiosByType((prev) => ({
                      ...prev, [type]: ratios.map(() => true),
                    }))}
                    className="text-[10px] uppercase tracking-wider text-biotuner-light/60 hover:text-biotuner-primary"
                  >
                    All
                  </button>
                  <span className="text-biotuner-light/30">·</span>
                  <button
                    onClick={() => setSelectedRatiosByType((prev) => ({
                      ...prev, [type]: ratios.map((_, i) => i === 0),
                    }))}
                    className="text-[10px] uppercase tracking-wider text-biotuner-light/60 hover:text-biotuner-primary"
                  >
                    First only
                  </button>
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {ratios.map((r, i) => {
                  const sel = (selectedRatiosByType[type] || ratios.map(() => true))[i] !== false
                  return (
                    <button
                      key={i}
                      onClick={() => {
                        setSelectedRatiosByType((prev) => {
                          const cur = prev[type] || ratios.map(() => true)
                          const next = [...cur]
                          next[i] = !next[i]
                          // Always keep at least one selected
                          if (!next.some(Boolean)) next[i] = true
                          return { ...prev, [type]: next }
                        })
                      }}
                      className={`px-2 py-1 rounded-md text-[11px] font-mono border transition-all
                        ${sel
                          ? 'bg-biotuner-accent/20 border-biotuner-accent/50 text-biotuner-accent'
                          : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/40 line-through'}`}
                      title={`Ratio #${i + 1}: ${r.toFixed(4)}`}
                    >
                      #{i + 1}: {r.toFixed(3)}
                    </button>
                  )
                })}
              </div>
              <p className="text-[10px] text-biotuner-light/40 leading-snug">
                Toggle which analysis ratios go into the backend. Try
                disabling all but two — the dominant ratio pair often gives
                the cleanest knots and L-systems.
              </p>
            </div>
          )}

          <h3 className="text-xs font-bold text-biotuner-primary/80 uppercase tracking-widest border-b border-biotuner-dark-600 pb-2">
            Parameters
          </h3>

          {/* Color + gradient + palette */}
          <div className="flex items-center gap-2 flex-wrap">
            <label className="text-xs text-biotuner-light/60 uppercase tracking-wider">
              Color
            </label>
            <input
              type="color"
              value={color}
              onChange={(e) => setColor(e.target.value)}
              className="w-10 h-9 rounded border border-biotuner-dark-600 bg-transparent cursor-pointer"
            />
            {geom.key !== 'chladni' && (
              <div className="flex gap-0.5 p-0.5 rounded-md bg-biotuner-dark-800 border border-biotuner-dark-600">
                {[
                  { v: 'solid',    label: 'Solid' },
                  { v: 'gradient', label: 'Gradient' },
                  { v: 'rainbow',  label: 'Rainbow' },
                ].map((opt) => (
                  <button
                    key={opt.v}
                    onClick={() => setColorMode(opt.v)}
                    aria-pressed={colorMode === opt.v}
                    className={`min-h-[28px] px-2 py-0.5 rounded text-[11px] font-medium transition-colors
                      ${colorMode === opt.v
                        ? 'bg-biotuner-primary text-biotuner-dark-900'
                        : 'text-biotuner-light/70 hover:text-biotuner-light'}`}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
            {colorMode === 'gradient' && geom.key !== 'chladni' && (
              <div
                className="w-8 h-8 rounded border border-biotuner-dark-600"
                style={{ background: `linear-gradient(90deg, ${color}, ${colorEnd})` }}
                title={`Auto end: ${colorEnd}`}
              />
            )}
            {colorMode === 'rainbow' && geom.key !== 'chladni' && (
              <div
                className="w-8 h-8 rounded border border-biotuner-dark-600"
                style={{ background: 'linear-gradient(90deg, #ff5e5e, #ffd955, #5eff8c, #5ed7ff, #8c5eff, #ff5ed7, #ff5e5e)' }}
                title="Hue cycle"
              />
            )}
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

          {/* Sliders — basic first, advanced behind a disclosure. Derived
              params are locked (read-only) when the source mode is not
              'manual', because the analysis owns those values. */}
          {(() => {
            const lockedDerived = !isPython && sourceMode !== 'manual'
            const basic = geom.paramSchema.filter((p) => !p.advanced)
            const advanced = geom.paramSchema.filter((p) => p.advanced)
            const renderRow = (p) => (
              <ParamControl
                key={paramId(geom.key, p.key)}
                schema={p}
                value={params[p.key]}
                onChange={(v) => setParam(p.key, v)}
                locked={lockedDerived && p.derived}
              />
            )
            return (
              <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
                {basic.map(renderRow)}
                {advanced.length > 0 && (
                  <>
                    <button
                      onClick={() => setShowAdvanced((s) => !s)}
                      className="w-full text-left text-[10px] uppercase tracking-wider text-biotuner-light/40 hover:text-biotuner-light/70 border-t border-biotuner-dark-600 pt-3 mt-1"
                    >
                      {showAdvanced ? '▼' : '▶'} Advanced ({advanced.length})
                    </button>
                    {showAdvanced && advanced.map(renderRow)}
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

function ParamControl({ schema, value, onChange, locked = false }) {
  const { key, label, type, min, max, step, format, options } = schema
  const display = format
    ? format(value)
    : type === 'slider' || type === 'int'
      ? (typeof value === 'number' ? +value.toFixed(step < 1 ? 3 : 0) : value)
      : ''
  return (
    <div className={locked ? 'opacity-70' : ''}>
      <div className="flex items-baseline justify-between mb-1">
        <label
          htmlFor={key}
          className="text-xs font-medium text-biotuner-light/60 uppercase tracking-wider flex items-center gap-1"
        >
          {label}
          {locked && (
            <span
              title="Derived from analysis ratios. Switch source to ‘Manual’ to edit."
              className="text-biotuner-accent/80"
            >
              <Lock className="w-3 h-3" />
            </span>
          )}
        </label>
        {display !== '' && (
          <span className={`text-xs font-mono tabular-nums ${
            locked ? 'text-biotuner-accent/80' : 'text-biotuner-primary/90'
          }`}>
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
          disabled={locked}
          onChange={(e) => {
            if (locked) return
            let v = parseFloat(e.target.value)
            if (type === 'int') v = Math.round(v)
            onChange(v)
          }}
          className={`w-full accent-biotuner-primary ${locked ? 'cursor-not-allowed' : 'cursor-pointer'}`}
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
