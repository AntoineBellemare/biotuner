/**
 * WavetableStudio — the interactive multi-frame wavetable view for the
 * Timbre tab. Three coordinated pieces:
 *
 *   1. Large single-cycle preview (the current frame, big and readable)
 *   2. Frame strip below — every frame as a thumbnail, click any to
 *      jump there, current frame highlighted
 *   3. Scrubber + Animate toggle + evolution-mode picker + frame count
 *
 * Pulls data from /api/timbre/wavetable using the same Timbre pipeline
 * as compute / export, so "what you see is what you export" — the
 * displayed frames are byte-for-byte the same as what would land in
 * the downloaded .wav wavetable.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { Loader2, Play, Pause } from 'lucide-react'

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''
const client = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

// Evolution-mode picker. Each entry maps to one of the per-frame helpers
// in biotuner.harmonic_timbre.exporters.to_wavetable.
const EVOLUTION_OPTIONS = [
  { value: 'tilt',             label: 'Spectral tilt sweep',     hint: 'Brightness ramp from dark to bright' },
  { value: 'harmonic_buildup', label: 'Harmonic buildup',        hint: 'Partials fade in one by one' },
  { value: 'amp_morph',        label: 'Amplitude morph',         hint: 'Random → matched amplitudes' },
  { value: 'phase_sweep',      label: 'Phase sweep',             hint: 'Partial phases rotate 0 → 2π' },
  { value: 'intermod_buildup', label: 'Intermod buildup',        hint: 'f₁±f₂ sidebands fade in' },
  { value: 'harmonic_stack',   label: 'Harmonic stack buildup',  hint: 'Overtones 2f, 3f, … fade in' },
  { value: 'formant_sweep',    label: 'Formant sweep',           hint: 'Vowel-like ah → ee' },
]

const FRAME_COUNT_OPTIONS = [8, 16, 32, 64, 128]

/** Draw a single waveform onto a canvas (used by both main + thumbnails). */
function drawWave(ctx, samples, w, h, opts = {}) {
  const color    = opts.color    || '#a78bfa'   // violet
  const lineW    = opts.lineWidth || 1.5
  const showAxis = opts.showAxis !== false
  const padTop   = opts.padTop   ?? 4
  const padBot   = opts.padBot   ?? 4
  ctx.clearRect(0, 0, w, h)
  if (showAxis) {
    ctx.strokeStyle = 'rgba(255,255,255,0.08)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke()
  }
  const usableH = h - padTop - padBot
  ctx.strokeStyle = color
  ctx.lineWidth = lineW
  ctx.beginPath()
  const N = samples.length
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * w
    const y = padTop + (1 - (samples[i] + 1) / 2) * usableH
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
}

// ============================================================================
// Main preview canvas (DPR-aware, draws the currently-selected frame)
// ============================================================================
function MainWaveCanvas({ samples, height = 200, animated = false, rate = 4 }) {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  // For animated playback, an internal frame index can be driven by a
  // parent later; here we just draw whatever `samples` is at any time.
  useEffect(() => {
    if (!canvasRef.current || !wrapRef.current) return
    const resize = () => {
      const c = canvasRef.current
      if (!c || !wrapRef.current) return
      const cssW = wrapRef.current.clientWidth
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      c.style.width  = `${cssW}px`
      c.style.height = `${height}px`
      c.width  = Math.round(cssW * dpr)
      c.height = Math.round(height * dpr)
      if (samples) {
        const ctx = c.getContext('2d')
        drawWave(ctx, samples, c.width, c.height, {
          lineWidth: 2 * dpr,
          padTop: 12 * dpr,
          padBot: 12 * dpr,
        })
      }
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [samples, height])
  return (
    <div ref={wrapRef} className="w-full bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600">
      <canvas ref={canvasRef} className="block w-full" />
    </div>
  )
}

// ============================================================================
// Strip of frame thumbnails (low-DPR, compact, click to jump)
// ============================================================================
function FrameStrip({ frames, currentIdx, onJump, height = 56 }) {
  const wrapRef = useRef(null)
  const [frameWidth, setFrameWidth] = useState(40)
  useEffect(() => {
    if (!wrapRef.current) return
    const compute = () => {
      const w = wrapRef.current.clientWidth
      // Aim for ~10–24 thumbnails visible; size each one to fit cleanly.
      // Scrollable horizontally if there are more than fit.
      const desired = Math.max(28, Math.min(64, Math.floor((w - 8) / frames.length) - 4))
      setFrameWidth(desired)
    }
    compute()
    const ro = new ResizeObserver(compute)
    ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [frames])

  return (
    <div ref={wrapRef} className="w-full overflow-x-auto">
      <div className="flex gap-1 pb-1" style={{ minWidth: 'min-content' }}>
        {frames.map((samples, i) => (
          <FrameThumb
            key={i}
            samples={samples}
            isCurrent={i === currentIdx}
            width={frameWidth}
            height={height}
            onClick={() => onJump(i)}
            idx={i}
          />
        ))}
      </div>
    </div>
  )
}

function FrameThumb({ samples, isCurrent, width, height, onClick, idx }) {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    ref.current.width  = Math.round(width * dpr)
    ref.current.height = Math.round(height * dpr)
    ref.current.style.width  = `${width}px`
    ref.current.style.height = `${height}px`
    const ctx = ref.current.getContext('2d')
    drawWave(ctx, samples, ref.current.width, ref.current.height, {
      color: isCurrent ? '#a78bfa' : 'rgba(167,139,250,0.4)',
      lineWidth: (isCurrent ? 1.5 : 1) * dpr,
      showAxis: false,
      padTop: 2 * dpr, padBot: 2 * dpr,
    })
  }, [samples, isCurrent, width, height])
  return (
    <button
      onClick={onClick}
      title={`Frame ${idx + 1}`}
      className={`flex-shrink-0 rounded border transition-colors
        ${isCurrent
          ? 'border-biotuner-primary bg-biotuner-primary/10'
          : 'border-biotuner-dark-600 bg-biotuner-dark-900 hover:border-biotuner-accent/50'}`}
    >
      <canvas ref={ref} className="block" />
    </button>
  )
}

// ============================================================================
// WavetableStudio — orchestrates fetching, scrubbing, and playback
// ============================================================================
export default function WavetableStudio({ requestPayload }) {
  const [evolution, setEvolution] = useState('tilt')
  const [nFrames, setNFrames]     = useState(32)
  const [currentIdx, setCurrentIdx] = useState(0)
  const [animPlaying, setAnimPlaying] = useState(false)
  const [animRate, setAnimRate]   = useState(8)   // frames per second
  const [wavetable, setWavetable] = useState(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const requestIdRef = useRef(0)
  const animRafRef = useRef(null)
  const lastFrameTimeRef = useRef(0)

  // Re-fetch when payload, evolution, or n_frames changes.
  useEffect(() => {
    if (!requestPayload) {
      setWavetable(null)
      return
    }
    const payload = {
      ...requestPayload,
      wavetable_config: {
        n_frames: nFrames,
        evolution,
        table_size: 512,
      },
    }
    const id = ++requestIdRef.current
    setLoading(true)
    setError(null)
    client.post('/api/timbre/wavetable', payload)
      .then((res) => {
        if (id !== requestIdRef.current) return
        setWavetable(res.data)
        // Keep current frame valid when nFrames shrinks.
        setCurrentIdx((i) => Math.min(i, res.data.frames.length - 1))
        setLoading(false)
      })
      .catch((e) => {
        if (id !== requestIdRef.current) return
        setError(e.response?.data?.detail || e.message || 'wavetable failed')
        setLoading(false)
      })
  }, [requestPayload, evolution, nFrames])

  // Animate the scrubber when playing — advances currentIdx at ``animRate``
  // frames per second, looping.
  useEffect(() => {
    if (!animPlaying || !wavetable) {
      if (animRafRef.current) cancelAnimationFrame(animRafRef.current)
      return
    }
    const N = wavetable.frames.length
    const step = (now) => {
      const dt = (now - lastFrameTimeRef.current) / 1000
      if (dt >= 1 / animRate) {
        lastFrameTimeRef.current = now
        setCurrentIdx((i) => (i + 1) % N)
      }
      animRafRef.current = requestAnimationFrame(step)
    }
    lastFrameTimeRef.current = performance.now()
    animRafRef.current = requestAnimationFrame(step)
    return () => {
      if (animRafRef.current) cancelAnimationFrame(animRafRef.current)
    }
  }, [animPlaying, wavetable, animRate])

  const currentSamples = useMemo(() => {
    if (!wavetable) return null
    const idx = Math.max(0, Math.min(currentIdx, wavetable.frames.length - 1))
    return wavetable.frames[idx]
  }, [wavetable, currentIdx])

  const evolutionOpt = EVOLUTION_OPTIONS.find((o) => o.value === evolution)

  return (
    <div className="bg-biotuner-dark-900/70 border border-biotuner-dark-600 rounded-xl p-4 space-y-3">
      <div className="flex items-baseline justify-between gap-2 flex-wrap">
        <h3 className="text-xs font-bold text-biotuner-accent/80 uppercase tracking-widest">
          Wavetable studio
        </h3>
        {wavetable && (
          <span className="text-[10px] text-biotuner-light/40">
            frame {currentIdx + 1} of {wavetable.frames.length} ·
            {' '}{evolutionOpt?.hint || wavetable.evolution_label}
          </span>
        )}
      </div>

      {/* Evolution + frame count picker */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 sm:gap-3">
        <div className="col-span-2 sm:col-span-2">
          <label className="block text-[10px] uppercase tracking-wider text-biotuner-light/50 mb-1">
            Evolution
          </label>
          <select
            value={evolution}
            onChange={(e) => setEvolution(e.target.value)}
            className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-2 text-sm"
          >
            {EVOLUTION_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-biotuner-light/50 mb-1">
            Frames
          </label>
          <select
            value={nFrames}
            onChange={(e) => setNFrames(parseInt(e.target.value, 10))}
            className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-2 text-sm"
          >
            {FRAME_COUNT_OPTIONS.map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Errors */}
      {error && (
        <div className="text-xs text-red-300 bg-red-500/10 border border-red-500/30 rounded p-2">
          {error}
        </div>
      )}

      {/* Main wave + loading overlay */}
      <div className="relative">
        {loading && (
          <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5 px-2 py-1
                          rounded-md bg-biotuner-dark-900/80 border border-biotuner-accent/30
                          text-xs text-biotuner-accent">
            <Loader2 className="w-3 h-3 animate-spin" /> computing wavetable…
          </div>
        )}
        <MainWaveCanvas samples={currentSamples} height={180} />
      </div>

      {/* Scrubber + transport */}
      {wavetable && wavetable.frames.length > 1 && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setAnimPlaying((p) => !p)}
              title={animPlaying ? 'Pause animation' : 'Play animation'}
              className={`min-h-[36px] px-3 flex items-center gap-1.5 rounded-md text-xs font-medium border
                ${animPlaying
                  ? 'bg-biotuner-accent/20 border-biotuner-accent text-biotuner-accent'
                  : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/70 hover:border-biotuner-accent/50'}`}
            >
              {animPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {animPlaying ? 'Pause' : 'Animate'}
            </button>
            <input
              type="range"
              min={0}
              max={wavetable.frames.length - 1}
              step={1}
              value={currentIdx}
              onChange={(e) => {
                setAnimPlaying(false)
                setCurrentIdx(parseInt(e.target.value, 10))
              }}
              className="flex-1 accent-biotuner-accent"
            />
            <span className="font-mono text-xs text-biotuner-light/60 w-14 text-right">
              {currentIdx + 1}/{wavetable.frames.length}
            </span>
          </div>
          {animPlaying && (
            <div className="flex items-center gap-2 text-[10px] text-biotuner-light/50">
              <span className="uppercase tracking-wider">Rate</span>
              <input
                type="range"
                min={1}
                max={30}
                step={1}
                value={animRate}
                onChange={(e) => setAnimRate(parseInt(e.target.value, 10))}
                className="flex-1 accent-biotuner-accent"
              />
              <span className="font-mono w-12 text-right">{animRate} fps</span>
            </div>
          )}
        </div>
      )}

      {/* Frame strip */}
      {wavetable && wavetable.frames.length > 1 && (
        <FrameStrip
          frames={wavetable.frames}
          currentIdx={currentIdx}
          onJump={(i) => { setAnimPlaying(false); setCurrentIdx(i) }}
        />
      )}
    </div>
  )
}
