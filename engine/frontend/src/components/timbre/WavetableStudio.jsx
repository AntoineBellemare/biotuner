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

import CompositeBuilder from './CompositeBuilder'

// Default layer stack when the user first picks "Composite" — a
// recognisable starting point that demonstrates non-trivial multi-axis
// behaviour without overwhelming them.
const DEFAULT_COMPOSITE_LAYERS = [
  {
    evolution: 'harmonic_stack',
    weight_curve: 'linear',
    weight_min: 0,
    weight_max: 4,
    params: { rolloff: 0.9 },
  },
  {
    evolution: 'wavefolding',
    weight_curve: 'ease_in',
    weight_min: 0.0,
    weight_max: 2.5,
    params: { output_drive: 1.0 },
  },
]

const API_BASE_URL = import.meta.env.VITE_API_URL || ''
const client = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

// Evolution-mode picker. Each entry maps to one of the per-frame helpers
// in biotuner.harmonic_timbre.exporters.to_wavetable.
const EVOLUTION_OPTIONS = [
  // Linear evolutions
  { value: 'tilt',             label: 'Spectral tilt sweep',     hint: 'Brightness ramp from dark to bright' },
  { value: 'harmonic_buildup', label: 'Harmonic buildup',        hint: 'Partials fade in one by one' },
  { value: 'amp_morph',        label: 'Amplitude morph',         hint: 'Random → matched amplitudes' },
  { value: 'phase_sweep',      label: 'Phase sweep',             hint: 'Partial phases rotate 0 → 2π' },
  { value: 'intermod_buildup', label: 'Intermod buildup',        hint: 'f₁±f₂ sidebands fade in' },
  { value: 'harmonic_stack',   label: 'Harmonic stack buildup',  hint: 'Overtones 2f, 3f, … fade in' },
  { value: 'formant_sweep',    label: 'Formant sweep',           hint: 'Vowel-like ah → ee' },
  // Nonlinear enrichments — Buchla folder + audio-rate FM. Both bake
  // their character into the per-frame cycle so the exported wavetable
  // carries it without needing the host synth to recreate.
  { value: 'wavefolding',      label: 'Wavefolding (nonlinear)', hint: 'Buchla-style sin folder · adds odd harmonics' },
  { value: 'fm_baked',         label: 'FM baked (nonlinear)',    hint: 'Audio-rate FM written into the cycle · bell character' },
  // Composite — multi-axis. Opens a layer-builder panel below the
  // dropdown when selected. Maxes out at 4 layers (UI choice).
  { value: 'composite',        label: 'Composite (multi-axis)',  hint: 'Chain 2–4 evolutions with per-layer curves' },
  // Biosignal-structure evolutions — exploit aspects of the bt that
  // synthetic sources can't access (FOOOF decomposition, IMFs, bands).
  { value: 'noise_to_structure', label: 'Noise → structure (bio)', hint: '1/f^k noise crystallises into your harmonic identity' },
  { value: 'imf_morph',          label: 'IMF morph (bio)',         hint: 'Walk through EMD intrinsic mode functions — high-freq → low-freq spectral content' },
  { value: 'band_morph',         label: 'Band morph (bio)',        hint: 'Slice the spectrum into bands, build a Timbre per band, morph across them' },
]

// Band-edge presets — common slicing conventions for typical signal
// types. Custom edges can be typed in directly.
const BAND_PRESETS = [
  { value: 'eeg',    label: 'EEG (δ θ α β γ)',   edges: [1, 4, 8, 13, 30, 100] },
  { value: 'hrv',    label: 'HRV (VLF / LF / HF)', edges: [0.003, 0.04, 0.15, 0.5] },
  { value: 'octave', label: 'Audible octaves',   edges: [40, 80, 160, 320, 640, 1280, 2560, 5120] },
  { value: 'log4',   label: 'Log (4 bands)',     edges: [1, 10, 100, 1000, 10000] },
  { value: 'custom', label: 'Custom (edit below)', edges: null },
]

// Blend modes for imf_morph / band_morph — exposed in a small dropdown
// next to the evolution picker when one of those modes is selected.
const TIMBRE_MORPH_BLENDS = [
  { value: 'linear_walk', label: 'Linear walk (smoothest)' },
  { value: 'pure',        label: 'Pure (stepwise)' },
  { value: 'gaussian',    label: 'Gaussian (multi-overlap)' },
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
// StackedFramesView — Ableton-style isometric wavetable stack.
//
// All frames rendered as polylines in a single canvas, each one offset
// in (x, y) so they recede into the distance. Drawn back-to-front (so
// the front frame visually overlaps the back ones). Color picks up a
// violet → cyan gradient from back to front; the current frame is
// emphasised with a thicker stroke and a cyan accent dot at frame 0.
//
// Click anywhere on the canvas → jump to the nearest frame's "anchor
// point" along the depth axis. Drag (mousemove with button held) →
// continuous scrub. The whole thing is a click-target so the user
// can grab any visible frame.
// ============================================================================
function StackedFramesView({ frames, currentIdx, onJump, height = 240 }) {
  const canvasRef = useRef(null)
  const wrapRef = useRef(null)
  // Layout constants — tuned to look like Ableton's Wavetable view.
  // The depth axis runs up-and-right from the bottom-left corner.
  const DEPTH_X_FRAC = 0.32   // fraction of width devoted to depth offset
  const DEPTH_Y_FRAC = 0.55   // fraction of height devoted to depth offset
  const AMP_FRAC     = 0.28   // fraction of height for each frame's amplitude
  const draggingRef = useRef(false)

  useEffect(() => {
    if (!canvasRef.current || !wrapRef.current) return
    const draw = () => {
      const c = canvasRef.current
      if (!c) return
      const cssW = wrapRef.current.clientWidth
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      c.style.width  = `${cssW}px`
      c.style.height = `${height}px`
      c.width  = Math.round(cssW * dpr)
      c.height = Math.round(height * dpr)
      const ctx = c.getContext('2d')
      const W = c.width
      const H = c.height
      ctx.clearRect(0, 0, W, H)

      const N = frames.length
      if (N === 0) return

      // Available footprint for the wave + depth offset.
      const padX = 10 * dpr
      const padY = 10 * dpr
      const usableW = W - 2 * padX
      const usableH = H - 2 * padY
      const depthX = usableW * DEPTH_X_FRAC
      const depthY = usableH * DEPTH_Y_FRAC
      const stepX = N > 1 ? depthX / (N - 1) : 0
      const stepY = N > 1 ? depthY / (N - 1) : 0
      const waveW = usableW - depthX
      const ampH  = usableH * AMP_FRAC

      // Subtle depth-axis line so the perspective reads even when the
      // wavetable is mostly silent.
      ctx.strokeStyle = 'rgba(255,255,255,0.05)'
      ctx.lineWidth = 1 * dpr
      ctx.beginPath()
      ctx.moveTo(padX, H - padY)
      ctx.lineTo(padX + depthX, H - padY - depthY)
      ctx.stroke()

      // Draw frames back-to-front so the front frame visually wins
      // overlapping pixels. "Back" = highest index here.
      for (let i = N - 1; i >= 0; i--) {
        const samples = frames[i]
        if (!samples || samples.length === 0) continue
        const ox = padX + i * stepX
        const oy = padY + (depthY - i * stepY) + (usableH - depthY) / 2
        const isCurrent = i === currentIdx
        // Color: hue interpolates violet (back) → cyan (front) with the
        // current frame snapped to a saturated cyan accent regardless of
        // position. Alpha rises toward the front so depth reads naturally.
        const t = N > 1 ? (1 - i / (N - 1)) : 1
        const hue = 280 - 100 * t   // 280 (violet) → 180 (cyan)
        const alpha = 0.18 + 0.55 * t
        if (isCurrent) {
          ctx.strokeStyle = '#06b6d4'
          ctx.lineWidth = 2.2 * dpr
          ctx.shadowColor = 'rgba(6, 182, 212, 0.7)'
          ctx.shadowBlur = 8 * dpr
        } else {
          ctx.strokeStyle = `hsla(${hue}, 70%, 62%, ${alpha})`
          ctx.lineWidth = 1.2 * dpr
          ctx.shadowBlur = 0
        }
        ctx.beginPath()
        const S = samples.length
        for (let s = 0; s < S; s++) {
          const sampleX = (s / (S - 1)) * waveW
          const sampleY = -samples[s] * (ampH / 2)
          const sx = ox + sampleX
          const sy = oy + sampleY
          if (s === 0) ctx.moveTo(sx, sy)
          else ctx.lineTo(sx, sy)
        }
        ctx.stroke()
      }
      ctx.shadowBlur = 0

      // Frame index labels at the two extremes — helps users orient
      // themselves to the depth axis when the wavetable is dense.
      ctx.fillStyle = 'rgba(255,255,255,0.4)'
      ctx.font = `${10 * dpr}px ui-monospace, monospace`
      ctx.textAlign = 'left'
      ctx.fillText('1', padX - 2 * dpr, H - padY + 12 * dpr)
      ctx.fillText(`${N}`, padX + depthX, padY + depthY / 2 - 4 * dpr)
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(wrapRef.current)
    return () => ro.disconnect()
  }, [frames, currentIdx, height])

  // Click / drag → jump to nearest frame.
  // Compute the inverse mapping: given pointer (px, py) inside canvas,
  // find the frame whose anchor (ox, oy) is closest.
  const handlePoint = (e) => {
    const c = canvasRef.current
    if (!c) return
    const rect = c.getBoundingClientRect()
    const dpr = c.width / rect.width
    const px = (e.clientX - rect.left) * dpr
    const py = (e.clientY - rect.top) * dpr
    const W = c.width
    const H = c.height
    const N = frames.length
    if (N === 0) return
    const padX = 10 * dpr
    const padY = 10 * dpr
    const usableW = W - 2 * padX
    const usableH = H - 2 * padY
    const depthX = usableW * DEPTH_X_FRAC
    const depthY = usableH * DEPTH_Y_FRAC
    const stepX = N > 1 ? depthX / (N - 1) : 0
    const stepY = N > 1 ? depthY / (N - 1) : 0
    let best = 0
    let bestD = Infinity
    for (let i = 0; i < N; i++) {
      const ox = padX + i * stepX + (usableW - depthX) * 0.5   // anchor = mid of wave
      const oy = padY + (depthY - i * stepY) + (usableH - depthY) / 2
      const d = (px - ox) * (px - ox) + (py - oy) * (py - oy)
      if (d < bestD) { bestD = d; best = i }
    }
    onJump(best)
  }
  const onDown = (e) => { draggingRef.current = true;  handlePoint(e) }
  const onMove = (e) => { if (draggingRef.current) handlePoint(e) }
  const onUp   = ()   => { draggingRef.current = false }

  // Touch handlers — preventDefault on touchmove stops the page from
  // scrolling while the user drags across the wavetable to scrub. Read
  // touches[0] for the primary finger; multi-touch zooming is left to
  // the browser since the parent layout doesn't need pinch handling.
  const touchToEvent = (e) => {
    const t = e.touches?.[0] || e.changedTouches?.[0]
    if (!t) return null
    return { clientX: t.clientX, clientY: t.clientY }
  }
  const onTouchStart = (e) => {
    const pt = touchToEvent(e)
    if (!pt) return
    e.preventDefault()
    draggingRef.current = true
    handlePoint(pt)
  }
  const onTouchMove = (e) => {
    if (!draggingRef.current) return
    const pt = touchToEvent(e)
    if (!pt) return
    e.preventDefault()
    handlePoint(pt)
  }
  const onTouchEnd = (e) => {
    e.preventDefault()
    draggingRef.current = false
  }

  return (
    <div ref={wrapRef} className="w-full bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600">
      <canvas
        ref={canvasRef}
        className="block w-full cursor-crosshair touch-none"
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={onUp}
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      />
    </div>
  )
}

// ============================================================================
// WavetableStudio — orchestrates fetching, scrubbing, and playback
// ============================================================================
export default function WavetableStudio({ requestPayload, sessionId, samplingRate }) {
  // Format the session's sampling rate as a friendly chip — "44.1 kHz"
  // for audio, "1 kHz" for EEG-ish, "256 Hz" for CSV defaults, etc.
  // Surfaced in the IMF and band panels so the user can sanity-check
  // that EMD / bandpass is operating on the rate they expect.
  const sfChip = samplingRate
    ? (samplingRate >= 1000
        ? `${(samplingRate / 1000).toFixed(samplingRate >= 10000 ? 0 : 1)} kHz`
        : `${Math.round(samplingRate)} Hz`)
    : null
  const [evolution, setEvolution] = useState('tilt')
  const [nFrames, setNFrames]     = useState(32)
  // Composite layers state — only consumed when evolution === 'composite'.
  // Initialised to a sensible 2-layer stack so the user sees something
  // working on first pick instead of an empty editor.
  const [compositeLayers, setCompositeLayers] = useState(DEFAULT_COMPOSITE_LAYERS)
  // IMF-morph state: cached IMF Timbres from the backend + blend mode.
  // Resets when session changes (different signal = invalid cache).
  const [imfTimbres, setImfTimbres]       = useState([])
  const [computingImfs, setComputingImfs] = useState(false)
  const [imfMethod,  setImfMethod]        = useState('EMD')   // 'EMD' | 'EEMD' | 'CEEMDAN'
  const [imfCount,   setImfCount]         = useState(5)
  // Index range — which of the cached IMFs actually feed the morph.
  // Defaults to "all of them" but the user can trim the noisy top
  // (first IMF often contains 60Hz hum or scanner noise) or the
  // residual bottom (last IMFs are trend/DC).
  const [imfRange,   setImfRange]         = useState([1, 99])
  // Clamp the range whenever the cached count changes so it stays
  // within bounds after re-computing with a different imfCount.
  useEffect(() => {
    if (imfTimbres.length === 0) return
    setImfRange(([lo, hi]) => [
      Math.max(1, Math.min(lo, imfTimbres.length)),
      Math.max(1, Math.min(hi, imfTimbres.length)),
    ])
  }, [imfTimbres.length])
  const [blendMode,  setBlendMode]        = useState('linear_walk')
  const [gaussianSigma, setGaussianSigma] = useState(0.5)

  useEffect(() => { setImfTimbres([]) }, [sessionId])
  // Band-morph state: cached band Timbres + the edges that produced
  // them + the user's preset choice + a textual edges string for the
  // "Custom" editor.
  const [bandTimbres, setBandTimbres]     = useState([])
  const [computingBands, setComputingBands] = useState(false)
  const [bandPreset, setBandPreset]       = useState('eeg')
  const [bandEdgesText, setBandEdgesText] = useState('1, 4, 8, 13, 30, 100')
  useEffect(() => { setBandTimbres([]) }, [sessionId])
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
        // Forward layers only for composite mode; the backend errors
        // if the list is empty when composite is selected.
        ...(evolution === 'composite' ? { layers: compositeLayers } : {}),
        // For IMF / band morph, forward the cached Timbre sequence
        // plus the blend params.
        ...(evolution === 'imf_morph' && imfTimbres.length
          ? {
              // Slice to the user-selected IMF range. Indices are 1-based
              // in the UI; convert here. Inclusive on both ends.
              timbre_sequence: imfTimbres.slice(
                Math.max(0, imfRange[0] - 1),
                Math.min(imfTimbres.length, imfRange[1]),
              ),
              blend_mode: blendMode,
              gaussian_sigma: gaussianSigma,
            }
          : {}),
        ...(evolution === 'band_morph' && bandTimbres.length
          ? {
              timbre_sequence: bandTimbres,
              blend_mode: blendMode,
              gaussian_sigma: gaussianSigma,
            }
          : {}),
      },
    }
    // Don't fetch when imf_morph / band_morph is selected but the
    // sequence cache hasn't been populated — backend would 400; the
    // UI shows the relevant compute button instead.
    if (evolution === 'imf_morph' && imfTimbres.length === 0) {
      setWavetable(null)
      setLoading(false)
      return
    }
    if (evolution === 'band_morph' && bandTimbres.length === 0) {
      setWavetable(null)
      setLoading(false)
      return
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
  }, [requestPayload, evolution, nFrames, compositeLayers,
      imfTimbres, imfRange, bandTimbres, blendMode, gaussianSigma])

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

  // ----- IMF computation (server-side EMD on the session signal) ------
  const handleComputeImfs = async () => {
    if (!sessionId) {
      setError('No active session — re-run the analysis first')
      return
    }
    setComputingImfs(true)
    setError(null)
    try {
      const res = await client.post('/api/timbre/imfs', {
        session_id: sessionId,
        n_imfs: imfCount,
        n_peaks_per_imf: 4,
        method: imfMethod,
      })
      setImfTimbres(res.data.imfs || [])
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'IMF extraction failed')
    } finally {
      setComputingImfs(false)
    }
  }

  // ----- Band computation (server-side bandpass on the session) -------
  const handleComputeBands = async () => {
    if (!sessionId) {
      setError('No active session — re-run the analysis first')
      return
    }
    // Parse the edges-text input ("1, 4, 8, 13, 30, 100") into a list
    // of positive floats, sorted ascending.
    const edges = bandEdgesText
      .split(/[,\s]+/)
      .map((s) => parseFloat(s))
      .filter((v) => Number.isFinite(v) && v > 0)
      .sort((a, b) => a - b)
    if (edges.length < 2) {
      setError('Need at least 2 band edges (e.g. "8, 13")')
      return
    }
    setComputingBands(true)
    setError(null)
    try {
      const res = await client.post('/api/timbre/bands', {
        session_id: sessionId,
        band_edges: edges,
        n_peaks_per_band: 4,
      })
      setBandTimbres(res.data.bands || [])
    } catch (e) {
      setError(e.response?.data?.detail || e.message || 'Band extraction failed')
    } finally {
      setComputingBands(false)
    }
  }

  // Switching presets repopulates the edges-text editor (custom keeps
  // whatever the user typed).
  const handleBandPresetChange = (val) => {
    setBandPreset(val)
    const preset = BAND_PRESETS.find((p) => p.value === val)
    if (preset?.edges) {
      setBandEdgesText(preset.edges.join(', '))
    }
  }

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

      {/* Composite layer builder — only visible when composite mode
          is selected. Shown above the wave preview so the cause/effect
          relationship is obvious (edit layers → preview updates). */}
      {evolution === 'composite' && (
        <div className="bg-biotuner-dark-900/60 border border-biotuner-dark-600 rounded-lg p-2.5">
          <div className="text-[10px] uppercase tracking-wider text-biotuner-accent/80 mb-2">
            Layers
          </div>
          <CompositeBuilder
            layers={compositeLayers}
            onChange={setCompositeLayers}
          />
        </div>
      )}

      {/* IMF-morph control panel — visible when imf_morph is selected.
          Lets the user pick EMD method + IMF count, fire the compute,
          and pick a blend mode for how the resulting Timbre sequence
          mixes across frames. */}
      {evolution === 'imf_morph' && (
        <div className="bg-biotuner-dark-900/60 border border-biotuner-dark-600 rounded-lg p-2.5 space-y-2">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wider text-biotuner-accent/80">
                IMF morph
              </span>
              {sfChip && (
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded
                                 bg-biotuner-dark-800 border border-biotuner-dark-600
                                 text-biotuner-light/60"
                      title="Session sample rate. EMD operates on the signal at this rate; the resulting IMFs span frequencies up to sf/2.">
                  sf: {sfChip}
                </span>
              )}
            </div>
            <div className="text-[10px] text-biotuner-light/40">
              {imfTimbres.length
                ? `${imfTimbres.length} IMFs cached · walk runs from high → low freq`
                : 'Click compute to extract IMFs from your signal'}
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            <div>
              <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
                EMD method
              </label>
              <select
                value={imfMethod}
                onChange={(e) => setImfMethod(e.target.value)}
                className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1 text-xs"
              >
                <option value="EMD">EMD (fast)</option>
                <option value="EEMD">EEMD (noise-assisted)</option>
                <option value="CEEMDAN">CEEMDAN (best)</option>
              </select>
            </div>
            <div>
              <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
                Max IMFs
              </label>
              <input
                type="number"
                min={2} max={10} step={1}
                value={imfCount}
                onChange={(e) => setImfCount(Math.max(2, Math.min(10, parseInt(e.target.value, 10) || 5)))}
                className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1 text-xs font-mono"
              />
            </div>
            <div>
              <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
                Blend
              </label>
              <select
                value={blendMode}
                onChange={(e) => setBlendMode(e.target.value)}
                className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1 text-xs"
              >
                {TIMBRE_MORPH_BLENDS.map((b) => (
                  <option key={b.value} value={b.value}>{b.label}</option>
                ))}
              </select>
            </div>
            {blendMode === 'gaussian' && (
              <div>
                <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
                  σ {gaussianSigma.toFixed(2)}
                </label>
                <input
                  type="range"
                  min={0.1} max={2.0} step={0.05}
                  value={gaussianSigma}
                  onChange={(e) => setGaussianSigma(parseFloat(e.target.value))}
                  className="w-full accent-biotuner-accent"
                />
              </div>
            )}
          </div>
          <button
            onClick={handleComputeImfs}
            disabled={computingImfs || !sessionId}
            className={`w-full min-h-[36px] flex items-center justify-center gap-2 px-3 rounded-md
              text-xs font-medium border transition-colors
              ${imfTimbres.length
                ? 'bg-biotuner-accent/15 border-biotuner-accent/50 text-biotuner-accent'
                : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/70 hover:border-biotuner-accent/50'}
              disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {computingImfs
              ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Decomposing signal…</>
              : (imfTimbres.length
                ? `✓ ${imfTimbres.length} IMFs cached — re-compute`
                : '↻ Compute IMFs from signal')}
          </button>

          {/* IMF range selector — appears once IMFs are cached.
              Lets the user trim noisy top IMFs (often 60Hz hum / scanner
              artefacts) and / or low residual IMFs (DC / drift). Each
              cached IMF shows its dominant frequency so the user can
              pick which ones to keep based on the actual content. */}
          {imfTimbres.length > 0 && (
            <div className="space-y-1.5 mt-1">
              <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-biotuner-light/50">
                <span>Use IMFs</span>
                <span className="font-mono text-biotuner-accent/70">
                  #{imfRange[0]} → #{imfRange[1]} ({Math.max(0, imfRange[1] - imfRange[0] + 1)} of {imfTimbres.length})
                </span>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-[9px] text-biotuner-light/40 mb-0.5">From IMF</label>
                  <input
                    type="number"
                    min={1}
                    max={imfTimbres.length}
                    value={imfRange[0]}
                    onChange={(e) => {
                      const v = Math.max(1, Math.min(imfTimbres.length, parseInt(e.target.value, 10) || 1))
                      setImfRange(([_, hi]) => [v, Math.max(v, hi)])
                    }}
                    className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1 text-xs font-mono"
                  />
                </div>
                <div>
                  <label className="block text-[9px] text-biotuner-light/40 mb-0.5">To IMF</label>
                  <input
                    type="number"
                    min={1}
                    max={imfTimbres.length}
                    value={imfRange[1]}
                    onChange={(e) => {
                      const v = Math.max(1, Math.min(imfTimbres.length, parseInt(e.target.value, 10) || imfTimbres.length))
                      setImfRange(([lo, _]) => [Math.min(lo, v), v])
                    }}
                    className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1 text-xs font-mono"
                  />
                </div>
              </div>
              {/* IMF mini-table — shows each cached IMF, its dominant
                  frequency, and whether it's currently included in the
                  morph (green dot) or excluded (gray dot). Clickable
                  rows snap the range to "use only this IMF and the
                  ones beyond it in the desired direction". */}
              <div className="flex flex-wrap gap-1 pt-1">
                {imfTimbres.map((t, i) => {
                  const inRange = i + 1 >= imfRange[0] && i + 1 <= imfRange[1]
                  const dominantHz = t.partials_hz?.[0] || 0
                  const hzLabel = dominantHz >= 1000
                    ? `${(dominantHz / 1000).toFixed(1)}k`
                    : `${dominantHz.toFixed(dominantHz < 10 ? 2 : 1)}Hz`
                  return (
                    <span
                      key={i}
                      className={`text-[9px] font-mono px-1.5 py-0.5 rounded border
                        ${inRange
                          ? 'bg-biotuner-accent/15 border-biotuner-accent/40 text-biotuner-accent'
                          : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/35'}`}
                      title={`IMF #${i + 1} — dominant freq ${dominantHz.toFixed(2)} Hz`}
                    >
                      #{i + 1}: {hzLabel}
                    </span>
                  )
                })}
              </div>
            </div>
          )}
          {!sessionId && (
            <p className="text-[10px] text-biotuner-light/40 italic">
              Run an analysis first — IMF extraction needs the session's raw signal.
            </p>
          )}
        </div>
      )}

      {/* Band-morph control panel — visible when band_morph is selected.
          Lets the user pick a slicing preset (EEG, HRV, octaves, log)
          or type custom Hz edges, fire the compute, and pick blend mode. */}
      {evolution === 'band_morph' && (
        <div className="bg-biotuner-dark-900/60 border border-biotuner-dark-600 rounded-lg p-2.5 space-y-2">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wider text-biotuner-accent/80">
                Band morph
              </span>
              {sfChip && (
                <span className="text-[9px] font-mono px-1.5 py-0.5 rounded
                                 bg-biotuner-dark-800 border border-biotuner-dark-600
                                 text-biotuner-light/60"
                      title="Session sample rate. Band edges above sf/2 are silently dropped.">
                  sf: {sfChip}
                </span>
              )}
            </div>
            <div className="text-[10px] text-biotuner-light/40">
              {bandTimbres.length
                ? `${bandTimbres.length} bands cached · walk runs low → high freq`
                : 'Pick slicing + click compute to extract band Timbres'}
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            <div>
              <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
                Preset
              </label>
              <select
                value={bandPreset}
                onChange={(e) => handleBandPresetChange(e.target.value)}
                className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1 text-xs"
              >
                {BAND_PRESETS.map((p) => (
                  <option key={p.value} value={p.value}>{p.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
                Blend
              </label>
              <select
                value={blendMode}
                onChange={(e) => setBlendMode(e.target.value)}
                className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1 text-xs"
              >
                {TIMBRE_MORPH_BLENDS.map((b) => (
                  <option key={b.value} value={b.value}>{b.label}</option>
                ))}
              </select>
            </div>
            {blendMode === 'gaussian' && (
              <div>
                <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
                  σ {gaussianSigma.toFixed(2)}
                </label>
                <input
                  type="range"
                  min={0.1} max={2.0} step={0.05}
                  value={gaussianSigma}
                  onChange={(e) => setGaussianSigma(parseFloat(e.target.value))}
                  className="w-full accent-biotuner-accent"
                />
              </div>
            )}
          </div>
          <div>
            <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
              Band edges (Hz, comma-separated)
            </label>
            <input
              type="text"
              value={bandEdgesText}
              onChange={(e) => {
                setBandEdgesText(e.target.value)
                // Editing edges manually flips the preset to 'custom'.
                if (bandPreset !== 'custom') setBandPreset('custom')
              }}
              placeholder="e.g. 1, 4, 8, 13, 30, 100"
              className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1 text-xs font-mono"
            />
            <p className="text-[9px] text-biotuner-light/30 mt-0.5">
              N values define N−1 bands. Sorted ascending automatically.
            </p>
          </div>
          <button
            onClick={handleComputeBands}
            disabled={computingBands || !sessionId}
            className={`w-full min-h-[36px] flex items-center justify-center gap-2 px-3 rounded-md
              text-xs font-medium border transition-colors
              ${bandTimbres.length
                ? 'bg-biotuner-accent/15 border-biotuner-accent/50 text-biotuner-accent'
                : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/70 hover:border-biotuner-accent/50'}
              disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {computingBands
              ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Filtering bands…</>
              : (bandTimbres.length
                ? `✓ ${bandTimbres.length} bands cached — re-compute`
                : '↻ Compute bands from signal')}
          </button>
          {!sessionId && (
            <p className="text-[10px] text-biotuner-light/40 italic">
              Run an analysis first — band extraction needs the session's raw signal.
            </p>
          )}
        </div>
      )}

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

      {/* Stacked frames view — Ableton-style isometric stack. Renders
          every frame as a polyline in one canvas, offset so they recede
          into depth. The current frame is highlighted in cyan with a
          glow; back frames fade through a violet→cyan gradient. Click
          or drag on the canvas to jump to a frame. */}
      {wavetable && wavetable.frames.length > 1 && (
        <div>
          <div className="flex items-baseline justify-between mb-1.5">
            <span className="text-[10px] uppercase tracking-wider text-biotuner-light/40">
              Wavetable stack (click or drag to scrub)
            </span>
            <span className="text-[10px] font-mono text-biotuner-light/40">
              frame 1 → frame {wavetable.frames.length}
            </span>
          </div>
          <StackedFramesView
            frames={wavetable.frames}
            currentIdx={currentIdx}
            onJump={(i) => { setAnimPlaying(false); setCurrentIdx(i) }}
            height={220}
          />
        </div>
      )}
    </div>
  )
}
