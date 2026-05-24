/**
 * SpectrumViz — interactive harmonic stem plot for the Timbre tab.
 *
 * Each partial is rendered as a vertical bar at its Hz position with
 * height proportional to amplitude. Above every bar floats a chip
 * showing either the ratio (5/4) or the cents value (+386¢), toggled
 * by the parent. Hovering shows full details. Click toggles mute,
 * Shift+click solos.
 *
 * When ``animated`` is true, the bars wobble in real time according
 * to the timbre's AM modulators on each partial — visually surfacing
 * the modulation that the synth applies. The waveform is driven by
 * a requestAnimationFrame loop and reads the live modulator state.
 */

import { useEffect, useMemo, useRef } from 'react'

/**
 * Find the closest small-integer fraction p/q within ``maxDen``.
 * Returns ``{n, d, err}`` so the caller can decide whether the fit is
 * meaningful or if we should fall back to a different label format.
 */
function bestFraction(ratio, maxDen) {
  if (!Number.isFinite(ratio) || ratio <= 0) return { n: 1, d: 1, err: Infinity }
  let bestN = 1, bestD = 1, bestErr = Infinity
  for (let d = 1; d <= maxDen; d++) {
    const n = Math.round(ratio * d)
    if (n <= 0) continue
    const err = Math.abs(ratio - n / d) / Math.max(1, ratio)
    if (err < bestErr) { bestErr = err; bestN = n; bestD = d }
  }
  return { n: bestN, d: bestD, err: bestErr }
}

/**
 * Octave-reduced ratio label. Pulls the ratio into [1, 2) by dividing
 * out powers of 2, finds a small-integer fraction within that range,
 * and tacks on "·8va" annotations when the original ratio was outside.
 * A 2 % relative-error cap keeps us from showing absurd fractions like
 * 202/17 for non-rational data; over the cap we fall back to "≈ N¢".
 */
function ratioFraction(ratio, maxDen = 12) {
  if (!Number.isFinite(ratio) || ratio <= 0) return '—'
  let r = ratio
  let octaveShift = 0
  while (r >= 2) { r /= 2; octaveShift += 1 }
  while (r < 1)  { r *= 2; octaveShift -= 1 }
  const { n, d, err } = bestFraction(r, maxDen)
  // Reject sloppy fits: prefer a cents readout when the closest
  // small-integer fraction is more than 2 % off the actual value.
  if (err > 0.02) {
    const cents = Math.round(Math.log2(ratio) * 1200)
    return `${cents >= 0 ? '+' : ''}${cents}¢`
  }
  const base = `${n}/${d}`
  if (octaveShift === 0) return base
  if (octaveShift > 0)   return `${base}·${octaveShift}va`
  return `${base}÷${-octaveShift}`
}

function centsBetween(freq, baseFreq) {
  if (!baseFreq || baseFreq <= 0) return 0
  return Math.log2(freq / baseFreq) * 1200
}

/**
 * Compute the live amplitude of a partial at time ``t`` accounting for
 * any enabled AM modulators. Mirrors the synth's math exactly so the
 * viz and the audio stay in lockstep.
 */
function liveAmpAt(baseAmp, modulators, partialIdx, t, modStrength) {
  let amp = baseAmp
  for (const m of modulators) {
    if (!m.enabled || m.carrier_idx !== partialIdx || m.type !== 'AM') continue
    const lfoVal = Math.sin(2 * Math.PI * m.mod_freq * t)
    amp += m.depth * baseAmp * modStrength * lfoVal
  }
  return Math.max(0, amp)
}

export default function SpectrumViz({
  timbre,
  badgeMode = 'ratio',     // 'ratio' | 'cents' | 'hz' | 'off'
  animated = false,
  modulationStrength = 1.0,
  mutedIndices = [],
  soloIndex = null,         // null = no solo
  onPartialClick,           // (idx, event) => void
  height = 280,
}) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const rafRef = useRef(null)
  const tStartRef = useRef(performance.now())

  const partials = timbre?.partials_hz || []
  const amps     = timbre?.amplitudes  || []
  const amMods   = timbre?.am_modulators || []
  const N = partials.length

  // Frequency axis bounds: log scale so harmonic ratios are visually
  // proportional. Tight fit around the actual partials — no fixed
  // 20Hz–20kHz clamping. Biosignal-scale spectra (peaks ≪ 20 Hz)
  // used to collapse to the left edge with the old clamping; now we
  // honour whatever range the data actually occupies, with ~half an
  // octave of breathing room on each side.
  const { fMin, fMax, fMinLog, fMaxLog } = useMemo(() => {
    if (!N) return { fMin: 20, fMax: 20000, fMinLog: Math.log2(20), fMaxLog: Math.log2(20000) }
    const fHi = Math.max(...partials)
    const fLo = Math.min(...partials)
    // Pad outward by sqrt(2) (= half an octave). Ensures even a single
    // partial gets centred in the view rather than slammed to one edge.
    const minHz = Math.max(0.01, fLo / Math.SQRT2)
    const maxHz = fHi * Math.SQRT2
    // Guarantee at least 1.5 octaves of visible range when all partials
    // are very close together, so the gridlines + labels stay readable.
    const fMinLog = Math.log2(minHz)
    const fMaxLog = Math.log2(maxHz)
    if (fMaxLog - fMinLog < 1.5) {
      const center = (fMinLog + fMaxLog) / 2
      return {
        fMin: Math.pow(2, center - 0.75),
        fMax: Math.pow(2, center + 0.75),
        fMinLog: center - 0.75,
        fMaxLog: center + 0.75,
      }
    }
    return { fMin: minHz, fMax: maxHz, fMinLog, fMaxLog }
  }, [partials, N])

  const xForFreq = (f, w) => {
    const x = (Math.log2(f) - fMinLog) / (fMaxLog - fMinLog)
    return Math.round(x * (w - 60)) + 30  // 30px padding each side
  }

  // ----- Canvas sizing (DPR-aware) ------------------------------------
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current) return
    const resize = () => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return
      const cssW = container.clientWidth
      const cssH = height
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      canvas.style.width  = `${cssW}px`
      canvas.style.height = `${cssH}px`
      canvas.width  = Math.round(cssW * dpr)
      canvas.height = Math.round(cssH * dpr)
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [height])

  // ----- Render loop --------------------------------------------------
  useEffect(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const dpr = canvas.width / parseFloat(canvas.style.width)

    const draw = () => {
      const W = canvas.width
      const H = canvas.height
      ctx.clearRect(0, 0, W, H)

      // Background grid + axis labels — adapt the tick density to the
      // visible span. Very narrow spans (< 3 octaves) get half-octave
      // ticks so biosignal-scale data has multiple labels visible.
      ctx.strokeStyle = 'rgba(255,255,255,0.06)'
      ctx.fillStyle   = 'rgba(255,255,255,0.35)'
      ctx.font = `${10 * dpr}px ui-monospace, monospace`
      ctx.textAlign = 'center'
      const span = fMaxLog - fMinLog
      const step = span < 3 ? 0.5 : 1.0
      const startLog = Math.ceil(fMinLog / step) * step
      const endLog   = Math.floor(fMaxLog / step) * step
      for (let l = startLog; l <= endLog + 1e-9; l += step) {
        const f = Math.pow(2, l)
        if (f < fMin || f > fMax) continue
        const x = xForFreq(f, W)
        ctx.beginPath()
        ctx.moveTo(x, 20 * dpr)
        ctx.lineTo(x, H - 30 * dpr)
        ctx.stroke()
        // Label formatting: kHz / Hz / sub-Hz depending on magnitude.
        let label
        if (f >= 1000)      label = `${(f / 1000).toFixed(f >= 10000 ? 0 : 1)}k`
        else if (f >= 100)  label = `${f.toFixed(0)}`
        else if (f >= 10)   label = `${f.toFixed(1)}`
        else                label = `${f.toFixed(2)}`
        ctx.fillText(label, x, H - 12 * dpr)
      }

      if (!N) {
        ctx.fillStyle = 'rgba(255,255,255,0.4)'
        ctx.textAlign = 'center'
        ctx.font = `${14 * dpr}px sans-serif`
        ctx.fillText('No timbre data', W / 2, H / 2)
        return
      }

      const t = animated ? (performance.now() - tStartRef.current) / 1000 : 0
      const maxAmp = Math.max(...amps, 0.0001)
      const padTop = 30 * dpr
      const padBot = 35 * dpr
      const usableH = H - padTop - padBot

      // Stems
      for (let i = 0; i < N; i++) {
        const f = partials[i]
        const baseAmp = amps[i] || 0
        const isMuted = mutedIndices.includes(i)
        const isSoloed = soloIndex !== null && soloIndex !== i
        const isHidden = isMuted || isSoloed
        const liveAmp = animated
          ? liveAmpAt(baseAmp, amMods, i, t, modulationStrength)
          : baseAmp
        const x = xForFreq(f, W)
        const barH = isHidden ? 6 * dpr : (liveAmp / maxAmp) * usableH
        const yTop = H - padBot - barH

        ctx.strokeStyle = isHidden ? 'rgba(255,255,255,0.15)' : '#06b6d4'
        ctx.fillStyle   = isHidden ? 'rgba(255,255,255,0.1)'  : '#06b6d4'
        ctx.lineWidth = 3 * dpr
        ctx.beginPath()
        ctx.moveTo(x, H - padBot)
        ctx.lineTo(x, yTop)
        ctx.stroke()
        // Dot at the tip
        ctx.beginPath()
        ctx.arc(x, yTop, 4 * dpr, 0, Math.PI * 2)
        ctx.fill()
      }

      if (animated) {
        rafRef.current = requestAnimationFrame(draw)
      }
    }
    draw()
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [
    partials, amps, amMods, N, fMin, fMax, fMinLog, fMaxLog,
    animated, modulationStrength, mutedIndices, soloIndex, height,
  ])

  // ----- Badges + click overlay (DOM, not canvas) ---------------------
  // Rendered as absolutely-positioned chips so they're crisp and selectable
  // and the click hit areas are easy. CSS pixel math (not DPR-scaled).
  const badges = useMemo(() => {
    if (!N || !containerRef.current) return []
    const W = containerRef.current.clientWidth
    const baseFreq = timbre?.base_freq || partials[0]
    return partials.map((f, i) => {
      const x = xForFreq(f, W)
      let label
      if (badgeMode === 'ratio')  label = ratioFraction(f / baseFreq)
      else if (badgeMode === 'cents') {
        const c = centsBetween(f, baseFreq)
        label = `${c >= 0 ? '+' : ''}${c.toFixed(0)}¢`
      }
      else if (badgeMode === 'hz') label = `${f.toFixed(0)}`
      else label = ''
      return { i, x, label, freq: f }
    })
  }, [partials, badgeMode, N, timbre?.base_freq])

  return (
    <div
      ref={containerRef}
      className="relative w-full bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 overflow-hidden"
      style={{ height }}
    >
      <canvas ref={canvasRef} className="block w-full h-full" />
      {/* Badge layer */}
      {badgeMode !== 'off' && badges.map((b) => {
        const isMuted = mutedIndices.includes(b.i)
        const isSoloed = soloIndex !== null && soloIndex !== b.i
        return (
          <div
            key={b.i}
            onClick={(e) => onPartialClick?.(b.i, e)}
            title={`Partial ${b.i + 1}: ${b.freq.toFixed(2)} Hz · click to ${
              isMuted ? 'unmute' : 'mute'
            } · shift+click to solo`}
            className={`absolute select-none cursor-pointer text-[10px] font-mono px-1.5 py-0.5
              rounded border transition-colors
              ${isMuted || isSoloed
                ? 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/40'
                : 'bg-biotuner-primary/20 border-biotuner-primary/50 text-biotuner-primary hover:bg-biotuner-primary/30'}`}
            style={{
              left: `${b.x}px`,
              top: '6px',
              transform: 'translateX(-50%)',
            }}
          >
            {b.label}
          </div>
        )
      })}
      {/* Mute / solo legend (small bottom-right) */}
      {N > 0 && (
        <div className="absolute bottom-2 right-3 text-[9px] uppercase tracking-wider text-biotuner-light/30 pointer-events-none">
          click chip to mute · shift+click to solo
        </div>
      )}
    </div>
  )
}
