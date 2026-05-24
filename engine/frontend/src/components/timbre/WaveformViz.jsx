/**
 * WaveformViz — single-cycle time-domain plot of the additive timbre.
 *
 * Sums sinusoids at each partial frequency with their amplitudes and
 * phases over one period of the fundamental. This is the timbre as
 * a *wave* — the same shape that a wavetable export would store.
 *
 * Pairs naturally with SpectrumViz (which shows the same data in the
 * frequency domain). Updates live as partials / amplitudes / phases /
 * mute / solo / mod-strength change. When ``animated`` is on, modulator
 * activity is rendered into the waveform too (slow temporal modulation
 * shows up as the wave morphing over time).
 */

import { useEffect, useMemo, useRef } from 'react'

/**
 * Render one period of the additive sum at ``nSamples`` points. The
 * result is normalised to ``[-1, 1]`` for display so a sparse timbre
 * (few partials, low total amplitude) still fills the view.
 */
function buildCycle(partials, amps, phases, fundamental, nSamples, t, animated, amMods, modStrength) {
  if (!partials.length || !fundamental) return new Float32Array(nSamples)
  const samples = new Float32Array(nSamples)
  const TWO_PI = Math.PI * 2
  // One full period of the fundamental.
  const period = 1.0 / fundamental
  for (let i = 0; i < nSamples; i++) {
    const t_local = (i / nSamples) * period
    let v = 0
    for (let k = 0; k < partials.length; k++) {
      const f = partials[k]
      let amp = amps[k] || 0
      const ph = phases?.[k] || 0
      // When animated, fold in AM modulator effect on this partial — same
      // math the synth uses so what you see matches what you hear.
      if (animated) {
        for (const m of amMods) {
          if (m.carrier_idx !== k || !m.enabled) continue
          amp += m.depth * (amps[k] || 0) * modStrength * Math.sin(TWO_PI * m.mod_freq * t)
        }
      }
      v += amp * Math.sin(TWO_PI * f * t_local + ph)
    }
    samples[i] = v
  }
  // Normalise to [-1, 1].
  let peak = 0
  for (let i = 0; i < nSamples; i++) {
    const a = Math.abs(samples[i])
    if (a > peak) peak = a
  }
  if (peak > 0) {
    const inv = 1 / peak
    for (let i = 0; i < nSamples; i++) samples[i] *= inv
  }
  return samples
}

export default function WaveformViz({
  timbre,
  animated = false,
  modulationStrength = 1.0,
  mutedIndices = [],
  soloIndex = null,
  height = 140,
}) {
  const canvasRef = useRef(null)
  const containerRef = useRef(null)
  const rafRef = useRef(null)
  const tStartRef = useRef(performance.now())

  const N = timbre?.partials_hz?.length || 0

  // Apply mute / solo BEFORE building the waveform so muted partials
  // don't contribute — same effect they have in the spectrum view.
  const effective = useMemo(() => {
    if (!timbre || !N) return { partials: [], amps: [], phases: [], amMods: [], fundamental: 0 }
    const partials = timbre.partials_hz
    const ampsRaw = timbre.amplitudes || []
    const amps = partials.map((_, i) => {
      if (mutedIndices.includes(i)) return 0
      if (soloIndex !== null && soloIndex !== i) return 0
      return ampsRaw[i] || 0
    })
    return {
      partials,
      amps,
      phases: timbre.phases || [],
      amMods: timbre.am_modulators || [],
      fundamental: Math.min(...partials),  // fundamental = lowest visible partial
    }
  }, [timbre, mutedIndices, soloIndex, N])

  // ----- Canvas sizing -----------------------------------------------
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

      // Zero crossing axis
      ctx.strokeStyle = 'rgba(255,255,255,0.08)'
      ctx.lineWidth = 1 * dpr
      ctx.beginPath()
      ctx.moveTo(0, H / 2)
      ctx.lineTo(W, H / 2)
      ctx.stroke()

      // Labels
      ctx.fillStyle = 'rgba(255,255,255,0.3)'
      ctx.font = `${9 * dpr}px ui-monospace, monospace`
      ctx.textAlign = 'left'
      ctx.fillText('+1', 4 * dpr, 12 * dpr)
      ctx.textAlign = 'left'
      ctx.fillText('−1', 4 * dpr, H - 4 * dpr)
      ctx.textAlign = 'right'
      ctx.fillText('1 cycle', W - 4 * dpr, H - 4 * dpr)

      if (!effective.partials.length || effective.amps.every((a) => a === 0)) {
        ctx.fillStyle = 'rgba(255,255,255,0.4)'
        ctx.textAlign = 'center'
        ctx.font = `${12 * dpr}px sans-serif`
        ctx.fillText('no audible partials', W / 2, H / 2 + 12 * dpr)
        return
      }

      const nSamples = Math.min(1024, Math.max(256, Math.round(W / dpr / 2)))
      const t = animated ? (performance.now() - tStartRef.current) / 1000 : 0
      const samples = buildCycle(
        effective.partials,
        effective.amps,
        effective.phases,
        effective.fundamental,
        nSamples,
        t,
        animated,
        effective.amMods,
        modulationStrength,
      )

      // Waveform path
      ctx.strokeStyle = '#10b981'   // emerald — distinct from the cyan spectrum
      ctx.lineWidth = 2 * dpr
      ctx.beginPath()
      for (let i = 0; i < nSamples; i++) {
        const x = (i / (nSamples - 1)) * W
        const y = H / 2 - samples[i] * (H / 2 - 14 * dpr)
        if (i === 0) ctx.moveTo(x, y)
        else ctx.lineTo(x, y)
      }
      ctx.stroke()

      if (animated) {
        rafRef.current = requestAnimationFrame(draw)
      }
    }
    draw()
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [effective, animated, modulationStrength, height])

  return (
    <div
      ref={containerRef}
      className="relative w-full bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 overflow-hidden"
      style={{ height }}
    >
      <canvas ref={canvasRef} className="block w-full h-full" />
    </div>
  )
}
