/**
 * Timbre tab — turn the analysed tuning into a playable instrument voice,
 * preview it, and export it to production tools (Vital, SFZ, Surge, etc.).
 *
 * Phase A1 (this file): spectrum + design controls + live preview synth +
 * single-click export. Phases A2–A5 add the wavetable studio, modulator
 * routing matrix, animated spectrum hookup, and bundle builder.
 *
 * The preview synth lives in services/timbre/synth.js and routes
 * AM/FM modulators per-partial so what the user hears matches what
 * gets baked into the exported preset file.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Loader2, Play, Square, Download, Sparkles, RefreshCw, Volume2, Music2,
} from 'lucide-react'

import SpectrumViz from '../timbre/SpectrumViz'
import { TimbreSynth } from '../../services/timbre/synth'
import {
  buildTimbreRequest, computeTimbre, exportTimbre, downloadBlob,
} from '../../services/timbre/api'

// ---------------------------------------------------------------------------
// Voice / matching presets
// ---------------------------------------------------------------------------

// Matching method options — these are the real biotuner methods
// dispatched by harmonic_timbre.matching.match_timbre. 'harmonic_input'
// is our HI adapter path (partials = peaks); the rest are biotuner's
// own algorithms that derive partials from the chosen scale's ratios.
const MATCHING_METHODS = [
  { value: 'harmonic_input',     label: 'Direct (partials = peaks)' },
  { value: 'consonance_weighted',label: 'Consonance-weighted' },
  { value: 'sethares',           label: 'Sethares (dissonance min)' },
  { value: 'harmonic_entropy',   label: 'Harmonic entropy' },
  { value: 'hybrid',             label: 'Hybrid (consonance + entropy)' },
  { value: 'direct',             label: 'Direct (matching-method)' },
]

// Scale variants — the canonical SCALE_KEYS vocabulary, plus an empty
// option that uses the default "first available" behaviour.
const SCALE_PRIORITY_OPTIONS = [
  { value: '',                          label: 'Default (best available)' },
  { value: 'peaks_ratios_cons',         label: 'Consonance-filtered peaks' },
  { value: 'peaks_ratios',              label: 'Raw peak ratios' },
  { value: 'extended_peaks_ratios_cons',label: 'Extended cons. ratios' },
  { value: 'extended_peaks_ratios',     label: 'Extended raw ratios' },
  { value: 'diss_scale',                label: 'Dissonance-curve minima' },
  { value: 'HE',                        label: 'Harmonic-entropy minima' },
  { value: 'euler_fokker',              label: 'Euler-Fokker' },
  { value: 'harm_tuning',               label: 'Harmonic tuning' },
  { value: 'harm_fit',                  label: 'Harmonic-fit tuning' },
]

// Single-click export targets. Each one is one call to /api/timbre/export.
const EXPORT_TARGETS = [
  { fmt: 'vital',         label: '.vital',  hint: 'Vital DAW preset' },
  { fmt: 'sfz',           label: '.sfz',    hint: 'SFZ sampler bundle' },
  { fmt: 'surge',         label: 'Surge',   hint: 'Surge XT bundle' },
  { fmt: 'wavetable',     label: '.wav (wavetable)', hint: 'Multi-frame wavetable' },
  { fmt: 'wav',           label: '.wav (samples)',   hint: 'Multi-pitch sample pack' },
  { fmt: 'csound',        label: '.csd',    hint: 'Csound document' },
  { fmt: 'supercollider', label: '.scd',    hint: 'SuperCollider script' },
  { fmt: 'tuning',        label: '.scl/.kbm', hint: 'Universal tuning files' },
]

// Default ADSR — matches the Keyboard tab's triangle preset for consistency.
const DEFAULT_ADSR = { attack: 0.03, decay: 0.15, sustain: 0.55, release: 0.4 }

export default function TimbreTab({ analysisResult }) {
  // --------- Local state -------------------------------------------------
  const [matchingMethod, setMatchingMethod] = useState('harmonic_input')
  const [scalePriority,  setScalePriority]  = useState('')
  const [adsr, setAdsr] = useState({ ...DEFAULT_ADSR })
  const [volumeDb,  setVolumeDb]  = useState(-8)
  const [modStrength, setModStrength] = useState(1.0)
  const [badgeMode, setBadgeMode] = useState('ratio')  // 'ratio' | 'cents' | 'hz' | 'off'
  const [animated,  setAnimated]  = useState(false)
  const [mutedIdx,  setMutedIdx]  = useState([])
  const [soloIdx,   setSoloIdx]   = useState(null)
  const [enabledMods, setEnabledMods] = useState({})  // mod_id -> bool
  // Enrichment: opt-in partial-spectrum expansion. Intermod ADDS
  // sidebands at f1±f2; harmonic stack ADDS 2f, 3f, ..., nf per partial.
  // Both produce visible new bars in the spectrum, distinct from
  // modulators which wobble existing bars.
  const [intermodOn,     setIntermodOn]     = useState(false)
  const [intermodDepth,  setIntermodDepth]  = useState(0.5)
  const [stackOn,        setStackOn]        = useState(false)
  const [stackN,         setStackN]         = useState(4)
  const [stackRolloff,   setStackRolloff]   = useState(0.9)
  const [previewFreq, setPreviewFreq] = useState(null)  // Hz; null = derive

  const [timbre, setTimbre] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [exporting, setExporting] = useState(null)  // fmt currently exporting

  const synthRef = useRef(null)
  const requestIdRef = useRef(0)

  // --------- Build the request payload (memoised) ------------------------
  const enrichmentObj = useMemo(() => ({
    intermod: { enabled: intermodOn, depth: intermodDepth },
    harmonic_stack: { enabled: stackOn, n: stackN, rolloff: stackRolloff },
  }), [intermodOn, intermodDepth, stackOn, stackN, stackRolloff])

  const designKey = `${matchingMethod}|${scalePriority}|${JSON.stringify(enabledMods)}|${JSON.stringify(enrichmentObj)}`
  const requestPayload = useMemo(() => {
    if (!analysisResult) return null
    return buildTimbreRequest(analysisResult, {
      matching_method: matchingMethod,
      scale_priority: scalePriority ? [scalePriority] : null,
      enabled_modulators: enabledMods,
      enrichment: enrichmentObj,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisResult, designKey])

  // --------- Default preview freq from the analysis ----------------------
  const defaultPreviewFreq = useMemo(() => {
    const p = analysisResult?.peaks?.[0]
    if (p && p >= 55 && p <= 880) return Math.round(p)
    if (p && p > 880) {
      // Octave-shift down into a comfortable register
      let f = p
      while (f > 880) f /= 2
      return Math.round(f)
    }
    return 220
  }, [analysisResult])

  useEffect(() => {
    if (previewFreq === null) setPreviewFreq(defaultPreviewFreq)
  }, [defaultPreviewFreq, previewFreq])

  // --------- Fetch the timbre snapshot whenever the design changes -------
  useEffect(() => {
    if (!requestPayload) {
      setTimbre(null)
      return
    }
    const id = ++requestIdRef.current
    setLoading(true)
    setError(null)
    computeTimbre(requestPayload)
      .then((data) => {
        if (id !== requestIdRef.current) return
        setTimbre(data)
        // Initialise enabledMods from response so toggles persist sensibly
        // across re-fetches (we don't reset user choices).
        setEnabledMods((prev) => {
          const next = { ...prev }
          for (const m of [...(data.am_modulators || []), ...(data.fm_modulators || [])]) {
            if (next[m.id] === undefined) next[m.id] = true
          }
          return next
        })
        setLoading(false)
      })
      .catch((e) => {
        if (id !== requestIdRef.current) return
        setError(e.response?.data?.detail || e.message || 'Failed to compute timbre')
        setLoading(false)
      })
  }, [requestPayload])

  // --------- Synth lifecycle --------------------------------------------
  useEffect(() => {
    return () => {
      if (synthRef.current) {
        synthRef.current.dispose()
        synthRef.current = null
      }
    }
  }, [])

  // Push live state into the synth without re-attacking.
  useEffect(() => {
    if (synthRef.current) {
      synthRef.current.setAdsr(adsr)
      synthRef.current.setVolume(volumeDb)
      synthRef.current.setModulationStrength(modStrength)
    }
  }, [adsr, volumeDb, modStrength])

  useEffect(() => {
    if (synthRef.current && timbre) {
      synthRef.current.loadTimbre(timbre)
    }
  }, [timbre])

  const handlePlay = async () => {
    if (!timbre) return
    if (!synthRef.current) {
      synthRef.current = new TimbreSynth()
      await synthRef.current.start()
      synthRef.current.setAdsr(adsr)
      synthRef.current.setVolume(volumeDb)
      synthRef.current.setModulationStrength(modStrength)
      synthRef.current.loadTimbre(timbre)
    }
    if (isPlaying) {
      synthRef.current.noteOff()
      setIsPlaying(false)
    } else {
      synthRef.current.noteOn(previewFreq || defaultPreviewFreq)
      setIsPlaying(true)
    }
  }

  // --------- Partial mute / solo ----------------------------------------
  const handlePartialClick = (idx, e) => {
    if (e.shiftKey) {
      setSoloIdx((cur) => (cur === idx ? null : idx))
      return
    }
    setMutedIdx((cur) => (
      cur.includes(idx) ? cur.filter((x) => x !== idx) : [...cur, idx]
    ))
  }

  // The synth doesn't yet honour mute/solo from the parent (Phase A2
  // task — needs per-partial gain handles exposed). For now muting hides
  // bars in the viz; export honours the user's selection by passing
  // enabled_modulators but not partial muting yet. We DO drop muted
  // partials from the export request via a future "muted_partials" field
  // — TODO when wiring the routing matrix.

  // --------- Reset helpers ----------------------------------------------
  const resetDesign = () => {
    setMatchingMethod('harmonic_input')
    setScalePriority('')
    setAdsr({ ...DEFAULT_ADSR })
    setVolumeDb(-8)
    setModStrength(1.0)
    setMutedIdx([])
    setSoloIdx(null)
    setEnabledMods({})
    setIntermodOn(false)
    setIntermodDepth(0.5)
    setStackOn(false)
    setStackN(4)
    setStackRolloff(0.9)
  }

  // --------- Export -----------------------------------------------------
  const handleExport = async (fmt) => {
    if (!requestPayload) return
    setExporting(fmt)
    try {
      const { blob, filename } = await exportTimbre(fmt, requestPayload)
      downloadBlob(blob, filename)
    } catch (e) {
      setError(e.response?.data?.detail || e.message || `Export ${fmt} failed`)
    } finally {
      setExporting(null)
    }
  }

  // ---------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------

  if (!analysisResult) {
    return (
      <div className="text-center py-12 text-biotuner-light/60">
        <Music2 className="w-12 h-12 mx-auto mb-3 opacity-30" />
        Run an analysis first to design a timbre from it.
      </div>
    )
  }

  const nPartials = timbre?.partials_hz?.length || 0
  const nMods = (timbre?.am_modulators?.length || 0) + (timbre?.fm_modulators?.length || 0)

  return (
    <div className="space-y-5">
      {/* Header + provenance chips */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-lg font-bold text-biotuner-primary flex items-center gap-2">
            <Music2 className="w-5 h-5" /> Timbre
          </h2>
          <p className="text-sm text-biotuner-light/70 mt-1">
            Design a playable voice from the analysed harmonic spectrum, preview
            it live, export to your DAW.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 items-start">
          {nPartials > 0 && (
            <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded
                             bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/70">
              {nPartials} partials
            </span>
          )}
          {nMods > 0 && (
            <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded
                             bg-biotuner-accent/10 border border-biotuner-accent/40 text-biotuner-accent">
              {nMods} modulators
            </span>
          )}
          {timbre?.scale_source && (
            <span className="text-[10px] uppercase tracking-wider px-2 py-1 rounded
                             bg-biotuner-primary/10 border border-biotuner-primary/30 text-biotuner-primary/80">
              scale: {timbre.scale_source}
            </span>
          )}
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/40 text-red-300 text-sm rounded-lg p-3">
          {error}
        </div>
      )}

      {/* Main layout: spectrum left, controls right */}
      <div className="flex flex-col lg:flex-row gap-4">
        <div className="flex-1 min-w-0 space-y-3">
          {/* Spectrum */}
          <div className="relative">
            {loading && (
              <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5 px-2 py-1 rounded-md
                              bg-biotuner-dark-900/80 border border-biotuner-primary/30 text-xs text-biotuner-primary">
                <Loader2 className="w-3 h-3 animate-spin" />
                computing…
              </div>
            )}
            <SpectrumViz
              timbre={timbre}
              badgeMode={badgeMode}
              animated={animated && nMods > 0}
              mutedIndices={mutedIdx}
              soloIndex={soloIdx}
              modulationStrength={modStrength}
              onPartialClick={handlePartialClick}
              height={300}
            />
          </div>

          {/* Spectrum toolbar */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 text-xs text-biotuner-light/60">
              <span className="mr-1 uppercase tracking-wider text-[10px]">Badges</span>
              {['ratio', 'cents', 'hz', 'off'].map((m) => (
                <button
                  key={m}
                  onClick={() => setBadgeMode(m)}
                  className={`min-h-[28px] px-2 rounded border text-[10px] uppercase tracking-wider
                    ${badgeMode === m
                      ? 'bg-biotuner-primary/20 border-biotuner-primary text-biotuner-primary'
                      : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/60'}`}
                >
                  {m}
                </button>
              ))}
            </div>
            <button
              onClick={() => setAnimated((a) => !a)}
              disabled={nMods === 0}
              aria-pressed={animated}
              title={nMods === 0
                ? 'No modulators to animate — enable PAC / CFC in the sidebar'
                : 'Animate the spectrum to show live modulator activity'}
              className={`min-h-[28px] px-2 rounded border text-[10px] uppercase tracking-wider
                ${animated && nMods > 0
                  ? 'bg-biotuner-accent/20 border-biotuner-accent text-biotuner-accent'
                  : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/60 hover:border-biotuner-accent/50'}
                disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              {animated && nMods > 0 ? '⟳ Animated' : '⟳ Animate'}
            </button>
            <div className="flex-1" />
            {(mutedIdx.length > 0 || soloIdx !== null) && (
              <button
                onClick={() => { setMutedIdx([]); setSoloIdx(null) }}
                className="text-[10px] uppercase tracking-wider min-h-[28px] px-2 rounded
                           bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/60
                           hover:border-biotuner-primary/50"
              >
                Clear mute / solo
              </button>
            )}
            <button
              onClick={resetDesign}
              title="Reset all timbre design choices"
              className="min-h-[28px] flex items-center gap-1 px-2 rounded border
                         bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/70
                         hover:border-biotuner-primary/50 text-[10px] uppercase tracking-wider"
            >
              <RefreshCw className="w-3 h-3" /> Reset
            </button>
          </div>

          {/* Preview transport + base freq */}
          <div className="bg-biotuner-dark-900/60 border border-biotuner-dark-600 rounded-lg p-3
                          flex flex-wrap items-center gap-3">
            <button
              onClick={handlePlay}
              disabled={!timbre}
              className={`min-h-[42px] px-4 flex items-center gap-2 rounded-lg font-medium text-sm transition
                ${isPlaying
                  ? 'bg-biotuner-primary text-biotuner-dark-900 border-2 border-biotuner-primary'
                  : 'bg-biotuner-primary/15 border border-biotuner-primary/40 text-biotuner-primary hover:bg-biotuner-primary/25'}
                disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              {isPlaying ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? 'Stop' : 'Preview'}
            </button>
            <div className="flex items-center gap-2 flex-1 min-w-[200px]">
              <label className="text-[10px] uppercase tracking-wider text-biotuner-light/50">
                Preview Hz
              </label>
              <input
                type="range"
                min={55}
                max={880}
                step={1}
                value={previewFreq || defaultPreviewFreq}
                onChange={(e) => setPreviewFreq(parseInt(e.target.value, 10))}
                className="flex-1 accent-biotuner-primary"
              />
              <input
                type="number"
                min={20}
                max={2000}
                step={1}
                value={previewFreq || defaultPreviewFreq}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10)
                  if (Number.isFinite(v)) setPreviewFreq(Math.max(20, Math.min(2000, v)))
                }}
                className="w-16 bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-1
                           text-xs text-right font-mono"
              />
            </div>
          </div>
        </div>

        {/* Right column: design controls */}
        <div className="lg:w-80 lg:flex-shrink-0 space-y-4">
          {/* Voice */}
          <div className="bg-biotuner-dark-900/70 border border-biotuner-dark-600 rounded-xl p-4 space-y-3">
            <h3 className="text-xs font-bold text-biotuner-accent/80 uppercase tracking-widest
                           flex items-center gap-2">
              <Sparkles className="w-3.5 h-3.5" /> Voice
            </h3>
            <div>
              <label className="block text-[10px] uppercase tracking-wider text-biotuner-light/50 mb-1">
                Matching method
              </label>
              <select
                value={matchingMethod}
                onChange={(e) => setMatchingMethod(e.target.value)}
                className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-2 text-sm"
              >
                {MATCHING_METHODS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-[10px] uppercase tracking-wider text-biotuner-light/50 mb-1">
                Scale source
              </label>
              <select
                value={scalePriority}
                onChange={(e) => setScalePriority(e.target.value)}
                className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-2 text-sm"
              >
                {SCALE_PRIORITY_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
          </div>

          {/* ADSR */}
          <div className="bg-biotuner-dark-900/70 border border-biotuner-dark-600 rounded-xl p-4 space-y-3">
            <h3 className="text-xs font-bold text-biotuner-accent/80 uppercase tracking-widest">
              Envelope
            </h3>
            {([
              ['attack',  0.001, 2,  0.001, 's'],
              ['decay',   0.01,  2,  0.01,  's'],
              ['sustain', 0,     1,  0.01,  '%'],
              ['release', 0.05,  5,  0.05,  's'],
            ]).map(([key, min, max, step, unit]) => (
              <div key={key}>
                <div className="flex items-baseline justify-between text-[10px] uppercase tracking-wider mb-0.5">
                  <span className="text-biotuner-light/50">{key}</span>
                  <span className="text-biotuner-primary/80 font-mono">
                    {unit === '%' ? `${(adsr[key] * 100).toFixed(0)}%` : `${adsr[key].toFixed(2)}s`}
                  </span>
                </div>
                <input
                  type="range"
                  min={min} max={max} step={step}
                  value={adsr[key]}
                  onChange={(e) => setAdsr({ ...adsr, [key]: parseFloat(e.target.value) })}
                  className="w-full accent-biotuner-primary"
                />
              </div>
            ))}
          </div>

          {/* Volume + modulation strength */}
          <div className="bg-biotuner-dark-900/70 border border-biotuner-dark-600 rounded-xl p-4 space-y-3">
            <h3 className="text-xs font-bold text-biotuner-accent/80 uppercase tracking-widest
                           flex items-center gap-2">
              <Volume2 className="w-3.5 h-3.5" /> Output
            </h3>
            <div>
              <div className="flex items-baseline justify-between text-[10px] uppercase tracking-wider mb-0.5">
                <span className="text-biotuner-light/50">Volume</span>
                <span className="text-biotuner-accent/80 font-mono">{volumeDb} dB</span>
              </div>
              <input
                type="range"
                min={-40} max={0} step={1}
                value={volumeDb}
                onChange={(e) => setVolumeDb(parseInt(e.target.value, 10))}
                className="w-full accent-biotuner-accent"
              />
            </div>
            <div>
              <div className="flex items-baseline justify-between text-[10px] uppercase tracking-wider mb-0.5">
                <span className="text-biotuner-light/50">Modulation strength</span>
                <span className="text-biotuner-accent/80 font-mono">
                  {modStrength.toFixed(2)}×
                </span>
              </div>
              <input
                type="range"
                min={0} max={2} step={0.01}
                value={modStrength}
                onChange={(e) => setModStrength(parseFloat(e.target.value))}
                className="w-full accent-biotuner-accent"
                disabled={nMods === 0}
              />
              {nMods === 0 && (
                <p className="text-[10px] text-biotuner-light/30 mt-1">
                  No modulators in this analysis. Enable PAC / CFC in the sidebar
                  to populate this slider.
                </p>
              )}
            </div>
          </div>

          {/* Enrichment: ADD partials (unlike modulators which wobble
              existing ones). Intermod adds f1±f2 sidebands for each
              intermodulation pair; harmonic stack adds 2f, 3f, … per
              partial. Together they turn a sparse biotuner output (a
              handful of peaks) into a rich, mixable timbre. */}
          <div className="bg-biotuner-dark-900/70 border border-biotuner-dark-600 rounded-xl p-4 space-y-3">
            <h3 className="text-xs font-bold text-biotuner-accent/80 uppercase tracking-widest">
              Enrich partials
            </h3>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="flex items-center gap-2 text-xs text-biotuner-light/80 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={intermodOn}
                    onChange={(e) => setIntermodOn(e.target.checked)}
                    disabled={!(analysisResult?.endogenous_intermodulations?.length)}
                    className="w-3.5 h-3.5 accent-biotuner-primary"
                  />
                  <span>Intermod sidebands</span>
                </label>
                <span className="text-[9px] text-biotuner-light/40 font-mono">
                  +f₁±f₂
                </span>
              </div>
              {!(analysisResult?.endogenous_intermodulations?.length) ? (
                <p className="text-[10px] text-biotuner-light/30 pl-5">
                  No intermod data in this analysis (enable in sidebar).
                </p>
              ) : intermodOn && (
                <div className="pl-5">
                  <div className="flex items-baseline justify-between text-[10px] uppercase tracking-wider mb-0.5">
                    <span className="text-biotuner-light/50">Depth</span>
                    <span className="text-biotuner-primary/80 font-mono">
                      {(intermodDepth * 100).toFixed(0)}%
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0} max={1} step={0.01}
                    value={intermodDepth}
                    onChange={(e) => setIntermodDepth(parseFloat(e.target.value))}
                    className="w-full accent-biotuner-primary"
                  />
                </div>
              )}
            </div>

            <div>
              <div className="flex items-center justify-between mb-1">
                <label className="flex items-center gap-2 text-xs text-biotuner-light/80 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={stackOn}
                    onChange={(e) => setStackOn(e.target.checked)}
                    className="w-3.5 h-3.5 accent-biotuner-primary"
                  />
                  <span>Harmonic stack</span>
                </label>
                <span className="text-[9px] text-biotuner-light/40 font-mono">
                  +2f, 3f, …
                </span>
              </div>
              {stackOn && (
                <div className="pl-5 space-y-2 mt-1">
                  <div>
                    <div className="flex items-baseline justify-between text-[10px] uppercase tracking-wider mb-0.5">
                      <span className="text-biotuner-light/50">Overtones</span>
                      <span className="text-biotuner-primary/80 font-mono">{stackN}</span>
                    </div>
                    <input
                      type="range"
                      min={1} max={8} step={1}
                      value={stackN}
                      onChange={(e) => setStackN(parseInt(e.target.value, 10))}
                      className="w-full accent-biotuner-primary"
                    />
                  </div>
                  <div>
                    <div className="flex items-baseline justify-between text-[10px] uppercase tracking-wider mb-0.5">
                      <span className="text-biotuner-light/50">Rolloff</span>
                      <span className="text-biotuner-primary/80 font-mono">
                        {stackRolloff.toFixed(2)}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={0.3} max={2.0} step={0.05}
                      value={stackRolloff}
                      onChange={(e) => setStackRolloff(parseFloat(e.target.value))}
                      className="w-full accent-biotuner-primary"
                    />
                    <p className="text-[9px] text-biotuner-light/30 mt-0.5">
                      0.7–1.2 sweet spot (warmer → duller)
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Export — bottom row, full width */}
      <div className="bg-biotuner-dark-900/70 border border-biotuner-dark-600 rounded-xl p-4">
        <h3 className="text-xs font-bold text-biotuner-accent/80 uppercase tracking-widest mb-3
                       flex items-center gap-2">
          <Download className="w-3.5 h-3.5" /> Export
        </h3>
        <div className="flex flex-wrap gap-2">
          {EXPORT_TARGETS.map(({ fmt, label, hint }) => (
            <button
              key={fmt}
              onClick={() => handleExport(fmt)}
              disabled={!timbre || exporting !== null}
              title={hint}
              className={`min-h-[40px] px-3 py-2 rounded-lg border text-sm font-medium transition
                ${exporting === fmt
                  ? 'bg-biotuner-primary text-biotuner-dark-900 border-biotuner-primary'
                  : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/80 hover:border-biotuner-primary/50'}
                disabled:opacity-40 disabled:cursor-not-allowed`}
            >
              {exporting === fmt ? (
                <span className="flex items-center gap-1.5">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  {label}
                </span>
              ) : (
                label
              )}
            </button>
          ))}
        </div>
        <p className="text-[10px] text-biotuner-light/40 mt-2 leading-relaxed">
          Each export bakes the current voice + envelope + modulator state into the
          target format. Bundle builder (all formats in one .zip) ships next.
        </p>
      </div>
    </div>
  )
}
