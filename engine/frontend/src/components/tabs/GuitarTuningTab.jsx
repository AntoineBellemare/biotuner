import { useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'
import { Play, Download, Save, Music, Volume2, Mic, MicOff, Lock, Sparkles } from 'lucide-react'
import { exportTuning as exportTuningLocal } from '../../services/tuningExport'
import { startPitchDetection } from '../../services/audio/pitchDetector'
import apiClient from '../../services/api'

// ---------------------------------------------------------------------------
// Instruments
// ---------------------------------------------------------------------------

const INSTRUMENTS = {
  guitar:       { label: 'Guitar (EADGBE)',      strings: [82.41, 110.00, 146.83, 196.00, 246.94, 329.63], names: ['E2','A2','D3','G3','B3','E4'] },
  guitar_dropd: { label: 'Guitar Drop D',        strings: [73.42, 110.00, 146.83, 196.00, 246.94, 329.63], names: ['D2','A2','D3','G3','B3','E4'] },
  bass:         { label: 'Bass (EADG)',          strings: [41.20, 55.00, 73.42, 98.00],                   names: ['E1','A1','D2','G2'] },
  ukulele:      { label: 'Ukulele (GCEA)',       strings: [392.00, 261.63, 329.63, 440.00],               names: ['G4','C4','E4','A4'] },
}

const STANDARD_FUNDAMENTAL_CHIPS = [82.41, 110, 220, 440]

// Pick the strongest peak from the analysis, then octave-shift it into a
// guitar-friendly range (~55 Hz → ~440 Hz) so it can drive the string
// mapping directly.
function deriveAudioFundamental(analysisResult) {
  const peaks = analysisResult?.peaks || []
  if (!peaks.length) return null
  const powers = analysisResult?.powers || []
  let idx = 0
  if (powers.length === peaks.length) {
    for (let i = 1; i < powers.length; i++) {
      if (powers[i] > powers[idx]) idx = i
    }
  }
  let f = peaks[idx]
  if (!Number.isFinite(f) || f <= 0) return null
  while (f >= 440) f /= 2
  while (f <  55)  f *= 2
  return f
}

// Harmonics depth: 1 = ratios only, N = also include r×2..r×N and r÷2..r÷N
// (with octave wrapping). Higher N → more candidate pitches → tighter fits.
const HARMONICS_OPTIONS = [
  { value: 1,  label: 'Off'   },
  { value: 2,  label: '×2'    },
  { value: 3,  label: '×3'    },
  { value: 5,  label: '×5'    },
  { value: 7,  label: '×7'    },
  { value: 10, label: '×10'   },
]

function centsBetween(c, s) {
  if (!c || !s) return 0
  return 1200 * Math.log2(c / s)
}

// Cents distance from a detected frequency to the nearest octave of a target.
// Used by the live tuner so playing a string at the wrong octave still snaps
// to the right string (positive or negative cents from the closest octave).
function octaveTolerantCents(detectedHz, targetHz) {
  if (!detectedHz || !targetHz) return null
  const k = Math.round(Math.log2(detectedHz / targetHz))
  return 1200 * Math.log2(detectedHz / (targetHz * Math.pow(2, k)))
}

// Auto-pick the string whose nearest octave is closest in cents to the
// detected pitch. Returns { index, cents } or null.
function findClosestString(detectedHz, strings) {
  if (!detectedHz || !strings?.length) return null
  let best = null
  for (let i = 0; i < strings.length; i++) {
    const c = octaveTolerantCents(detectedHz, strings[i].target)
    if (c == null) continue
    if (best == null || Math.abs(c) < Math.abs(best.cents)) {
      best = { index: i, cents: c }
    }
  }
  return best
}

function colorForCents(absCents) {
  if (absCents <= 20) return 'text-emerald-300'
  if (absCents <= 50) return 'text-yellow-300'
  return 'text-red-400'
}

function buildMapping({ tuning, fundamental, instrument, harmonicsDepth, intermod }) {
  const inst = INSTRUMENTS[instrument] || INSTRUMENTS.guitar
  if (!tuning?.length || !fundamental) return { strings: [], inst }

  const OCT_RANGE = 5  // candidates span ±5 octaves of any base ratio

  // De-duplicated candidate set keyed by quantized frequency.
  const set = new Map()
  const add = (ratio, source) => {
    if (!Number.isFinite(ratio) || ratio <= 0) return
    for (let k = -OCT_RANGE; k <= OCT_RANGE; k++) {
      const freq = fundamental * ratio * (2 ** k)
      if (freq <= 0 || !Number.isFinite(freq)) continue
      const key = freq.toFixed(4)
      if (!set.has(key)) {
        set.set(key, { freq, ratio, octave_shift: k, source })
      }
    }
  }

  // Base ratios from the analysis
  for (const r of tuning) add(r, 'ratio')

  // Harmonics + subharmonics
  if (harmonicsDepth > 1) {
    for (const r of tuning) {
      for (let n = 2; n <= harmonicsDepth; n++) {
        add(r * n, `×${n}`)
        add(r / n, `÷${n}`)
      }
    }
  }

  // Intermodulation: pairwise sum, difference, product on the base ratios
  if (intermod) {
    for (let i = 0; i < tuning.length; i++) {
      for (let j = i + 1; j < tuning.length; j++) {
        const a = tuning[i]
        const b = tuning[j]
        add(a + b,           'IM+')
        add(Math.abs(a - b), 'IM−')
        add(a * b,           'IM×')
      }
    }
  }

  const candidates = Array.from(set.values())

  const strings = inst.strings.map((s, idx) => {
    let best = candidates[0]
    let bestDist = Math.abs(centsBetween(best.freq, s))
    for (const c of candidates) {
      const d = Math.abs(centsBetween(c.freq, s))
      if (d < bestDist) { best = c; bestDist = d }
    }
    return {
      index: idx,
      name: inst.names[idx],
      standard: s,
      target: best.freq,
      cents_offset: centsBetween(best.freq, s),
      ratio_used: best.ratio,
      octave_shift: best.octave_shift,
      source: best.source,
    }
  })

  return { strings, inst, candidateCount: candidates.length }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function GuitarTuningTab({
  sessionId,
  analysisResult,
  onSaveToLibrary,
  reducedTuning,
  onReducedTuningChange,
}) {
  const [instrument, setInstrument] = useState('guitar')
  const audioFundamental = useMemo(
    () => deriveAudioFundamental(analysisResult),
    [analysisResult]
  )
  const [fundamental, setFundamental] = useState(audioFundamental || 82.41)
  const [harmonicsDepth, setHarmonicsDepth] = useState(3)
  const [intermod, setIntermod] = useState(false)
  // 'full' = use analysisResult.tuning, 'reduced' = use reducedTuning.reduced_tuning
  const [tuningSource, setTuningSource] = useState('full')
  const [reduceNSteps, setReduceNSteps] = useState(12)
  const [reducing, setReducing] = useState(false)
  const synthRef = useRef(null)

  // --- Live tuner state ---
  const [tunerActive, setTunerActive] = useState(false)
  const [tunerError, setTunerError] = useState(null)
  const [detectedHz, setDetectedHz] = useState(null)
  const [lockedIdx, setLockedIdx] = useState(null)  // null = auto-detect
  const [inTuneFlash, setInTuneFlash] = useState(false)
  const tunerHandleRef = useRef(null)
  const inTuneSinceRef = useRef(null)
  const lastHapticAtRef = useRef(0)

  // Whenever a new analysis comes in, snap the fundamental to its strongest
  // peak. The user can still override with chips or the manual input.
  useEffect(() => {
    if (audioFundamental) setFundamental(audioFundamental)
  }, [audioFundamental])

  // Lifecycle: start/stop pitch detection when the tuner toggle flips.
  useEffect(() => {
    if (!tunerActive) {
      const h = tunerHandleRef.current
      tunerHandleRef.current = null
      if (h) h.stop().catch(() => {})
      setDetectedHz(null)
      inTuneSinceRef.current = null
      return
    }
    let cancelled = false
    setTunerError(null)
    startPitchDetection({
      onPitch: ({ frequency }) => {
        if (cancelled) return
        setDetectedHz(frequency)
      },
    })
      .then((handle) => {
        if (cancelled) {
          handle.stop().catch(() => {})
        } else {
          tunerHandleRef.current = handle
        }
      })
      .catch((err) => {
        console.error('Pitch detection failed:', err)
        let msg = err?.message || 'Could not start the live tuner.'
        if (err?.name === 'NotAllowedError') {
          msg = 'Microphone permission was denied.'
        }
        setTunerError(msg)
        setTunerActive(false)
      })
    return () => {
      cancelled = true
    }
  }, [tunerActive])

  // Always release mic on unmount.
  useEffect(() => () => {
    if (tunerHandleRef.current) {
      tunerHandleRef.current.stop().catch(() => {})
      tunerHandleRef.current = null
    }
  }, [])

  const fullTuning = analysisResult?.tuning || []
  const reducedRatios = reducedTuning?.reduced_tuning || null

  // When a reduction lands and the user is still on 'full', auto-switch them
  // to 'reduced' — that's almost always what they want. They can flip back.
  useEffect(() => {
    if (reducedRatios && reducedRatios.length && tuningSource === 'full') {
      setTuningSource('reduced')
    }
    // Intentionally only fire when a *new* reduction appears.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reducedRatios?.length])

  const tuning = tuningSource === 'reduced' && reducedRatios?.length
    ? reducedRatios
    : fullTuning

  const { strings, inst, candidateCount } = useMemo(
    () => buildMapping({ tuning, fundamental, instrument, harmonicsDepth, intermod }),
    [tuning, fundamental, instrument, harmonicsDepth, intermod]
  )

  async function handleReduceInline() {
    if (!sessionId) return
    setReducing(true)
    try {
      const r = await apiClient.reduceTuning(sessionId, reduceNSteps, 2.0)
      onReducedTuningChange?.(r)
      setTuningSource('reduced')
    } catch (err) {
      console.error('Inline reduction failed:', err)
    } finally {
      setReducing(false)
    }
  }

  // Derive which string the tuner is reading + the cents offset.
  const tunerReading = useMemo(() => {
    if (!detectedHz || !strings.length) {
      return { stringIdx: null, cents: null }
    }
    if (lockedIdx != null && strings[lockedIdx]) {
      return {
        stringIdx: lockedIdx,
        cents: octaveTolerantCents(detectedHz, strings[lockedIdx].target),
      }
    }
    const closest = findClosestString(detectedHz, strings)
    return { stringIdx: closest?.index ?? null, cents: closest?.cents ?? null }
  }, [detectedHz, strings, lockedIdx])

  // In-tune logic: |cents| ≤ 5 for ≥ 500 ms triggers the flash + haptic.
  useEffect(() => {
    const c = tunerReading.cents
    if (c == null) {
      inTuneSinceRef.current = null
      setInTuneFlash(false)
      return
    }
    if (Math.abs(c) <= 5) {
      if (inTuneSinceRef.current == null) {
        inTuneSinceRef.current = performance.now()
      } else if (performance.now() - inTuneSinceRef.current >= 500) {
        const now = performance.now()
        if (now - lastHapticAtRef.current > 1200) {
          lastHapticAtRef.current = now
          if (typeof navigator.vibrate === 'function') {
            try { navigator.vibrate(40) } catch { /* ignore */ }
          }
          setInTuneFlash(true)
          setTimeout(() => setInTuneFlash(false), 600)
        }
      }
    } else {
      inTuneSinceRef.current = null
      setInTuneFlash(false)
    }
  }, [tunerReading.cents])

  async function ensureSynth() {
    await Tone.start()
    if (!synthRef.current) {
      synthRef.current = new Tone.Synth({
        oscillator: { type: 'triangle' },
        envelope: { attack: 0.03, decay: 0.2, sustain: 0.6, release: 0.4 },
      }).toDestination()
    }
    return synthRef.current
  }

  async function playFreq(freq, dur = 1.5) {
    const synth = await ensureSynth()
    synth.triggerAttackRelease(freq, dur)
  }

  async function playArpeggio() {
    const synth = await ensureSynth()
    const now = Tone.now()
    strings.forEach((s, i) => {
      synth.triggerAttackRelease(s.target, 0.5, now + i * 0.45)
    })
  }

  function handleExportScl() {
    exportTuningLocal('scl', tuning, {
      description: 'Biotuner-derived tuning',
      filename: 'biotuner_tuning',
    })
  }

  function handleExportJson() {
    exportTuningLocal('json', tuning, {
      description: 'Biotuner-derived tuning',
      filename: 'biotuner_tuning',
    })
  }

  function handleCopyFreqs() {
    const text = strings.map((s) => `${s.name}\t${s.target.toFixed(2)} Hz`).join('\n')
    navigator.clipboard?.writeText(text).catch(() => {})
  }

  if (!tuning.length) {
    return (
      <div className="text-center text-biotuner-light/60 py-12">
        <Music className="w-12 h-12 mx-auto mb-3 opacity-50" />
        <p>No tuning to map yet. Run an analysis first.</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
            Instrument
          </label>
          <select
            value={instrument}
            onChange={(e) => setInstrument(e.target.value)}
            className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 min-h-[48px]"
          >
            {Object.entries(INSTRUMENTS).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
            Fundamental (Hz)
          </label>
          <input
            type="number"
            step="0.01"
            min="20"
            max="2000"
            value={fundamental}
            onChange={(e) => setFundamental(parseFloat(e.target.value) || 0)}
            className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 min-h-[48px]"
          />

          {/* Two reference sources, both single-shot.
              "From data" picks the strongest-peak fundamental (already
              octave-shifted into a guitar-friendly 55–440 Hz window by
              deriveAudioFundamental). The earlier multi-octave picker
              (÷4 … ×4) was misleading — string targets are matched in
              absolute Hz, so shifting the reference by a power of two
              produced the *exact same* assignment table. One button
              keeps the choice meaningful. */}
          {audioFundamental && (
            <div className="mt-2">
              <div className="text-[10px] uppercase tracking-wider text-biotuner-light/40 mb-1">
                From data
              </div>
              <button
                onClick={() => setFundamental(audioFundamental)}
                className={`min-h-[40px] px-3 rounded-md text-xs font-mono border transition-all
                  ${Math.abs(fundamental - audioFundamental) < 0.05
                    ? 'bg-biotuner-primary text-biotuner-dark-900 border-biotuner-primary'
                    : 'bg-biotuner-primary/10 text-biotuner-primary border-biotuner-primary/40 hover:border-biotuner-primary'}`}
              >
                {audioFundamental.toFixed(2)} Hz
              </button>
            </div>
          )}

          <div className="mt-2">
            <div className="text-[10px] uppercase tracking-wider text-biotuner-light/40 mb-1">
              Standard references
            </div>
            <div className="flex gap-2 flex-wrap">
              {STANDARD_FUNDAMENTAL_CHIPS.map((f) => (
                <button
                  key={f}
                  onClick={() => setFundamental(f)}
                  className={`min-h-[40px] px-3 rounded-md text-xs font-medium border transition-all
                    ${Math.abs(fundamental - f) < 0.01
                      ? 'bg-biotuner-primary text-biotuner-dark-900 border-biotuner-primary'
                      : 'bg-biotuner-dark-800 text-biotuner-light/80 border-biotuner-dark-600 hover:border-biotuner-primary/50'}`}
                >
                  {f}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="flex items-end gap-3 flex-wrap">
        <div className="flex-1 min-w-[180px]">
          <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
            Harmonics depth
          </label>
          <select
            value={harmonicsDepth}
            onChange={(e) => setHarmonicsDepth(parseInt(e.target.value, 10))}
            className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 min-h-[48px]"
          >
            {HARMONICS_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        <label className="flex items-center gap-2 text-sm text-biotuner-light/80 cursor-pointer min-h-[48px] px-3 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600">
          <input
            type="checkbox"
            checked={intermod}
            onChange={(e) => setIntermod(e.target.checked)}
            className="w-4 h-4"
          />
          Intermodulation
        </label>

        <button
          onClick={playArpeggio}
          className="min-h-[48px] flex items-center gap-2 px-4 py-2 rounded-lg bg-gradient-to-r from-biotuner-primary to-biotuner-secondary text-biotuner-dark-900 font-medium"
        >
          <Play className="w-4 h-4" /> Play all
        </button>

        <button
          onClick={() => {
            if (tunerActive) {
              setTunerActive(false)
            } else {
              setLockedIdx(null)
              setTunerActive(true)
            }
          }}
          aria-pressed={tunerActive}
          className={`
            min-h-[48px] flex items-center gap-2 px-4 py-2 rounded-lg font-medium border
            ${tunerActive
              ? 'bg-red-500/20 border-red-400/50 text-red-200'
              : 'bg-biotuner-dark-800 border-biotuner-dark-600 text-biotuner-light/80 hover:border-biotuner-primary/50'}
          `}
        >
          {tunerActive ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
          {tunerActive ? 'Stop tuner' : 'Live tuner'}
        </button>
      </div>

      <p className="text-xs text-biotuner-light/40 -mt-2">
        Always picks the absolute closest candidate — no cents constraint.
        Add harmonics and/or intermodulation to grow the candidate pool
        ({candidateCount} pitches available).
      </p>

      {/* Source tuning: full vs reduced (consonance-maximised) */}
      <div className="bg-biotuner-dark-800/50 border border-biotuner-dark-600 rounded-lg p-4">
        <div className="flex items-start sm:items-center gap-3 flex-col sm:flex-row">
          <div className="flex-shrink-0">
            <div className="text-xs font-medium text-biotuner-light/60 uppercase tracking-wider mb-1">
              Source tuning
            </div>
            <div className="text-xs text-biotuner-light/40">
              Mapping draws candidates from this set of ratios.
            </div>
          </div>
          <div className="flex gap-1 p-1 rounded-lg bg-biotuner-dark-900 border border-biotuner-dark-600 sm:ml-auto">
            <button
              onClick={() => setTuningSource('full')}
              aria-pressed={tuningSource === 'full'}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all min-h-[36px]
                ${tuningSource === 'full'
                  ? 'bg-biotuner-primary text-biotuner-dark-900'
                  : 'text-biotuner-light/70 hover:text-biotuner-light'}`}
            >
              Full ({fullTuning.length})
            </button>
            <button
              onClick={() => setTuningSource('reduced')}
              disabled={!reducedRatios?.length}
              aria-pressed={tuningSource === 'reduced'}
              className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all min-h-[36px] disabled:opacity-40 disabled:cursor-not-allowed
                ${tuningSource === 'reduced'
                  ? 'bg-biotuner-primary text-biotuner-dark-900'
                  : 'text-biotuner-light/70 hover:text-biotuner-light'}`}
            >
              Reduced{reducedRatios?.length ? ` (${reducedRatios.length})` : ''}
            </button>
          </div>
        </div>

        {/* Inline reduce control */}
        {!reducedRatios?.length && sessionId && (
          <div className="flex items-center gap-2 flex-wrap mt-3 pt-3 border-t border-biotuner-dark-600">
            <span className="text-xs text-biotuner-light/60">
              Reduce to most consonant subset:
            </span>
            <input
              type="number"
              min="3"
              max="32"
              value={reduceNSteps}
              onChange={(e) => setReduceNSteps(parseInt(e.target.value, 10) || 12)}
              className="w-16 bg-biotuner-dark-900 text-biotuner-light border border-biotuner-dark-600 rounded p-1.5 text-sm"
            />
            <span className="text-xs text-biotuner-light/40">steps</span>
            <button
              onClick={handleReduceInline}
              disabled={reducing}
              className="ml-auto sm:ml-0 min-h-[36px] flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-biotuner-accent/15 border border-biotuner-accent/40 text-biotuner-accent text-sm disabled:opacity-50"
            >
              <Sparkles className="w-3.5 h-3.5" />
              {reducing ? 'Reducing…' : 'Reduce'}
            </button>
          </div>
        )}

        {/* Show consonance gain when reduction is loaded */}
        {reducedRatios?.length > 0 && reducedTuning?.original_consonance != null && (
          <div className="mt-3 pt-3 border-t border-biotuner-dark-600 text-xs text-biotuner-light/60 flex items-center gap-3 flex-wrap">
            <span>
              Original consonance{' '}
              <span className="font-mono text-biotuner-light/80">
                {reducedTuning.original_consonance.toFixed(2)}
              </span>
            </span>
            <span>→</span>
            <span>
              Reduced{' '}
              <span className="font-mono text-emerald-300">
                {reducedTuning.reduced_consonance?.toFixed(2)}
              </span>
            </span>
            {reducedTuning.reduced_consonance && reducedTuning.original_consonance > 0 && (
              <span className="ml-auto sm:ml-0 px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-300">
                {((reducedTuning.reduced_consonance / reducedTuning.original_consonance - 1) * 100).toFixed(1)}%
              </span>
            )}
          </div>
        )}
      </div>

      {/* Live tuner error */}
      {tunerError && !tunerActive && (
        <div className="bg-red-900/20 border border-red-500/40 rounded-lg p-3 text-sm text-red-300">
          {tunerError}
        </div>
      )}

      {/* Live tuner strip */}
      {tunerActive && (
        <TunerStrip
          reading={tunerReading}
          detectedHz={detectedHz}
          string={tunerReading.stringIdx != null ? strings[tunerReading.stringIdx] : null}
          locked={lockedIdx != null}
          inTuneFlash={inTuneFlash}
          onUnlock={() => setLockedIdx(null)}
        />
      )}

      {/* String cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {strings.map((s) => {
          const cents = s.cents_offset
          const abs = Math.abs(cents)
          const color = colorForCents(abs)
          const isActiveString = tunerActive && tunerReading.stringIdx === s.index
          const isLocked = tunerActive && lockedIdx === s.index
          return (
            <div
              key={s.index}
              onClick={() => {
                if (!tunerActive) return
                setLockedIdx((cur) => (cur === s.index ? null : s.index))
              }}
              role={tunerActive ? 'button' : undefined}
              className={`
                bg-biotuner-dark-800/70 border rounded-lg p-4 flex flex-col gap-2 transition-all
                ${isLocked
                  ? 'border-biotuner-accent/80 bg-biotuner-accent/10'
                  : isActiveString
                    ? 'border-biotuner-primary/80'
                    : 'border-biotuner-dark-600'}
                ${tunerActive ? 'cursor-pointer hover:border-biotuner-primary/50' : ''}
              `}
            >
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold text-biotuner-light flex items-center gap-1.5">
                  {s.name}
                  {isLocked && <Lock className="w-3.5 h-3.5 text-biotuner-accent" />}
                </span>
                <span className="text-xs text-biotuner-light/40 font-mono">
                  std {s.standard.toFixed(2)} Hz
                </span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-mono text-biotuner-primary">
                  {s.target.toFixed(2)}
                </span>
                <span className="text-sm text-biotuner-light/40">Hz</span>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className={`font-mono ${color}`}>
                  {cents >= 0 ? '+' : ''}{cents.toFixed(1)} ¢
                </span>
                <span className="text-xs text-biotuner-light/40 font-mono">
                  r={s.ratio_used?.toFixed(3)} · 2^{s.octave_shift} · {s.source}
                </span>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); playFreq(s.target) }}
                className="mt-1 min-h-[44px] flex items-center justify-center gap-1.5 px-3 py-2 rounded-md bg-biotuner-dark-900 border border-biotuner-dark-600 hover:border-biotuner-primary/50 text-sm text-biotuner-light/80"
              >
                <Volume2 className="w-4 h-4" /> Play reference
              </button>
            </div>
          )
        })}
      </div>

      {/* Export buttons */}
      <div className="flex flex-wrap gap-2 pt-2">
        <button
          onClick={handleExportScl}
          className="min-h-[44px] flex items-center gap-2 px-4 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 hover:border-biotuner-primary/50 text-sm text-biotuner-light"
        >
          <Download className="w-4 h-4" /> Export .scl
        </button>
        <button
          onClick={handleExportJson}
          className="min-h-[44px] flex items-center gap-2 px-4 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 hover:border-biotuner-primary/50 text-sm text-biotuner-light"
        >
          <Download className="w-4 h-4" /> Export JSON
        </button>
        <button
          onClick={handleCopyFreqs}
          className="min-h-[44px] flex items-center gap-2 px-4 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 hover:border-biotuner-primary/50 text-sm text-biotuner-light"
        >
          Copy frequencies
        </button>
        {onSaveToLibrary && (
          <button
            onClick={() => onSaveToLibrary({
              ratios: tuning,
              fundamental,
              instrument,
              strings,
            })}
            className="min-h-[44px] flex items-center gap-2 px-4 py-2 rounded-lg bg-biotuner-primary/20 border border-biotuner-primary/40 text-biotuner-primary text-sm"
          >
            <Save className="w-4 h-4" /> Save to library
          </button>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// TunerStrip — needle meter at the top of the Guitar tab while the live
// tuner is active. Renders a -50..+50 ¢ scale, a colored needle, the active
// string + target frequency, and a green flash when the user is in tune.
// ---------------------------------------------------------------------------

function TunerStrip({ reading, detectedHz, string, locked, inTuneFlash, onUnlock }) {
  const cents = reading?.cents
  const hasReading = cents != null && string != null

  // Map cents → [0, 100] %, clamped to the visible ±50¢ window.
  const needlePct = hasReading
    ? Math.max(0, Math.min(100, 50 + cents))
    : 50

  const absCents = hasReading ? Math.abs(cents) : Infinity
  const needleColor =
    absCents <= 5  ? 'bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.6)]' :
    absCents <= 20 ? 'bg-emerald-300' :
    absCents <= 50 ? 'bg-yellow-300' :
                     'bg-red-400'

  const directionHint =
    !hasReading           ? null :
    absCents <= 5         ? null :
    cents < 0             ? 'tune ↑ up' :
                            'tune ↓ down'

  return (
    <div
      className={`
        rounded-xl border-2 p-4 sm:p-5 transition-colors duration-300
        ${inTuneFlash
          ? 'border-emerald-400 bg-emerald-500/15'
          : 'border-biotuner-primary/40 bg-biotuner-dark-900/80'}
      `}
    >
      {/* Top row: detected note + locked indicator */}
      <div className="flex items-center justify-between mb-3 gap-3">
        <div className="flex items-baseline gap-3">
          <div className="text-3xl sm:text-4xl font-bold text-biotuner-light tabular-nums">
            {string ? string.name : '—'}
          </div>
          <div className="text-xs sm:text-sm text-biotuner-light/60">
            target {string ? `${string.target.toFixed(2)} Hz` : ''}
            {locked && (
              <span className="ml-2 inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-biotuner-accent/20 text-biotuner-accent uppercase tracking-wider text-[10px]">
                <Lock className="w-3 h-3" /> locked
              </span>
            )}
          </div>
        </div>
        <div className="text-right font-mono text-sm text-biotuner-light/80">
          {detectedHz ? `${detectedHz.toFixed(2)} Hz` : 'listening…'}
        </div>
      </div>

      {/* Needle meter */}
      <div className="relative h-20 select-none">
        {/* Center axis (target line) */}
        <div className="absolute inset-x-0 top-1/2 h-px bg-biotuner-light/15" />

        {/* Tick marks at -50, -25, 0, +25, +50 */}
        {[-50, -25, 0, 25, 50].map((t) => (
          <div
            key={t}
            className="absolute top-1/2 -translate-y-1/2 flex flex-col items-center"
            style={{ left: `${50 + t}%`, transform: 'translateX(-50%)' }}
          >
            <div className={t === 0
              ? 'w-px h-10 bg-biotuner-primary'
              : 'w-px h-5 bg-biotuner-light/30'} />
            <div className="text-[10px] text-biotuner-light/40 mt-1 font-mono">
              {t === 0 ? '0¢' : `${t > 0 ? '+' : ''}${t}`}
            </div>
          </div>
        ))}

        {/* Needle — no CSS transition on `left`; smoothing is done in JS so a
            transition here would only add visible lag on top of it. */}
        {hasReading && (
          <div
            className="absolute top-1/2"
            style={{
              left: `${needlePct}%`,
              transform: 'translate(-50%, -50%)',
              transition: 'background-color 200ms',
              willChange: 'left',
            }}
          >
            <div className={`w-1.5 h-14 rounded-full ${needleColor}`} />
          </div>
        )}
      </div>

      {/* Bottom row: cents readout + direction hint + unlock */}
      <div className="flex items-center justify-between gap-3 mt-2 min-h-[24px]">
        <div className={`font-mono text-sm ${
          inTuneFlash ? 'text-emerald-300 font-bold' :
          !hasReading ? 'text-biotuner-light/30' :
          absCents <= 5 ? 'text-emerald-300' :
          absCents <= 20 ? 'text-emerald-200' :
          absCents <= 50 ? 'text-yellow-200' :
                           'text-red-300'
        }`}>
          {hasReading
            ? `${cents >= 0 ? '+' : ''}${cents.toFixed(1)} ¢`
            : '—'}
          {inTuneFlash && '  ✓ in tune'}
        </div>
        <div className="flex items-center gap-2">
          {directionHint && (
            <span className="text-xs text-biotuner-light/50">{directionHint}</span>
          )}
          {locked && (
            <button
              onClick={onUnlock}
              className="min-h-[32px] px-3 rounded-md text-xs bg-biotuner-dark-800 border border-biotuner-dark-600 hover:border-biotuner-primary/50 text-biotuner-light/80"
            >
              Unlock
            </button>
          )}
        </div>
      </div>

      {!locked && (
        <p className="text-[11px] text-biotuner-light/40 mt-2">
          Auto-detecting closest string. Tap a string card to lock it.
        </p>
      )}
    </div>
  )
}
