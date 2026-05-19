import { useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'
import { Play, Download, Save, Music, Volume2 } from 'lucide-react'
import { exportTuning as exportTuningLocal } from '../../services/tuningExport'

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

export default function GuitarTuningTab({ analysisResult, onSaveToLibrary }) {
  const [instrument, setInstrument] = useState('guitar')
  const audioFundamental = useMemo(
    () => deriveAudioFundamental(analysisResult),
    [analysisResult]
  )
  const [fundamental, setFundamental] = useState(audioFundamental || 82.41)
  const [harmonicsDepth, setHarmonicsDepth] = useState(3)
  const [intermod, setIntermod] = useState(false)
  const synthRef = useRef(null)

  // Whenever a new analysis comes in, snap the fundamental to its strongest
  // peak. The user can still override with chips or the manual input.
  useEffect(() => {
    if (audioFundamental) setFundamental(audioFundamental)
  }, [audioFundamental])

  const tuning = analysisResult?.tuning || []

  const { strings, inst, candidateCount } = useMemo(
    () => buildMapping({ tuning, fundamental, instrument, harmonicsDepth, intermod }),
    [tuning, fundamental, instrument, harmonicsDepth, intermod]
  )

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

          {audioFundamental && (
            <div className="mt-2">
              <div className="text-[10px] uppercase tracking-wider text-biotuner-light/40 mb-1">
                From audio (strongest peak, octave-shifted)
              </div>
              <div className="flex gap-2 flex-wrap">
                {[0.25, 0.5, 1, 2, 4].map((m) => {
                  const f = audioFundamental * m
                  return (
                    <button
                      key={m}
                      onClick={() => setFundamental(f)}
                      className={`min-h-[40px] px-3 rounded-md text-xs font-mono border transition-all
                        ${Math.abs(fundamental - f) < 0.05
                          ? 'bg-biotuner-primary text-biotuner-dark-900 border-biotuner-primary'
                          : 'bg-biotuner-primary/10 text-biotuner-primary border-biotuner-primary/40 hover:border-biotuner-primary'}`}
                    >
                      {m === 1 ? '×1' : (m < 1 ? `÷${1/m}` : `×${m}`)} {f.toFixed(2)}
                    </button>
                  )
                })}
              </div>
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
      </div>

      <p className="text-xs text-biotuner-light/40 -mt-2">
        Always picks the absolute closest candidate — no cents constraint.
        Add harmonics and/or intermodulation to grow the candidate pool
        ({candidateCount} pitches available).
      </p>

      {/* String cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {strings.map((s) => {
          const cents = s.cents_offset
          const abs = Math.abs(cents)
          const color = colorForCents(abs)
          return (
            <div
              key={s.index}
              className="bg-biotuner-dark-800/70 border border-biotuner-dark-600 rounded-lg p-4 flex flex-col gap-2"
            >
              <div className="flex items-center justify-between">
                <span className="text-lg font-bold text-biotuner-light">{s.name}</span>
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
                onClick={() => playFreq(s.target)}
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
