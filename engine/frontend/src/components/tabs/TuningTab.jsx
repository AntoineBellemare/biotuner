import { useState, useEffect, useRef } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Download, Play, Volume2 } from 'lucide-react'
import * as Tone from 'tone'
import apiClient from '../../services/api'
import ConsonanceMatrix from '../ConsonanceMatrix'

// Convert decimal to fraction using max denominator
function decimalToFraction(decimal, maxDenom = 100) {
  let bestNum = 1
  let bestDen = 1
  let minError = Math.abs(decimal - 1)
  
  for (let den = 1; den <= maxDenom; den++) {
    const num = Math.round(decimal * den)
    const error = Math.abs(decimal - num / den)
    if (error < minError) {
      minError = error
      bestNum = num
      bestDen = den
    }
  }
  
  // Reduce fraction
  const gcd = (a, b) => b === 0 ? a : gcd(b, a % b)
  const divisor = gcd(bestNum, bestDen)
  return `${bestNum / divisor}/${bestDen / divisor}`
}

// Find closest interval name from catalog
function findIntervalName(ratio, intervalCatalog, tolerance = 0.01) {
  if (!intervalCatalog || intervalCatalog.length === 0) return ''
  
  let closestInterval = null
  let minDiff = Infinity
  
  for (const interval of intervalCatalog) {
    const diff = Math.abs(interval.ratio - ratio)
    if (diff < minDiff && diff < tolerance) {
      minDiff = diff
      closestInterval = interval.name
    }
  }
  
  return closestInterval || ''
}

// Modern Consonance Gauge Component
function ConsonanceGauge({ consonance }) {
  // Consonance is in 0-50 range
  const value = Math.min(50, Math.max(0, consonance))
  const percentage = (value / 50) * 100
  
  // Determine color based on value
  const getColor = () => {
    if (value < 15) return { from: '#ef4444', to: '#dc2626', glow: 'rgba(239, 68, 68, 0.3)' } // Red
    if (value < 35) return { from: '#f59e0b', to: '#d97706', glow: 'rgba(245, 158, 11, 0.3)' } // Amber
    return { from: '#10b981', to: '#059669', glow: 'rgba(16, 185, 129, 0.3)' } // Green
  }
  
  const colors = getColor()
  const radius = 80
  const strokeWidth = 12
  const normalizedRadius = radius - strokeWidth / 2
  const circumference = normalizedRadius * 2 * Math.PI
  const strokeDashoffset = circumference - (percentage / 100) * circumference
  
  return (
    <div className="relative flex flex-col items-center justify-center p-8">
      {/* Circular progress */}
      <div className="relative w-64 h-64">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 180 180">
          <defs>
            <linearGradient id="consonanceGradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" style={{ stopColor: colors.from, stopOpacity: 1 }} />
              <stop offset="100%" style={{ stopColor: colors.to, stopOpacity: 1 }} />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
          
          {/* Background circle */}
          <circle
            cx="90"
            cy="90"
            r={normalizedRadius}
            fill="none"
            stroke="#1f2937"
            strokeWidth={strokeWidth}
          />
          
          {/* Progress circle */}
          <circle
            cx="90"
            cy="90"
            r={normalizedRadius}
            fill="none"
            stroke="url(#consonanceGradient)"
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={`${circumference} ${circumference}`}
            strokeDashoffset={strokeDashoffset}
            filter="url(#glow)"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        
        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <div 
            className="text-6xl font-bold mb-2 transition-colors duration-500"
            style={{ color: colors.from }}
          >
            {value.toFixed(1)}
          </div>
          <div className="text-sm text-gray-400 uppercase tracking-wider">
            Tuning Consonance
          </div>
          <div className="mt-4 flex items-center gap-2">
            <div className="text-xs text-gray-500">0</div>
            <div className="w-32 h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div 
                className="h-full rounded-full transition-all duration-1000 ease-out"
                style={{ 
                  width: `${percentage}%`,
                  background: `linear-gradient(90deg, ${colors.from}, ${colors.to})`
                }}
              />
            </div>
            <div className="text-xs text-gray-500">50</div>
          </div>
        </div>
      </div>
      
      {/* Status label */}
      <div className="mt-4 px-4 py-2 rounded-full text-sm font-medium" style={{ 
        backgroundColor: colors.glow,
        color: colors.from 
      }}>
        {value < 15 ? '🔴 Low Consonance' : value < 35 ? '🟡 Moderate Consonance' : '🟢 High Consonance'}
      </div>
    </div>
  )
}

export default function TuningTab({
  sessionId,
  analysisResult,
  fileInfo,
  reducedTuning: reducedTuningProp,
  onReducedTuningChange,
}) {
  // Prefer the lifted state when provided so other tabs (Guitar) see the
  // reduction. Fall back to local state if rendered outside the App tree.
  const [reducedTuningLocal, setReducedTuningLocal] = useState(null)
  const reducedTuning = reducedTuningProp !== undefined ? reducedTuningProp : reducedTuningLocal
  const setReducedTuning = onReducedTuningChange || setReducedTuningLocal
  const [nSteps, setNSteps] = useState(12)
  const [loading, setLoading] = useState(false)
  const [playingTuning, setPlayingTuning] = useState(false)
  const [playingReduced, setPlayingReduced] = useState(false)
  const [playingRandomChords, setPlayingRandomChords] = useState(false)
  const [audioRef, setAudioRef] = useState(null)
  const [intervalCatalog, setIntervalCatalog] = useState([])
  // Ratios currently sounding — used to highlight matching rows in the
  // tuning table while Play Scale or Play Chords runs.
  const [activeRatios, setActiveRatios] = useState([])
  const synthRef = useRef(null)
  const playbackTimeoutsRef = useRef([])

  // Load interval catalog on mount
  useEffect(() => {
    const loadCatalog = async () => {
      try {
        const data = await apiClient.getIntervalCatalog()
        setIntervalCatalog(data.intervals || [])
      } catch (error) {
        console.error('Failed to load interval catalog:', error)
      }
    }
    loadCatalog()
  }, [])

  // Prepare chart data
  const peaksData = analysisResult.peaks?.map((peak, idx) => ({
    name: `Peak ${idx + 1}`,
    frequency: peak,
    power: analysisResult.powers?.[idx] || 0,
  })) || []

  const tuningData = analysisResult.tuning?.map((ratio, idx) => ({
    name: `Note ${idx + 1}`,
    ratio: ratio,
    cents: Math.log2(ratio) * 1200,
  })) || []

  // -------------------------------------------------------------------------
  // Frontend chord/scale playback (Tone.js) — gives us per-note timing so
  // we can highlight the matching rows in the tuning table as they sound.
  // -------------------------------------------------------------------------

  async function ensureSynth() {
    await Tone.start()
    if (!synthRef.current) {
      synthRef.current = new Tone.PolySynth(Tone.Synth, {
        oscillator: { type: 'triangle' },
        envelope: { attack: 0.05, decay: 0.2, sustain: 0.5, release: 0.6 },
        volume: -8,
      }).toDestination()
    }
    return synthRef.current
  }

  function clearPlaybackTimers() {
    for (const t of playbackTimeoutsRef.current) clearTimeout(t)
    playbackTimeoutsRef.current = []
  }

  // Clean up audio + timers on unmount
  useEffect(() => {
    return () => {
      clearPlaybackTimers()
      if (synthRef.current) {
        try { synthRef.current.releaseAll() } catch { /* ignore */ }
        try { synthRef.current.dispose() } catch { /* ignore */ }
        synthRef.current = null
      }
    }
  }, [])

  // Handle tuning reduction
  const handleReduceTuning = async () => {
    try {
      setLoading(true)
      const result = await apiClient.reduceTuning(sessionId, nSteps, 2.0)
      setReducedTuning(result)
      setLoading(false)
    } catch (error) {
      console.error('Reduction error:', error)
      setLoading(false)
    }
  }

  // Play tuning — scale, one note at a time, highlights each row as it sounds.
  const handlePlayTuning = async (tuning, isReduced = false) => {
    try {
      if (!tuning?.length) return
      clearPlaybackTimers()
      const synth = await ensureSynth()
      const baseFreq = 220
      const noteDuration = 0.45  // seconds per note
      const gap = 0.05

      if (isReduced) setPlayingReduced(true)
      else setPlayingTuning(true)
      setActiveRatios([])

      const startAudio = Tone.now() + 0.05
      tuning.forEach((ratio, i) => {
        const at = startAudio + i * (noteDuration + gap)
        synth.triggerAttackRelease(baseFreq * ratio, noteDuration, at)
        const visualOffsetMs = i * (noteDuration + gap) * 1000
        playbackTimeoutsRef.current.push(
          setTimeout(() => setActiveRatios([ratio]), visualOffsetMs),
        )
      })

      const totalMs = tuning.length * (noteDuration + gap) * 1000
      playbackTimeoutsRef.current.push(setTimeout(() => {
        setActiveRatios([])
        if (isReduced) setPlayingReduced(false)
        else setPlayingTuning(false)
      }, totalMs + 100))
    } catch (error) {
      console.error('Play error:', error)
      setActiveRatios([])
      if (isReduced) setPlayingReduced(false)
      else setPlayingTuning(false)
    }
  }

  // Play random chords — N chords of 3–4 notes each, highlights all rows
  // belonging to the currently-sounding chord.
  const handlePlayRandomChords = async () => {
    try {
      const tuning = reducedTuning?.reduced_tuning || analysisResult.tuning
      if (!tuning?.length) return
      clearPlaybackTimers()
      const synth = await ensureSynth()

      const baseFreq = 220
      const chordDuration = 1.5
      const numChords = 8
      const chords = []
      for (let i = 0; i < numChords; i++) {
        const size = Math.min(tuning.length, 3 + Math.floor(Math.random() * 2))
        const picked = new Set()
        while (picked.size < size) {
          picked.add(Math.floor(Math.random() * tuning.length))
        }
        chords.push([...picked].sort((a, b) => a - b).map((idx) => tuning[idx]))
      }

      setPlayingRandomChords(true)
      setActiveRatios([])
      const startAudio = Tone.now() + 0.05
      chords.forEach((ratios, i) => {
        const freqs = ratios.map((r) => baseFreq * r)
        const at = startAudio + i * chordDuration
        synth.triggerAttackRelease(freqs, chordDuration - 0.15, at)
        playbackTimeoutsRef.current.push(
          setTimeout(() => setActiveRatios(ratios), i * chordDuration * 1000),
        )
      })

      const totalMs = numChords * chordDuration * 1000
      playbackTimeoutsRef.current.push(setTimeout(() => {
        setActiveRatios([])
        setPlayingRandomChords(false)
      }, totalMs + 100))
    } catch (error) {
      console.error('Play random chords error:', error)
      setPlayingRandomChords(false)
      setActiveRatios([])
    }
  }

  // Stop everything currently sounding (also covers the scale playback)
  const handleStopRandomChords = () => {
    clearPlaybackTimers()
    if (synthRef.current) {
      try { synthRef.current.releaseAll() } catch { /* ignore */ }
    }
    setActiveRatios([])
    setPlayingRandomChords(false)
    setPlayingTuning(false)
    setPlayingReduced(false)
  }

  // Download SCL file
  const downloadSCL = () => {
    const tuning = reducedTuning?.reduced_tuning || analysisResult.tuning
    
    let scl = `! Biotuner Scale\n`
    scl += `! Generated from ${fileInfo.filename}\n`
    scl += `!\n`
    scl += `${tuning.length}\n`
    scl += `!\n`
    
    tuning.forEach((ratio) => {
      const cents = Math.log2(ratio) * 1200
      scl += `${cents.toFixed(6)}\n`
    })

    const blob = new Blob([scl], { type: 'text/plain' })
    apiClient.downloadBlob(blob, 'biotuning.scl')
  }

  return (
    <div className="space-y-6 sm:space-y-8">
      {/* Analysis Info */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        <div className="bg-biotuner-dark-900 p-3 sm:p-4 rounded-lg border border-biotuner-primary/30">
          <h3 className="text-xs text-biotuner-light/60 uppercase tracking-wider mb-1">Peak Method</h3>
          <p className="text-base sm:text-xl font-bold text-biotuner-primary truncate">
            {analysisResult.method}
          </p>
        </div>
        <div className="bg-biotuner-dark-900 p-3 sm:p-4 rounded-lg border border-biotuner-secondary/30">
          <h3 className="text-xs text-biotuner-light/60 uppercase tracking-wider mb-1">Tuning Method</h3>
          <p className="text-base sm:text-xl font-bold text-biotuner-secondary truncate">
            {analysisResult.tuning_method || 'peaks_ratios'}
          </p>
        </div>
        <div className="bg-biotuner-dark-900 p-3 sm:p-4 rounded-lg border border-biotuner-accent/30">
          <h3 className="text-xs text-biotuner-light/60 uppercase tracking-wider mb-1">Peaks Found</h3>
          <p className="text-base sm:text-xl font-bold text-biotuner-accent">
            {analysisResult.n_peaks}
          </p>
        </div>
        <div className="bg-biotuner-dark-900 p-3 sm:p-4 rounded-lg border border-biotuner-accent/30">
          <h3 className="text-xs text-biotuner-light/60 uppercase tracking-wider mb-1">Consonance</h3>
          <p className="text-base sm:text-xl font-bold text-biotuner-accent">
            {analysisResult.metrics?.consonance?.toFixed(3) || 'N/A'}
          </p>
        </div>
      </div>

      {/* Parameters Used */}
      <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-4 sm:p-6">
        <h3 className="text-sm font-semibold text-biotuner-light/60 uppercase tracking-wider mb-4">
          Analysis Parameters
        </h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 text-sm">
          <div>
            <span className="text-biotuner-light/40">Precision:</span>
            <span className="ml-2 text-biotuner-primary font-mono">{analysisResult.precision || 1} Hz</span>
          </div>
          <div>
            <span className="text-biotuner-light/40">Max Freq:</span>
            <span className="ml-2 text-biotuner-primary font-mono">{analysisResult.max_freq || 60} Hz</span>
          </div>
          <div>
            <span className="text-biotuner-light/40">Max Denom:</span>
            <span className="ml-2 text-biotuner-primary font-mono">{analysisResult.max_denominator || 100}</span>
          </div>
          <div>
            <span className="text-biotuner-light/40">Harmonics:</span>
            <span className="ml-2 text-biotuner-primary font-mono">{analysisResult.n_harm || 10}</span>
          </div>
        </div>
      </div>

      {/* Peaks Chart */}
      <div>
        <h3 className="text-lg sm:text-xl font-bold mb-4">🎵 Frequency Peaks</h3>
        <div className="bg-gray-800 p-2 sm:p-4 rounded-lg">
          <ResponsiveContainer width="100%" height={250} className="sm:!h-[300px]">
            <BarChart data={peaksData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="name" stroke="#f5deb3" tick={{ fontSize: 11 }} />
              <YAxis stroke="#f5deb3" label={{ value: 'Frequency (Hz)', angle: -90, position: 'insideLeft', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f1f1f', border: '1px solid #6A5ACD' }}
                labelStyle={{ color: '#f5deb3' }}
              />
              <Bar dataKey="frequency" fill="#6A5ACD" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Consonance Matrix */}
      {analysisResult.tuning && analysisResult.tuning.length > 1 && (
        <ConsonanceMatrix 
          tuning={analysisResult.tuning} 
          maxDenominator={analysisResult.max_denominator || 100}
        />
      )}

      {/* Tuning Values Table and Consonance Gauge */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Tuning Table */}
        <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-4 sm:p-6">
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-4">
            <h3 className="text-lg sm:text-xl font-bold">📊 Tuning Values</h3>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => handlePlayTuning(analysisResult.tuning)}
                disabled={playingTuning}
                className="bg-biotuner-primary text-white px-3 sm:px-4 py-2 rounded-lg hover:bg-biotuner-secondary disabled:opacity-50 flex items-center gap-2 text-sm"
              >
                <Play className="w-4 h-4" />
                {playingTuning ? 'Playing...' : 'Play Scale'}
              </button>
              {!playingRandomChords ? (
                <button
                  onClick={handlePlayRandomChords}
                  className="bg-green-600 text-white px-3 sm:px-4 py-2 rounded-lg hover:bg-green-700 flex items-center gap-2 text-sm"
                >
                  <Play className="w-4 h-4" />
                  <span className="hidden sm:inline">Play </span>Chords
                </button>
              ) : (
                <button
                  onClick={handleStopRandomChords}
                  className="bg-red-600 text-white px-3 sm:px-4 py-2 rounded-lg hover:bg-red-700 flex items-center gap-2 animate-pulse text-sm"
                >
                  <Volume2 className="w-4 h-4" />
                  Stop
                </button>
              )}
            </div>
          </div>
        
        <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
          <table className="w-full text-sm min-w-[400px]">
            <thead>
              <tr className="border-b border-biotuner-dark-600">
                <th className="py-2 px-2 sm:px-4 text-left text-biotuner-light/60">#</th>
                <th className="py-2 px-2 sm:px-4 text-left text-biotuner-light/60">Ratio</th>
                <th className="py-2 px-2 sm:px-4 text-left text-biotuner-light/60">Cents</th>
                <th className="py-2 px-2 sm:px-4 text-left text-biotuner-light/60">Interval Name</th>
              </tr>
            </thead>
            <tbody>
              {analysisResult.tuning?.map((ratio, idx) => {
                const fraction = decimalToFraction(ratio, analysisResult.max_denominator || 100)
                const intervalName = findIntervalName(ratio, intervalCatalog, 0.01)

                // Check if this ratio is in the reduced tuning
                const isReduced = reducedTuning?.reduced_tuning?.some(
                  r => Math.abs(r - ratio) < 0.0001
                )
                // Highlight while this ratio is currently being played
                const isActive = activeRatios.some(
                  r => Math.abs(r - ratio) < 0.0001
                )

                return (
                  <tr
                    key={idx}
                    className={`border-b border-biotuner-dark-700 transition-all duration-150
                      ${isActive
                        ? 'bg-biotuner-accent/30 ring-1 ring-biotuner-accent/70 shadow-[inset_0_0_12px_rgba(16,185,129,0.25)]'
                        : isReduced
                          ? 'bg-biotuner-primary/20 border-biotuner-primary/50 hover:bg-biotuner-dark-800'
                          : 'hover:bg-biotuner-dark-800'}
                    `}
                  >
                    <td className={`py-2 px-4 font-bold ${isReduced ? 'text-biotuner-pink' : 'text-biotuner-accent'}`}>
                      {idx}
                    </td>
                    <td className={`py-2 px-4 font-mono ${isReduced ? 'text-biotuner-pink font-bold' : 'text-biotuner-primary'}`}>
                      {fraction}
                    </td>
                    <td className={`py-2 px-4 font-mono ${isReduced ? 'text-biotuner-pink' : 'text-biotuner-secondary'}`}>
                      {(Math.log2(ratio) * 1200).toFixed(1)}¢
                    </td>
                    <td className={`py-2 px-4 italic ${isReduced ? 'text-biotuner-pink/90' : 'text-biotuner-light/80'}`}>
                      {intervalName || '-'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>      
      {/* Right: Consonance Gauge */}
      <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-4 sm:p-6 flex items-center justify-center">
        <ConsonanceGauge consonance={analysisResult.metrics?.consonance || 0} />
      </div>
    </div>
      {/* Tuning Reduction */}
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg border border-biotuner-purple/30">
        <h3 className="text-lg sm:text-xl font-bold mb-4">🔄 Tuning Reduction</h3>
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4">
          <div className="flex-1">
            <label className="block text-sm mb-2">Number of Steps</label>
            <input
              type="number"
              min="3"
              max="24"
              value={nSteps}
              onChange={(e) => setNSteps(parseInt(e.target.value) || 12)}
              className="w-full bg-gray-700 border border-biotuner-purple/50 rounded-lg p-2"
            />
          </div>
          <button
            onClick={handleReduceTuning}
            disabled={loading}
            className="sm:mt-6 bg-biotuner-purple text-white px-4 sm:px-6 py-2 rounded-lg hover:bg-biotuner-pink hover:text-black disabled:opacity-50 w-full sm:w-auto"
          >
            {loading ? 'Reducing...' : 'Reduce Scale'}
          </button>
        </div>

        {reducedTuning && (
          <div className="mt-4 sm:mt-6 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
              <div className="bg-biotuner-dark-900 p-3 sm:p-4 rounded-lg border border-biotuner-primary/30">
                <h4 className="text-xs text-biotuner-light/60 uppercase tracking-wider mb-1">Original Consonance</h4>
                <p className="text-xl sm:text-2xl font-bold text-biotuner-primary">
                  {reducedTuning.original_consonance?.toFixed(2) || 'N/A'}
                </p>
              </div>
              <div className="bg-biotuner-dark-900 p-3 sm:p-4 rounded-lg border border-biotuner-secondary/30">
                <h4 className="text-xs text-biotuner-light/60 uppercase tracking-wider mb-1">Reduced Consonance</h4>
                <p className="text-xl sm:text-2xl font-bold text-biotuner-secondary">
                  {reducedTuning.reduced_consonance?.toFixed(2) || 'N/A'}
                </p>
              </div>
              <div className="bg-biotuner-dark-900 p-3 sm:p-4 rounded-lg border border-biotuner-accent/30">
                <h4 className="text-xs text-biotuner-light/60 uppercase tracking-wider mb-1">Improvement</h4>
                <p className="text-xl sm:text-2xl font-bold text-biotuner-accent">
                  {reducedTuning.reduced_consonance && reducedTuning.original_consonance
                    ? `${((reducedTuning.reduced_consonance / reducedTuning.original_consonance - 1) * 100).toFixed(1)}%`
                    : 'N/A'
                  }
                </p>
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-biotuner-dark-900 p-3 sm:p-4 rounded-lg border border-biotuner-pink/30">
              <div>
                <h4 className="font-semibold text-biotuner-pink mb-1 text-sm sm:text-base">
                  Reduced Tuning ({reducedTuning.n_steps} steps selected)
                </h4>
                <p className="text-xs sm:text-sm text-biotuner-light/60">
                  Highlighted rows in the table above show the most consonant intervals
                </p>
              </div>
              <button
                onClick={() => handlePlayTuning(reducedTuning.reduced_tuning, true)}
                disabled={playingReduced}
                className="bg-biotuner-pink text-black px-4 sm:px-6 py-2 rounded-lg hover:bg-biotuner-purple hover:text-white disabled:opacity-50 flex items-center justify-center gap-2 w-full sm:w-auto"
              >
                <Volume2 className="w-4 h-4" />
                {playingReduced ? 'Playing...' : 'Play Reduced'}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Download Button */}
      <div className="flex justify-center">
        <button
          onClick={downloadSCL}
          className="bg-green-600 text-white px-6 sm:px-8 py-3 rounded-lg font-semibold hover:bg-green-500 flex items-center gap-2 w-full sm:w-auto justify-center"
        >
          <Download className="w-5 h-5" />
          Download .SCL File
        </button>
      </div>
    </div>
  )
}
