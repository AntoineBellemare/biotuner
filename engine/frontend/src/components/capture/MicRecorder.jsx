import { useEffect, useRef, useState } from 'react'
import { Mic, Square, AlertCircle } from 'lucide-react'
import { startRecording } from '../../services/audio/recordWav'

const PERMISSION_FLAG = 'mic_permission_explained'
const MAX_SECONDS_KEY = 'mic_max_seconds'

const DURATION_OPTIONS = [10, 20, 30, 60]

export default function MicRecorder({ onRecording, disabled }) {
  const [state, setState] = useState('idle')      // idle | recording | error
  const [error, setError] = useState(null)
  const [level, setLevel] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [maxSeconds, setMaxSeconds] = useState(() => {
    const stored = parseInt(localStorage.getItem(MAX_SECONDS_KEY) || '20', 10)
    return DURATION_OPTIONS.includes(stored) ? stored : 20
  })
  const [showExplainer, setShowExplainer] = useState(
    () => !localStorage.getItem(PERMISSION_FLAG)
  )

  const handleRef = useRef(null)
  const tickRef = useRef(null)

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (handleRef.current && handleRef.current.getState() === 'recording') {
        handleRef.current.stop().catch(() => {})
      }
      if (tickRef.current) clearInterval(tickRef.current)
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(MAX_SECONDS_KEY, String(maxSeconds))
  }, [maxSeconds])

  const beginRecording = async () => {
    setError(null)
    try {
      // Must remain synchronous w.r.t. the user gesture for iOS Safari to unlock
      // the AudioContext.
      const handle = await startRecording({ maxSeconds })
      handleRef.current = handle
      setState('recording')
      setElapsed(0)
      setLevel(0)

      const unsub = handle.level$.subscribe(setLevel)

      tickRef.current = setInterval(() => {
        const dur = handle.getDurationSec()
        setElapsed(dur)
        if (handle.getState() !== 'recording') {
          // Hit max-seconds limit internally
          clearInterval(tickRef.current)
        }
      }, 100)

      // Capture auto-stop (max length reached)
      handle.stopped.then((result) => {
        unsub()
        clearInterval(tickRef.current)
        setState('idle')
        setLevel(0)
        if (result?.blob) {
          onRecording?.(result.blob, {
            sampleRate: result.sampleRate,
            durationSec: result.durationSec,
          })
        }
      })
    } catch (err) {
      console.error(err)
      let msg = err?.message || 'Could not start recording.'
      if (err?.name === 'NotAllowedError') {
        msg = 'Microphone permission was denied. Enable it in your browser settings.'
      } else if (err?.name === 'NotFoundError') {
        msg = 'No microphone detected on this device.'
      }
      setError(msg)
      setState('error')
    }
  }

  const handleRecordClick = async () => {
    if (state === 'recording') {
      await handleRef.current?.stop()
      return
    }
    if (showExplainer) return
    await beginRecording()
  }

  const continueFromExplainer = async () => {
    localStorage.setItem(PERMISSION_FLAG, '1')
    setShowExplainer(false)
    // Defer to next tick so the click-handler that triggered this still counts
    // as the activating gesture on iOS.
    setTimeout(() => beginRecording(), 0)
  }

  const recording = state === 'recording'
  const pct = Math.min(1, elapsed / maxSeconds)

  if (showExplainer) {
    return (
      <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-primary/30 p-5 sm:p-6">
        <div className="flex items-start gap-3 mb-3">
          <div className="w-10 h-10 rounded-full bg-biotuner-primary/15 flex items-center justify-center flex-shrink-0">
            <Mic className="w-5 h-5 text-biotuner-primary" />
          </div>
          <div>
            <h3 className="text-biotuner-light font-semibold text-base sm:text-lg">
              Why microphone access?
            </h3>
            <p className="text-biotuner-light/70 text-sm mt-1 leading-relaxed">
              Biotuner analyzes audio you record locally on your device. The
              recording stays on your phone unless you send it for analysis.
            </p>
          </div>
        </div>
        <div className="flex flex-col sm:flex-row gap-2 mt-4">
          <button
            onClick={continueFromExplainer}
            className="flex-1 px-4 py-3 rounded-lg bg-gradient-to-r from-biotuner-primary to-biotuner-secondary text-biotuner-dark-900 font-semibold min-h-[48px]"
          >
            Continue
          </button>
          <button
            onClick={() => {
              localStorage.setItem(PERMISSION_FLAG, '1')
              setShowExplainer(false)
            }}
            className="flex-1 px-4 py-3 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/80 min-h-[48px]"
          >
            Not now
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-5 sm:p-8">
      <div className="flex flex-col items-center gap-5">
        {/* Big circular record button */}
        <button
          onClick={handleRecordClick}
          disabled={disabled || state === 'error'}
          aria-label={recording ? 'Stop recording' : 'Start recording'}
          className={`
            relative w-28 h-28 sm:w-32 sm:h-32 rounded-full flex items-center justify-center
            transition-all duration-300 select-none
            ${recording
              ? 'bg-gradient-to-br from-red-500 to-red-700 shadow-lg shadow-red-500/40 scale-105'
              : 'bg-gradient-to-br from-biotuner-primary to-biotuner-secondary shadow-lg shadow-biotuner-primary/30 hover:scale-105'}
            ${(disabled || state === 'error') ? 'opacity-50 cursor-not-allowed' : ''}
          `}
        >
          {recording ? (
            <Square className="w-10 h-10 text-white" fill="currentColor" />
          ) : (
            <Mic className="w-10 h-10 text-biotuner-dark-900" />
          )}
          {recording && (
            <span className="absolute inset-0 rounded-full border-4 border-red-300/40 animate-ping" />
          )}
        </button>

        {/* VU meter */}
        <div className="w-full max-w-sm">
          <div className="h-3 rounded-full bg-biotuner-dark-800 overflow-hidden">
            <div
              className="h-full transition-[width] duration-75 bg-gradient-to-r from-biotuner-accent via-biotuner-primary to-red-400"
              style={{ width: `${Math.round(level * 100)}%` }}
            />
          </div>
        </div>

        {/* Duration */}
        <div className="text-center">
          <div className="text-3xl font-mono text-biotuner-primary tabular-nums">
            {elapsed.toFixed(1)}<span className="text-biotuner-light/40">s</span>
          </div>
          <div className="text-xs text-biotuner-light/40 uppercase tracking-wider mt-1">
            {recording ? 'recording' : 'ready'} · max {maxSeconds}s
          </div>
          {recording && (
            <div className="w-40 h-1 bg-biotuner-dark-800 rounded-full mt-2 overflow-hidden">
              <div
                className="h-full bg-biotuner-primary transition-all"
                style={{ width: `${pct * 100}%` }}
              />
            </div>
          )}
        </div>

        {/* Max-length selector */}
        {!recording && (
          <div className="flex items-center gap-2 flex-wrap justify-center">
            <span className="text-xs text-biotuner-light/40 uppercase tracking-wider">
              Length
            </span>
            {DURATION_OPTIONS.map((s) => (
              <button
                key={s}
                onClick={() => setMaxSeconds(s)}
                className={`
                  min-w-[48px] min-h-[40px] px-3 rounded-md text-sm font-medium transition-all
                  ${maxSeconds === s
                    ? 'bg-biotuner-primary text-biotuner-dark-900'
                    : 'bg-biotuner-dark-800 text-biotuner-light/70 border border-biotuner-dark-600 hover:border-biotuner-primary/50'}
                `}
              >
                {s}s
              </button>
            ))}
          </div>
        )}

        {error && (
          <div className="w-full max-w-sm flex items-start gap-2 p-3 rounded-md bg-red-900/20 border border-red-500/40 text-red-300 text-sm">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>
    </div>
  )
}
