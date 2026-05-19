import { useEffect, useRef, useState } from 'react'
import { Smartphone, Square, AlertCircle } from 'lucide-react'
import { startMotion, isMagnetometerAvailable, isIos } from '../../services/sensors/motionRecorder'

const PERMISSION_FLAG = 'motion_permission_explained'
const MAX_SECONDS_KEY = 'motion_max_seconds'
const SELECTED_SENSORS_KEY = 'motion_selected_sensors'

const DURATION_OPTIONS = [10, 20, 30, 60]

const SENSORS = [
  { id: 'accel',       label: 'Accelerometer', desc: 'x/y/z motion' },
  { id: 'gyro',        label: 'Gyroscope',     desc: 'rotation rate' },
  { id: 'orientation', label: 'Orientation',   desc: 'tilt α/β/γ' },
  { id: 'mag',         label: 'Magnetometer',  desc: 'raw 3-axis' },
]

export default function SensorRecorder({ onRecording, disabled }) {
  const [state, setState] = useState('idle')
  const [error, setError] = useState(null)
  const [elapsed, setElapsed] = useState(0)
  const [latest, setLatest] = useState(null)
  const [maxSeconds, setMaxSeconds] = useState(() => {
    const stored = parseInt(localStorage.getItem(MAX_SECONDS_KEY) || '20', 10)
    return DURATION_OPTIONS.includes(stored) ? stored : 20
  })
  const [showExplainer, setShowExplainer] = useState(
    () => !localStorage.getItem(PERMISSION_FLAG)
  )
  const [selected, setSelected] = useState(() => {
    try {
      const raw = localStorage.getItem(SELECTED_SENSORS_KEY)
      if (raw) {
        const parsed = JSON.parse(raw)
        if (Array.isArray(parsed) && parsed.length) return parsed
      }
    } catch { /* fall through */ }
    return ['accel']
  })

  const handleRef = useRef(null)
  const tickRef = useRef(null)
  const magAvailable = isMagnetometerAvailable()
  const ios = isIos()

  useEffect(() => {
    return () => {
      if (handleRef.current?.getState() === 'recording') {
        handleRef.current.stop().catch(() => {})
      }
      if (tickRef.current) clearInterval(tickRef.current)
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(MAX_SECONDS_KEY, String(maxSeconds))
  }, [maxSeconds])

  useEffect(() => {
    localStorage.setItem(SELECTED_SENSORS_KEY, JSON.stringify(selected))
  }, [selected])

  const toggleSensor = (id) => {
    setSelected((cur) => {
      if (cur.includes(id)) {
        return cur.length > 1 ? cur.filter((x) => x !== id) : cur
      }
      return [...cur, id]
    })
  }

  const begin = async () => {
    setError(null)
    try {
      const handle = await startMotion({
        sensors: selected,
        hz: 50,
        maxSeconds,
      })
      handleRef.current = handle
      setState('recording')
      setElapsed(0)

      const unsub = handle.samples$.subscribe(({ latest }) => setLatest(latest))

      tickRef.current = setInterval(() => {
        setElapsed(handle.getDurationSec())
      }, 100)

      handle.stopped.then((result) => {
        unsub()
        clearInterval(tickRef.current)
        setState('idle')
        if (result?.blob) {
          onRecording?.(result.blob, {
            sampleRate: result.sampleRate,
            durationSec: result.durationSec,
            columns: result.columns,
          })
        }
      })
    } catch (err) {
      console.error(err)
      setError(err?.message || 'Could not start sensor capture.')
      setState('error')
    }
  }

  const onClick = async () => {
    if (state === 'recording') {
      await handleRef.current?.stop()
      return
    }
    if (showExplainer) return
    await begin()
  }

  const continueFromExplainer = () => {
    localStorage.setItem(PERMISSION_FLAG, '1')
    setShowExplainer(false)
    setTimeout(() => begin(), 0)
  }

  const recording = state === 'recording'
  const pct = Math.min(1, elapsed / maxSeconds)

  if (showExplainer) {
    return (
      <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-primary/30 p-5 sm:p-6">
        <div className="flex items-start gap-3 mb-3">
          <div className="w-10 h-10 rounded-full bg-biotuner-primary/15 flex items-center justify-center flex-shrink-0">
            <Smartphone className="w-5 h-5 text-biotuner-primary" />
          </div>
          <div>
            <h3 className="text-biotuner-light font-semibold text-base sm:text-lg">
              Why motion sensor access?
            </h3>
            <p className="text-biotuner-light/70 text-sm mt-1 leading-relaxed">
              Biotuner reads accelerometer, gyroscope, and orientation streams
              from your phone, locally. Data stays on your device unless you
              send it for analysis.
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
        {/* Sensor toggle row */}
        {!recording && (
          <div className="w-full grid grid-cols-2 gap-2">
            {SENSORS.map((s) => {
              const isMag = s.id === 'mag'
              const disabledBecauseIos = isMag && !magAvailable
              const isOn = selected.includes(s.id)
              return (
                <button
                  key={s.id}
                  onClick={() => !disabledBecauseIos && toggleSensor(s.id)}
                  disabled={disabledBecauseIos}
                  title={disabledBecauseIos
                    ? 'Raw magnetometer is not exposed on iOS.'
                    : s.desc}
                  className={`
                    text-left px-3 py-3 min-h-[56px] rounded-lg border transition-all
                    ${disabledBecauseIos
                      ? 'border-biotuner-dark-600 bg-biotuner-dark-800/50 opacity-50 cursor-not-allowed'
                      : isOn
                        ? 'border-biotuner-primary bg-biotuner-primary/10'
                        : 'border-biotuner-dark-600 bg-biotuner-dark-800/50 hover:border-biotuner-primary/50'
                    }
                  `}
                >
                  <div className="text-sm font-medium text-biotuner-light">{s.label}</div>
                  <div className="text-xs text-biotuner-light/50">
                    {disabledBecauseIos ? 'unavailable on iOS' : s.desc}
                  </div>
                </button>
              )
            })}
          </div>
        )}

        {/* Big circular record button */}
        <button
          onClick={onClick}
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
            <Smartphone className="w-10 h-10 text-biotuner-dark-900" />
          )}
          {recording && (
            <span className="absolute inset-0 rounded-full border-4 border-red-300/40 animate-ping" />
          )}
        </button>

        {/* Duration */}
        <div className="text-center">
          <div className="text-3xl font-mono text-biotuner-primary tabular-nums">
            {elapsed.toFixed(1)}<span className="text-biotuner-light/40">s</span>
          </div>
          <div className="text-xs text-biotuner-light/40 uppercase tracking-wider mt-1">
            {recording ? 'capturing' : 'ready'} · max {maxSeconds}s
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

        {/* Live readout */}
        {recording && latest && (
          <div className="w-full max-w-md grid grid-cols-3 gap-2 text-center text-xs font-mono">
            {selected.includes('accel') && (
              <>
                <div className="bg-biotuner-dark-800 rounded p-2">
                  <div className="text-biotuner-light/40">ax</div>
                  <div className="text-biotuner-primary">{latest.accel.x.toFixed(2)}</div>
                </div>
                <div className="bg-biotuner-dark-800 rounded p-2">
                  <div className="text-biotuner-light/40">ay</div>
                  <div className="text-biotuner-primary">{latest.accel.y.toFixed(2)}</div>
                </div>
                <div className="bg-biotuner-dark-800 rounded p-2">
                  <div className="text-biotuner-light/40">az</div>
                  <div className="text-biotuner-primary">{latest.accel.z.toFixed(2)}</div>
                </div>
              </>
            )}
          </div>
        )}

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

        {ios && selected.includes('mag') === false && magAvailable === false && (
          <p className="text-xs text-biotuner-light/40 text-center max-w-sm">
            On iOS, raw 3-axis magnetometer data isn't exposed. Use orientation
            or compass-derived heading instead.
          </p>
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
