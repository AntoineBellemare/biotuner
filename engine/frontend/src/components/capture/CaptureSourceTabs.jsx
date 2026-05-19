import { useEffect, useState } from 'react'
import { Upload, Mic, Smartphone } from 'lucide-react'
import FileUpload from '../FileUpload'
import MicRecorder from './MicRecorder'
import SensorRecorder from './SensorRecorder'

const SOURCES = [
  { id: 'file',   label: 'File',   icon: Upload },
  { id: 'mic',    label: 'Mic',    icon: Mic },
  { id: 'sensor', label: 'Sensor', icon: Smartphone },
]

export default function CaptureSourceTabs({
  active,
  onActiveChange,
  onFileUpload,
  onRecording,
  loading,
  fileInfo,
}) {
  const [internalActive, setInternalActive] = useState(active || 'file')
  const current = active ?? internalActive

  useEffect(() => {
    if (active && active !== internalActive) setInternalActive(active)
  }, [active, internalActive])

  const setActive = (id) => {
    setInternalActive(id)
    onActiveChange?.(id)
  }

  return (
    <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-4 sm:p-6 lg:p-8">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
        <h2 className="text-xs font-semibold text-biotuner-light/60 uppercase tracking-wider">
          Capture Source
        </h2>

        <div className="flex gap-1 p-1 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600">
          {SOURCES.map((s) => {
            const Icon = s.icon
            const isActive = current === s.id
            return (
              <button
                key={s.id}
                onClick={() => setActive(s.id)}
                aria-pressed={isActive}
                className={`
                  flex items-center gap-1.5 px-3 sm:px-4 py-2 rounded-md text-sm font-medium
                  min-h-[40px] min-w-[64px] justify-center transition-all
                  ${isActive
                    ? 'bg-biotuner-primary text-biotuner-dark-900 shadow'
                    : 'text-biotuner-light/70 hover:text-biotuner-light'}
                `}
              >
                <Icon className="w-4 h-4" />
                <span>{s.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      <div>
        {current === 'file' && (
          <FileUpload
            onFileUpload={onFileUpload}
            loading={loading}
            fileInfo={fileInfo}
            embedded
          />
        )}
        {current === 'mic' && (
          <MicRecorder
            onRecording={(blob, meta) => onRecording?.(blob, { ...meta, source: 'mic' })}
            disabled={loading}
          />
        )}
        {current === 'sensor' && (
          <SensorRecorder
            onRecording={(blob, meta) => onRecording?.(blob, { ...meta, source: 'sensor' })}
            disabled={loading}
          />
        )}
      </div>
    </div>
  )
}
