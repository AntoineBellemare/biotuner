import { useEffect, useRef, useState } from 'react'
import { X, Download, Upload, Trash2, Play, Bookmark, FileAudio } from 'lucide-react'
import { library } from '../../services/storage'
import apiClient from '../../services/api'
import { exportTuning as exportTuningLocal } from '../../services/tuningExport'

export default function LibraryDrawer({ open, onClose, onLoadTuning, onLoadRecording }) {
  const [tunings, setTunings] = useState([])
  const [recordings, setRecordings] = useState([])
  const [tab, setTab] = useState('tunings')
  const [busy, setBusy] = useState(false)
  const importInput = useRef(null)

  const refresh = async () => {
    setTunings(await library.listTunings())
    setRecordings(await library.listRecordings())
  }

  useEffect(() => {
    if (open) refresh()
  }, [open])

  const handleDeleteTuning = async (id) => {
    await library.deleteTuning(id)
    refresh()
  }

  const handleDeleteRecording = async (id) => {
    await library.deleteRecording(id)
    refresh()
  }

  const handleExportScl = (t) => {
    exportTuningLocal('scl', t.ratios || [], {
      description: t.name || 'Biotuner tuning',
      filename: sanitize(t.name),
    })
  }

  const handleExportJson = (t) => {
    const blob = new Blob([JSON.stringify(t, null, 2)], { type: 'application/json' })
    apiClient.downloadBlob(blob, `${sanitize(t.name)}.json`)
  }

  const handleExportAll = async () => {
    setBusy(true)
    try {
      const bundle = await library.exportAll()
      const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: 'application/json' })
      apiClient.downloadBlob(blob, `biotuner_library_${Date.now()}.json`)
    } finally {
      setBusy(false)
    }
  }

  const handleImport = (file) => {
    if (!file) return
    const reader = new FileReader()
    reader.onload = async () => {
      try {
        const json = JSON.parse(String(reader.result))
        await library.importAll(json, { mode: 'merge' })
        refresh()
      } catch (err) {
        console.error('Import failed:', err)
        alert('Import failed — file is not a valid biotuner library bundle.')
      }
    }
    reader.readAsText(file)
  }

  const handleLoadRecording = async (meta) => {
    const full = await library.getRecording(meta.id)
    if (!full?.blob) return
    onLoadRecording?.(full)
    onClose?.()
  }

  return (
    <>
      {open && (
        <div className="fixed inset-0 bg-black/60 z-40" onClick={onClose} />
      )}
      <aside
        className={`
          fixed top-0 right-0 z-50 h-full w-full sm:w-[420px] bg-biotuner-dark-900
          border-l border-biotuner-dark-600 shadow-2xl flex flex-col
          transform transition-transform duration-300 ease-in-out
          ${open ? 'translate-x-0' : 'translate-x-full'}
        `}
        aria-hidden={!open}
      >
        <div className="flex items-center justify-between p-4 border-b border-biotuner-dark-600">
          <h2 className="text-lg font-semibold text-biotuner-light flex items-center gap-2">
            <Bookmark className="w-5 h-5 text-biotuner-primary" />
            Library
          </h2>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-biotuner-dark-800" aria-label="Close">
            <X className="w-5 h-5 text-biotuner-light/60" />
          </button>
        </div>

        {/* Tab switcher */}
        <div className="flex border-b border-biotuner-dark-600">
          <button
            onClick={() => setTab('tunings')}
            className={`flex-1 py-3 text-sm font-medium uppercase tracking-wider transition-colors ${
              tab === 'tunings' ? 'text-biotuner-primary border-b-2 border-biotuner-primary' : 'text-biotuner-light/40'
            }`}
          >
            Tunings ({tunings.length})
          </button>
          <button
            onClick={() => setTab('recordings')}
            className={`flex-1 py-3 text-sm font-medium uppercase tracking-wider transition-colors ${
              tab === 'recordings' ? 'text-biotuner-primary border-b-2 border-biotuner-primary' : 'text-biotuner-light/40'
            }`}
          >
            Recordings ({recordings.length})
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {tab === 'tunings' && (
            tunings.length === 0 ? (
              <EmptyState message="No tunings saved yet." />
            ) : tunings.map((t) => (
              <div key={t.id} className="bg-biotuner-dark-800/70 border border-biotuner-dark-600 rounded-lg p-3">
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-biotuner-light truncate">{t.name}</div>
                    <div className="text-xs text-biotuner-light/40 mt-0.5">
                      {new Date(t.createdAt).toLocaleString()} · {t.modality || 'tuning'} · {t.ratios?.length || 0} notes
                    </div>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 mt-3">
                  <button
                    onClick={() => { onLoadTuning?.(t); onClose?.() }}
                    className="min-h-[36px] flex items-center gap-1 px-3 py-1.5 rounded-md bg-biotuner-primary/15 border border-biotuner-primary/40 text-biotuner-primary text-xs"
                  >
                    <Play className="w-3 h-3" /> Load
                  </button>
                  <button
                    onClick={() => handleExportScl(t)}
                    className="min-h-[36px] flex items-center gap-1 px-3 py-1.5 rounded-md bg-biotuner-dark-900 border border-biotuner-dark-600 text-xs text-biotuner-light/80"
                  >
                    <Download className="w-3 h-3" /> .scl
                  </button>
                  <button
                    onClick={() => handleExportJson(t)}
                    className="min-h-[36px] flex items-center gap-1 px-3 py-1.5 rounded-md bg-biotuner-dark-900 border border-biotuner-dark-600 text-xs text-biotuner-light/80"
                  >
                    <Download className="w-3 h-3" /> JSON
                  </button>
                  <button
                    onClick={() => handleDeleteTuning(t.id)}
                    className="min-h-[36px] flex items-center gap-1 px-3 py-1.5 rounded-md bg-red-900/20 border border-red-500/30 text-red-300 text-xs ml-auto"
                  >
                    <Trash2 className="w-3 h-3" /> Delete
                  </button>
                </div>
              </div>
            ))
          )}

          {tab === 'recordings' && (
            recordings.length === 0 ? (
              <EmptyState message="No recordings saved yet." />
            ) : recordings.map((r) => (
              <div key={r.id} className="bg-biotuner-dark-800/70 border border-biotuner-dark-600 rounded-lg p-3">
                <div className="flex items-start gap-2">
                  <FileAudio className="w-4 h-4 text-biotuner-primary mt-0.5" />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm font-medium text-biotuner-light truncate">{r.name}</div>
                    <div className="text-xs text-biotuner-light/40 mt-0.5">
                      {new Date(r.createdAt).toLocaleString()} · {r.modality} · {r.durationSec?.toFixed?.(1) || '?'}s
                    </div>
                  </div>
                </div>
                <div className="flex flex-wrap gap-2 mt-3">
                  <button
                    onClick={() => handleLoadRecording(r)}
                    className="min-h-[36px] flex items-center gap-1 px-3 py-1.5 rounded-md bg-biotuner-primary/15 border border-biotuner-primary/40 text-biotuner-primary text-xs"
                  >
                    <Play className="w-3 h-3" /> Load
                  </button>
                  <button
                    onClick={() => handleDeleteRecording(r.id)}
                    className="min-h-[36px] flex items-center gap-1 px-3 py-1.5 rounded-md bg-red-900/20 border border-red-500/30 text-red-300 text-xs ml-auto"
                  >
                    <Trash2 className="w-3 h-3" /> Delete
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        <div className="p-3 border-t border-biotuner-dark-600 flex gap-2">
          <button
            onClick={handleExportAll}
            disabled={busy}
            className="flex-1 min-h-[44px] flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-biotuner-dark-800 border border-biotuner-dark-600 text-sm text-biotuner-light/80"
          >
            <Download className="w-4 h-4" /> Export library
          </button>
          <button
            onClick={() => importInput.current?.click()}
            className="flex-1 min-h-[44px] flex items-center justify-center gap-2 px-3 py-2 rounded-md bg-biotuner-dark-800 border border-biotuner-dark-600 text-sm text-biotuner-light/80"
          >
            <Upload className="w-4 h-4" /> Import
          </button>
          <input
            ref={importInput}
            type="file"
            accept="application/json,.json"
            onChange={(e) => handleImport(e.target.files?.[0])}
            className="hidden"
          />
        </div>
      </aside>
    </>
  )
}

function EmptyState({ message }) {
  return (
    <div className="text-center text-biotuner-light/40 text-sm py-12">
      {message}
    </div>
  )
}

function sanitize(name) {
  return (name || 'biotuner_tuning')
    .replace(/[^a-zA-Z0-9_-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 64) || 'biotuner_tuning'
}
