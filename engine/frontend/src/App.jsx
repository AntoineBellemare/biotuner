import { useEffect, useMemo, useState } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import ModalitySelector from './components/ModalitySelector'
import SignalPreview from './components/SignalPreview'
import TabsContainer from './components/TabsContainer'
import CaptureSourceTabs from './components/capture/CaptureSourceTabs'
import LibraryDrawer from './components/library/LibraryDrawer'
import SaveTuningDialog from './components/library/SaveTuningDialog'
import apiClient from './services/api'
import { ANALYSIS_DEFAULTS, configMatchesPreset, getPreset, presetKey } from './services/presets'
import { library, prefs } from './services/storage'
import { suggestionsFor } from './services/errorHints'

const MODALITY_TO_DEFAULT_SOURCE = {
  audio:    'mic',
  brain:    'file',
  heart:    'file',
  sensors:  'sensor',
  plant:    'file',
  creative: 'file',
}

function App() {
  // Global state
  const [sessionId, setSessionId] = useState(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [libraryOpen, setLibraryOpen] = useState(false)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)
  const [fileInfo, setFileInfo] = useState(null)

  const [selectedModality, setSelectedModality] = useState(() => prefs.get('modality', null))
  const [captureSource, setCaptureSource] = useState(() => prefs.get('captureSource', 'file'))
  const [presetMode, setPresetMode] = useState(() => prefs.get('presetMode', 'auto')) // 'auto' | 'custom'

  const [analysisConfig, setAnalysisConfig] = useState(() => {
    const stored = prefs.get('customConfig', null)
    if (stored) return stored
    return ANALYSIS_DEFAULTS
  })

  const [analysisResult, setAnalysisResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [cropSettings, setCropSettings] = useState({ start_time: null, end_time: null })

  // Track the last-uploaded blob so we can save it to the recordings library.
  const [lastRecording, setLastRecording] = useState(null)

  // ---------------------------------------------------------------------------
  // Presets: when modality / source changes, snap config to that preset unless
  // the user has explicitly chosen "Custom".
  // ---------------------------------------------------------------------------
  const activePresetKey = useMemo(
    () => presetKey(selectedModality, captureSource),
    [selectedModality, captureSource]
  )

  useEffect(() => {
    if (presetMode !== 'auto') return
    if (!selectedModality) return
    const preset = getPreset(selectedModality, captureSource)
    setAnalysisConfig((prev) => ({ ...prev, ...preset }))
  }, [activePresetKey, presetMode, selectedModality, captureSource])

  // Persist user choices
  useEffect(() => { prefs.set('modality', selectedModality) }, [selectedModality])
  useEffect(() => { prefs.set('captureSource', captureSource) }, [captureSource])
  useEffect(() => { prefs.set('presetMode', presetMode) }, [presetMode])
  useEffect(() => {
    if (presetMode === 'custom') prefs.set('customConfig', analysisConfig)
  }, [analysisConfig, presetMode])

  // Sidebar config-change handler: any manual tweak flips to "custom".
  const handleConfigChange = (next) => {
    setAnalysisConfig(next)
    if (presetMode !== 'custom') {
      const preset = getPreset(selectedModality, captureSource)
      if (!configMatchesPreset(next, preset)) setPresetMode('custom')
    }
  }

  // ---------------------------------------------------------------------------
  // WebSocket
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (sessionId) {
      apiClient.connectWebSocket(sessionId, {
        onProgress: (data) => { console.log('Progress:', data) },
      })
      return () => { apiClient.closeWebSocket() }
    }
  }, [sessionId])

  // ---------------------------------------------------------------------------
  // Uploads
  // ---------------------------------------------------------------------------
  const handleFileUpload = async (file, meta = {}) => {
    try {
      setLoading(true)
      setError(null)
      const result = await apiClient.uploadFile(file, sessionId)
      setSessionId(result.session_id)
      setFileInfo(result)
      setLoading(false)

      // Stash for save-to-library
      setLastRecording({
        blob: file,
        name: file.name || 'recording',
        mimeType: file.type || (meta.source === 'sensor' ? 'text/csv' : 'audio/wav'),
        modality: `${selectedModality || 'data'}.${captureSource || 'file'}`,
        durationSec: meta.durationSec,
        sampleRate: meta.sampleRate,
      })
    } catch (error) {
      console.error('Upload error:', error)
      setError(error.response?.data?.detail || 'Failed to upload file. Please try again.')
      setLoading(false)
    }
  }

  // Recordings (mic / sensor) wrap into a File and reuse the upload path.
  const handleRecording = async (blob, meta = {}) => {
    const ext = blob.type?.startsWith('audio/') ? 'wav' : 'csv'
    const stamp = new Date().toISOString().replace(/[:.]/g, '-')
    const filename = `${meta.source || 'capture'}-${stamp}.${ext}`
    const file = new File([blob], filename, { type: blob.type })
    await handleFileUpload(file, meta)
  }

  // ---------------------------------------------------------------------------
  // Analysis
  // ---------------------------------------------------------------------------
  const handleAnalyze = async () => {
    if (!sessionId) return
    try {
      setLoading(true)
      setError(null)
      const result = await apiClient.analyze({
        session_id: sessionId,
        ...analysisConfig,
        ...cropSettings,
      })
      setAnalysisResult(result)
      setLoading(false)
    } catch (error) {
      console.error('Analysis error:', error)
      setError(error.response?.data?.detail || 'Analysis failed. Please check your parameters.')
      setLoading(false)
    }
  }

  const handleCrop = (crop) => setCropSettings(crop)

  const handleColumnChange = async (colIndex) => {
    try {
      const result = await apiClient.selectColumn(sessionId, colIndex)
      setFileInfo((prev) => ({
        ...prev,
        preview_data: result.preview_data,
        data_points: result.data_points,
        duration: result.duration,
      }))
    } catch (err) {
      console.error('Column change error:', err)
    }
  }

  // ---------------------------------------------------------------------------
  // Modality selection: default the capture source for that modality.
  // ---------------------------------------------------------------------------
  const handleSelectModality = (id) => {
    setSelectedModality(id)
    if (presetMode === 'custom') setPresetMode('auto')
    const next = MODALITY_TO_DEFAULT_SOURCE[id]
    if (next) setCaptureSource(next)
  }

  // ---------------------------------------------------------------------------
  // Library actions
  // ---------------------------------------------------------------------------
  const handleSaveTuning = (extras = {}) => {
    if (!analysisResult) return
    setSaveDialogOpen(true)
    // We use a state trick: the dialog reads `saveDefaults` from current values.
  }

  const handleSaveDialogConfirm = async ({ name, notes }) => {
    setSaveDialogOpen(false)
    if (!analysisResult) return
    let recordingId
    if (lastRecording?.blob) {
      try {
        recordingId = await library.saveRecording({
          name: lastRecording.name,
          modality: lastRecording.modality,
          mimeType: lastRecording.mimeType,
          blob: lastRecording.blob,
          durationSec: lastRecording.durationSec,
          sampleRate: lastRecording.sampleRate,
        })
      } catch (err) {
        console.warn('Could not persist recording blob:', err)
      }
    }
    await library.saveTuning({
      kind: 'plain',
      name,
      notes,
      modality: `${selectedModality || 'data'}.${captureSource || 'file'}`,
      preset: activePresetKey,
      ratios: analysisResult.tuning || [],
      peaks: analysisResult.peaks || [],
      powers: analysisResult.powers || [],
      metrics: analysisResult.metrics || {},
      derivedFrom: recordingId ? { recordingId } : undefined,
    })
  }

  const handleLoadTuning = (saved) => {
    setAnalysisResult({
      tuning: saved.ratios || [],
      peaks: saved.peaks || [],
      powers: saved.powers || [],
      metrics: saved.metrics || {},
    })
  }

  const handleLoadRecording = async (rec) => {
    if (!rec?.blob) return
    const file = new File([rec.blob], rec.name || 'recording', { type: rec.mimeType })
    await handleFileUpload(file, {
      durationSec: rec.durationSec,
      sampleRate: rec.sampleRate,
    })
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-biotuner-dark-900 text-biotuner-light flex flex-col">
      <Header
        onMenuToggle={() => setSidebarOpen(!sidebarOpen)}
        onLibraryToggle={() => setLibraryOpen(true)}
      />

      <div className="flex flex-1 overflow-hidden relative">
        {sidebarOpen && (
          <div
            className="fixed inset-0 bg-black/60 z-40 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
        )}

        <div className={`
          fixed lg:relative inset-y-0 left-0 z-50 transform transition-transform duration-300 ease-in-out
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}>
          <Sidebar
            config={analysisConfig}
            onConfigChange={handleConfigChange}
            fileInfo={fileInfo}
            onClose={() => setSidebarOpen(false)}
            presetMode={presetMode}
            onPresetModeChange={setPresetMode}
            activePresetKey={activePresetKey}
            onResetPreset={() => {
              setPresetMode('auto')
              const p = getPreset(selectedModality, captureSource)
              setAnalysisConfig((prev) => ({ ...prev, ...p }))
            }}
          />
        </div>

        <main className="flex-1 overflow-y-auto bg-biotuner-dark-800">
          <div className="p-4 sm:p-6 lg:p-8 space-y-6 sm:space-y-8">
            {error && (() => {
              const hints = suggestionsFor(error, analysisConfig)
              const applyHint = (patch) => {
                const next = { ...analysisConfig, ...patch }
                handleConfigChange(next)
                setError(null)
                // Auto-retry if we already have a session — the whole point
                // of a suggestion chip is to act, not just to set state.
                if (sessionId) {
                  setTimeout(() => {
                    // run analyze with the patched config (state may not have
                    // flushed yet; pass the patched values directly).
                    apiClient
                      .analyze({ session_id: sessionId, ...next, ...cropSettings })
                      .then((result) => setAnalysisResult(result))
                      .catch((err) => setError(err.response?.data?.detail || 'Analysis failed.'))
                  }, 0)
                }
              }
              return (
                <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className="text-red-400 mt-0.5">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <h3 className="text-red-400 font-semibold mb-1">Error</h3>
                      <p className="text-red-300/90 text-sm whitespace-pre-line">{error}</p>
                      {hints.length > 0 && (
                        <div className="mt-3 flex flex-wrap gap-2">
                          <span className="text-xs uppercase tracking-wider text-red-300/60 self-center">
                            Try:
                          </span>
                          {hints.map((h) => (
                            <button
                              key={h.label}
                              onClick={() => applyHint(h.patch)}
                              className="min-h-[36px] px-3 py-1.5 rounded-md bg-red-500/10 hover:bg-red-500/20 border border-red-500/40 text-red-200 text-xs font-medium transition-colors"
                            >
                              {h.label}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                    <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300 transition-colors">
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    </button>
                  </div>
                </div>
              )
            })()}

            <ModalitySelector
              selected={selectedModality}
              onSelect={handleSelectModality}
            />

            <CaptureSourceTabs
              active={captureSource}
              onActiveChange={setCaptureSource}
              onFileUpload={handleFileUpload}
              onRecording={handleRecording}
              loading={loading}
              fileInfo={fileInfo}
            />

            {fileInfo && (
              <SignalPreview
                sessionId={sessionId}
                fileInfo={fileInfo}
                onCrop={handleCrop}
                onColumnChange={handleColumnChange}
              />
            )}

            {fileInfo && !analysisResult && (
              <div className="flex justify-center">
                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="relative group px-8 sm:px-12 py-3 sm:py-4 rounded-lg font-semibold text-base sm:text-lg tracking-wide overflow-hidden transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed w-full sm:w-auto max-w-sm"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-biotuner-primary to-biotuner-secondary opacity-100 group-hover:opacity-90 transition-opacity"></div>
                  <div className="relative z-10 text-biotuner-dark-900">
                    {loading ? '⚡ ANALYZING...' : '▶ ANALYZE SIGNAL'}
                  </div>
                </button>
              </div>
            )}

            {analysisResult && (
              <div className="relative bg-gradient-to-br from-indigo-950/60 via-purple-950/50 to-slate-900/60 rounded-xl border-2 border-biotuner-primary/40 p-4 sm:p-6 lg:p-8 shadow-2xl backdrop-blur-sm">
                <div className="absolute top-0 left-0 w-16 sm:w-32 h-16 sm:h-32 bg-gradient-to-br from-biotuner-primary/20 to-transparent rounded-tl-xl pointer-events-none"></div>
                <div className="absolute bottom-0 right-0 w-16 sm:w-32 h-16 sm:h-32 bg-gradient-to-tl from-biotuner-secondary/20 to-transparent rounded-br-xl pointer-events-none"></div>

                <div className="relative z-10">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-biotuner-primary rounded-full animate-pulse shadow-lg shadow-biotuner-primary/50"></div>
                      <h2 className="text-xl sm:text-2xl font-bold text-biotuner-primary drop-shadow-lg">Analysis Results</h2>
                    </div>
                    <div className="flex flex-wrap gap-2 w-full sm:w-auto">
                      <button
                        onClick={() => setSaveDialogOpen(true)}
                        className="px-4 py-2 rounded-lg bg-biotuner-primary/15 border border-biotuner-primary/40 text-biotuner-primary text-sm min-h-[44px]"
                      >
                        Save tuning
                      </button>
                      <button
                        onClick={handleAnalyze}
                        disabled={loading}
                        title="Re-run analysis with the current sidebar settings, keeping the same recording"
                        className="px-4 py-2 rounded-lg bg-biotuner-dark-900/80 hover:bg-biotuner-dark-800 border border-biotuner-accent/40 text-biotuner-accent text-sm min-h-[44px] disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {loading ? '⏳ Re-analyzing…' : '↻ Re-analyze'}
                      </button>
                      <button
                        onClick={() => setAnalysisResult(null)}
                        className="px-4 py-2 rounded-lg bg-biotuner-dark-900/80 hover:bg-biotuner-dark-800 border border-biotuner-primary/40 text-sm min-h-[44px]"
                      >
                        ← New Analysis
                      </button>
                    </div>
                  </div>
                  <TabsContainer
                    sessionId={sessionId}
                    analysisResult={analysisResult}
                    analysisConfig={analysisConfig}
                    fileInfo={fileInfo}
                    onSaveTuning={handleSaveTuning}
                  />
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      <LibraryDrawer
        open={libraryOpen}
        onClose={() => setLibraryOpen(false)}
        onLoadTuning={handleLoadTuning}
        onLoadRecording={handleLoadRecording}
      />

      <SaveTuningDialog
        open={saveDialogOpen}
        onClose={() => setSaveDialogOpen(false)}
        onSave={handleSaveDialogConfirm}
        defaults={{
          modality: selectedModality ? `${selectedModality}.${captureSource}` : 'tuning',
        }}
      />
    </div>
  )
}

export default App
