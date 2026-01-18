import { useState, useEffect } from 'react'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import FileUpload from './components/FileUpload'
import ModalitySelector from './components/ModalitySelector'
import SignalPreview from './components/SignalPreview'
import TabsContainer from './components/TabsContainer'
import apiClient from './services/api'

function App() {
  // Global state
  const [sessionId, setSessionId] = useState(null)
  const [fileInfo, setFileInfo] = useState(null)
  const [analysisConfig, setAnalysisConfig] = useState({
    method: 'harmonic_recurrence',
    n_peaks: 5,
    precision: 0.1,
    max_freq: 100,
    tuning_method: 'peaks_ratios',
    max_denominator: 100,
    n_harm: 10,
  })
  const [analysisResult, setAnalysisResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [selectedModality, setSelectedModality] = useState(null)
  const [cropSettings, setCropSettings] = useState({ start_time: null, end_time: null })

  // WebSocket connection
  useEffect(() => {
    if (sessionId) {
      apiClient.connectWebSocket(sessionId, {
        onProgress: (data) => {
          console.log('Progress:', data)
        }
      })

      return () => {
        apiClient.closeWebSocket()
      }
    }
  }, [sessionId])

  // Handle file upload
  const handleFileUpload = async (file) => {
    try {
      setLoading(true)
      setError(null)
      const result = await apiClient.uploadFile(file, sessionId)
      setSessionId(result.session_id)
      setFileInfo(result)
      setLoading(false)
    } catch (error) {
      console.error('Upload error:', error)
      setError(error.response?.data?.detail || 'Failed to upload file. Please try again.')
      setLoading(false)
    }
  }

  // Handle analysis
  const handleAnalyze = async () => {
    if (!sessionId) return

    try {
      setLoading(true)
      setError(null)
      const result = await apiClient.analyze({
        session_id: sessionId,
        ...analysisConfig,
        ...cropSettings,  // Include crop settings
      })
      setAnalysisResult(result)
      setLoading(false)
    } catch (error) {
      console.error('Analysis error:', error)
      const errorMsg = error.response?.data?.detail || 'Analysis failed. Please check your parameters.'
      setError(errorMsg)
      setLoading(false)
    }
  }

  const handleCrop = (crop) => {
    setCropSettings(crop)
  }

  const handleColumnChange = async (colIndex) => {
    try {
      const result = await apiClient.selectColumn(sessionId, colIndex)
      // Update fileInfo with new preview data
      setFileInfo(prev => ({
        ...prev,
        preview_data: result.preview_data,
        data_points: result.data_points,
        duration: result.duration
      }))
    } catch (error) {
      console.error('Column change error:', error)
    }
  }

  return (
    <div className="min-h-screen bg-biotuner-dark-900 text-biotuner-light flex flex-col">
      {/* Header */}
      <Header />

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar
          config={analysisConfig}
          onConfigChange={setAnalysisConfig}
          fileInfo={fileInfo}
        />

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto bg-biotuner-dark-800">
          <div className="p-8 space-y-8">
            {/* Error Banner */}
            {error && (
              <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <div className="text-red-400 mt-0.5">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h3 className="text-red-400 font-semibold mb-1">Analysis Error</h3>
                    <p className="text-red-300/90 text-sm whitespace-pre-line">{error}</p>
                  </div>
                  <button
                    onClick={() => setError(null)}
                    className="text-red-400 hover:text-red-300 transition-colors"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                  </button>
                </div>
              </div>
            )}

            {/* Modality Selector */}
            <ModalitySelector
              selected={selectedModality}
              onSelect={setSelectedModality}
            />

            {/* File Upload */}
            <FileUpload
              onFileUpload={handleFileUpload}
              loading={loading}
              fileInfo={fileInfo}
            />

            {/* Signal Preview */}
            {fileInfo && (
              <SignalPreview
                sessionId={sessionId}
                fileInfo={fileInfo}
                onCrop={handleCrop}
                onColumnChange={handleColumnChange}
              />
            )}

            {/* Analysis Button */}
            {fileInfo && !analysisResult && (
              <div className="flex justify-center">
                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="relative group px-12 py-4 rounded-lg font-semibold text-lg tracking-wide overflow-hidden transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-biotuner-primary to-biotuner-secondary opacity-100 group-hover:opacity-90 transition-opacity"></div>
                  <div className="relative z-10 text-biotuner-dark-900">
                    {loading ? '⚡ ANALYZING...' : '▶ ANALYZE SIGNAL'}
                  </div>
                </button>
              </div>
            )}

            {/* Results Section with distinct background */}
            {analysisResult && (
              <div className="relative bg-gradient-to-br from-indigo-950/60 via-purple-950/50 to-slate-900/60 rounded-xl border-2 border-biotuner-primary/40 p-8 shadow-2xl backdrop-blur-sm">
                {/* Decorative corner accent */}
                <div className="absolute top-0 left-0 w-32 h-32 bg-gradient-to-br from-biotuner-primary/20 to-transparent rounded-tl-xl pointer-events-none"></div>
                <div className="absolute bottom-0 right-0 w-32 h-32 bg-gradient-to-tl from-biotuner-secondary/20 to-transparent rounded-br-xl pointer-events-none"></div>
                
                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <div className="w-3 h-3 bg-biotuner-primary rounded-full animate-pulse shadow-lg shadow-biotuner-primary/50"></div>
                      <h2 className="text-2xl font-bold text-biotuner-primary drop-shadow-lg">Analysis Results</h2>
                    </div>
                    <button
                      onClick={() => setAnalysisResult(null)}
                      className="relative px-4 py-2 rounded-lg bg-biotuner-dark-900/80 hover:bg-biotuner-dark-800 border border-biotuner-primary/40 text-sm transition-all hover:border-biotuner-primary/60 animate-moving-glow overflow-hidden group"
                    >
                      {/* Animated shine overlay */}
                      <div className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-1000 bg-gradient-to-r from-transparent via-white/10 to-transparent"></div>
                      <span className="relative z-10">← New Analysis</span>
                    </button>
                  </div>
                  <TabsContainer
                    sessionId={sessionId}
                    analysisResult={analysisResult}
                    analysisConfig={analysisConfig}
                    fileInfo={fileInfo}
                  />
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
