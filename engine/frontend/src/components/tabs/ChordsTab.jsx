import { useState, useEffect } from 'react'
import { Play, Download, Music } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, BarChart, Bar, Cell } from 'recharts'
import apiClient from '../../services/api'

export default function ChordsTab({ sessionId, analysisResult, analysisConfig, fileInfo }) {
  const [chordConfig, setChordConfig] = useState({
    n_segments: 24,
    n_peaks: 5,
    time_resolution: 10,
    frequency_resolution: 1000,
    prominence: 0.5,
    n_oct_up: 7
  })
  const [chords, setChords] = useState(null)
  const [loading, setLoading] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)
  const [showScore, setShowScore] = useState(false)

  // Load and render MusicXML score using Verovio
  useEffect(() => {
    if (showScore && chords) {
      const loadScore = async () => {
        const container = document.getElementById('score-container')
        if (!container) return
        
        try {
          container.innerHTML = '<div class="text-center text-gray-600 py-8"><p>Loading score...</p></div>'
          
          // Fetch MusicXML from backend
          const response = await apiClient.client.post('/api/export-musicxml', {
            session_id: sessionId,
            chords: chords.chords,
            bound_times: chords.bound_times,
            total_duration: fileInfo?.duration,
            n_oct_up: 7
          }, {
            responseType: 'text'
          })
          
          const xmlText = response.data
          
          // Load Verovio dynamically if not already loaded
          if (!window.verovio) {
            await new Promise((resolve, reject) => {
              const script = document.createElement('script')
              script.src = 'https://www.verovio.org/javascript/latest/verovio-toolkit-wasm.js'
              script.onload = resolve
              script.onerror = reject
              document.head.appendChild(script)
            })
            
            // Wait for module to initialize
            await new Promise(resolve => setTimeout(resolve, 500))
          }
          
          await renderScore(xmlText)
        } catch (error) {
          console.error('Error loading score:', error)
          container.innerHTML = `<div class="text-center text-red-600 py-8"><p class="font-bold">Error loading score</p><p class="text-sm">${error.message}</p></div>`
        }
      }
      
      const renderScore = async (xmlText) => {
        try {
          // Initialize Verovio toolkit with WASM
          const vrvToolkit = new window.verovio.toolkit()
          
          // Set rendering options
          const options = {
            pageHeight: 2000,
            pageWidth: 1400,
            scale: 40,
            adjustPageHeight: true
          }
          
          // Load data and render
          vrvToolkit.setOptions(options)
          vrvToolkit.loadData(xmlText)
          const svg = vrvToolkit.renderToSVG(1, {})
          
          const container = document.getElementById('score-container')
          if (container) {
            container.innerHTML = svg
          }
        } catch (error) {
          console.error('Error rendering score:', error)
          const container = document.getElementById('score-container')
          if (container) {
            container.innerHTML = `<div class="text-center text-orange-600 py-8">
              <p class="font-bold mb-2">Score rendering unavailable</p>
              <p class="text-sm">Try downloading the MusicXML file instead</p>
            </div>`
          }
        }
      }
      
      loadScore()
    }
  }, [showScore, chords, sessionId, fileInfo])

  // Generate chords
  const handleGenerateChords = async () => {
    try {
      setLoading(true)
      const result = await apiClient.generateChords({
        session_id: sessionId,
        method: analysisConfig?.method || 'harmonic_recurrence',
        ...chordConfig,
      })
      console.log(`Received ${result.chords.length} chords, ${result.bound_times?.length} boundaries`)
      console.log('Bound times:', result.bound_times)
      
      // Validate bound_times for NaN values
      const nanCount = result.bound_times?.filter(t => isNaN(t) || t == null).length || 0
      if (nanCount > 0) {
        console.error(`WARNING: ${nanCount} NaN/null values in bound_times out of ${result.bound_times?.length}`)
        console.error('Invalid boundaries at indices:', 
          result.bound_times?.map((t, i) => isNaN(t) || t == null ? i : null).filter(i => i !== null))
      } else {
        console.log('✓ All boundary times are valid numbers')
      }
      
      setChords(result)
      setLoading(false)
    } catch (error) {
      console.error('Chord generation error:', error)
      console.error('Error details:', error.response?.data)
      alert(`Error: ${error.response?.data?.detail || error.message}`)
      setLoading(false)
    }
  }

  // Play chord progression
  const handlePlayChords = async () => {
    if (!chords || !chords.chords || chords.chords.length === 0) return

    try {
      // Create audio from first few chords (for demo)
      const tuning = analysisResult.tuning || []
      const audioBlob = await apiClient.getChordAudio(
        sessionId,
        tuning,
        3, // num_chords
        440, // base_freq
        1.0 // duration
      )

      // Create URL and play
      const url = URL.createObjectURL(audioBlob)
      setAudioUrl(url)

      const audio = new Audio(url)
      audio.play()
    } catch (error) {
      console.error('Playback error:', error)
    }
  }

  // Export as MIDI
  const handleExportMIDI = async () => {
    if (!chords) return

    try {
      const midiBlob = await apiClient.exportMidi(
        sessionId,
        chords.chords,
        chords.bound_times,
        fileInfo.duration
      )

      apiClient.downloadBlob(midiBlob, 'biotuner_chords.mid')
    } catch (error) {
      console.error('MIDI export error:', error)
    }
  }

  // Export as MusicXML
  const handleExportMusicXML = async () => {
    if (!chords) return

    try {
      const response = await apiClient.client.post('/api/export-musicxml', {
        session_id: sessionId,
        chords: chords.chords,
        bound_times: chords.bound_times,
        total_duration: fileInfo?.duration,
        n_oct_up: 7
      }, {
        responseType: 'blob'
      })

      apiClient.downloadBlob(response.data, 'biotuner_chords.musicxml')
    } catch (error) {
      console.error('MusicXML export error:', error)
    }
  }

  return (
    <div className="space-y-8">
      {/* Configuration */}
      <div className="bg-gray-800 p-6 rounded-lg border border-biotuner-purple/30">
        <h3 className="text-xl font-bold mb-4">⚙️ Chord Generation Settings</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm mb-2">
              Number of Segments
              <span className="text-xs text-gray-400 ml-2">(Target: how many temporal chunks)</span>
            </label>
            <input
              type="number"
              min="4"
              max="512"
              value={chordConfig.n_segments}
              onChange={(e) => setChordConfig({ ...chordConfig, n_segments: parseInt(e.target.value) })}
              className="w-full bg-gray-700 border border-biotuner-purple/50 rounded-lg p-2"
            />
            <p className="text-xs text-gray-400 mt-1">
              💡 More segments = shorter, more granular chords. Actual count may vary based on signal.
            </p>
          </div>

          <div>
            <label className="block text-sm mb-2">
              Peaks per Chord
              <span className="text-xs text-gray-400 ml-2">(Notes in each chord)</span>
            </label>
            <input
              type="number"
              min="3"
              max="10"
              value={chordConfig.n_peaks}
              onChange={(e) => setChordConfig({ ...chordConfig, n_peaks: parseInt(e.target.value) })}
              className="w-full bg-gray-700 border border-biotuner-purple/50 rounded-lg p-2"
            />
            <p className="text-xs text-gray-400 mt-1">
              💡 How many frequency peaks to extract from each segment.
            </p>
          </div>

          <div>
            <label className="block text-sm mb-2">
              Time Resolution (ms)
              <span className="text-xs text-gray-400 ml-2">(Temporal detail)</span>
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={chordConfig.time_resolution}
              onChange={(e) => setChordConfig({ ...chordConfig, time_resolution: parseInt(e.target.value) })}
              className="w-full bg-gray-700 border border-biotuner-purple/50 rounded-lg p-2"
            />
            <p className="text-xs text-gray-400 mt-1">
              💡 Lower = more temporal precision (try 10-50ms). Higher = smoother segments.
            </p>
          </div>

          <div>
            <label className="block text-sm mb-2">
              Frequency Resolution
              <span className="text-xs text-gray-400 ml-2">(FFT window size)</span>
            </label>
            <input
              type="number"
              min="100"
              max="4096"
              step="100"
              value={chordConfig.frequency_resolution}
              onChange={(e) => setChordConfig({ ...chordConfig, frequency_resolution: parseInt(e.target.value) })}
              className="w-full bg-gray-700 border border-biotuner-purple/50 rounded-lg p-2"
            />
            <p className="text-xs text-gray-400 mt-1">
              💡 Higher = better frequency detail (1024-2048 for most signals).
            </p>
          </div>
        </div>

        {/* Advanced Settings */}
        <details className="mt-6 bg-gray-700/50 rounded-lg border border-biotuner-purple/30">
          <summary className="cursor-pointer p-4 font-semibold text-biotuner-purple hover:text-biotuner-pink">
            ⚙️ Advanced Chord Settings
          </summary>
          <div className="p-4 grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm mb-2">
                Peak Prominence
                <span className="text-xs text-gray-400 ml-2">(Minimum height)</span>
              </label>
              <input
                type="number"
                min="0.1"
                max="2"
                step="0.1"
                value={chordConfig.prominence}
                onChange={(e) => setChordConfig({ ...chordConfig, prominence: parseFloat(e.target.value) })}
                className="w-full bg-gray-700 border border-biotuner-purple/50 rounded-lg p-2"
              />
              <p className="text-xs text-gray-400 mt-1">
                💡 Lower = detect weaker peaks (0.3-1.0 recommended). Higher = only strong peaks.
              </p>
            </div>

            <div>
              <label className="block text-sm mb-2">
                MIDI Octave Shift
                <span className="text-xs text-gray-400 ml-2">(n_oct_up)</span>
              </label>
              <input
                type="number"
                min="0"
                max="10"
                value={chordConfig.n_oct_up}
                onChange={(e) => setChordConfig({ ...chordConfig, n_oct_up: parseInt(e.target.value) })}
                className="w-full bg-gray-700 border border-biotuner-purple/50 rounded-lg p-2"
              />
              <p className="text-xs text-gray-400 mt-1">
                💡 Shift frequencies up by N octaves for audible MIDI range (7 = x128).
              </p>
            </div>
          </div>
        </details>

        <button
          onClick={handleGenerateChords}
          disabled={loading}
          className="mt-6 w-full bg-biotuner-purple text-white px-6 py-3 rounded-lg font-semibold hover:bg-biotuner-pink hover:text-black disabled:opacity-50"
        >
          {loading ? '🔄 Generating Chords...' : '🎹 Generate Chord Progression'}
        </button>
      </div>

      {/* Results */}
      {chords && (
        <>
          {/* Chord Info */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
              <h3 className="text-sm text-gray-400 mb-1">Total Chords</h3>
              <p className="text-2xl font-bold text-biotuner-purple">
                {chords.n_segments}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                (requested {chordConfig.n_segments})
              </p>
            </div>

            <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
              <h3 className="text-sm text-gray-400 mb-1">Non-Empty Chords</h3>
              <p className="text-2xl font-bold text-biotuner-pink">
                {chords.chords.filter(c => c && c.length > 0).length}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {((chords.chords.filter(c => c && c.length > 0).length / chords.n_segments) * 100).toFixed(0)}% success rate
              </p>
            </div>

            <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
              <h3 className="text-sm text-gray-400 mb-1">Octave Shift</h3>
              <p className="text-2xl font-bold text-green-400">
                {chords.n_oct_up || 7}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                x{Math.pow(2, chords.n_oct_up || 7)} multiplier
              </p>
            </div>

            {chords.midi_range && (
              <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
                <h3 className="text-sm text-gray-400 mb-1">MIDI Range</h3>
                <p className="text-2xl font-bold text-blue-400">
                  {chords.midi_range.min} - {chords.midi_range.max}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Median: {chords.midi_range.median}
                </p>
              </div>
            )}
          </div>

          {/* Duration Info */}
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
              <h3 className="text-sm text-gray-400 mb-1">Avg Duration</h3>
              <p className="text-2xl font-bold text-biotuner-pink">
                {chords.segment_durations 
                  ? (chords.segment_durations.reduce((a, b) => a + b, 0) / chords.segment_durations.length).toFixed(2) 
                  : 'N/A'
                }s
              </p>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
              <h3 className="text-sm text-gray-400 mb-1">Peaks Method</h3>
              <p className="text-2xl font-bold text-green-400">
                {chords.method || 'N/A'}
              </p>
            </div>
          </div>

          {/* Signal Visualization with Segment Boundaries */}
          {fileInfo && fileInfo.preview_data && (
            <div>
              <h3 className="text-xl font-bold mb-4">📊 Signal with Segment Boundaries</h3>
              <div className="bg-gray-800 p-4 rounded-lg">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart 
                    data={(() => {
                      // Use the maximum boundary time as the reference duration
                      // This ensures signal and boundaries use the same time scale
                      const maxBoundTime = chords.bound_times && chords.bound_times.length > 0
                        ? Math.max(...chords.bound_times.filter(t => !isNaN(t) && isFinite(t)))
                        : fileInfo.duration
                      
                      return fileInfo.preview_data.map((val, idx) => ({
                        time: (idx / fileInfo.preview_data.length) * maxBoundTime,
                        value: val
                      }))
                    })()}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                    {(() => {
                      // Calculate domain based on actual boundary extent
                      const maxBoundary = chords.bound_times && chords.bound_times.length > 0 
                        ? Math.max(...chords.bound_times.filter(t => !isNaN(t) && isFinite(t)))
                        : 0
                      const minBoundary = chords.bound_times && chords.bound_times.length > 0
                        ? Math.min(...chords.bound_times.filter(t => !isNaN(t) && isFinite(t)))
                        : 0
                      
                      // Add 1% padding to ensure all boundaries are visible
                      const padding = (maxBoundary - minBoundary) * 0.01
                      const domainMin = Math.max(0, minBoundary - padding)
                      const domainMax = maxBoundary + padding
                      
                      console.log(`Domain calc: minBoundary=${minBoundary}, maxBoundary=${maxBoundary}, domain=[${domainMin}, ${domainMax}]`)
                      
                      return (
                        <XAxis 
                          dataKey="time" 
                          stroke="#888"
                          label={{ value: 'Time (s)', position: 'insideBottom', offset: -5, fill: '#888' }}
                          domain={[domainMin, domainMax]}
                          type="number"
                        />
                      )
                    })()}
                    <YAxis 
                      stroke="#888"
                      label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: '#888' }}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #00d9ff' }}
                      labelStyle={{ color: '#00d9ff' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#00d9ff" 
                      dot={false}
                      strokeWidth={1}
                      opacity={0.6}
                    />
                    {/* Add vertical reference lines at segment boundaries */}
                    {chords.bound_times && Array.isArray(chords.bound_times) && chords.bound_times.map((time, idx) => {
                      // Convert to number if it's a string
                      const timeNum = typeof time === 'string' ? parseFloat(time) : time
                      
                      // Skip invalid values with detailed logging
                      if (timeNum == null || isNaN(timeNum) || !isFinite(timeNum) || typeof timeNum !== 'number') {
                        console.warn(`⚠️ Skipping boundary ${idx}/${chords.bound_times.length}: time=${time} (type: ${typeof time}), parsed=${timeNum}`)
                        return null
                      }
                      
                      const isStart = idx === 0
                      const isEnd = idx === chords.bound_times.length - 1
                      const isGreen = isStart || isEnd
                      
                      // Debug: Log first/last few and green lines
                      if (isGreen || idx < 3 || idx > chords.bound_times.length - 4) {
                        console.log(`Boundary ${idx}: time=${timeNum.toFixed(3)}s, isGreen=${isGreen}`)
                      }
                      
                      return (
                        <ReferenceLine 
                          key={`boundary-${idx}`}
                          x={timeNum}
                          stroke={isGreen ? "#00ff00" : "#ff0000"}
                          strokeDasharray="5 5"
                          strokeWidth={isGreen ? 4 : 3}
                          strokeOpacity={isGreen ? 1.0 : 0.9}
                          ifOverflow="extendDomain"
                          label={
                            isStart ? { value: `Start (${chords.bound_times.length} bounds)`, position: 'top', fill: '#00ff00', fontSize: 11, fontWeight: 'bold' } : 
                            isEnd ? { value: `End (${timeNum.toFixed(1)}s)`, position: 'top', fill: '#00ff00', fontSize: 11, fontWeight: 'bold' } :
                            idx % 3 === 0 ? { value: `#${idx}`, position: 'top', fill: '#ff0000', fontSize: 9, offset: 5 } :
                            null
                          }
                        />
                      )
                    })}
                  </LineChart>
                </ResponsiveContainer>
                <p className="text-sm text-gray-400 mt-2 text-center">
                  {chords.bound_times.length} temporal segment boundaries: 
                  <span className="text-green-400"> 2 green</span> (start/end) + 
                  <span className="text-red-400"> {chords.bound_times.length - 2} red</span> (internal)
                  {chords.bound_times.length !== chords.n_segments + 1 && (
                    <span className="text-yellow-400"> ⚠️ Expected {chords.n_segments + 1} boundaries for {chords.n_segments} segments!</span>
                  )}
                </p>
                <p className="text-xs text-gray-500 mt-1 text-center">
                  Duration range: 0s - {(chords.bound_times && chords.bound_times.length > 0
                    ? Math.max(fileInfo?.duration || 0, ...chords.bound_times)
                    : (fileInfo?.duration || 0)
                  ).toFixed(2)}s
                  {' | '}Green lines = start/end, Red lines = segment boundaries
                  {' | '}Labels shown every 3 boundaries
                </p>
              </div>
            </div>
          )}

          {/* Segment Length Distribution */}
          <div>
            <h3 className="text-xl font-bold mb-4">📊 Segment Length Distribution</h3>
            
            {/* Check for highly skewed distribution */}
            {(() => {
              const durations = chords.segment_durations
              const max = Math.max(...durations)
              const avg = durations.reduce((a,b) => a+b, 0) / durations.length
              const isSkewed = max > avg * 10 // If max is 10x larger than average
              
              return isSkewed && (
                <div className="bg-yellow-900/30 border border-yellow-500/50 rounded-lg p-4 mb-4">
                  <h4 className="text-yellow-400 font-bold mb-2">⚠️ Highly Uneven Distribution Detected</h4>
                  <p className="text-sm text-yellow-200 mb-2">
                    Your segments are very unbalanced (max: {max.toFixed(2)}s, avg: {avg.toFixed(2)}s). 
                    This might indicate suboptimal parameters.
                  </p>
                  <div className="text-xs text-yellow-100 space-y-1">
                    <p>🔧 <strong>Try these fixes:</strong></p>
                    <ul className="list-disc list-inside ml-2 space-y-1">
                      <li><strong>Reduce Time Resolution</strong> to 10-30ms for more temporal detail</li>
                      <li><strong>Adjust Frequency Resolution</strong> to 1024-2048 for better feature extraction</li>
                      <li><strong>Reduce Number of Segments</strong> if signal is uniform</li>
                      <li><strong>Check your signal</strong> - very uniform signals produce poor segmentation</li>
                    </ul>
                  </div>
                </div>
              )
            })()}
            
            <div className="bg-gray-800 p-4 rounded-lg">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart 
                  data={chords.segment_durations.map((dur, idx) => ({
                    segment: idx + 1,
                    duration: dur
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis 
                    dataKey="segment" 
                    stroke="#888"
                    label={{ value: 'Segment #', position: 'insideBottom', offset: -5, fill: '#888' }}
                  />
                  <YAxis 
                    stroke="#888"
                    label={{ value: 'Duration (s)', angle: -90, position: 'insideLeft', fill: '#888' }}
                  />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1a1a1a', border: '1px solid #7c3aed' }}
                    labelStyle={{ color: '#7c3aed' }}
                    formatter={(value) => [`${value.toFixed(3)}s`, 'Duration']}
                  />
                  <Bar dataKey="duration" fill="#7c3aed">
                    {chords.segment_durations.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={`hsl(${270 + index * 2}, 70%, ${50 + (index % 20)}%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-gray-400">
                <div>
                  <span className="text-biotuner-primary">Min:</span> {Math.min(...chords.segment_durations).toFixed(3)}s
                </div>
                <div>
                  <span className="text-biotuner-secondary">Avg:</span> {(chords.segment_durations.reduce((a,b) => a+b, 0) / chords.segment_durations.length).toFixed(3)}s
                </div>
                <div>
                  <span className="text-green-400">Max:</span> {Math.max(...chords.segment_durations).toFixed(3)}s
                </div>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-4 justify-center flex-wrap">
            <button
              onClick={handlePlayChords}
              className="bg-green-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-green-500 flex items-center gap-2"
            >
              <Play className="w-5 h-5" />
              Play Sample Chords
            </button>
            
            <button
              onClick={handleExportMIDI}
              className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-blue-500 flex items-center gap-2"
            >
              <Download className="w-5 h-5" />
              Download MIDI
            </button>

            <button
              onClick={handleExportMusicXML}
              className="bg-purple-600 text-white px-8 py-3 rounded-lg font-semibold hover:bg-purple-500 flex items-center gap-2"
            >
              <Download className="w-5 h-5" />
              Download MusicXML
            </button>

            <button
              onClick={() => setShowScore(!showScore)}
              className="bg-biotuner-secondary text-white px-8 py-3 rounded-lg font-semibold hover:bg-biotuner-pink flex items-center gap-2"
            >
              <Music className="w-5 h-5" />
              {showScore ? 'Hide' : 'Show'} Musical Score
            </button>
          </div>

          {/* Musical Score Viewer */}
          {showScore && (
            <div className="bg-gray-800 p-6 rounded-lg border border-biotuner-secondary/30">
              <h3 className="text-xl font-bold mb-4">🎼 Musical Score</h3>
              <div className="bg-white rounded-lg p-4" id="score-container">
                <div className="text-center text-gray-600 py-8">
                  <p className="mb-2">Loading Verovio music engraver...</p>
                  <p className="text-sm">Musical score rendering will appear here</p>
                </div>
              </div>
              <p className="text-xs text-gray-400 mt-2 text-center">
                Powered by Verovio - Open-source music notation rendering
              </p>
            </div>
          )}

          {/* Audio Player */}
          {audioUrl && (
            <div className="bg-gray-800 p-4 rounded-lg">
              <audio
                src={audioUrl}
                controls
                className="w-full"
                style={{ filter: 'hue-rotate(270deg)' }}
              />
            </div>
          )}
        </>
      )}
    </div>
  )
}
