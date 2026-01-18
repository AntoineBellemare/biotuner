import { useState } from 'react'
import { Download, Palette } from 'lucide-react'
import apiClient from '../../services/api'

export default function BiocolorsTab({ sessionId, analysisResult }) {
  const [colorPalette, setColorPalette] = useState(null)
  const [paletteType, setPaletteType] = useState('tuning')  // 'tuning' or 'peaks'
  const [fundamental, setFundamental] = useState(440)
  const [loading, setLoading] = useState(false)
  const [selectedColors, setSelectedColors] = useState(new Set())

  // Generate color palette
  const handleGenerateColors = async (fund = fundamental) => {
    try {
      setLoading(true)
      
      const config = {
        session_id: sessionId,
        palette_type: paletteType,
        fundamental: fund,
      }
      
      if (paletteType === 'tuning') {
        config.tuning = analysisResult.tuning
      } else {
        config.peaks = analysisResult.peaks
        config.powers = analysisResult.powers
      }
      
      const result = await apiClient.generateBiocolors(config)
      setColorPalette(result)
      setLoading(false)
    } catch (error) {
      console.error('Color generation error:', error)
      setLoading(false)
    }
  }
  
  // Handle fundamental frequency change with real-time update
  const handleFundamentalChange = async (newFund) => {
    setFundamental(newFund)
    if (paletteType === 'tuning' && colorPalette) {
      // Real-time update for tuning mode
      await handleGenerateColors(newFund)
    }
  }

  // Toggle color selection
  const toggleColorSelection = (name) => {
    setSelectedColors(prev => {
      const newSet = new Set(prev)
      if (newSet.has(name)) {
        newSet.delete(name)
      } else {
        newSet.add(name)
      }
      return newSet
    })
  }

  // Select all colors
  const selectAllColors = () => {
    if (colorPalette) {
      setSelectedColors(new Set(Object.keys(colorPalette.palette)))
    }
  }

  // Deselect all colors
  const deselectAllColors = () => {
    setSelectedColors(new Set())
  }

  // Export palette (only selected colors or all if none selected)
  const handleExportPalette = async (format) => {
    if (!colorPalette) return

    try {
      // Use selected colors if any, otherwise use all
      const namesToExport = selectedColors.size > 0 
        ? Array.from(selectedColors)
        : Object.keys(colorPalette.palette)
      
      const colors = Object.fromEntries(
        namesToExport.map(name => [name, colorPalette.palette[name].hex])
      )

      const blob = await apiClient.exportPalette(format, colors, 'biotuner_palette')
      apiClient.downloadBlob(blob, `biotuner_palette.${format}`)
    } catch (error) {
      console.error('Export error:', error)
    }
  }

  return (
    <div className="space-y-8">
      {/* Configuration */}
      <div className="bg-gray-800 p-6 rounded-lg border border-biotuner-purple/30">
        <h3 className="text-xl font-bold mb-4">🎨 Color Palette Settings</h3>
        
        {/* Palette Type Toggle */}
        <div className="mb-6">
          <label className="block text-sm mb-2 font-semibold">Palette Source</label>
          <div className="flex gap-2">
            <button
              onClick={() => {
                setPaletteType('tuning')
                setColorPalette(null)
              }}
              className={`flex-1 px-4 py-3 rounded-lg font-semibold transition-all ${
                paletteType === 'tuning'
                  ? 'bg-biotuner-purple text-white ring-2 ring-biotuner-pink'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              🎵 From Tuning (Ratios)
            </button>
            <button
              onClick={() => {
                setPaletteType('peaks')
                setColorPalette(null)
              }}
              className={`flex-1 px-4 py-3 rounded-lg font-semibold transition-all ${
                paletteType === 'peaks'
                  ? 'bg-biotuner-purple text-white ring-2 ring-biotuner-pink'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              📊 From Peaks (Direct)
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2">
            {paletteType === 'tuning' 
              ? '💡 Colors based on harmonic ratios with consonance-driven saturation'
              : '💡 Colors directly from signal peaks with amplitude-driven saturation'
            }
          </p>
        </div>
        
        {/* Fundamental Frequency Slider (only for tuning mode) */}
        {paletteType === 'tuning' && (
          <div className="mb-6">
            <label className="block text-sm mb-2 font-semibold">
              Fundamental Frequency: <span className="text-biotuner-pink">{fundamental} Hz</span>
            </label>
            <input
              type="range"
              min="20"
              max="2000"
              step="1"
              value={fundamental}
              onChange={(e) => handleFundamentalChange(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-biotuner-purple"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>20 Hz</span>
              <span>440 Hz (A4)</span>
              <span>2000 Hz</span>
            </div>
            <p className="text-xs text-gray-400 mt-2">
              ⚡ Drag to shift colors in real-time (octave multiplication affects wavelength mapping)
            </p>
          </div>
        )}
        
        <button
          onClick={() => handleGenerateColors()}
          disabled={loading}
          className="w-full bg-biotuner-purple text-white px-8 py-3 rounded-lg font-semibold hover:bg-biotuner-pink hover:text-black disabled:opacity-50 transition-all"
        >
          {loading ? '🔄 Generating...' : `🎨 Generate ${paletteType === 'tuning' ? 'Tuning' : 'Peaks'} Palette`}
        </button>
      </div>

      {/* Color Palette Display */}
      {colorPalette && (
        <>
          {/* Palette Info */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
              <h3 className="text-sm text-gray-400 mb-1">Total Colors</h3>
              <p className="text-2xl font-bold text-biotuner-purple">
                {colorPalette.n_colors}
              </p>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
              <h3 className="text-sm text-gray-400 mb-1">Fundamental</h3>
              <p className="text-2xl font-bold text-biotuner-pink">
                {colorPalette.fundamental} Hz
              </p>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg border border-biotuner-purple/30">
              <h3 className="text-sm text-gray-400 mb-1">Color Space</h3>
              <p className="text-2xl font-bold text-green-400">
                RGB/HSV
              </p>
            </div>
          </div>

          {/* Color Swatches */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold">🌈 Color Palette</h3>
              <div className="flex gap-2">
                <button
                  onClick={selectAllColors}
                  className="text-sm bg-gray-700 hover:bg-biotuner-purple px-4 py-2 rounded-lg transition-colors"
                >
                  Select All
                </button>
                <button
                  onClick={deselectAllColors}
                  className="text-sm bg-gray-700 hover:bg-biotuner-pink hover:text-black px-4 py-2 rounded-lg transition-colors"
                >
                  Deselect All
                </button>
                {selectedColors.size > 0 && (
                  <span className="text-sm bg-biotuner-purple px-4 py-2 rounded-lg">
                    {selectedColors.size} selected
                  </span>
                )}
              </div>
            </div>
            
            {/* Large swatches */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-8">
              {Object.entries(colorPalette.palette).map(([name, colorData]) => {
                const isSelected = selectedColors.has(name)
                return (
                  <div
                    key={name}
                    onClick={() => toggleColorSelection(name)}
                    className={`group relative overflow-hidden rounded-xl shadow-lg transition-all duration-300 hover:scale-110 cursor-pointer ${
                      isSelected ? 'ring-4 ring-biotuner-pink scale-105' : ''
                    }`}
                    style={{ aspectRatio: '1/1' }}
                  >
                    <div
                      className="w-full h-full"
                      style={{ backgroundColor: colorData.hex }}
                    />
                    {isSelected && (
                      <div className="absolute top-2 right-2 bg-biotuner-pink text-black rounded-full w-6 h-6 flex items-center justify-center font-bold text-sm">
                        ✓
                      </div>
                    )}
                    <div className="absolute inset-0 bg-black/80 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex flex-col items-center justify-center p-2 text-xs">
                      <p className="font-bold text-white mb-1">{colorData.hex}</p>
                      <p className="text-white/80">{colorData.frequency.toFixed(2)} Hz</p>
                      <p className="text-white/80">{colorData.wavelength.toFixed(0)} nm</p>
                      {colorData.consonance !== undefined && (
                        <p className="text-white/80">Consonance: {colorData.consonance.toFixed(3)}</p>
                      )}
                      {colorData.saturation !== undefined && (
                        <p className="text-white/80">Saturation: {colorData.saturation.toFixed(3)}</p>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Linear palette view */}
            <div className="h-24 rounded-lg overflow-hidden shadow-lg flex">
              {Object.entries(colorPalette.palette).map(([name, colorData]) => (
                <div
                  key={name}
                  className="flex-1 relative group cursor-pointer"
                  style={{ backgroundColor: colorData.hex }}
                  title={`${colorData.hex} - ${colorData.frequency.toFixed(2)} Hz`}
                >
                  <div className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                    <span className="text-white text-xs font-bold">
                      {colorData.hex}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Color Details Table */}
          <div>
            <h3 className="text-xl font-bold mb-4">📊 Color Details</h3>
            <div className="bg-gray-800 rounded-lg overflow-hidden">
              <table className="w-full">
                <thead className="bg-biotuner-purple/30">
                  <tr>
                    <th className="px-4 py-3 text-left">Color</th>
                    <th className="px-4 py-3 text-left">Hex</th>
                    <th className="px-4 py-3 text-left">RGB</th>
                    <th className="px-4 py-3 text-left">Frequency</th>
                    <th className="px-4 py-3 text-left">Wavelength</th>
                    <th className="px-4 py-3 text-left">{paletteType === 'tuning' ? 'Consonance' : 'Saturation'}</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(colorPalette.palette).map(([name, colorData], idx) => (
                    <tr key={name} className={idx % 2 === 0 ? 'bg-gray-700/50' : ''}>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div
                            className="w-8 h-8 rounded"
                            style={{ backgroundColor: colorData.hex }}
                          />
                          <span className="text-sm">{name.split('_').slice(-1)}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 font-mono text-sm">{colorData.hex}</td>
                      <td className="px-4 py-3 font-mono text-sm">
                        {colorData.rgb.join(', ')}
                      </td>
                      <td className="px-4 py-3">{colorData.frequency.toFixed(2)} Hz</td>
                      <td className="px-4 py-3">{colorData.wavelength.toFixed(0)} nm</td>
                      <td className="px-4 py-3">
                        {colorData.consonance !== undefined ? (
                          <div className="flex items-center gap-2">
                            <div className="flex-1 bg-gray-600 rounded-full h-2">
                              <div
                                className="bg-biotuner-pink h-2 rounded-full"
                                style={{ width: `${colorData.consonance * 100}%` }}
                              />
                            </div>
                            <span className="text-sm">{colorData.consonance.toFixed(3)}</span>
                          </div>
                        ) : colorData.saturation !== undefined ? (
                          <div className="flex items-center gap-2">
                            <div className="flex-1 bg-gray-600 rounded-full h-2">
                              <div
                                className="bg-biotuner-purple h-2 rounded-full"
                                style={{ width: `${colorData.saturation * 100}%` }}
                              />
                            </div>
                            <span className="text-sm">{colorData.saturation.toFixed(3)}</span>
                          </div>
                        ) : (
                          <span className="text-gray-500">N/A</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Export Buttons */}
          <div>
            <h3 className="text-xl font-bold mb-4">
              💾 Export Palette
              {selectedColors.size > 0 && (
                <span className="text-sm text-biotuner-pink ml-2">
                  (exporting {selectedColors.size} selected colors)
                </span>
              )}
            </h3>
            <div className="flex flex-wrap gap-3 justify-center">
              {['ase', 'json', 'svg', 'css', 'gpl'].map((format) => (
                <button
                  key={format}
                  onClick={() => handleExportPalette(format)}
                  className="bg-biotuner-purple text-white px-6 py-3 rounded-lg font-semibold hover:bg-biotuner-pink hover:text-black flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  {format.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
