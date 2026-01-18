import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Brush, ReferenceArea } from 'recharts'
import { Activity, Scissors, RefreshCw } from 'lucide-react'
import apiClient from '../services/api'

export default function SignalPreview({ sessionId, fileInfo, onCrop, onColumnChange }) {
  const [chartData, setChartData] = useState([])
  const [selectedColumn, setSelectedColumn] = useState(0)
  const [cropRange, setCropRange] = useState({ start: 0, end: 100 })
  const [startInput, setStartInput] = useState('0.00')
  const [endInput, setEndInput] = useState('1.00')
  const [appliedCrop, setAppliedCrop] = useState(null) // Track applied crop for visual feedback
  const [fullChartData, setFullChartData] = useState([]) // Keep full data for reference

  useEffect(() => {
    if (fileInfo?.preview_data) {
      // Convert preview data to chart format
      const data = fileInfo.preview_data.map((value, index) => ({
        index,
        time: parseFloat((index * (fileInfo.duration / fileInfo.preview_data.length)).toFixed(3)),
        value: value
      }))
      setChartData(data)
      setFullChartData(data)
      
      // Reset crop when new file is loaded
      setAppliedCrop(null)
      setCropRange({ start: 0, end: 100 })
      
      // Initialize input values based on duration
      if (fileInfo.duration) {
        setStartInput('0.00')
        setEndInput(fileInfo.duration.toFixed(2))
      }
    }
  }, [fileInfo])

  const handleColumnChange = async (colIndex) => {
    setSelectedColumn(colIndex)
    
    if (onColumnChange) {
      await onColumnChange(colIndex)
    }
  }

  const handleApplyCrop = () => {
    if (onCrop) {
      const startTime = (cropRange.start / 100) * fileInfo.duration
      const endTime = (cropRange.end / 100) * fileInfo.duration
      const cropData = { start_time: startTime, end_time: endTime }
      setAppliedCrop(cropData) // Store applied crop for visual feedback only
      onCrop(cropData)
    }
  }

  const handleReset = () => {
    setCropRange({ start: 0, end: 100 })
    setStartInput('0.00')
    setEndInput(fileInfo?.duration.toFixed(2) || '1.00')
    setAppliedCrop(null) // Clear applied crop
    
    if (onCrop) {
      onCrop({ start_time: 0, end_time: fileInfo.duration })
    }
  }

  return (
    <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-biotuner-primary" />
          <h3 className="text-sm font-semibold text-biotuner-light uppercase tracking-wider">
            Signal Preview
          </h3>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Column Selector for CSV */}
          {fileInfo?.columns && fileInfo.columns.length > 1 && (
            <div className="flex items-center gap-2">
              <label className="text-xs text-biotuner-light/60">Column:</label>
              <select
                value={selectedColumn}
                onChange={(e) => handleColumnChange(parseInt(e.target.value))}
                className="bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded px-2 py-1 text-sm"
              >
                {fileInfo.columns.map((col, idx) => (
                  <option key={idx} value={idx}>{col}</option>
                ))}
              </select>
            </div>
          )}
          
          <button
            onClick={handleReset}
            className="flex items-center gap-1 px-3 py-1 rounded bg-biotuner-dark-800 hover:bg-biotuner-dark-700 text-biotuner-light text-sm transition-colors"
          >
            <RefreshCw className="w-3 h-3" />
            Reset
          </button>
        </div>
      </div>

      {/* Signal Chart */}
      <div className="bg-biotuner-dark-800 rounded-lg p-4 mb-4 relative">
        {/* Applied Crop Visual Indicator */}
        {appliedCrop && (
          <div className="absolute top-2 left-2 z-10 bg-biotuner-primary/90 text-white text-xs px-3 py-1.5 rounded-full flex items-center gap-2 shadow-lg">
            <Scissors className="w-3 h-3" />
            <span>
              Cropped: {appliedCrop.start_time.toFixed(2)}s - {appliedCrop.end_time.toFixed(2)}s
            </span>
          </div>
        )}
        
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#252525" />
            <XAxis 
              dataKey="time" 
              stroke="#6b7280"
              tickFormatter={(value) => parseFloat(value).toFixed(1)}
              domain={['dataMin', 'dataMax']}
              type="number"
            />
            <YAxis 
              stroke="#6b7280"
              label={{ value: 'Amplitude', angle: -90, position: 'insideLeft', fill: '#9ca3af' }}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1a1a1a', 
                border: '1px solid #252525',
                borderRadius: '8px' 
              }}
              labelStyle={{ color: '#00d9ff' }}
            />
            
            {/* Highlight cropped region if applied */}
            {appliedCrop && (
              <ReferenceArea
                x1={appliedCrop.start_time}
                x2={appliedCrop.end_time}
                strokeOpacity={0.5}
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.2}
              />
            )}
            
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#00d9ff" 
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Brush 
              dataKey="time" 
              height={30} 
              stroke="#00d9ff"
              fill="#121212"
              onChange={(range) => {
                if (range && range.startIndex !== undefined && range.endIndex !== undefined) {
                  const startPercent = (range.startIndex / chartData.length) * 100
                  const endPercent = (range.endIndex / chartData.length) * 100
                  setCropRange({ start: startPercent, end: endPercent })
                }
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Crop Controls */}
      <div className="flex items-center gap-4">
        <div className="flex-1 grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-biotuner-light/60 mb-1">Start Time (s)</label>
            <input
              type="number"
              min="0"
              max={fileInfo?.duration || 0}
              step="0.1"
              value={startInput}
              onChange={(e) => {
                const value = e.target.value
                setStartInput(value)
                
                const startTime = parseFloat(value) || 0
                if (!isNaN(startTime) && fileInfo?.duration) {
                  const startPercent = (startTime / fileInfo.duration) * 100
                  setCropRange({ ...cropRange, start: Math.max(0, Math.min(startPercent, cropRange.end)) })
                }
              }}
              onBlur={(e) => {
                // Format on blur
                const startTime = parseFloat(e.target.value) || 0
                setStartInput(startTime.toFixed(2))
              }}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded px-3 py-2 text-sm"
            />
          </div>
          
          <div>
            <label className="block text-xs text-biotuner-light/60 mb-1">End Time (s)</label>
            <input
              type="number"
              min="0"
              max={fileInfo?.duration || 0}
              step="0.1"
              value={endInput}
              onChange={(e) => {
                const value = e.target.value
                setEndInput(value)
                
                const endTime = parseFloat(value) || 0
                if (!isNaN(endTime) && fileInfo?.duration) {
                  const endPercent = (endTime / fileInfo.duration) * 100
                  setCropRange({ ...cropRange, end: Math.max(cropRange.start, Math.min(100, endPercent)) })
                }
              }}
              onBlur={(e) => {
                // Format on blur
                const endTime = parseFloat(e.target.value) || 0
                setEndInput(endTime.toFixed(2))
              }}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded px-3 py-2 text-sm"
            />
          </div>
        </div>
        
        <button
          onClick={handleApplyCrop}
          className="flex items-center gap-2 px-6 py-2 rounded bg-gradient-to-r from-biotuner-primary to-biotuner-secondary text-biotuner-dark-900 font-semibold hover:opacity-90 transition-opacity"
        >
          <Scissors className="w-4 h-4" />
          Apply Crop
        </button>
      </div>

      {/* Info */}
      <div className="mt-4 p-3 bg-biotuner-dark-800/50 rounded border border-biotuner-dark-600">
        <p className="text-xs text-biotuner-light/40">
          Selected: {((cropRange.end - cropRange.start) / 100 * (fileInfo?.duration || 0)).toFixed(2)}s 
          ({((cropRange.end - cropRange.start)).toFixed(1)}% of signal)
        </p>
      </div>
    </div>
  )
}
