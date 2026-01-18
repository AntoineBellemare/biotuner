import { Settings, Sliders, Info } from 'lucide-react'

export default function Sidebar({ config, onConfigChange, fileInfo }) {
  const peakMethods = [
    { value: 'harmonic_recurrence', label: 'Harmonic Recurrence' },
    { value: 'cepstrum', label: 'Cepstrum' },
    { value: 'EIMC', label: 'EIMC (Intermodulation)' },
    { value: 'EMD', label: 'EMD' },
    { value: 'fixed', label: 'EEG Bands' },
    { value: 'FOOOF', label: 'FOOOF' },
  ]

  const precisionOptions = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

  return (
    <aside className="w-80 bg-biotuner-dark-900 border-r border-biotuner-dark-600 flex flex-col">
      <div className="p-6 border-b border-biotuner-dark-600">
        <div className="flex items-center gap-3 mb-2">
          <Settings className="w-5 h-5 text-biotuner-primary" />
          <h2 className="text-lg font-semibold tracking-wide text-biotuner-light uppercase">
            Configuration
          </h2>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* Peak Extraction Section */}
        <div className="space-y-4">
          <h3 className="text-xs font-bold text-biotuner-primary/80 uppercase tracking-widest border-b border-biotuner-dark-600 pb-2">
            Peak Extraction
          </h3>
          
          {/* Peak Extraction Method */}
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Method
            </label>
            <select
              value={config.method}
              onChange={(e) => onConfigChange({ ...config, method: e.target.value })}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 focus:ring-2 focus:ring-biotuner-primary focus:border-biotuner-primary transition-all"
            >
              {peakMethods.map((method) => (
                <option key={method.value} value={method.value}>
                  {method.label}
                </option>
              ))}
            </select>
          </div>

          {/* Precision */}
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Precision (Hz)
            </label>
            <select
              value={config.precision}
              onChange={(e) => onConfigChange({ ...config, precision: parseFloat(e.target.value) })}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 focus:ring-2 focus:ring-biotuner-primary focus:border-biotuner-primary transition-all"
            >
              {precisionOptions.map((val) => (
                <option key={val} value={val}>
                  {val} Hz
                </option>
              ))}
            </select>
          </div>

          {/* Number of Peaks */}
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Number of Peaks
            </label>
            <input
              type="number"
              min="1"
              max="50"
              value={config.n_peaks}
              onChange={(e) => onConfigChange({ ...config, n_peaks: parseInt(e.target.value) })}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 focus:ring-2 focus:ring-biotuner-primary focus:border-biotuner-primary transition-all"
            />
          </div>

          {/* Max Frequency */}
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Max Frequency (Hz)
            </label>
            <input
              type="number"
              min="1"
              max="1000"
              value={config.max_freq}
              onChange={(e) => onConfigChange({ ...config, max_freq: parseFloat(e.target.value) })}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 focus:ring-2 focus:ring-biotuner-primary focus:border-biotuner-primary transition-all"
            />
          </div>
        </div>

        {/* Tuning Analysis Section */}
        <div className="space-y-4">
          <h3 className="text-xs font-bold text-biotuner-secondary/80 uppercase tracking-widest border-b border-biotuner-dark-600 pb-2">
            Tuning Analysis
          </h3>
          
          {/* Tuning Method */}
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Tuning Method
            </label>
            <select
              value={config.tuning_method || 'peaks_ratios'}
              onChange={(e) => onConfigChange({ ...config, tuning_method: e.target.value })}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 focus:ring-2 focus:ring-biotuner-secondary focus:border-biotuner-secondary transition-all"
            >
              <option value="peaks_ratios">Peaks Ratios</option>
              <option value="harmonic_fit">Harmonic Fit</option>
              <option value="diss_curve">Dissonance Curve</option>
            </select>
          </div>

          {/* Max Denominator */}
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Max Denominator
            </label>
            <input
              type="number"
              min="1"
              max="1000"
              value={config.max_denominator || 100}
              onChange={(e) => onConfigChange({ ...config, max_denominator: parseInt(e.target.value) })}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 focus:ring-2 focus:ring-biotuner-secondary focus:border-biotuner-secondary transition-all"
            />
          </div>

          {/* Number of Harmonics */}
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Number of Harmonics
            </label>
            <input
              type="number"
              min="1"
              max="256"
              value={config.n_harm || 10}
              onChange={(e) => onConfigChange({ ...config, n_harm: parseInt(e.target.value) })}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 focus:ring-2 focus:ring-biotuner-secondary focus:border-biotuner-secondary transition-all"
            />
          </div>
        </div>
      </div>

      {/* File Info */}
      {fileInfo && (
        <div className="p-6 border-t border-biotuner-dark-600 bg-biotuner-dark-800/50">
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-4 h-4 text-biotuner-primary" />
            <h3 className="text-xs font-semibold text-biotuner-light/60 uppercase tracking-wider">
              Source Data
            </h3>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-biotuner-light/40">File</span>
              <span className="text-biotuner-light font-mono text-xs truncate max-w-[180px]">
                {fileInfo.filename}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-biotuner-light/40">Duration</span>
              <span className="text-biotuner-primary">{fileInfo.duration?.toFixed(2)}s</span>
            </div>
            <div className="flex justify-between">
              <span className="text-biotuner-light/40">Sample Rate</span>
              <span className="text-biotuner-primary">{fileInfo.sampling_rate} Hz</span>
            </div>
            <div className="flex justify-between">
              <span className="text-biotuner-light/40">Data Points</span>
              <span className="text-biotuner-primary">{fileInfo.data_points?.toLocaleString()}</span>
            </div>
          </div>
        </div>
      )}
    </aside>
  )
}
