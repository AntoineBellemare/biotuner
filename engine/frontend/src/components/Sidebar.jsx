import { Settings, Sliders, Info, X } from 'lucide-react'
import { presetLabel } from '../services/presets'
import HelpHint from './HelpHint'

export default function Sidebar({
  config,
  onConfigChange,
  fileInfo,
  onClose,
  presetMode = 'auto',
  onPresetModeChange,
  activePresetKey,
  onResetPreset,
}) {
  const peakMethods = [
    { value: 'harmonic_recurrence', label: 'Harmonic Recurrence' },
    { value: 'cepstrum',            label: 'Cepstrum' },
    { value: 'EIMC',                label: 'EIMC (Intermodulation)' },
    { value: 'EMD',                 label: 'EMD' },
    { value: 'fixed',               label: 'EEG Bands' },
    { value: 'FOOOF',               label: 'FOOOF' },
    { value: 'SMS',                 label: 'SMS (sinusoidal modeling)' },
  ]

  // spectrum_method only affects extractors that consume a PSD.
  const SPECTRUM_AWARE_METHODS = new Set(['FOOOF', 'harmonic_recurrence', 'EIMC'])
  const showSpectrumPicker = SPECTRUM_AWARE_METHODS.has(config.method)

  const precisionOptions = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10]

  return (
    <aside className="w-80 max-w-[85vw] h-full bg-biotuner-dark-900 border-r border-biotuner-dark-600 flex flex-col">
      <div className="p-4 sm:p-6 border-b border-biotuner-dark-600">
        <div className="flex items-center justify-between gap-3 mb-2">
          <div className="flex items-center gap-3">
            <Settings className="w-5 h-5 text-biotuner-primary" />
            <h2 className="text-lg font-semibold tracking-wide text-biotuner-light uppercase">
              Configuration
            </h2>
          </div>
          {/* Mobile close button */}
          <button
            onClick={onClose}
            className="lg:hidden p-2 rounded-lg hover:bg-biotuner-dark-800 transition-colors"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5 text-biotuner-light/60" />
          </button>
        </div>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
        {/* Preset Selector */}
        {onPresetModeChange && (
          <div className="space-y-2">
            <h3 className="text-xs font-bold text-biotuner-primary/80 uppercase tracking-widest border-b border-biotuner-dark-600 pb-2">
              Preset
            </h3>
            <div className="flex items-center gap-2">
              <div className="flex-1 text-sm text-biotuner-light/80">
                {presetMode === 'custom' ? 'Custom (manual)' : presetLabel(activePresetKey)}
              </div>
              {presetMode === 'custom' && onResetPreset && (
                <button
                  onClick={onResetPreset}
                  className="text-xs px-2 py-1 rounded-md bg-biotuner-dark-800 border border-biotuner-dark-600 hover:border-biotuner-primary/50 text-biotuner-primary"
                >
                  Reset
                </button>
              )}
            </div>
            <p className="text-xs text-biotuner-light/40">
              Defaults update with your modality + source. Tweaking any value
              switches you to Custom.
            </p>
          </div>
        )}

        {/* Peak Extraction Section */}
        <div className="space-y-4">
          <h3 className="text-xs font-bold text-biotuner-primary/80 uppercase tracking-widest border-b border-biotuner-dark-600 pb-2">
            Peak Extraction
          </h3>
          
          {/* Peak Extraction Method */}
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Method
              <HelpHint
                text="How peaks are detected. Harmonic Recurrence picks peaks with a matching harmonic series (great for tonal signals). FOOOF separates 1/f noise from peaks (great for EEG). EMD decomposes into modes (good for non-stationary signals). Cepstrum finds the fundamental of a pitched sound. SMS tracks partials through time (best for evolving timbres)."
                example="EMD for forest soundscapes, FOOOF for EEG."
              />
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
              <HelpHint
                text="Width of one frequency bin. Smaller = finer resolution (you can tell apart close peaks) but requires longer recordings and more compute. Larger = smoother, less sensitive to noise."
                example="0.1 Hz for EEG/heart, 1–5 Hz for audio."
              />
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
              <HelpHint
                text="How many of the strongest peaks to keep. More peaks = richer tuning but more noise. Fewer peaks = cleaner but you may miss harmonic structure."
                example="5 is a sensible default; bells and bowls often want 7–10."
              />
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
              <HelpHint
                text="Upper bound for peak search. Anything above this is ignored. Keep it just above where you expect your top peak — too high wastes resolution on empty spectrum, too low cuts off real peaks."
                example="50 Hz for EEG, 5 Hz for heart, 8000 Hz for audio."
              />
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

          {/* Spectrum estimator — only shown for PSD-consuming methods */}
          {showSpectrumPicker && (
            <div>
              <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
                Spectrum estimator
                <HelpHint
                  text="How the power spectrum is computed before peaks are picked. FFT (Welch) is the standard. Multitaper averages several DPSS-tapered spectra — lower variance, good for short or noisy recordings (neuroscience uses it by default)."
                  example="FFT for clean audio, Multitaper for EEG and short bursts."
                />
              </label>
              <select
                value={config.spectrum_method || 'fft'}
                onChange={(e) => onConfigChange({ ...config, spectrum_method: e.target.value })}
                className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 focus:ring-2 focus:ring-biotuner-primary focus:border-biotuner-primary transition-all"
              >
                <option value="fft">FFT (Welch)</option>
                <option value="multitaper">Multitaper (DPSS)</option>
              </select>
              <p className="text-[10px] text-biotuner-light/40 mt-1 leading-snug">
                Multitaper averages K DPSS-tapered spectra — lower variance on
                short or noisy signals (esp. EEG / HRV).
              </p>
            </div>
          )}
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
              <HelpHint
                text="How peak frequencies are turned into a tuning. Peaks Ratios uses the ratios between the strongest peaks directly. Harmonic Fit replaces peaks with the nearest simple harmonic ratio. Dissonance Curve picks ratios at the local minima of a dissonance curve over the spectrum."
                example="Peaks Ratios for natural audio, Dissonance Curve for chord-friendly scales."
              />
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
              <HelpHint
                text="Upper bound for the denominator when approximating each peak ratio as a simple fraction. Lower = simpler, more 'consonant' ratios (3/2, 5/4). Higher = more accurate to the exact peak frequencies."
                example="50 for 'just intonation' feel, 1000 for high fidelity."
              />
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
              <HelpHint
                text="How many integer harmonics each peak is expanded to when computing harmonic-fit tunings. More harmonics = richer search but slower; not used by the Peaks Ratios method."
                example="10 is the default; bump up if your signal has a long harmonic series."
              />
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
        <div className="p-4 sm:p-6 border-t border-biotuner-dark-600 bg-biotuner-dark-800/50">
          <div className="flex items-center gap-2 mb-3">
            <Info className="w-4 h-4 text-biotuner-primary" />
            <h3 className="text-xs font-semibold text-biotuner-light/60 uppercase tracking-wider">
              Source Data
            </h3>
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-biotuner-light/40">File</span>
              <span className="text-biotuner-light font-mono text-xs truncate max-w-[140px] sm:max-w-[180px]">
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
