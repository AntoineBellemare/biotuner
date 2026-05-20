/**
 * Modality-aware analysis presets.
 *
 * Keys follow `{modality}.{captureSource}` — every entry must produce a config
 * compatible with the backend `AnalysisConfig` schema so the existing
 * `/api/analyze` flow keeps working unchanged.
 */

export const PRESETS = {
  'audio.file':    { method: 'EMD',                precision: 1.0,  max_freq: 8000, n_peaks: 5, tuning_method: 'peaks_ratios' },
  'audio.mic':     { method: 'EMD',                precision: 1.0,  max_freq: 8000, n_peaks: 5, tuning_method: 'peaks_ratios' },
  'brain.file':    { method: 'FOOOF',              precision: 0.1,  max_freq: 50,   n_peaks: 5, tuning_method: 'peaks_ratios' },
  'heart.file':    { method: 'harmonic_recurrence', precision: 0.01, max_freq: 5,    n_peaks: 5, tuning_method: 'peaks_ratios' },
  'heart.ppg':     { method: 'harmonic_recurrence', precision: 0.01, max_freq: 5,    n_peaks: 5, tuning_method: 'peaks_ratios' },
  'sensors.file':  { method: 'harmonic_recurrence', precision: 0.05, max_freq: 25,   n_peaks: 5, tuning_method: 'peaks_ratios' },
  'sensors.sensor':{ method: 'harmonic_recurrence', precision: 0.05, max_freq: 25,   n_peaks: 5, tuning_method: 'peaks_ratios' },
  'plant.file':    { method: 'EMD',                precision: 0.01, max_freq: 2,    n_peaks: 5, tuning_method: 'peaks_ratios' },
  'creative.file': { method: 'harmonic_recurrence', precision: 1.0,  max_freq: 100,  n_peaks: 5, tuning_method: 'peaks_ratios' },
  'object.tap':    { method: 'FOOOF',              precision: 1.0,  max_freq: 12000, n_peaks: 7, tuning_method: 'peaks_ratios' },
}

const LABELS = {
  'audio.file':    'Audio file',
  'audio.mic':     'Audio (mic recording)',
  'brain.file':    'EEG resting state',
  'heart.file':    'Heart (ECG/RR file)',
  'heart.ppg':     'Heart (camera PPG)',
  'sensors.file':  'Smartphone sensors (file)',
  'sensors.sensor':'Smartphone sensors (live)',
  'plant.file':    'Plant signal',
  'creative.file': 'Creative / generic',
  'object.tap':    'Object resonance',
}

const DEFAULT_KEY = 'audio.file'

export const ANALYSIS_DEFAULTS = {
  method: 'harmonic_recurrence',
  n_peaks: 5,
  precision: 0.1,
  max_freq: 100,
  tuning_method: 'peaks_ratios',
  max_denominator: 100,
  n_harm: 10,
  spectrum_method: 'fft',
}

export function presetKey(modality, source) {
  if (!modality) return DEFAULT_KEY
  const src = source || 'file'
  if (PRESETS[`${modality}.${src}`]) return `${modality}.${src}`
  if (PRESETS[`${modality}.file`]) return `${modality}.file`
  return DEFAULT_KEY
}

export function getPreset(modality, source) {
  const k = presetKey(modality, source)
  return { key: k, ...ANALYSIS_DEFAULTS, ...PRESETS[k] }
}

export function presetLabel(key) {
  return LABELS[key] || key
}

export function listPresets() {
  return Object.keys(PRESETS).map((k) => ({ key: k, label: LABELS[k] || k }))
}

/**
 * Returns true when `config` (whatever fields the sidebar exposes) matches
 * `preset` for the overlapping keys. Used to detect "user has tweaked" → flip
 * the preset selector to "Custom".
 */
export function configMatchesPreset(config, preset) {
  if (!config || !preset) return false
  for (const k of ['method', 'precision', 'max_freq', 'n_peaks', 'tuning_method']) {
    if (preset[k] === undefined) continue
    if (config[k] !== preset[k]) return false
  }
  return true
}
