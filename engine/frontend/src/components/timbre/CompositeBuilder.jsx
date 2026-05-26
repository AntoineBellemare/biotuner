/**
 * CompositeBuilder — multi-axis wavetable layer editor.
 *
 * Each row represents one ``WavetableLayer`` on the backend:
 *   - Evolution dropdown (any single-axis mode; ``composite`` excluded)
 *   - Weight curve dropdown (linear / ease_in / ease_out / sine / constant)
 *   - Weight range (min, max) — meaning depends on the chosen evolution
 *   - Per-evolution extras when applicable (rolloff for harmonic_stack,
 *     output_drive for wavefolding, cm_ratio + target_partial_idx for
 *     fm_baked, width_hz + gain_db for formant_sweep)
 *   - Up / down / delete actions to reorder layers
 *
 * Layers run in the order the user lists them: spectral edits
 * accumulate into a running Timbre, then a render happens, then
 * waveform-domain edits apply to the rendered cycle. The recommended
 * order banner reminds the user to think about this.
 */

import { useMemo } from 'react'
import { ChevronUp, ChevronDown, X, Plus, Info } from 'lucide-react'

// Per-evolution metadata: the user-facing label, the default
// (weight_min, weight_max) for new layers, and any extra param widgets
// that appear inline beneath the range slider.
const EVOLUTION_META = {
  tilt:             { label: 'Spectral tilt',     range: [0.0, 2.5],  extras: [] },
  harmonic_buildup: { label: 'Harmonic buildup',  range: [1, 8],      extras: [] },
  amp_morph:        { label: 'Amplitude morph',   range: [0.0, 1.0],  extras: [] },
  phase_sweep:      { label: 'Phase sweep',       range: [0.0, 6.28], extras: [] },
  intermod_buildup: { label: 'Intermod sidebands',range: [0.0, 0.6],  extras: [] },
  harmonic_stack:   { label: 'Harmonic stack',    range: [0, 4],      extras: ['rolloff'] },
  formant_sweep:    { label: 'Formant sweep (Hz)',range: [800, 2800], extras: ['width_hz', 'gain_db'] },
  wavefolding:      { label: 'Wavefolding',       range: [0.0, 4.0],  extras: ['output_drive'] },
  fm_baked:         { label: 'FM (baked)',        range: [0.0, 3.0],  extras: ['cm_ratio', 'target_partial_idx'] },
}

const WEIGHT_CURVES = [
  { value: 'linear',   label: 'Linear' },
  { value: 'ease_in',  label: 'Ease in' },
  { value: 'ease_out', label: 'Ease out' },
  { value: 'sine',     label: 'Sine' },
  { value: 'constant', label: 'Constant (midpoint)' },
]

const EXTRA_PARAM_META = {
  rolloff:            { label: 'Rolloff', min: 0.3, max: 2.0, step: 0.05, def: 0.9 },
  output_drive:       { label: 'Drive',   min: 0.7, max: 1.3, step: 0.05, def: 1.0 },
  cm_ratio:           { label: 'C:M',     min: 0.5, max: 5.0, step: 0.1,  def: 2.0 },
  target_partial_idx: { label: 'Target #',min: -1,  max: 16,  step: 1,    def: 0 },
  width_hz:           { label: 'Width Hz',min: 100, max: 3000,step: 50,   def: 800 },
  gain_db:            { label: 'Gain dB', min: 0,   max: 18,  step: 0.5,  def: 4 },
}

const MAX_LAYERS = 4

/** Build a fresh layer config with sensible defaults for the chosen evolution. */
function defaultLayer(evolution = 'harmonic_stack') {
  const meta = EVOLUTION_META[evolution]
  const extras = {}
  for (const key of meta.extras) extras[key] = EXTRA_PARAM_META[key].def
  return {
    evolution,
    weight_curve: 'linear',
    weight_min: meta.range[0],
    weight_max: meta.range[1],
    params: extras,
  }
}

export default function CompositeBuilder({ layers, onChange }) {
  // When a layer's evolution changes, refresh its range + extras
  // defaults — keeping the OLD range from a different evolution
  // would almost always be wrong (e.g. tilt range [0, 2.5] applied
  // to formant Hz would be 0–2.5 Hz, silent).
  const setLayer = (idx, patch) => {
    const next = layers.map((l, i) => {
      if (i !== idx) return l
      if ('evolution' in patch && patch.evolution !== l.evolution) {
        // Switching evolution → reset range + extras to that mode's defaults.
        return { ...defaultLayer(patch.evolution), weight_curve: l.weight_curve }
      }
      return { ...l, ...patch }
    })
    onChange(next)
  }
  const addLayer = () => {
    if (layers.length >= MAX_LAYERS) return
    onChange([...layers, defaultLayer()])
  }
  const removeLayer = (idx) => {
    onChange(layers.filter((_, i) => i !== idx))
  }
  const moveLayer = (idx, dir) => {
    const j = idx + dir
    if (j < 0 || j >= layers.length) return
    const next = layers.slice()
    ;[next[idx], next[j]] = [next[j], next[idx]]
    onChange(next)
  }

  return (
    <div className="space-y-2.5">
      {layers.length === 0 && (
        <div className="text-[11px] text-biotuner-light/40 italic px-1">
          No layers yet — add one to start composing.
        </div>
      )}
      {layers.map((layer, idx) => (
        <LayerRow
          key={idx}
          idx={idx}
          layer={layer}
          isFirst={idx === 0}
          isLast={idx === layers.length - 1}
          onChange={(patch) => setLayer(idx, patch)}
          onRemove={() => removeLayer(idx)}
          onMoveUp={() => moveLayer(idx, -1)}
          onMoveDown={() => moveLayer(idx, +1)}
        />
      ))}

      <div className="flex items-center justify-between gap-2 pt-1">
        <button
          onClick={addLayer}
          disabled={layers.length >= MAX_LAYERS}
          className="min-h-[32px] flex items-center gap-1.5 px-3 rounded-md
            bg-biotuner-accent/10 border border-biotuner-accent/40 text-biotuner-accent
            text-xs font-medium hover:bg-biotuner-accent/20
            disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <Plus className="w-3.5 h-3.5" />
          Add layer{layers.length >= MAX_LAYERS ? ` (max ${MAX_LAYERS})` : ''}
        </button>
        <div className="flex items-center gap-1.5 text-[10px] text-biotuner-light/40">
          <Info className="w-3 h-3" />
          Recommended order: enrichments → shape → nonlinear
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// One layer row
// ============================================================================
function LayerRow({
  idx, layer, isFirst, isLast,
  onChange, onRemove, onMoveUp, onMoveDown,
}) {
  const meta = EVOLUTION_META[layer.evolution]
  const updateParam = (key, val) => {
    onChange({ params: { ...layer.params, [key]: val } })
  }

  // Memoise the formatted range string so it doesn't re-compute on
  // every keystroke; it's also used in the row's compact summary chip.
  const summary = useMemo(() => {
    const ev = meta?.label || layer.evolution
    return `#${idx + 1} · ${ev} · ${layer.weight_min} → ${layer.weight_max}`
  }, [idx, meta, layer.evolution, layer.weight_min, layer.weight_max])

  return (
    <div className="bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-2.5 space-y-2">
      {/* Top row: index + reorder + summary + delete */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] font-mono text-biotuner-light/40 px-1.5">
          {idx + 1}
        </span>
        {/* Reorder + delete touch targets — bumped to 28×24 px (each
            button) so they're tappable on phones. Apple says 44 px is
            ideal, but two stacked 24 px buttons fit the layout better
            than two side-by-side 44 px ones, and the chip's bordered
            container makes the targets feel discoverable enough. */}
        <div className="flex flex-col">
          <button
            onClick={onMoveUp}
            disabled={isFirst}
            className="h-6 w-7 flex items-center justify-center text-biotuner-light/40
              hover:text-biotuner-primary hover:bg-biotuner-dark-700 rounded-t
              disabled:opacity-20 disabled:cursor-not-allowed"
            title="Move up"
          >
            <ChevronUp className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onMoveDown}
            disabled={isLast}
            className="h-6 w-7 flex items-center justify-center text-biotuner-light/40
              hover:text-biotuner-primary hover:bg-biotuner-dark-700 rounded-b
              disabled:opacity-20 disabled:cursor-not-allowed"
            title="Move down"
          >
            <ChevronDown className="w-3.5 h-3.5" />
          </button>
        </div>
        <span className="flex-1 text-[10px] text-biotuner-light/50 truncate">{summary}</span>
        <button
          onClick={onRemove}
          className="h-7 w-7 flex items-center justify-center text-biotuner-light/40
            hover:text-red-400 hover:bg-biotuner-dark-700 rounded"
          title="Remove layer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Evolution + curve dropdowns */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
            Evolution
          </label>
          <select
            value={layer.evolution}
            onChange={(e) => onChange({ evolution: e.target.value })}
            className="w-full bg-biotuner-dark-900 border border-biotuner-dark-600 rounded p-1 text-xs"
          >
            {Object.entries(EVOLUTION_META).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
            Curve
          </label>
          <select
            value={layer.weight_curve}
            onChange={(e) => onChange({ weight_curve: e.target.value })}
            className="w-full bg-biotuner-dark-900 border border-biotuner-dark-600 rounded p-1 text-xs"
          >
            {WEIGHT_CURVES.map((c) => (
              <option key={c.value} value={c.value}>{c.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Range — min and max, numeric inputs for precision */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
            Min
          </label>
          <input
            type="number"
            value={layer.weight_min}
            step={meta?.range && Math.abs(meta.range[1] - meta.range[0]) >= 10 ? 1 : 0.1}
            onChange={(e) => onChange({ weight_min: parseFloat(e.target.value) || 0 })}
            className="w-full bg-biotuner-dark-900 border border-biotuner-dark-600 rounded p-1 text-xs font-mono"
          />
        </div>
        <div>
          <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
            Max
          </label>
          <input
            type="number"
            value={layer.weight_max}
            step={meta?.range && Math.abs(meta.range[1] - meta.range[0]) >= 10 ? 1 : 0.1}
            onChange={(e) => onChange({ weight_max: parseFloat(e.target.value) || 0 })}
            className="w-full bg-biotuner-dark-900 border border-biotuner-dark-600 rounded p-1 text-xs font-mono"
          />
        </div>
      </div>

      {/* Per-evolution extras (rolloff, drive, cm_ratio, etc.) */}
      {meta?.extras?.length > 0 && (
        <div className="grid grid-cols-2 gap-2 pt-0.5">
          {meta.extras.map((key) => {
            const m = EXTRA_PARAM_META[key]
            return (
              <div key={key}>
                <label className="block text-[9px] uppercase tracking-wider text-biotuner-light/50 mb-0.5">
                  {m.label}
                </label>
                <input
                  type="number"
                  value={layer.params?.[key] ?? m.def}
                  min={m.min}
                  max={m.max}
                  step={m.step}
                  onChange={(e) => updateParam(key, parseFloat(e.target.value))}
                  className="w-full bg-biotuner-dark-900 border border-biotuner-dark-600 rounded p-1 text-xs font-mono"
                />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
