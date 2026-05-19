import { useEffect, useState } from 'react'
import { X, Save } from 'lucide-react'

export default function SaveTuningDialog({ open, onClose, defaults, onSave }) {
  const [name, setName] = useState('')
  const [notes, setNotes] = useState('')

  useEffect(() => {
    if (open) {
      const date = new Date().toISOString().slice(0, 10)
      const prefix = defaults?.modality || 'tuning'
      setName(defaults?.name || `${prefix} ${date}`)
      setNotes(defaults?.notes || '')
    }
  }, [open, defaults])

  if (!open) return null

  const submit = () => {
    onSave?.({
      name: name.trim() || 'Untitled tuning',
      notes: notes.trim() || undefined,
    })
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/70 p-4">
      <div className="w-full max-w-md bg-biotuner-dark-900 border border-biotuner-dark-600 rounded-lg shadow-2xl">
        <div className="flex items-center justify-between p-4 border-b border-biotuner-dark-600">
          <h3 className="text-base font-semibold text-biotuner-light flex items-center gap-2">
            <Save className="w-4 h-4 text-biotuner-primary" />
            Save tuning
          </h3>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-biotuner-dark-800">
            <X className="w-5 h-5 text-biotuner-light/60" />
          </button>
        </div>
        <div className="p-4 space-y-4">
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Name
            </label>
            <input
              type="text"
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 min-h-[48px]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-2 text-biotuner-light/60 uppercase tracking-wider">
              Notes (optional)
            </label>
            <textarea
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              className="w-full bg-biotuner-dark-800 text-biotuner-light border border-biotuner-dark-600 rounded-lg p-3 resize-none"
            />
          </div>
        </div>
        <div className="p-4 border-t border-biotuner-dark-600 flex justify-end gap-2">
          <button
            onClick={onClose}
            className="min-h-[44px] px-4 py-2 rounded-lg bg-biotuner-dark-800 border border-biotuner-dark-600 text-biotuner-light/80"
          >
            Cancel
          </button>
          <button
            onClick={submit}
            className="min-h-[44px] px-4 py-2 rounded-lg bg-gradient-to-r from-biotuner-primary to-biotuner-secondary text-biotuner-dark-900 font-medium"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  )
}
