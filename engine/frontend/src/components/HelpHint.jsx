import { useEffect, useRef, useState } from 'react'
import { HelpCircle } from 'lucide-react'

/**
 * Small (?) icon next to a label. Hover on desktop or tap on mobile to reveal
 * a short plain-English explanation of the parameter.
 *
 * Usage:
 *   <label>Precision <HelpHint text="Frequency-bin width in Hz..." /></label>
 */
export default function HelpHint({ text, example }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return
    const onDocClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [open])

  return (
    <span
      ref={ref}
      className="relative inline-flex items-center align-middle ml-1"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <button
        type="button"
        onClick={(e) => { e.preventDefault(); setOpen((o) => !o) }}
        aria-label="Help"
        className="text-biotuner-light/40 hover:text-biotuner-primary focus:outline-none focus:text-biotuner-primary"
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute z-50 left-5 top-1/2 -translate-y-1/2 w-64 max-w-[80vw] p-3 rounded-md
                     bg-biotuner-dark-900 border border-biotuner-primary/40 shadow-xl
                     text-xs text-biotuner-light/90 leading-relaxed normal-case tracking-normal font-normal"
        >
          {text}
          {example && (
            <span className="block mt-1.5 text-biotuner-light/60">
              <span className="text-biotuner-primary/80">e.g.</span> {example}
            </span>
          )}
        </span>
      )}
    </span>
  )
}
