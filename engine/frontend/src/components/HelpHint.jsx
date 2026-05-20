import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { HelpCircle } from 'lucide-react'

/**
 * Small (?) icon next to a label. Hover on desktop or tap on mobile to reveal
 * a short plain-English explanation of the parameter.
 *
 * The tooltip is rendered into document.body via a portal so it escapes any
 * scrolling parent's overflow clipping (e.g. the configuration sidebar).
 * Position is computed from the icon's bounding rect and auto-flips when it
 * would overflow the viewport edge.
 */
const TOOLTIP_WIDTH = 280
const GAP = 8

export default function HelpHint({ text, example }) {
  const [open, setOpen] = useState(false)
  const [coords, setCoords] = useState({ left: 0, top: 0, placement: 'right' })
  const btnRef = useRef(null)
  const tipRef = useRef(null)

  const computePosition = () => {
    if (!btnRef.current) return
    const r = btnRef.current.getBoundingClientRect()
    const vw = window.innerWidth
    const vh = window.innerHeight

    // Prefer right of the icon; flip left if it would overflow.
    const placeRight = r.right + GAP + TOOLTIP_WIDTH + 8 < vw
    const left = placeRight ? r.right + GAP : r.left - GAP - TOOLTIP_WIDTH
    let top = r.top + r.height / 2

    // Clamp vertically once the tooltip has rendered (measure its height).
    const tipH = tipRef.current?.offsetHeight ?? 120
    const minTop = 8 + tipH / 2
    const maxTop = vh - 8 - tipH / 2
    if (top < minTop) top = minTop
    if (top > maxTop) top = maxTop

    setCoords({
      left: Math.max(8, left),
      top,
      placement: placeRight ? 'right' : 'left',
    })
  }

  useEffect(() => {
    if (!open) return
    computePosition()
    // Recompute after first paint when tooltip height is known.
    const id = requestAnimationFrame(computePosition)

    const onDocClick = (e) => {
      if (
        btnRef.current && !btnRef.current.contains(e.target) &&
        tipRef.current && !tipRef.current.contains(e.target)
      ) setOpen(false)
    }
    const onScrollOrResize = () => computePosition()

    document.addEventListener('mousedown', onDocClick)
    window.addEventListener('scroll', onScrollOrResize, true)
    window.addEventListener('resize', onScrollOrResize)
    return () => {
      cancelAnimationFrame(id)
      document.removeEventListener('mousedown', onDocClick)
      window.removeEventListener('scroll', onScrollOrResize, true)
      window.removeEventListener('resize', onScrollOrResize)
    }
  }, [open])

  return (
    <span className="relative inline-flex items-center align-middle ml-1">
      <button
        ref={btnRef}
        type="button"
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); setOpen((o) => !o) }}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={(e) => {
          // Keep open if pointer moves into the tooltip itself.
          if (tipRef.current && tipRef.current.contains(e.relatedTarget)) return
          setOpen(false)
        }}
        aria-label="Help"
        className="text-biotuner-light/40 hover:text-biotuner-primary focus:outline-none focus:text-biotuner-primary"
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>

      {open && createPortal(
        <div
          ref={tipRef}
          role="tooltip"
          style={{
            position: 'fixed',
            left: `${coords.left}px`,
            top: `${coords.top}px`,
            width: `${TOOLTIP_WIDTH}px`,
            transform: 'translateY(-50%)',
            zIndex: 9999,
          }}
          onMouseEnter={() => setOpen(true)}
          onMouseLeave={() => setOpen(false)}
          className="p-3 rounded-md bg-biotuner-dark-900 border border-biotuner-primary/40 shadow-xl
                     text-xs text-biotuner-light/90 leading-relaxed normal-case tracking-normal font-normal pointer-events-auto"
        >
          {text}
          {example && (
            <span className="block mt-1.5 text-biotuner-light/60">
              <span className="text-biotuner-primary/80">e.g.</span> {example}
            </span>
          )}
        </div>,
        document.body
      )}
    </span>
  )
}
