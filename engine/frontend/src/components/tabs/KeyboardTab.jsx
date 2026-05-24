/**
 * Keyboard tab — play the analysed tuning interactively.
 *
 * Three input modes, all live simultaneously:
 *   1. Mouse / touch on the on-screen keys.
 *   2. Computer keyboard (a s d f g h j k l ; ' ) — one key per scale step
 *      across the visible octaves; z/x shift base octave.
 *   3. WebMIDI — plug in a hardware MIDI keyboard and every note gets
 *      remapped from 12-TET to the user's custom tuning.
 *
 * Polyphonic via Tone.PolySynth. Voice / base-frequency / octaves are
 * user-controlled and persist into Tone's PolySynth so the user can
 * audition the tuning with whatever timbre suits.
 *
 * MIDI mapping convention: MIDI 60 (middle C) = scale step 0. Every
 * step above adds one degree of the custom tuning. After N degrees
 * (where N = tuning length), we wrap to the next octave (× 2). This
 * is "wrapped chromatic" — exposes every degree on the keyboard
 * regardless of how many notes the tuning has.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import * as Tone from 'tone'
import { Cable, Music2 } from 'lucide-react'

// Oscillator + envelope presets. Each rebuilds the PolySynth on change
// because Tone.PolySynth's oscillator type is set at construction time.
const VOICES = {
  sine:     { type: 'sine',     env: { attack: 0.03, decay: 0.15, sustain: 0.4, release: 0.6 } },
  triangle: { type: 'triangle', env: { attack: 0.03, decay: 0.15, sustain: 0.5, release: 0.6 } },
  square:   { type: 'square',   env: { attack: 0.01, decay: 0.12, sustain: 0.25, release: 0.4 } },
  sawtooth: { type: 'sawtooth', env: { attack: 0.01, decay: 0.12, sustain: 0.3, release: 0.4 } },
  fatsaw:   { type: 'fatsawtooth', env: { attack: 0.02, decay: 0.15, sustain: 0.4, release: 0.7 } },
  amsine:   { type: 'amsine',   env: { attack: 0.05, decay: 0.2,  sustain: 0.5, release: 0.8 } },
}

// Computer-keyboard row used as note shortcuts. Mapped in order to the
// visible keys (degree 0 of octave 0 = 'a', degree 1 = 's', etc.).
// 11 chars handles tunings up to N=11 within a single row; multi-octave
// views extend onto additional rows via SHORTCUT_ROW_2.
const SHORTCUT_ROW_1 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'"]
const SHORTCUT_ROW_2 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[']

export default function KeyboardTab({ analysisResult }) {
  const tuning = useMemo(() => analysisResult?.tuning || [], [analysisResult])
  const N = tuning.length

  // Pick a sensible default base — the first peak from the analysis if
  // it's audible, else A3 (220 Hz). Fundamental that's already in the
  // signal makes everything sound naturally pitched to the source.
  const defaultBase = useMemo(() => {
    const peak = analysisResult?.peaks?.[0]
    if (peak && peak >= 55 && peak <= 880) return Math.round(peak)
    return 220
  }, [analysisResult])

  const [baseFreq, setBaseFreq] = useState(defaultBase)
  const [voice, setVoice] = useState('triangle')
  const [octaves, setOctaves] = useState(2)
  const [baseOctaveShift, setBaseOctaveShift] = useState(0)
  // String IDs so on-screen keys, computer-shortcut keys and MIDI notes
  // can all share one "is pressed" Set.
  const [activeIds, setActiveIds] = useState(() => new Set())
  // WebMIDI state.
  const [midiInputs, setMidiInputs] = useState([])
  const [midiInputId, setMidiInputId] = useState('')
  const [midiAccessState, setMidiAccessState] = useState('idle') // idle | granted | denied | unsupported

  const synthRef = useRef(null)
  const midiAccessRef = useRef(null)
  // For MIDI release we need the freq we attacked at — store note → freq.
  const midiActiveFreqsRef = useRef(new Map())
  // The voice that the currently-built synth uses; lets us lazy-rebuild
  // only when the user actually changes it.
  const builtVoiceRef = useRef(null)

  // Sync defaultBase when a new analysis result arrives.
  useEffect(() => {
    setBaseFreq(defaultBase)
  }, [defaultBase])

  // ----- Visible keys = octaves × N degrees -----
  // Each key holds everything needed to render + play: freq, ratio,
  // cents, octave/degree indices, and the computer-keyboard shortcut.
  const keys = useMemo(() => {
    if (!N) return []
    const out = []
    for (let oct = 0; oct < octaves; oct++) {
      for (let deg = 0; deg < N; deg++) {
        const ratio = tuning[deg] * Math.pow(2, oct)
        const freq = baseFreq * Math.pow(2, baseOctaveShift) * ratio
        const cents = (oct * 1200) + (Math.log2(tuning[deg] || 1) * 1200)
        const flatIdx = oct * N + deg
        const shortcut =
          flatIdx < SHORTCUT_ROW_1.length ? SHORTCUT_ROW_1[flatIdx]
          : flatIdx < SHORTCUT_ROW_1.length + SHORTCUT_ROW_2.length
            ? SHORTCUT_ROW_2[flatIdx - SHORTCUT_ROW_1.length]
            : null
        out.push({
          id: `o${oct}-d${deg}`,
          deg, oct, ratio, freq, cents, shortcut,
        })
      }
    }
    return out
  }, [tuning, baseFreq, octaves, baseOctaveShift, N])

  // ----- Synth lifecycle -----
  // Lazy: only created on first user gesture. Tone.start() must run
  // inside a gesture handler (browser AudioContext policy).
  async function ensureSynth() {
    await Tone.start()
    if (synthRef.current && builtVoiceRef.current === voice) return synthRef.current
    if (synthRef.current) killSynth()  // voice changed → rebuild
    const v = VOICES[voice] || VOICES.triangle
    synthRef.current = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: v.type },
      envelope: v.env,
      volume: -8,
    }).toDestination()
    builtVoiceRef.current = voice
    return synthRef.current
  }

  // Hard dispose — necessary because PolySynth's oscillator type can't
  // be hot-swapped via .set(). Also used on unmount and on voice change.
  function killSynth() {
    if (!synthRef.current) return
    try { synthRef.current.releaseAll() } catch { /* ignore */ }
    try { synthRef.current.disconnect() } catch { /* ignore */ }
    try { synthRef.current.dispose() } catch { /* ignore */ }
    synthRef.current = null
    builtVoiceRef.current = null
    midiActiveFreqsRef.current.clear()
    setActiveIds(new Set())
  }

  // Voice change → drop the current synth so the next note builds fresh.
  useEffect(() => {
    if (synthRef.current && builtVoiceRef.current !== voice) {
      killSynth()
    }
  }, [voice])

  // Unmount: tear everything down so background notes don't bleed
  // when the user switches tabs or leaves the page.
  useEffect(() => {
    return () => {
      killSynth()
      if (midiAccessRef.current) {
        midiAccessRef.current.inputs.forEach((input) => { input.onmidimessage = null })
      }
    }
  }, [])

  // ----- Press / release ------------------------------------------------
  const pressKey = async (key) => {
    if (activeIds.has(key.id)) return
    const synth = await ensureSynth()
    try { synth.triggerAttack(key.freq) } catch (err) { console.warn(err) }
    setActiveIds((prev) => {
      const next = new Set(prev)
      next.add(key.id)
      return next
    })
  }

  const releaseKey = (key) => {
    if (!activeIds.has(key.id)) return
    if (synthRef.current) {
      try { synthRef.current.triggerRelease(key.freq) } catch { /* ignore */ }
    }
    setActiveIds((prev) => {
      const next = new Set(prev)
      next.delete(key.id)
      return next
    })
  }

  // ----- Computer keyboard listener ------------------------------------
  // Looks up the pressed character in the visible keys' shortcut map.
  // Skips events from input/select/textarea so typing in the sidebar
  // doesn't honk notes at the user.
  useEffect(() => {
    if (!keys.length) return
    const byChar = new Map()
    keys.forEach((k) => { if (k.shortcut) byChar.set(k.shortcut, k) })

    const isTypingTarget = (el) => {
      if (!el) return false
      const tag = el.tagName
      return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || el.isContentEditable
    }

    const onDown = (e) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return
      if (isTypingTarget(e.target)) return
      const ch = e.key.toLowerCase()
      if (ch === 'z') { setBaseOctaveShift((s) => Math.max(-3, s - 1)); return }
      if (ch === 'x') { setBaseOctaveShift((s) => Math.min(3, s + 1)); return }
      const key = byChar.get(ch)
      if (key) {
        e.preventDefault()
        pressKey(key)
      }
    }
    const onUp = (e) => {
      if (isTypingTarget(e.target)) return
      const ch = e.key.toLowerCase()
      const key = byChar.get(ch)
      if (key) {
        e.preventDefault()
        releaseKey(key)
      }
    }
    // Failsafe: blur the window (tab switch, alt-tab) → release everything
    // to prevent stuck notes.
    const onBlur = () => {
      if (synthRef.current) {
        try { synthRef.current.releaseAll() } catch { /* ignore */ }
      }
      setActiveIds(new Set())
      midiActiveFreqsRef.current.clear()
    }
    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
    window.addEventListener('blur', onBlur)
    return () => {
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup', onUp)
      window.removeEventListener('blur', onBlur)
    }
    // pressKey/releaseKey close over activeIds + synth so they don't
    // need explicit deps here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keys])

  // ----- WebMIDI ----------------------------------------------------------
  const requestMidi = async () => {
    if (!navigator.requestMIDIAccess) {
      setMidiAccessState('unsupported')
      return
    }
    try {
      const access = await navigator.requestMIDIAccess({ sysex: false })
      midiAccessRef.current = access
      const inputs = Array.from(access.inputs.values())
      setMidiInputs(inputs)
      setMidiAccessState('granted')
      if (inputs.length && !midiInputId) setMidiInputId(inputs[0].id)
      // React to live device changes (plug/unplug).
      access.onstatechange = () => {
        setMidiInputs(Array.from(access.inputs.values()))
      }
    } catch {
      setMidiAccessState('denied')
    }
  }

  // Map an incoming MIDI note number to a frequency in the custom tuning.
  // Convention: MIDI 60 (Middle C) = degree 0 at the user's current base
  // octave. Notes above wrap through the N tuning degrees, jumping an
  // octave (× 2) every N steps. Notes below mirror downward.
  const midiNoteToFreq = (noteNum) => {
    if (!N) return baseFreq
    const offset = noteNum - 60
    const deg = ((offset % N) + N) % N
    const octFromMidi = Math.floor(offset / N)
    const totalOct = octFromMidi + baseOctaveShift
    const ratio = tuning[deg] * Math.pow(2, totalOct)
    return baseFreq * ratio
  }

  // Subscribe to the selected MIDI input. Cleans up + re-subscribes when
  // the user picks a different device.
  useEffect(() => {
    if (!midiAccessRef.current || !midiInputId) return
    const input = midiAccessRef.current.inputs.get(midiInputId)
    if (!input) return

    const handler = async (e) => {
      const [status, note, vel] = e.data
      const cmd = status & 0xf0
      if (cmd === 0x90 && vel > 0) {                     // note on
        const freq = midiNoteToFreq(note)
        const synth = await ensureSynth()
        try { synth.triggerAttack(freq) } catch { /* ignore */ }
        midiActiveFreqsRef.current.set(note, freq)
        setActiveIds((prev) => {
          const next = new Set(prev)
          next.add(`midi-${note}`)
          return next
        })
      } else if (cmd === 0x80 || (cmd === 0x90 && vel === 0)) {   // note off
        const freq = midiActiveFreqsRef.current.get(note)
        if (freq && synthRef.current) {
          try { synthRef.current.triggerRelease(freq) } catch { /* ignore */ }
        }
        midiActiveFreqsRef.current.delete(note)
        setActiveIds((prev) => {
          const next = new Set(prev)
          next.delete(`midi-${note}`)
          return next
        })
      }
    }
    input.onmidimessage = handler
    return () => { input.onmidimessage = null }
    // midiNoteToFreq depends on tuning/baseFreq/baseOctaveShift via closure;
    // including the source values keeps the handler in sync.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [midiInputId, tuning, baseFreq, baseOctaveShift, voice])

  // Visible-key highlight when a MIDI note plays a frequency that matches
  // a visible key (within 0.5 Hz). Lets the user see what their hardware
  // is doing on the on-screen keyboard.
  const midiHighlights = useMemo(() => {
    const set = new Set()
    if (!midiActiveFreqsRef.current.size) return set
    for (const freq of midiActiveFreqsRef.current.values()) {
      for (const k of keys) {
        if (Math.abs(k.freq - freq) < 0.5) {
          set.add(k.id)
          break
        }
      }
    }
    return set
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeIds, keys])

  // ----- Empty state ----------------------------------------------------
  if (!N) {
    return (
      <div className="text-center py-12 text-biotuner-light/60">
        <Music2 className="w-12 h-12 mx-auto mb-3 opacity-30" />
        Run an analysis first to generate a tuning to play.
      </div>
    )
  }

  // ----- Render ---------------------------------------------------------
  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <h2 className="text-lg font-bold text-biotuner-primary flex items-center gap-2">
          <Music2 className="w-5 h-5" /> Keyboard
        </h2>
        <p className="text-sm text-biotuner-light/70 mt-1">
          Play your {N}-note tuning across {octaves} octave{octaves !== 1 ? 's' : ''} —
          mouse, touch, computer keyboard ({SHORTCUT_ROW_1[0]}–{SHORTCUT_ROW_1[Math.min(N - 1, SHORTCUT_ROW_1.length - 1)]}),
          or a MIDI controller.
        </p>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
        <div>
          <label className="block text-[10px] uppercase tracking-wider text-biotuner-light/50 mb-1">
            Voice
          </label>
          <select
            value={voice}
            onChange={(e) => setVoice(e.target.value)}
            className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-2 text-sm"
          >
            {Object.keys(VOICES).map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-[10px] uppercase tracking-wider text-biotuner-light/50 mb-1">
            Base Hz
          </label>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={55}
              max={880}
              step={1}
              value={baseFreq}
              onChange={(e) => setBaseFreq(parseInt(e.target.value, 10))}
              className="flex-1 accent-biotuner-primary"
            />
            <input
              type="number"
              min={20}
              max={2000}
              step={1}
              value={baseFreq}
              onChange={(e) => {
                const v = parseInt(e.target.value, 10)
                if (Number.isFinite(v)) setBaseFreq(Math.max(20, Math.min(2000, v)))
              }}
              className="w-16 bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-1 text-xs text-right font-mono"
            />
          </div>
        </div>

        <div>
          <label className="block text-[10px] uppercase tracking-wider text-biotuner-light/50 mb-1">
            Octaves
          </label>
          <select
            value={octaves}
            onChange={(e) => setOctaves(parseInt(e.target.value, 10))}
            className="w-full bg-biotuner-dark-800 border border-biotuner-dark-600 rounded-md p-2 text-sm"
          >
            {[1, 2, 3, 4].map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-[10px] uppercase tracking-wider text-biotuner-light/50 mb-1">
            Octave shift
          </label>
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => setBaseOctaveShift((s) => Math.max(-3, s - 1))}
              className="min-h-[40px] flex-1 px-2 bg-biotuner-dark-800 border border-biotuner-dark-600 rounded text-sm hover:border-biotuner-primary/50"
              title="Shift base octave down (z)"
            >
              −
            </button>
            <span className="flex-1 text-center font-mono text-sm">
              {baseOctaveShift > 0 ? '+' : ''}{baseOctaveShift}
            </span>
            <button
              onClick={() => setBaseOctaveShift((s) => Math.min(3, s + 1))}
              className="min-h-[40px] flex-1 px-2 bg-biotuner-dark-800 border border-biotuner-dark-600 rounded text-sm hover:border-biotuner-primary/50"
              title="Shift base octave up (x)"
            >
              +
            </button>
          </div>
        </div>
      </div>

      {/* MIDI panel */}
      <div className="bg-biotuner-dark-900/60 border border-biotuner-dark-600 rounded-lg p-3 space-y-2">
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div className="flex items-center gap-2">
            <Cable className="w-4 h-4 text-biotuner-accent" />
            <span className="text-sm font-medium text-biotuner-accent">MIDI input</span>
          </div>
          {midiAccessState === 'idle' && (
            <button
              onClick={requestMidi}
              className="text-xs min-h-[36px] px-3 py-1.5 rounded bg-biotuner-accent/10 border border-biotuner-accent/40 text-biotuner-accent hover:bg-biotuner-accent/20"
            >
              Connect MIDI
            </button>
          )}
          {midiAccessState === 'granted' && midiInputs.length > 0 && (
            <select
              value={midiInputId}
              onChange={(e) => setMidiInputId(e.target.value)}
              className="flex-1 min-w-0 max-w-xs bg-biotuner-dark-800 border border-biotuner-dark-600 rounded p-1.5 text-xs"
            >
              {midiInputs.map((input) => (
                <option key={input.id} value={input.id}>
                  {input.name || `Device ${input.id.slice(0, 6)}`}
                </option>
              ))}
            </select>
          )}
          {midiAccessState === 'granted' && midiInputs.length === 0 && (
            <span className="text-xs text-biotuner-light/50">No MIDI devices detected</span>
          )}
          {midiAccessState === 'denied' && (
            <span className="text-xs text-red-400">Permission denied</span>
          )}
          {midiAccessState === 'unsupported' && (
            <span className="text-xs text-biotuner-light/50">
              WebMIDI unsupported (try Chrome or Edge)
            </span>
          )}
        </div>
        <p className="text-[10px] text-biotuner-light/40 leading-snug">
          MIDI note 60 (Middle&nbsp;C) = scale step 0. Notes above/below cycle through the {N}
          ratios across octaves, so every key on a hardware keyboard maps to a degree of your
          custom tuning.
        </p>
      </div>

      {/* The keys */}
      <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
        <div className="flex gap-1 min-w-fit pb-2">
          {keys.map((k) => {
            const isActive = activeIds.has(k.id) || midiHighlights.has(k.id)
            // Mark the start of each octave with a left accent border so
            // the user can see the octave boundaries at a glance.
            const isOctaveStart = k.deg === 0 && k.oct > 0
            return (
              <button
                key={k.id}
                onMouseDown={(e) => { e.preventDefault(); pressKey(k) }}
                onMouseUp={() => releaseKey(k)}
                onMouseLeave={() => releaseKey(k)}
                onTouchStart={(e) => { e.preventDefault(); pressKey(k) }}
                onTouchEnd={(e) => { e.preventDefault(); releaseKey(k) }}
                onContextMenu={(e) => e.preventDefault()}
                className={`
                  flex-shrink-0 w-12 sm:w-14 h-32 sm:h-40 rounded-md border-2
                  flex flex-col justify-end items-center pb-2 px-1 select-none
                  transition-all duration-75 touch-none
                  ${isActive
                    ? 'bg-biotuner-primary/40 border-biotuner-primary shadow-[inset_0_0_18px_rgba(0,217,255,0.55)] -translate-y-0.5'
                    : 'bg-biotuner-dark-700 border-biotuner-dark-600 hover:border-biotuner-primary/50'}
                  ${isOctaveStart ? 'ml-2' : ''}
                `}
                aria-label={`Step ${k.deg + 1}, octave ${k.oct + 1}, ${k.freq.toFixed(1)} Hz`}
              >
                {k.shortcut && (
                  <span className={`text-[9px] font-mono mb-1 ${
                    isActive ? 'text-biotuner-dark-900' : 'text-biotuner-light/30'
                  }`}>
                    {k.shortcut === ';' ? ';' : k.shortcut.toUpperCase()}
                  </span>
                )}
                <span className={`text-[10px] font-mono ${
                  isActive ? 'text-biotuner-dark-900' : 'text-biotuner-light/50'
                }`}>
                  #{k.deg + 1}
                </span>
                <span className={`text-[10px] sm:text-xs font-mono font-bold mt-0.5 ${
                  isActive ? 'text-biotuner-dark-900' : 'text-biotuner-primary'
                }`}>
                  {k.ratio.toFixed(2)}
                </span>
                <span className={`text-[9px] font-mono mt-0.5 ${
                  isActive ? 'text-biotuner-dark-900' : 'text-biotuner-accent/70'
                }`}>
                  {k.freq.toFixed(0)} Hz
                </span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Shortcut hint */}
      <div className="text-[10px] sm:text-xs text-biotuner-light/40 bg-biotuner-dark-900/50 rounded p-2 leading-relaxed">
        <span className="font-bold uppercase tracking-wider mr-2 text-biotuner-light/60">
          Shortcuts
        </span>
        <kbd className="font-mono">a s d f g h j k l ; '</kbd> play notes ·
        <kbd className="font-mono ml-2">q w e r t y …</kbd> second octave ·
        <kbd className="font-mono ml-2">z</kbd>/<kbd className="font-mono">x</kbd> octave shift
      </div>
    </div>
  )
}
