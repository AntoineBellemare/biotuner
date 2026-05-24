/**
 * TimbreSynth — Web Audio additive synth with per-partial-per-modulator
 * routing, mirroring biotuner.harmonic_timbre's render_modulated path.
 *
 * Why we build this ourselves instead of layering Tone.js:
 *
 *   - The Timbre tab's professional / intuitive / elegant promise hinges
 *     on "what you hear matches what you export". To honour that, AM/FM
 *     modulators must route to the EXACT carrier partial they were
 *     attached to, not to a global LFO summed over all partials.
 *   - Tone's PolySynth uses one oscillator per voice; here each "note"
 *     spawns N oscillators (one per partial) plus 2K LFOs (one per
 *     AM/FM modulator on each partial). Native Web Audio gives us the
 *     fine-grained routing graph for free.
 *   - The audio graph stays cheap: typical biotuner output is 5–20
 *     partials, modulators only present when PAC/CFC ran. A 5-partial
 *     note with one PAC pair = 6 oscillators total.
 *
 * Lifecycle (mirrors KeyboardTab's killSynth pattern):
 *   const synth = new TimbreSynth()
 *   await synth.start()                        // unlocks AudioContext
 *   synth.loadTimbre(computeResponse)          // payload from /api/timbre/compute
 *   synth.setAdsr({ attack, decay, sustain, release })
 *   synth.setVolume(-8)                        // dB
 *   synth.setModulationStrength(1.0)           // master scale on all mod depths
 *   synth.noteOn(220)                          // play at fundamental (Hz)
 *   synth.noteOff()
 *   synth.dispose()                            // tear everything down on unmount
 *
 * All AM/FM connections are torn down on noteOff so we don't leak
 * oscillators across notes.
 */

const DEFAULT_ADSR = { attack: 0.03, decay: 0.15, sustain: 0.55, release: 0.4 }

/** Convert dB to linear gain (Tone.js convention: 0dB = unity). */
function dbToGain(db) {
  return Math.pow(10, db / 20)
}

export class TimbreSynth {
  constructor() {
    this.ctx = null
    this.master = null              // GainNode — final output stage
    this.timbre = null              // last compute_timbre response
    this.adsr = { ...DEFAULT_ADSR }
    this.volumeDb = -8
    this.modulationStrength = 1.0   // master scale on every modulator depth
    this.voice = null               // current voice (single-voice for simplicity)
    this._started = false
  }

  /** Lazy AudioContext creation — must run inside a user gesture. */
  async start() {
    if (this._started) return
    this.ctx = new (window.AudioContext || window.webkitAudioContext)()
    if (this.ctx.state === 'suspended') {
      await this.ctx.resume()
    }
    this.master = this.ctx.createGain()
    this.master.gain.value = 0
    this.master.connect(this.ctx.destination)
    this._started = true
  }

  /** Store a /api/timbre/compute response. Doesn't make sound on its own. */
  loadTimbre(timbreData) {
    this.timbre = timbreData
  }

  setAdsr({ attack, decay, sustain, release }) {
    if (attack  != null) this.adsr.attack  = Math.max(0.001, attack)
    if (decay   != null) this.adsr.decay   = Math.max(0.001, decay)
    if (sustain != null) this.adsr.sustain = Math.max(0,     Math.min(1, sustain))
    if (release != null) this.adsr.release = Math.max(0.001, release)
  }

  setVolume(db) {
    this.volumeDb = db
    // If a note is currently sustaining, slide to the new volume
    // smoothly so the user can ride the gain without zipper noise.
    if (this.voice && this.master) {
      const t = this.ctx.currentTime
      const target = dbToGain(db) * this.adsr.sustain
      this.master.gain.cancelScheduledValues(t)
      this.master.gain.linearRampToValueAtTime(target, t + 0.05)
    }
  }

  setModulationStrength(strength) {
    this.modulationStrength = Math.max(0, strength)
    // Live update for any sustaining voice: scale every modulator's
    // depth-gain by (newStrength / oldStrength). Avoids re-attacking.
    if (this.voice) {
      const t = this.ctx.currentTime
      for (const mod of [...this.voice.amMods, ...this.voice.fmMods]) {
        const targetDepth = mod.baseDepth * this.modulationStrength
        mod.depthNode.gain.cancelScheduledValues(t)
        mod.depthNode.gain.linearRampToValueAtTime(targetDepth, t + 0.03)
      }
    }
  }

  /** True when a note is currently attacking or sustaining. */
  isPlaying() {
    return !!this.voice
  }

  /**
   * Trigger a note. ``freq`` is the fundamental in Hz; partials are
   * stretched proportionally — the analysis's first partial scales to
   * the chosen freq, and every other partial keeps its harmonic ratio.
   *
   * Calling noteOn while a previous note is sustaining tears it down
   * cleanly (single-voice; chord-playing would need a noteId map).
   */
  noteOn(freq) {
    if (!this._started) return
    if (!this.timbre || !this.timbre.partials_hz?.length) return
    if (this.voice) this._teardownVoice(0)

    const ctx = this.ctx
    const t = ctx.currentTime
    const t0 = this.timbre

    // Stretch ratio — apply uniformly to every partial so harmonic
    // relationships are preserved when the user plays at a different
    // fundamental than the analysis's natural base_freq.
    const stretch = freq / (t0.base_freq || t0.partials_hz[0] || freq)

    // Per-partial chain: osc -> partialGain -> master
    const oscs = []
    const partialGains = []
    const amMods = []
    const fmMods = []

    for (let i = 0; i < t0.partials_hz.length; i++) {
      const partialHz = t0.partials_hz[i] * stretch
      const amp = (t0.amplitudes?.[i] ?? 1.0)

      const osc = ctx.createOscillator()
      osc.type = 'sine'
      osc.frequency.value = partialHz

      // Per-partial gain — also the AM modulation target.
      const pGain = ctx.createGain()
      pGain.gain.value = amp

      // Wire AM modulators that target this partial.
      const amForThis = (t0.am_modulators || []).filter(
        (m) => m.enabled && m.carrier_idx === i,
      )
      for (const m of amForThis) {
        const lfo = ctx.createOscillator()
        lfo.type = 'sine'
        lfo.frequency.value = m.mod_freq
        // AM depth is normalized 0..1 in biotuner; convert to amplitude
        // units against THIS partial's amp so depth=1 fully modulates.
        const baseDepth = m.depth * amp
        const depthNode = ctx.createGain()
        depthNode.gain.value = baseDepth * this.modulationStrength
        lfo.connect(depthNode)
        depthNode.connect(pGain.gain)
        lfo.start(t)
        amMods.push({ lfo, depthNode, baseDepth, id: m.id })
      }

      // Wire FM modulators that target this partial.
      const fmForThis = (t0.fm_modulators || []).filter(
        (m) => m.enabled && m.carrier_idx === i,
      )
      for (const m of fmForThis) {
        const lfo = ctx.createOscillator()
        lfo.type = 'sine'
        lfo.frequency.value = m.mod_freq
        const baseDepth = m.depth  // Hz deviation
        const depthNode = ctx.createGain()
        depthNode.gain.value = baseDepth * this.modulationStrength
        lfo.connect(depthNode)
        depthNode.connect(osc.frequency)
        lfo.start(t)
        fmMods.push({ lfo, depthNode, baseDepth, id: m.id })
      }

      osc.connect(pGain)
      pGain.connect(this.master)
      osc.start(t)
      oscs.push(osc)
      partialGains.push(pGain)
    }

    // Apply spectral_tilt as a per-partial gain attenuation (1/f^k
    // weighting against the partial frequencies). Match exporter
    // semantics: gain *= (f0 / f) ^ tilt for each partial.
    if (t0.spectral_tilt != null && t0.spectral_tilt > 0) {
      const f0 = t0.partials_hz[0] * stretch
      const k = t0.spectral_tilt
      partialGains.forEach((g, i) => {
        const f = t0.partials_hz[i] * stretch
        g.gain.value *= Math.pow(f0 / f, k)
      })
    }

    // ADSR on the master.
    const peak = dbToGain(this.volumeDb)
    const { attack, decay, sustain } = this.adsr
    this.master.gain.cancelScheduledValues(t)
    this.master.gain.setValueAtTime(0, t)
    this.master.gain.linearRampToValueAtTime(peak, t + attack)
    this.master.gain.linearRampToValueAtTime(peak * sustain, t + attack + decay)

    this.voice = { oscs, partialGains, amMods, fmMods, startedAt: t }
  }

  /** Release the current note. Schedules teardown after release time. */
  noteOff() {
    if (!this.voice) return
    const t = this.ctx.currentTime
    const r = this.adsr.release
    this.master.gain.cancelScheduledValues(t)
    this.master.gain.setValueAtTime(this.master.gain.value, t)
    this.master.gain.linearRampToValueAtTime(0, t + r)
    this._teardownVoice(r + 0.05)
  }

  /** Internal: stop oscillators after ``delay`` seconds and clear voice. */
  _teardownVoice(delay) {
    const voice = this.voice
    if (!voice) return
    const stopAt = this.ctx.currentTime + Math.max(0, delay)
    for (const o of voice.oscs) {
      try { o.stop(stopAt) } catch { /* ignore */ }
    }
    for (const m of [...voice.amMods, ...voice.fmMods]) {
      try { m.lfo.stop(stopAt) } catch { /* ignore */ }
    }
    this.voice = null
  }

  /** Full teardown — call on unmount. */
  dispose() {
    if (this.voice) this._teardownVoice(0)
    if (this.master) {
      try { this.master.disconnect() } catch { /* ignore */ }
    }
    if (this.ctx) {
      try { this.ctx.close() } catch { /* ignore */ }
    }
    this.ctx = null
    this.master = null
    this._started = false
  }
}
