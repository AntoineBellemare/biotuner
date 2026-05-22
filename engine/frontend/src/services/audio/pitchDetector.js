/**
 * Real-time pitch detection from the microphone.
 *
 * Uses `pitchy`'s McLeod Pitch Method on a continuous AnalyserNode buffer.
 * Returns a `{ stop, getState }` handle. The onPitch callback fires on every
 * tick (~60 Hz via requestAnimationFrame) with either { frequency, clarity }
 * when a confident pitch is found or { frequency: null, clarity } otherwise.
 *
 * Output is exponentially smoothed in cents (perceptual scale) so the needle
 * doesn't twitch on every micro-fluctuation, but jumps quickly on real pitch
 * changes.
 */

import { PitchDetector } from 'pitchy'

const FFT_SIZE = 2048
const MIN_CLARITY = 0.9
const MIN_HZ = 30
const MAX_HZ = 5000

export async function startPitchDetection({
  onPitch,
  minClarity = MIN_CLARITY,
  smoothingFactor = 0.55,   // 0 = no smoothing, 1 = freeze (don't use 1).
} = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Microphone access not available in this browser.')
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  })

  const AudioCtx = window.AudioContext || window.webkitAudioContext
  const ctx = new AudioCtx()
  if (ctx.state === 'suspended') {
    try { await ctx.resume() } catch { /* ignore */ }
  }

  const source = ctx.createMediaStreamSource(stream)
  const analyser = ctx.createAnalyser()
  analyser.fftSize = FFT_SIZE
  source.connect(analyser)

  const detector = PitchDetector.forFloat32Array(analyser.fftSize)
  detector.minVolumeDecibels = -45  // softer threshold than default
  const buffer = new Float32Array(detector.inputLength)

  let smoothedFreq = null
  let running = true
  let rafId = null

  const tick = () => {
    if (!running) return
    analyser.getFloatTimeDomainData(buffer)
    const [pitch, clarity] = detector.findPitch(buffer, ctx.sampleRate)

    if (clarity >= minClarity && pitch >= MIN_HZ && pitch <= MAX_HZ) {
      if (smoothedFreq == null) {
        smoothedFreq = pitch
      } else {
        // Smooth on the log scale (cents-equivalent) so equal smoothing weight
        // feels consistent across registers.
        const cents = 1200 * Math.log2(pitch / smoothedFreq)
        // If the new pitch differs by > 200 cents, snap (string change /
        // octave jump). Otherwise blend.
        if (Math.abs(cents) > 200) {
          smoothedFreq = pitch
        } else {
          const blendCents = cents * (1 - smoothingFactor)
          smoothedFreq = smoothedFreq * Math.pow(2, blendCents / 1200)
        }
      }
      onPitch?.({ frequency: smoothedFreq, clarity })
    } else {
      // Decay smoothing memory so a long silence resets to "nothing detected"
      smoothedFreq = null
      onPitch?.({ frequency: null, clarity })
    }

    rafId = requestAnimationFrame(tick)
  }
  rafId = requestAnimationFrame(tick)

  const stop = async () => {
    if (!running) return
    running = false
    if (rafId) cancelAnimationFrame(rafId)
    try { analyser.disconnect() } catch { /* ignore */ }
    try { source.disconnect() } catch { /* ignore */ }
    for (const t of stream.getTracks()) t.stop()
    try { await ctx.close() } catch { /* ignore */ }
  }

  return {
    stop,
    getState: () => (running ? 'running' : 'stopped'),
  }
}
