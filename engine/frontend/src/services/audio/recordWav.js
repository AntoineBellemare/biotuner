/**
 * Microphone capture → WAV blob.
 *
 * Returns a `{ stop, level$, getState }` handle.
 *   stop()           → resolves to { blob, sampleRate, durationSec }
 *   level$.subscribe → callback receives 0..1 VU level updates (~50 Hz)
 *   getState()       → 'recording' | 'stopped'
 *
 * We deliberately disable echoCancellation / noiseSuppression / autoGainControl
 * because the downstream biotuner pipeline wants raw spectrum, not phone-call
 * conditioned audio.
 */

const FALLBACK_BUFFER_SIZE = 4096

function createLevelStream() {
  const subs = new Set()
  return {
    subscribe(fn) {
      subs.add(fn)
      return () => subs.delete(fn)
    },
    emit(v) {
      for (const fn of subs) fn(v)
    },
    clear() {
      subs.clear()
    },
  }
}

function encodeWav(samples, sampleRate) {
  // Mono 16-bit PCM WAV.
  const numSamples = samples.length
  const buffer = new ArrayBuffer(44 + numSamples * 2)
  const view = new DataView(buffer)

  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + numSamples * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)            // fmt chunk size
  view.setUint16(20, 1, true)             // PCM
  view.setUint16(22, 1, true)             // channels
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)// byte rate
  view.setUint16(32, 2, true)             // block align
  view.setUint16(34, 16, true)            // bits per sample
  writeString(36, 'data')
  view.setUint32(40, numSamples * 2, true)

  let off = 44
  for (let i = 0; i < numSamples; i++) {
    let s = Math.max(-1, Math.min(1, samples[i]))
    s = s < 0 ? s * 0x8000 : s * 0x7fff
    view.setInt16(off, s, true)
    off += 2
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

export async function startRecording({ maxSeconds = 60 } = {}) {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Microphone capture is not supported in this browser.')
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  })

  // AudioContext must be created/resumed inside the user gesture that triggered
  // this call — callers must invoke startRecording() synchronously from a click.
  const AudioCtx = window.AudioContext || window.webkitAudioContext
  const ctx = new AudioCtx()
  if (ctx.state === 'suspended') {
    try { await ctx.resume() } catch { /* ignore */ }
  }

  const source = ctx.createMediaStreamSource(stream)
  const analyser = ctx.createAnalyser()
  analyser.fftSize = 1024
  source.connect(analyser)

  const buffers = []      // array of Float32Array chunks
  let totalSamples = 0
  const level$ = createLevelStream()
  let state = 'recording'
  let stopFn = null
  let resolveStop = null
  const stopped = new Promise((res) => { resolveStop = res })

  // Level meter polling (using analyser, decoupled from capture)
  const levelData = new Uint8Array(analyser.frequencyBinCount)
  const levelInterval = setInterval(() => {
    if (state !== 'recording') return
    analyser.getByteTimeDomainData(levelData)
    let sum = 0
    for (let i = 0; i < levelData.length; i++) {
      const v = (levelData[i] - 128) / 128
      sum += v * v
    }
    const rms = Math.sqrt(sum / levelData.length)
    level$.emit(Math.min(1, rms * 2.5))
  }, 50)

  let processor = null
  let workletNode = null

  const onSamples = (chunk) => {
    if (state !== 'recording') return
    buffers.push(chunk)
    totalSamples += chunk.length
    if (totalSamples / ctx.sampleRate >= maxSeconds) {
      stopFn?.()
    }
  }

  // Prefer AudioWorklet; fall back to ScriptProcessor on older Safari.
  let useWorklet = false
  if (ctx.audioWorklet && typeof ctx.audioWorklet.addModule === 'function') {
    try {
      const workletCode = `
        class CaptureProcessor extends AudioWorkletProcessor {
          process(inputs) {
            const input = inputs[0]
            if (input && input[0]) {
              // copy because the buffer is reused by the engine
              this.port.postMessage(input[0].slice(0))
            }
            return true
          }
        }
        registerProcessor('biotuner-capture', CaptureProcessor)
      `
      const blob = new Blob([workletCode], { type: 'application/javascript' })
      const url = URL.createObjectURL(blob)
      await ctx.audioWorklet.addModule(url)
      URL.revokeObjectURL(url)

      workletNode = new AudioWorkletNode(ctx, 'biotuner-capture')
      workletNode.port.onmessage = (e) => onSamples(e.data)
      source.connect(workletNode)
      // Worklet doesn't need to reach the destination; connect to a muted gain
      // so the graph stays "live" on all browsers.
      const muted = ctx.createGain()
      muted.gain.value = 0
      workletNode.connect(muted).connect(ctx.destination)
      useWorklet = true
    } catch (err) {
      console.warn('AudioWorklet unavailable, falling back to ScriptProcessor:', err)
    }
  }

  if (!useWorklet) {
    // eslint-disable-next-line no-restricted-globals
    processor = ctx.createScriptProcessor(FALLBACK_BUFFER_SIZE, 1, 1)
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0)
      onSamples(new Float32Array(input))
    }
    source.connect(processor)
    const muted = ctx.createGain()
    muted.gain.value = 0
    processor.connect(muted).connect(ctx.destination)
  }

  stopFn = async () => {
    if (state !== 'recording') return
    state = 'stopped'
    clearInterval(levelInterval)
    level$.emit(0)

    try { workletNode?.disconnect() } catch { /* ignore */ }
    try { processor?.disconnect() } catch { /* ignore */ }
    try { source.disconnect() } catch { /* ignore */ }
    try { analyser.disconnect() } catch { /* ignore */ }
    for (const track of stream.getTracks()) track.stop()

    // Flatten Float32 chunks
    const merged = new Float32Array(totalSamples)
    let offset = 0
    for (const c of buffers) {
      merged.set(c, offset)
      offset += c.length
    }

    const sampleRate = ctx.sampleRate
    const blob = encodeWav(merged, sampleRate)

    try { await ctx.close() } catch { /* ignore */ }

    const result = {
      blob,
      sampleRate,
      durationSec: merged.length / sampleRate,
    }
    resolveStop(result)
    return result
  }

  return {
    stop: stopFn,
    stopped,
    level$,
    getState: () => state,
    getDurationSec: () => totalSamples / ctx.sampleRate,
  }
}
