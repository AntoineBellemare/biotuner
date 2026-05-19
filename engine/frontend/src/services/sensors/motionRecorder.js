/**
 * Device motion / orientation capture → CSV blob.
 *
 *   startMotion({ sensors: ['accel'|'gyro'|'orientation'|'mag'], hz, maxSeconds })
 *     → { stop, stopped, samples$, getState, getDurationSec }
 *
 *   stop() resolves to { blob, sampleRate, durationSec, columns }
 *
 * The recorder runs listeners at the device's native rate, then decimates to
 * the requested target rate (default 50 Hz) when building the CSV.
 *
 * iOS Safari requires DeviceMotionEvent.requestPermission() / DeviceOrientation
 * Event.requestPermission() to be called from a user gesture.
 */

const DEFAULT_HZ = 50
const COLUMNS = {
  accel:       ['ax', 'ay', 'az'],
  gyro:        ['gx', 'gy', 'gz'],
  orientation: ['alpha', 'beta', 'gamma'],
  mag:         ['mx', 'my', 'mz'],
}

function createStream() {
  const subs = new Set()
  return {
    subscribe(fn) { subs.add(fn); return () => subs.delete(fn) },
    emit(v) { for (const fn of subs) fn(v) },
  }
}

export function isMagnetometerAvailable() {
  return typeof window !== 'undefined' && 'Magnetometer' in window
}

export function isIos() {
  return typeof navigator !== 'undefined' &&
    /iP(hone|ad|od)/.test(navigator.userAgent)
}

async function requestMotionPermission() {
  if (typeof DeviceMotionEvent !== 'undefined' &&
      typeof DeviceMotionEvent.requestPermission === 'function') {
    const r = await DeviceMotionEvent.requestPermission()
    if (r !== 'granted') throw new Error('Motion permission denied.')
  }
}

async function requestOrientationPermission() {
  if (typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function') {
    const r = await DeviceOrientationEvent.requestPermission()
    if (r !== 'granted') throw new Error('Orientation permission denied.')
  }
}

export async function startMotion({
  sensors = ['accel'],
  hz = DEFAULT_HZ,
  maxSeconds = 30,
} = {}) {
  if (typeof window === 'undefined') throw new Error('Not in a browser')

  const needMotion = sensors.some((s) => s === 'accel' || s === 'gyro')
  const needOrient = sensors.includes('orientation')
  const needMag = sensors.includes('mag')

  if (needMotion) await requestMotionPermission()
  if (needOrient) await requestOrientationPermission()
  if (needMag && !isMagnetometerAvailable()) {
    throw new Error('Magnetometer is not exposed on this device (common on iOS).')
  }

  // Internal "latest reading" registers — listeners write here at native rate,
  // the sampler reads them at target hz.
  const latest = {
    accel:       { x: 0, y: 0, z: 0 },
    gyro:        { x: 0, y: 0, z: 0 },
    orientation: { alpha: 0, beta: 0, gamma: 0 },
    mag:         { x: 0, y: 0, z: 0 },
  }

  const onMotion = (e) => {
    if (sensors.includes('accel')) {
      const a = e.accelerationIncludingGravity || e.acceleration
      if (a) { latest.accel = { x: a.x ?? 0, y: a.y ?? 0, z: a.z ?? 0 } }
    }
    if (sensors.includes('gyro')) {
      const r = e.rotationRate
      if (r) { latest.gyro = { x: r.alpha ?? 0, y: r.beta ?? 0, z: r.gamma ?? 0 } }
    }
  }
  const onOrient = (e) => {
    latest.orientation = {
      alpha: e.alpha ?? 0,
      beta:  e.beta ?? 0,
      gamma: e.gamma ?? 0,
    }
  }

  if (needMotion) window.addEventListener('devicemotion', onMotion)
  if (needOrient) window.addEventListener('deviceorientation', onOrient)

  let magSensor = null
  if (needMag) {
    // eslint-disable-next-line no-undef
    magSensor = new Magnetometer({ frequency: hz })
    magSensor.addEventListener('reading', () => {
      latest.mag = { x: magSensor.x ?? 0, y: magSensor.y ?? 0, z: magSensor.z ?? 0 }
    })
    magSensor.start()
  }

  // Build column list
  const columns = ['time']
  for (const s of sensors) columns.push(...COLUMNS[s])

  const rows = []     // each row is [t, ...values]
  const samples$ = createStream()
  let state = 'recording'
  let resolveStop = null
  const stopped = new Promise((res) => { resolveStop = res })

  const startedAt = performance.now()
  const period = 1000 / hz

  const sampler = setInterval(() => {
    if (state !== 'recording') return
    const t = (performance.now() - startedAt) / 1000
    const row = [t]
    for (const s of sensors) {
      if (s === 'accel') row.push(latest.accel.x, latest.accel.y, latest.accel.z)
      else if (s === 'gyro') row.push(latest.gyro.x, latest.gyro.y, latest.gyro.z)
      else if (s === 'orientation') row.push(latest.orientation.alpha, latest.orientation.beta, latest.orientation.gamma)
      else if (s === 'mag') row.push(latest.mag.x, latest.mag.y, latest.mag.z)
    }
    rows.push(row)
    samples$.emit({ t, latest })
    if (t >= maxSeconds) stop()
  }, period)

  function buildCsv() {
    const header = columns.join(',')
    const body = rows.map((r) =>
      r.map((v, i) => (i === 0 ? v.toFixed(4) : Number(v).toFixed(5))).join(',')
    ).join('\n')
    return new Blob([header + '\n' + body + '\n'], { type: 'text/csv' })
  }

  function stop() {
    if (state !== 'recording') return Promise.resolve(null)
    state = 'stopped'
    clearInterval(sampler)
    if (needMotion) window.removeEventListener('devicemotion', onMotion)
    if (needOrient) window.removeEventListener('deviceorientation', onOrient)
    if (magSensor) try { magSensor.stop() } catch { /* ignore */ }

    const durationSec = rows.length ? rows[rows.length - 1][0] : 0
    const blob = buildCsv()
    const result = { blob, sampleRate: hz, durationSec, columns }
    resolveStop(result)
    return Promise.resolve(result)
  }

  return {
    stop,
    stopped,
    samples$,
    getState: () => state,
    getDurationSec: () => rows.length ? rows[rows.length - 1][0] : 0,
  }
}
