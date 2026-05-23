/**
 * Lazy-loaded Three.js viewer for biotuner GeometryData (mesh_3d /
 * point_cloud_3d / 3D tree). The user can rotate by drag, zoom by wheel,
 * pan by right-drag, via OrbitControls.
 *
 * Three.js is dynamically imported on mount so the main bundle stays light
 * for users who never open this tab.
 */

import { useEffect, useRef, useState } from 'react'

export default function ThreeViewer({
  geometry,           // GeometryData JSON from /api/harmonic-geometry
  color = '#06b6d4',
  colorEnd = null,    // when set + gradient, interpolate per-vertex
  gradient = false,
  background = '#0a0a0a',
  pointSize = 0.03,
  wireframe = false,
  autoRotate = true,
}) {
  const mountRef = useRef(null)
  const stateRef = useRef({})   // holds three.js instances across renders
  const [status, setStatus] = useState('loading')

  // ---- Mount / unmount: build the scene once ----
  useEffect(() => {
    let cancelled = false
    let cleanup = null

    ;(async () => {
      try {
        const THREE = await import('three')
        const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js')

        if (cancelled || !mountRef.current) return

        const mount = mountRef.current
        const width = mount.clientWidth
        const height = mount.clientHeight || Math.max(width, 400)

        const scene = new THREE.Scene()
        scene.background = new THREE.Color(background)

        const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000)
        camera.position.set(0, 0, 4)

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
        renderer.setSize(width, height, false)
        // Make the WebGL canvas fill its container exactly. setSize(..., false)
        // skips touching the canvas CSS, so the default inline display would
        // otherwise leave the canvas mis-sized / off-center on mobile.
        renderer.domElement.style.display = 'block'
        renderer.domElement.style.width = '100%'
        renderer.domElement.style.height = '100%'
        mount.appendChild(renderer.domElement)

        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.08
        controls.autoRotate = autoRotate
        controls.autoRotateSpeed = 0.6

        // Lights for mesh styles
        const ambient = new THREE.AmbientLight(0xffffff, 0.5)
        scene.add(ambient)
        const key = new THREE.DirectionalLight(0xffffff, 0.9)
        key.position.set(3, 4, 5)
        scene.add(key)
        const fill = new THREE.DirectionalLight(0xa3b3cc, 0.3)
        fill.position.set(-3, -2, -2)
        scene.add(fill)

        // Resize observer
        const ro = new ResizeObserver(() => {
          const w = mount.clientWidth
          const h = mount.clientHeight || Math.max(w, 400)
          renderer.setSize(w, h, false)
          camera.aspect = w / h
          camera.updateProjectionMatrix()
        })
        ro.observe(mount)

        // rAF loop
        let rafId
        const tick = () => {
          controls.update()
          renderer.render(scene, camera)
          rafId = requestAnimationFrame(tick)
        }
        rafId = requestAnimationFrame(tick)

        stateRef.current = {
          THREE, scene, camera, renderer, controls,
          currentObject: null,
        }
        setStatus('ready')

        cleanup = () => {
          if (rafId) cancelAnimationFrame(rafId)
          ro.disconnect()
          controls.dispose()
          if (stateRef.current.currentObject) {
            disposeObject(stateRef.current.currentObject)
          }
          renderer.dispose()
          if (renderer.domElement?.parentNode === mount) {
            mount.removeChild(renderer.domElement)
          }
        }
      } catch (err) {
        console.error('ThreeViewer init failed:', err)
        if (!cancelled) setStatus('error')
      }
    })()

    return () => {
      cancelled = true
      if (cleanup) cleanup()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])  // setup once

  // ---- Whenever geometry / styling changes, rebuild the rendered object ----
  useEffect(() => {
    const { THREE, scene, camera, controls } = stateRef.current
    if (!THREE || !geometry) return

    // Dispose any prior object
    if (stateRef.current.currentObject) {
      disposeObject(stateRef.current.currentObject)
      scene.remove(stateRef.current.currentObject)
      stateRef.current.currentObject = null
    }

    const obj = buildObject(THREE, geometry, {
      color, colorEnd, gradient, pointSize, wireframe,
    })
    if (!obj) return

    // Fit camera to the geometry
    const box = new THREE.Box3().setFromObject(obj)
    const size = new THREE.Vector3()
    box.getSize(size)
    const center = new THREE.Vector3()
    box.getCenter(center)
    const maxDim = Math.max(size.x, size.y, size.z) || 1
    const distance = maxDim * 2.2 + 0.5
    camera.position.set(center.x + distance * 0.7, center.y + distance * 0.5, center.z + distance)
    if (controls) {
      controls.target.copy(center)
      controls.update()
    }

    scene.add(obj)
    stateRef.current.currentObject = obj
  }, [geometry, color, colorEnd, gradient, pointSize, wireframe])

  // ---- Reflect autoRotate / background changes ----
  useEffect(() => {
    if (stateRef.current.controls) {
      stateRef.current.controls.autoRotate = autoRotate
    }
  }, [autoRotate])

  useEffect(() => {
    const { THREE, scene } = stateRef.current
    if (THREE && scene) scene.background = new THREE.Color(background)
  }, [background])

  return (
    <div ref={mountRef} className="relative w-full h-full">
      {status === 'loading' && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-biotuner-light/60">
          Loading 3D engine…
        </div>
      )}
      {status === 'error' && (
        <div className="absolute inset-0 flex items-center justify-center text-sm text-red-400">
          Failed to load Three.js
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildObject(THREE, geometry, opts) {
  const { coordinates, faces, edges, geom_type } = geometry
  if (!coordinates?.length) return null

  const colorStart = new THREE.Color(opts.color)
  const colorFinal = opts.colorEnd ? new THREE.Color(opts.colorEnd) : null
  const useGradient = opts.gradient && colorFinal

  // ---- Build vertex color buffer when gradient is on. We interpolate by
  // height (Y) for surfaces & points, by vertex order for lines/trees. ----
  let vertexColors = null
  if (useGradient) {
    vertexColors = new Float32Array(coordinates.length * 3)
    let yMin = Infinity, yMax = -Infinity
    for (const c of coordinates) {
      const y = c[1] ?? 0
      if (y < yMin) yMin = y
      if (y > yMax) yMax = y
    }
    const yRange = (yMax - yMin) || 1
    const tmp = new THREE.Color()
    for (let i = 0; i < coordinates.length; i++) {
      const c = coordinates[i]
      const yT = ((c[1] ?? 0) - yMin) / yRange  // 0..1 by height
      tmp.copy(colorStart).lerp(colorFinal, yT)
      vertexColors[i * 3]     = tmp.r
      vertexColors[i * 3 + 1] = tmp.g
      vertexColors[i * 3 + 2] = tmp.b
    }
  }

  const positionsFor = (count, lookup) => {
    const arr = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      const c = lookup(i)
      arr[i * 3]     = c?.[0] ?? 0
      arr[i * 3 + 1] = c?.[1] ?? 0
      arr[i * 3 + 2] = c?.[2] ?? 0
    }
    return arr
  }

  // ---- mesh_3d: triangulated surface ----
  if (geom_type === 'mesh_3d' && faces?.length) {
    const positions = positionsFor(coordinates.length, (i) => coordinates[i])
    const indices = new Uint32Array(faces.length * 3)
    for (let i = 0; i < faces.length; i++) {
      const f = faces[i]
      indices[i * 3]     = f[0]
      indices[i * 3 + 1] = f[1]
      indices[i * 3 + 2] = f[2]
    }
    const buf = new THREE.BufferGeometry()
    buf.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    if (vertexColors) buf.setAttribute('color', new THREE.BufferAttribute(vertexColors, 3))
    buf.setIndex(new THREE.BufferAttribute(indices, 1))
    buf.computeVertexNormals()

    const material = new THREE.MeshStandardMaterial({
      color: vertexColors ? 0xffffff : colorStart,
      vertexColors: !!vertexColors,
      wireframe: opts.wireframe,
      flatShading: false,
      metalness: 0.05,
      roughness: 0.55,
      side: THREE.DoubleSide,
    })
    return new THREE.Mesh(buf, material)
  }

  // ---- point_cloud_3d ----
  if (geom_type === 'point_cloud_3d' ||
      (!faces?.length && !edges?.length && Array.isArray(coordinates[0]))) {
    const positions = positionsFor(coordinates.length, (i) => coordinates[i])
    const buf = new THREE.BufferGeometry()
    buf.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    if (vertexColors) buf.setAttribute('color', new THREE.BufferAttribute(vertexColors, 3))
    const material = new THREE.PointsMaterial({
      color: vertexColors ? 0xffffff : colorStart,
      vertexColors: !!vertexColors,
      size: opts.pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    })
    return new THREE.Points(buf, material)
  }

  // ---- graph / tree / curve_3d / lsystem: line segments ----
  if (edges?.length) {
    const positions = new Float32Array(edges.length * 2 * 3)
    const edgeColors = useGradient ? new Float32Array(edges.length * 2 * 3) : null
    const tmp = useGradient ? new THREE.Color() : null
    for (let i = 0; i < edges.length; i++) {
      const [aIdx, bIdx] = edges[i]
      const pa = coordinates[aIdx] || [0, 0, 0]
      const pb = coordinates[bIdx] || [0, 0, 0]
      positions[i * 6]     = pa[0] ?? 0
      positions[i * 6 + 1] = pa[1] ?? 0
      positions[i * 6 + 2] = pa[2] ?? 0
      positions[i * 6 + 3] = pb[0] ?? 0
      positions[i * 6 + 4] = pb[1] ?? 0
      positions[i * 6 + 5] = pb[2] ?? 0
      if (useGradient) {
        const tA = aIdx / Math.max(1, coordinates.length - 1)
        const tB = bIdx / Math.max(1, coordinates.length - 1)
        tmp.copy(colorStart).lerp(colorFinal, tA)
        edgeColors[i * 6]     = tmp.r
        edgeColors[i * 6 + 1] = tmp.g
        edgeColors[i * 6 + 2] = tmp.b
        tmp.copy(colorStart).lerp(colorFinal, tB)
        edgeColors[i * 6 + 3] = tmp.r
        edgeColors[i * 6 + 4] = tmp.g
        edgeColors[i * 6 + 5] = tmp.b
      }
    }
    const buf = new THREE.BufferGeometry()
    buf.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    if (edgeColors) buf.setAttribute('color', new THREE.BufferAttribute(edgeColors, 3))
    const material = new THREE.LineBasicMaterial({
      color: edgeColors ? 0xffffff : colorStart,
      vertexColors: !!edgeColors,
      linewidth: 1,
    })
    return new THREE.LineSegments(buf, material)
  }

  // Fallback: points
  const positions = positionsFor(coordinates.length, (i) => coordinates[i])
  const buf = new THREE.BufferGeometry()
  buf.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  if (vertexColors) buf.setAttribute('color', new THREE.BufferAttribute(vertexColors, 3))
  return new THREE.Points(buf, new THREE.PointsMaterial({
    color: vertexColors ? 0xffffff : colorStart,
    vertexColors: !!vertexColors,
    size: opts.pointSize,
  }))
}

function disposeObject(obj) {
  if (!obj) return
  if (obj.geometry) obj.geometry.dispose()
  if (obj.material) {
    if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose())
    else obj.material.dispose()
  }
}
