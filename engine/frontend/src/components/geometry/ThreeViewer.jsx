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

    const obj = buildObject(THREE, geometry, { color, pointSize, wireframe })
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
  }, [geometry, color, pointSize, wireframe])

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
  const color = new THREE.Color(opts.color)

  if (!coordinates?.length) return null

  // mesh_3d: triangulated surface
  if (geom_type === 'mesh_3d' && faces?.length) {
    const positions = new Float32Array(coordinates.length * 3)
    for (let i = 0; i < coordinates.length; i++) {
      const c = coordinates[i]
      positions[i * 3]     = c[0] ?? 0
      positions[i * 3 + 1] = c[1] ?? 0
      positions[i * 3 + 2] = c[2] ?? 0
    }
    const indices = new Uint32Array(faces.length * 3)
    for (let i = 0; i < faces.length; i++) {
      const f = faces[i]
      indices[i * 3]     = f[0]
      indices[i * 3 + 1] = f[1]
      indices[i * 3 + 2] = f[2]
    }
    const buf = new THREE.BufferGeometry()
    buf.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    buf.setIndex(new THREE.BufferAttribute(indices, 1))
    buf.computeVertexNormals()

    const material = new THREE.MeshStandardMaterial({
      color,
      wireframe: opts.wireframe,
      flatShading: false,
      metalness: 0.05,
      roughness: 0.55,
      side: THREE.DoubleSide,
    })
    return new THREE.Mesh(buf, material)
  }

  // point_cloud_3d: Points
  if (geom_type === 'point_cloud_3d' || (!faces?.length && !edges?.length && Array.isArray(coordinates[0]))) {
    const positions = new Float32Array(coordinates.length * 3)
    for (let i = 0; i < coordinates.length; i++) {
      const c = coordinates[i]
      positions[i * 3]     = c[0] ?? 0
      positions[i * 3 + 1] = c[1] ?? 0
      positions[i * 3 + 2] = c[2] ?? 0
    }
    const buf = new THREE.BufferGeometry()
    buf.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const material = new THREE.PointsMaterial({
      color,
      size: opts.pointSize,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    })
    return new THREE.Points(buf, material)
  }

  // graph / tree / curve_3d / lsystem: line segments
  if (edges?.length) {
    const positions = new Float32Array(edges.length * 2 * 3)
    for (let i = 0; i < edges.length; i++) {
      const [a, b] = edges[i]
      const pa = coordinates[a] || [0, 0, 0]
      const pb = coordinates[b] || [0, 0, 0]
      positions[i * 6]     = pa[0] ?? 0
      positions[i * 6 + 1] = pa[1] ?? 0
      positions[i * 6 + 2] = pa[2] ?? 0
      positions[i * 6 + 3] = pb[0] ?? 0
      positions[i * 6 + 4] = pb[1] ?? 0
      positions[i * 6 + 5] = pb[2] ?? 0
    }
    const buf = new THREE.BufferGeometry()
    buf.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    const material = new THREE.LineBasicMaterial({ color, linewidth: 1 })
    return new THREE.LineSegments(buf, material)
  }

  // Fallback to a point cloud rendering
  const positions = new Float32Array(coordinates.length * 3)
  for (let i = 0; i < coordinates.length; i++) {
    const c = coordinates[i]
    positions[i * 3]     = c[0] ?? 0
    positions[i * 3 + 1] = c[1] ?? 0
    positions[i * 3 + 2] = c[2] ?? 0
  }
  const buf = new THREE.BufferGeometry()
  buf.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  return new THREE.Points(buf, new THREE.PointsMaterial({ color, size: opts.pointSize }))
}

function disposeObject(obj) {
  if (!obj) return
  if (obj.geometry) obj.geometry.dispose()
  if (obj.material) {
    if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose())
    else obj.material.dispose()
  }
}
