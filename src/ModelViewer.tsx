import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import type { GlbMeasurements } from './types'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js'
import { Line2 } from 'three/examples/jsm/lines/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js'

// ─── CFD shaders ─────────────────────────────────────────────────────────────

const CFD_VERT = /* glsl */`
  varying vec3 vWorldNormal;
  void main() {
    // World-space normal: camera-independent, so pressure is fixed
    // regardless of orbit angle. modelMatrix = mesh's matrixWorld.
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const CFD_FRAG = /* glsl */`
  varying vec3 vWorldNormal;
  uniform float cdValue;

  void main() {
    // Wind blows in -Z world direction; surfaces with +Z world normals
    // face the wind → stagnation (high pressure, red/orange).
    float pressure = vWorldNormal.z;

    vec3 color;
    if (pressure > 0.6) {
      color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), (pressure - 0.6) / 0.4);
    } else if (pressure > 0.2) {
      color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (pressure - 0.2) / 0.4);
    } else if (pressure > -0.2) {
      color = mix(vec3(0.0, 0.8, 0.4), vec3(1.0, 1.0, 0.0), (pressure + 0.2) / 0.4);
    } else if (pressure > -0.6) {
      color = mix(vec3(0.0, 0.5, 1.0), vec3(0.0, 0.8, 0.4), (pressure + 0.6) / 0.4);
    } else {
      color = mix(vec3(0.0, 0.0, 0.8), vec3(0.0, 0.5, 1.0), (pressure + 1.0) / 0.4);
    }

    gl_FragColor = vec4(color, 1.0);
  }
`

// ─── Types ────────────────────────────────────────────────────────────────────

interface ModelViewerProps {
  modelUrl: string | null
  cdValue: number
  loading: boolean
  loadingMessage: string
  onGeometryMeasured?: (m: GlbMeasurements) => void
}

interface StreamData {
  line: Line2
  geo: LineGeometry
  posArr:    Float32Array  // TRAIL_LENGTH * 3 — working positions
  colArr:    Float32Array  // TRAIL_LENGTH * 3 — working colors
  posSegBuf: THREE.InterleavedBuffer  // Line2's internal interleaved position buffer
  colSegBuf: THREE.InterleavedBuffer  // Line2's internal interleaved color buffer
  eDistArr: Float32Array  // TRAIL_LENGTH — body ellipsoid distance per vertex
  nzArr:    Float32Array  // TRAIL_LENGTH — normalized Z (pressure proxy) per vertex
  headX: number
  headY: number
  headZ: number
  vx: number
  vy: number
}

interface ViewerState {
  renderer: THREE.WebGLRenderer
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  controls: OrbitControls
  loader: GLTFLoader
  currentModel: THREE.Group | null
  streams: StreamData[]
  streamMat: LineMaterial
  animId: number
  cdMaterials: THREE.ShaderMaterial[]
  center: THREE.Vector3
  size: THREE.Vector3
}

// ─── Constants ────────────────────────────────────────────────────────────────

const TRAIL_LENGTH    = 50
const NUM_STREAMS     = 280
const WIND_SPEED      = 0.022
const BODY_RX         = 0.58
const BODY_RY         = 0.58
const BODY_RZ         = 0.65
const DEFLECT_FORCE   = 0.028
const LATERAL_DAMPING = 0.90

// ─── Helpers ─────────────────────────────────────────────────────────────────

function makeCFDMaterial(cdValue: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader: CFD_VERT,
    fragmentShader: CFD_FRAG,
    uniforms: { cdValue: { value: cdValue } },
    side: THREE.DoubleSide,
  })
}

function applyShader(root: THREE.Object3D, mat: THREE.ShaderMaterial) {
  root.traverse(child => {
    if (child instanceof THREE.Mesh) child.material = mat
  })
}

// Per-vertex color: undisturbed cyan/green blended toward CFD pressure
// color as proximity to body increases. Matches the model surface shader exactly.
function vertexColor(normY: number, eDist: number, nz: number): [number, number, number] {
  // Undisturbed base: cyan (#4fc3f7) → green (var(--green)) by height
  const t  = Math.max(0, Math.min(1, normY))
  const br = (0x4f + (0x00 - 0x4f) * t) / 255
  const bg = (0xc3 + (0xff - 0xc3) * t) / 255
  const bb = (0xf7 + (0x88 - 0xf7) * t) / 255

  // Blend factor: 0 = undisturbed, 1 = fully in body's pressure field
  const blend = Math.max(0, Math.min(1, (1.6 - eDist) / 1.1))
  if (blend < 0.01) return [br, bg, bb]

  // Pressure color matching fixed CFD shader (nz acts as the pressure proxy)
  const p = nz * 1.1
  let pr: number, pg: number, pb: number
  if (p > 0.6) {
    const s = (p - 0.6) / 0.4
    pr = 1.0; pg = 0.5 - 0.5 * s; pb = 0.0           // orange→red
  } else if (p > 0.2) {
    const s = (p - 0.2) / 0.4
    pr = 1.0; pg = 1.0 - 0.5 * s; pb = 0.0           // yellow→orange
  } else if (p > -0.2) {
    const s = (p + 0.2) / 0.4
    pr = s * 0.1; pg = 0.8 + 0.2 * s; pb = 0.4 * (1 - s)  // green
  } else if (p > -0.6) {
    const s = (p + 0.6) / 0.4
    pr = 0.0; pg = s * 0.5; pb = 0.5 + s * 0.3       // blue→teal
  } else {
    pr = 0.0; pg = 0.0; pb = 0.8                      // deep blue (wake)
  }

  return [
    br + (pr - br) * blend,
    bg + (pg - bg) * blend,
    bb + (pb - bb) * blend,
  ]
}

// Sync posArr/colArr into Line2's internal interleaved buffers without allocation.
// Each segment i needs [start.xyz, end.xyz] = posArr[i*3..i*3+6] (6 floats).
function syncLine2Buffers(sd: StreamData) {
  const posArr = sd.posSegBuf.array as Float32Array
  const colArr = sd.colSegBuf.array as Float32Array
  for (let i = 0; i < TRAIL_LENGTH - 1; i++) {
    posArr.set(sd.posArr.subarray(i * 3, i * 3 + 6), i * 6)
    colArr.set(sd.colArr.subarray(i * 3, i * 3 + 6), i * 6)
  }
  sd.posSegBuf.needsUpdate = true
  sd.colSegBuf.needsUpdate = true
}

function initTrail(sd: StreamData) {
  for (let j = 0; j < TRAIL_LENGTH; j++) {
    sd.posArr[j * 3]     = sd.headX
    sd.posArr[j * 3 + 1] = sd.headY
    sd.posArr[j * 3 + 2] = sd.headZ + (TRAIL_LENGTH - 1 - j) * WIND_SPEED
    sd.eDistArr[j] = 10.0
    sd.nzArr[j]    = 1.0
  }
  sd.colArr.fill(0)
  syncLine2Buffers(sd)
}

function spawnStreams(
  scene: THREE.Scene,
  mat: LineMaterial,
  center: THREE.Vector3,
  size: THREE.Vector3,
): StreamData[] {
  const streams: StreamData[] = []
  for (let i = 0; i < NUM_STREAMS; i++) {
    const posArr   = new Float32Array(TRAIL_LENGTH * 3)
    const colArr   = new Float32Array(TRAIL_LENGTH * 3)
    const eDistArr = new Float32Array(TRAIL_LENGTH).fill(10.0)
    const nzArr    = new Float32Array(TRAIL_LENGTH).fill(1.0)

    const headX = (Math.random() - 0.5) * size.x * 2.5 + center.x
    const headY = (Math.random() - 0.5) * size.y * 1.3 + center.y
    const headZ = Math.random() * size.z * 3 + center.z + size.z
    for (let j = 0; j < TRAIL_LENGTH; j++) {
      posArr[j * 3]     = headX
      posArr[j * 3 + 1] = headY
      posArr[j * 3 + 2] = headZ + (TRAIL_LENGTH - 1 - j) * WIND_SPEED
    }

    const geo = new LineGeometry()
    geo.setPositions(posArr)
    geo.setColors(colArr)

    // Grab the underlying interleaved buffers for zero-alloc per-frame updates
    const posSegBuf = (geo.getAttribute('instanceStart') as THREE.InterleavedBufferAttribute).data as THREE.InterleavedBuffer
    const colSegBuf = (geo.getAttribute('instanceColorStart') as THREE.InterleavedBufferAttribute).data as THREE.InterleavedBuffer

    const line = new Line2(geo, mat)
    line.frustumCulled = false
    line.computeLineDistances()
    scene.add(line)

    const sd: StreamData = {
      line, geo, posArr, colArr, posSegBuf, colSegBuf,
      eDistArr, nzArr, headX, headY, headZ, vx: 0, vy: 0,
    }
    streams.push(sd)
  }
  return streams
}

// Convert the viewer's normalised bounding box back to real-world metres.
// The model is scaled so its tallest dimension = 2 world units ≈ 1.75 m real height.
// This gives us actual body width and front-to-back depth from the 3D geometry —
// more accurate than the 2D shoulder-landmark estimate from the photo.
function measureGlb(state: ViewerState): GlbMeasurements {
  const REAL_HEIGHT = 1.75  // assumed average adult height (m)
  const wh = state.size.y   // world-space height (≈ 2.0 after normalisation)
  const realWidth = (state.size.x / wh) * REAL_HEIGHT
  const realDepth = (state.size.z / wh) * REAL_HEIGHT
  return {
    realWidth,
    realDepth,
    frontalArea: realWidth * REAL_HEIGHT * 0.73,  // 0.73 = standard fill factor
    depthToWidthRatio: realDepth / realWidth,
  }
}

function clearModel(state: ViewerState) {
  if (state.currentModel) {
    state.scene.remove(state.currentModel)
    state.currentModel = null
  }
  for (const sd of state.streams) {
    state.scene.remove(sd.line)
    sd.geo.dispose()
  }
  state.streams   = []
  state.cdMaterials = []
}

function setupModel(model: THREE.Group, state: ViewerState, cdValue: number) {
  const box = new THREE.Box3().setFromObject(model)
  const sz = new THREE.Vector3(), ctr = new THREE.Vector3()
  box.getSize(sz); box.getCenter(ctr)

  const scale = 2.0 / Math.max(sz.x, sz.y, sz.z)
  model.scale.setScalar(scale)
  model.position.set(-ctr.x * scale, -ctr.y * scale, -ctr.z * scale)

  new THREE.Box3().setFromObject(model).getSize(state.size)
  new THREE.Box3().setFromObject(model).getCenter(state.center)

  const mat = makeCFDMaterial(cdValue)
  state.cdMaterials = [mat]
  applyShader(model, mat)
  state.scene.add(model)
  state.currentModel = model

  const camDist = state.size.y * 1.6 + state.size.z
  state.camera.position.set(state.center.x, state.center.y * 0.8, state.center.z + camDist)
  state.camera.lookAt(state.center)
  state.controls.target.copy(state.center)

  state.streams = spawnStreams(state.scene, state.streamMat, state.center, state.size)
}

function loadFallback(state: ViewerState, cdValue: number) {
  const geo = new THREE.CapsuleGeometry(0.45, 1.3, 10, 20)
  const mat = makeCFDMaterial(cdValue)
  state.cdMaterials = [mat]
  const mesh = new THREE.Mesh(geo, mat)
  const group = new THREE.Group()
  group.add(mesh)
  state.center.set(0, 0, 0)
  state.size.set(0.9, 1.75, 0.9)
  state.scene.add(group)
  state.currentModel = group
  state.streams = spawnStreams(state.scene, state.streamMat, state.center, state.size)
}

// ─── Color scale bar overlay ─────────────────────────────────────────────────

function ColorScaleBar() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', userSelect: 'none' }}>
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 7, color: '#888', letterSpacing: 1,
        textTransform: 'uppercase', marginBottom: 5, whiteSpace: 'nowrap',
        writingMode: 'vertical-lr', transform: 'rotate(180deg)',
      }}>
        HIGH PRESSURE
      </div>
      <div style={{
        width: 11, height: 140,
        background: 'linear-gradient(to bottom, #ff0000, #ff8800, #ffff00, #00cc66, #0066ff, #0000cc)',
        border: '1px solid #2a2a2a',
      }} />
      <div style={{
        fontFamily: 'var(--mono)', fontSize: 7, color: '#888', letterSpacing: 1,
        textTransform: 'uppercase', marginTop: 5, whiteSpace: 'nowrap',
        writingMode: 'vertical-lr',
      }}>
        LOW PRESSURE
      </div>
    </div>
  )
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ModelViewer({ modelUrl, cdValue, loading, loadingMessage, onGeometryMeasured }: ModelViewerProps) {
  const mountRef       = useRef<HTMLDivElement>(null)
  const stateRef       = useRef<ViewerState | null>(null)
  const cdRef          = useRef(cdValue)
  const loadIdRef      = useRef(0)
  const activeUrlRef   = useRef<string | null>(null)
  // Ref so load-callbacks always call the latest prop without re-running effects
  const geoCallbackRef = useRef(onGeometryMeasured)
  useEffect(() => { geoCallbackRef.current = onGeometryMeasured }, [onGeometryMeasured])

  useEffect(() => { cdRef.current = cdValue }, [cdValue])

  // ── Initialize Three.js once ──────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(mount.clientWidth, mount.clientHeight)
    renderer.setClearColor(0x080808)
    mount.appendChild(renderer.domElement)

    const scene  = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, mount.clientWidth / mount.clientHeight, 0.01, 500)
    camera.position.set(0, 0, 5)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping   = true
    controls.dampingFactor   = 0.06
    controls.autoRotate      = true
    controls.autoRotateSpeed = 0.7

    scene.add(new THREE.AmbientLight(0xffffff, 0.04))

    const dracoLoader = new DRACOLoader()
    dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/')
    const gltfLoader = new GLTFLoader()
    gltfLoader.setDRACOLoader(dracoLoader)

    const streamMat = new LineMaterial({
      vertexColors: true,
      linewidth: 3,
      worldUnits: false,
      depthWrite: false,
      transparent: true,
      blending: THREE.AdditiveBlending,
      resolution: new THREE.Vector2(mount.clientWidth, mount.clientHeight),
    })

    const state: ViewerState = {
      renderer, scene, camera, controls, loader: gltfLoader,
      currentModel: null, streams: [], streamMat,
      animId: 0, cdMaterials: [],
      center: new THREE.Vector3(), size: new THREE.Vector3(1, 2, 1),
    }
    stateRef.current = state

    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth, h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
      state.streamMat.resolution.set(w, h)
    })
    ro.observe(mount)

    const animate = () => {
      state.animId = requestAnimationFrame(animate)
      controls.update()

      const { streams, center, size } = state
      const hx = size.x * BODY_RX
      const hy = size.y * BODY_RY
      const hz = size.z * BODY_RZ

      for (const sd of streams) {
        // ── Advance head ────────────────────────────────────────────
        sd.headZ -= WIND_SPEED
        sd.headX += sd.vx
        sd.headY += sd.vy

        // ── Ellipsoid proximity & deflection ────────────────────────
        const dx = sd.headX - center.x
        const dy = sd.headY - center.y
        const dz = sd.headZ - center.z
        const enx = dx / hx
        const eny = dy / hy
        const enz = dz / hz
        const eDist = Math.sqrt(enx * enx + eny * eny + enz * enz)
        // nz is the normalized Z component: >0 = upstream, <0 = downstream (wake)
        const nz = enz

        if (eDist < 1.0 && nz > -0.3) {
          const strength = (1.0 - eDist) * DEFLECT_FORCE
          const lat = Math.sqrt(dx * dx + dy * dy)
          if (lat > 0.001) {
            sd.vx += (dx / lat) * strength
            sd.vy += (dy / lat) * strength
          } else {
            sd.vy += strength
          }
        }

        sd.vx *= LATERAL_DAMPING
        sd.vy *= LATERAL_DAMPING

        // ── Shift trail arrays, append new head ─────────────────────
        sd.posArr.copyWithin(0, 3)
        sd.eDistArr.copyWithin(0, 1)
        sd.nzArr.copyWithin(0, 1)

        const last3 = (TRAIL_LENGTH - 1) * 3
        sd.posArr[last3]     = sd.headX
        sd.posArr[last3 + 1] = sd.headY
        sd.posArr[last3 + 2] = sd.headZ
        sd.eDistArr[TRAIL_LENGTH - 1] = eDist
        sd.nzArr[TRAIL_LENGTH - 1]    = nz

        // ── Per-vertex color: historical pressure × fade ─────────────
        for (let j = 0; j < TRAIL_LENGTH; j++) {
          const normY = Math.max(0, Math.min(1,
            (sd.posArr[j * 3 + 1] - center.y + size.y * 0.65) / (size.y * 1.3)
          ))
          const [r, g, b] = vertexColor(normY, sd.eDistArr[j], sd.nzArr[j])
          // ^0.35 keeps the trail bright longer; *0.35 caps per-line brightness
          // so additive blending of ~3 overlapping lines reaches ~1.0
          const brightness = Math.pow((j + 1) / TRAIL_LENGTH, 0.35) * 0.35
          sd.colArr[j * 3]     = r * brightness
          sd.colArr[j * 3 + 1] = g * brightness
          sd.colArr[j * 3 + 2] = b * brightness
        }

        syncLine2Buffers(sd)

        // ── Reset when head exits the back of volume ─────────────────
        if (sd.headZ < center.z - size.z * 1.5) {
          sd.headX = (Math.random() - 0.5) * size.x * 2.5 + center.x
          sd.headY = (Math.random() - 0.5) * size.y * 1.3 + center.y
          sd.headZ = center.z + size.z * 1.5
          sd.vx = 0
          sd.vy = 0
          initTrail(sd)
        }
      }

      renderer.render(scene, camera)
    }
    animate()

    const defaultUrl = '/default-human2.glb'
    activeUrlRef.current = defaultUrl
    loadIdRef.current++
    const myId = loadIdRef.current
    gltfLoader.load(defaultUrl, gltf => {
      if (loadIdRef.current !== myId || !stateRef.current) return
      clearModel(state)
      setupModel(gltf.scene, state, cdRef.current)
      geoCallbackRef.current?.(measureGlb(state))
    }, undefined, () => {
      if (loadIdRef.current !== myId || !stateRef.current) return
      clearModel(state)
      loadFallback(state, cdRef.current)
      geoCallbackRef.current?.(measureGlb(state))
    })

    return () => {
      cancelAnimationFrame(state.animId)
      ro.disconnect()
      streamMat.dispose()
      renderer.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
      stateRef.current = null
    }
  }, [])

  // ── Swap model when URL changes ───────────────────────────────────────────
  useEffect(() => {
    const state = stateRef.current
    if (!state) return
    const url = modelUrl ?? '/default-human2.glb'
    if (url === activeUrlRef.current) return
    activeUrlRef.current = url

    loadIdRef.current++
    const myId = loadIdRef.current
    clearModel(state)

    state.loader.load(url, gltf => {
      if (loadIdRef.current !== myId || !stateRef.current) return
      clearModel(state)
      setupModel(gltf.scene, state, cdRef.current)
      geoCallbackRef.current?.(measureGlb(state))
    }, undefined, err => {
      console.error('[AeroMaxx] GLB load failed:', err)
      if (loadIdRef.current !== myId || !stateRef.current) return
      clearModel(state)
      loadFallback(state, cdRef.current)
      geoCallbackRef.current?.(measureGlb(state))
    })
  }, [modelUrl])

  // ── Update Cd uniform live ─────────────────────────────────────────────────
  useEffect(() => {
    if (!stateRef.current) return
    for (const mat of stateRef.current.cdMaterials) {
      mat.uniforms.cdValue.value = cdValue
    }
  }, [cdValue])

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', background: '#080808' }}>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      <div style={{
        position: 'absolute', right: 18, bottom: 24,
        display: 'flex', flexDirection: 'column', alignItems: 'center',
      }}>
        <ColorScaleBar />
      </div>

      {loading && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(8,8,8,0.80)',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: 18,
        }}>
          <div style={{ fontFamily: 'var(--mono)', color: 'var(--green)', fontSize: 13, letterSpacing: 3 }}
            className="pulse">
            {loadingMessage || 'GENERATING 3D MODEL...'}
          </div>
          <div style={{ fontFamily: 'var(--mono)', fontSize: 10, color: '#2a2a2a', letterSpacing: 4 }}>
            ░░░░░░░░░░
          </div>
        </div>
      )}
    </div>
  )
}
