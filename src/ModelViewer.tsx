import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js'

// ─── CFD shaders ─────────────────────────────────────────────────────────────

const CFD_VERT = /* glsl */`
  varying vec3 vNormal;
  varying vec3 vPosition;
  void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`

const CFD_FRAG = /* glsl */`
  varying vec3 vNormal;
  varying vec3 vPosition;
  uniform float cdValue;

  void main() {
    vec3 windDir = normalize(vec3(0.0, 0.0, -1.0));
    float pressure = dot(vNormal, windDir);

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
}

interface ArrowData {
  helper: THREE.ArrowHelper
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
  arrowData: ArrowData[]
  animId: number
  cdMaterials: THREE.ShaderMaterial[]
  center: THREE.Vector3
  size: THREE.Vector3
}

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

function spawnArrows(
  scene: THREE.Scene,
  center: THREE.Vector3,
  size: THREE.Vector3
): ArrowData[] {
  const arrowData: ArrowData[] = []
  const dir = new THREE.Vector3(0, 0, -1)
  const len = Math.max(size.x, size.y) * 0.11

  for (let i = 0; i < 200; i++) {
    const pos = new THREE.Vector3(
      (Math.random() - 0.5) * size.x * 2.5 + center.x,
      (Math.random() - 0.5) * size.y * 1.3 + center.y,
      Math.random() * size.z * 3 + center.z + size.z
    )
    const t = Math.max(0, Math.min(1, (pos.y - center.y + size.y * 0.65) / (size.y * 1.3)))
    const color = new THREE.Color(0x4fc3f7).lerp(new THREE.Color(0x00ff88), t)
    const arrow = new THREE.ArrowHelper(dir, pos, len, color, len * 0.35, len * 0.18)
    ;(arrow.line.material as THREE.LineBasicMaterial).transparent = true
    ;(arrow.line.material as THREE.LineBasicMaterial).opacity = 0.45
    ;(arrow.cone.material as THREE.MeshBasicMaterial).transparent = true
    ;(arrow.cone.material as THREE.MeshBasicMaterial).opacity = 0.45
    scene.add(arrow)
    arrowData.push({ helper: arrow, vx: 0, vy: 0 })
  }
  return arrowData
}

function clearModel(state: ViewerState) {
  if (state.currentModel) {
    state.scene.remove(state.currentModel)
    state.currentModel = null
  }
  for (const ad of state.arrowData) state.scene.remove(ad.helper)
  state.arrowData = []
  state.cdMaterials = []
}

function setupModel(model: THREE.Group, state: ViewerState, cdValue: number) {
  const box = new THREE.Box3().setFromObject(model)
  const sz = new THREE.Vector3()
  const ctr = new THREE.Vector3()
  box.getSize(sz)
  box.getCenter(ctr)

  const scale = 2.0 / Math.max(sz.x, sz.y, sz.z)
  model.scale.setScalar(scale)
  model.position.set(-ctr.x * scale, -ctr.y * scale, -ctr.z * scale)

  const box2 = new THREE.Box3().setFromObject(model)
  box2.getSize(state.size)
  box2.getCenter(state.center)

  const mat = makeCFDMaterial(cdValue)
  state.cdMaterials = [mat]
  applyShader(model, mat)

  state.scene.add(model)
  state.currentModel = model

  const camDist = state.size.y * 1.6 + state.size.z
  state.camera.position.set(state.center.x, state.center.y * 0.8, state.center.z + camDist)
  state.camera.lookAt(state.center)
  state.controls.target.copy(state.center)

  state.arrowData = spawnArrows(state.scene, state.center, state.size)
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
  state.arrowData = spawnArrows(state.scene, state.center, state.size)
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

// Ellipsoid half-axes as fraction of body size
const BODY_RX = 0.58   // X half-axis (shoulder width radius)
const BODY_RY = 0.58   // Y half-axis (height radius)
const BODY_RZ = 0.65   // Z half-axis (depth radius, slightly larger for bow wave)
const WIND_SPEED = 0.022
const DEFLECT_FORCE = 0.028
const LATERAL_DAMPING = 0.90
const RESET_DIR = new THREE.Vector3(0, 0, -1)

export default function ModelViewer({ modelUrl, cdValue, loading, loadingMessage }: ModelViewerProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const stateRef = useRef<ViewerState | null>(null)
  const cdRef = useRef(cdValue)
  const loadIdRef = useRef(0)
  const activeUrlRef = useRef<string | null>(null)

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

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, mount.clientWidth / mount.clientHeight, 0.01, 500)
    camera.position.set(0, 0, 5)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.autoRotate = true
    controls.autoRotateSpeed = 0.7

    scene.add(new THREE.AmbientLight(0xffffff, 0.04))

    const dracoLoader = new DRACOLoader()
    dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.6/')
    const gltfLoader = new GLTFLoader()
    gltfLoader.setDRACOLoader(dracoLoader)

    const state: ViewerState = {
      renderer, scene, camera, controls, loader: gltfLoader,
      currentModel: null, arrowData: [],
      animId: 0, cdMaterials: [],
      center: new THREE.Vector3(), size: new THREE.Vector3(1, 2, 1),
    }
    stateRef.current = state

    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth, h = mount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    })
    ro.observe(mount)

    const _tmpDir = new THREE.Vector3()

    const animate = () => {
      state.animId = requestAnimationFrame(animate)
      controls.update()

      const { arrowData, center, size } = state
      // Half-axes of the deflection ellipsoid in world units
      const hx = size.x * BODY_RX
      const hy = size.y * BODY_RY
      const hz = size.z * BODY_RZ

      for (const ad of arrowData) {
        const pos = ad.helper.position

        pos.z -= WIND_SPEED

        // Normalized position inside the deflection ellipsoid (0 = surface, <0 = inside)
        const nx = (pos.x - center.x) / hx
        const ny = (pos.y - center.y) / hy
        const nz = (pos.z - center.z) / hz
        const ellipsoidDist = Math.sqrt(nx * nx + ny * ny + nz * nz)

        // Deflect if inside the ellipsoid AND arrow is still on the upstream half
        // (nz > 0 means the arrow is in front of the body center)
        if (ellipsoidDist < 1.0 && nz > -0.3) {
          const strength = (1.0 - ellipsoidDist) * DEFLECT_FORCE
          const dx = pos.x - center.x
          const dy = pos.y - center.y
          const lateralLen = Math.sqrt(dx * dx + dy * dy)
          if (lateralLen > 0.001) {
            ad.vx += (dx / lateralLen) * strength
            ad.vy += (dy / lateralLen) * strength
          } else {
            // Arrow dead-center: ride over the top
            ad.vy += strength
          }
        }

        pos.x += ad.vx
        pos.y += ad.vy
        ad.vx *= LATERAL_DAMPING
        ad.vy *= LATERAL_DAMPING

        _tmpDir.set(ad.vx, ad.vy, -WIND_SPEED).normalize()
        ad.helper.setDirection(_tmpDir)

        if (pos.z < center.z - size.z * 1.5) {
          pos.z = center.z + size.z * 1.5
          pos.x = (Math.random() - 0.5) * size.x * 2.5 + center.x
          pos.y = (Math.random() - 0.5) * size.y * 1.3 + center.y
          ad.vx = 0
          ad.vy = 0
          ad.helper.setDirection(RESET_DIR)
        }
      }

      renderer.render(scene, camera)
    }
    animate()

    // Load default model on startup
    const defaultUrl = '/default-human2.glb'
    activeUrlRef.current = defaultUrl
    loadIdRef.current++
    const myId = loadIdRef.current
    gltfLoader.load(defaultUrl, gltf => {
      if (loadIdRef.current !== myId || !stateRef.current) return
      clearModel(state)
      setupModel(gltf.scene, state, cdRef.current)
    }, undefined, () => {
      if (loadIdRef.current !== myId || !stateRef.current) return
      clearModel(state)
      loadFallback(state, cdRef.current)
    })

    return () => {
      cancelAnimationFrame(state.animId)
      ro.disconnect()
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
    }, undefined, err => {
      console.error('[AeroMaxx] GLB load failed:', err)
      if (loadIdRef.current !== myId || !stateRef.current) return
      clearModel(state)
      loadFallback(state, cdRef.current)
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

      {/* Pressure color scale — bottom-right corner */}
      <div style={{
        position: 'absolute', right: 18, bottom: 24,
        display: 'flex', flexDirection: 'column', alignItems: 'center',
      }}>
        <ColorScaleBar />
      </div>

      {/* Loading overlay */}
      {loading && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(8,8,8,0.80)',
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: 18,
        }}>
          <div style={{ fontFamily: 'var(--mono)', color: '#00ff88', fontSize: 13, letterSpacing: 3 }}
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
