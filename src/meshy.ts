const BASE = 'https://api.meshy.ai'

function headers() {
  return {
    Authorization: `Bearer ${import.meta.env.VITE_MESHY_API_KEY}`,
    'Content-Type': 'application/json',
  }
}

interface MeshyTask {
  id: string
  status: 'PENDING' | 'IN_PROGRESS' | 'SUCCEEDED' | 'FAILED' | 'EXPIRED'
  model_urls?: { glb?: string }
  progress?: number
  created_at?: number  // unix ms
}

export interface MeshyListItem {
  id: string
  glbUrl: string
  createdAt: number
}

export async function listSucceededTasks(): Promise<MeshyListItem[]> {
  const results: MeshyListItem[] = []
  for (let page = 1; page <= 5; page++) {
    const res = await fetch(
      `${BASE}/openapi/v1/image-to-3d?page_num=${page}&page_size=24&sort_by=-created_at`,
      { headers: headers() }
    )
    if (!res.ok) break
    const data: MeshyTask[] = await res.json()
    if (!data.length) break
    for (const t of data) {
      if (t.status === 'SUCCEEDED' && t.model_urls?.glb) {
        results.push({ id: t.id, glbUrl: t.model_urls.glb, createdAt: t.created_at ?? 0 })
      }
    }
  }
  return results
}

export async function createImageTo3DTask(imageUrl: string): Promise<string> {
   console.log('[AeroMaxx] Sending image URL to Meshy:', imageUrl)
  const res = await fetch(`${BASE}/openapi/v1/image-to-3d`, {
    method: 'POST',
    headers: headers(),
  body: JSON.stringify({
    image_url: imageUrl,
    enable_pbr: false,
    ai_model: 'meshy-6',  // latest model, better at humans
    symmetry_mode: 'on',   // helps with bilateral human body
})
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Meshy create failed: ${res.status} — ${body}`)
  }
  const data = await res.json()
  return data.result as string
}

export async function pollTask(
  taskId: string,
  onProgress?: (pct: number) => void
): Promise<string> {
  const deadline = Date.now() + 5 * 60 * 1000

  while (Date.now() < deadline) {
    await new Promise(r => setTimeout(r, 3000))

    const res = await fetch(`${BASE}/openapi/v1/image-to-3d/${taskId}`, { headers: headers() })
    if (!res.ok) throw new Error(`Meshy poll failed: ${res.status}`)
    const task: MeshyTask = await res.json()

    if (task.progress !== undefined) onProgress?.(task.progress)

    if (task.status === 'SUCCEEDED') {
      const glb = task.model_urls?.glb
      if (!glb) throw new Error('Meshy succeeded but returned no GLB URL')
      return glb
    }
    if (task.status === 'FAILED' || task.status === 'EXPIRED') {
      throw new Error(`Meshy task ${task.status}`)
    }
  }

  throw new Error('Meshy timed out after 5 minutes')
}