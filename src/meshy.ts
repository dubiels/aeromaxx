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
}

export async function createImageTo3DTask(imageUrl: string): Promise<string> {
  const res = await fetch(`${BASE}/v1/image-to-3d`, {
    method: 'POST',
    headers: headers(),
    body: JSON.stringify({ image_url: imageUrl, enable_pbr: false, ai_model: 'meshy-4' }),
  })
  if (!res.ok) throw new Error(`Meshy create failed: ${res.status}`)
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

    const res = await fetch(`${BASE}/v1/image-to-3d/${taskId}`, { headers: headers() })
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
