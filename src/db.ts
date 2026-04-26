import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  import.meta.env.VITE_SUPABASE_URL as string,
  import.meta.env.VITE_SUPABASE_ANON_KEY as string,
)

export interface SubjectRecord {
  id: string
  image_url: string
  glb_url: string
  cd_score: number
  created_at: string
}

export async function uploadGlb(blob: Blob, filename: string): Promise<string> {
  const { error } = await supabase.storage
    .from('glb-models')
    .upload(filename, blob, { contentType: 'model/gltf-binary', upsert: true })
  if (error) throw new Error(`GLB storage upload failed: ${error.message}`)
  const { data } = supabase.storage.from('glb-models').getPublicUrl(filename)
  return data.publicUrl
}

export async function saveSubject(
  imageUrl: string,
  glbUrl: string,
  cdScore: number,
): Promise<void> {
  const { error } = await supabase
    .from('subjects')
    .insert({ image_url: imageUrl, glb_url: glbUrl, cd_score: cdScore })
  if (error) console.error('[AeroMaxx] DB save failed:', error.message)
}

export async function getLeaderboard(): Promise<SubjectRecord[]> {
  const { data, error } = await supabase
    .from('subjects')
    .select('*')
    .order('cd_score', { ascending: true })
    .limit(100)
  if (error) console.error('[AeroMaxx] DB fetch failed:', error.message)
  return data ?? []
}
