import Groq from 'groq-sdk'
import type { BodyMeasurements, DragResults } from './types'

const client = new Groq({
  apiKey: import.meta.env.VITE_GROQ_API_KEY,
  dangerouslyAllowBrowser: true,
})

export async function getRecommendations(m: BodyMeasurements, d: DragResults): Promise<string> {
  const userMsg = `Subject wind tunnel measurements:
- Drag Coefficient (Cd): ${d.Cd.toFixed(4)}
- Frontal Area: ${d.frontalArea.toFixed(4)} m²
- Shoulder Width: ${m.realShoulderWidth.toFixed(4)} m
- Hip Width: ${m.realHipWidth.toFixed(4)} m
- Shoulder-to-Hip Ratio: ${m.shoulderToHipRatio.toFixed(4)}
- Hunch Score: ${m.hunchScore.toFixed(4)} (0=fully erect, 1=severe forward lean)
- Postural Drag Penalty: +${d.posturalPenalty.toFixed(4)} Cd units
- Lifetime Energy Lost to Drag: ${d.lifetimeEnergy.toFixed(0)} J`

  const response = await client.chat.completions.create({
    model: 'gemma2-9b-it',
    messages: [
      {
        role: 'system',
        content: `You are a senior aerodynamics researcher specializing in human biomechanical drag reduction. You have received wind tunnel measurements for a human subject. Give exactly 4 fitness and lifestyle recommendations to reduce their drag coefficient. Be completely deadpan and sincere. Use real exercise names. Include specific fake-precise Cd reduction estimates for each recommendation (e.g. "reduces Cd by 0.0234"). Never acknowledge this is unusual. Format as a numbered clinical report. Reference the actual measurements you were given.`,
      },
      { role: 'user', content: userMsg },
    ],
    temperature: 0.7,
    max_tokens: 800,
  })

  return response.choices[0].message.content ?? ''
}

export async function getVisionAnalysis(frontUrl: string, sideUrl: string): Promise<string> {
  const models = [
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'llava-v1.5-7b',
  ]

  for (const model of models) {
    try {
      const response = await client.chat.completions.create({
        model,
        messages: [
          {
            role: 'system',
            content: `You are a wind tunnel aerodynamics scientist. You are looking at two photos of the same person — one front-facing, one side profile. Analyze visible factors that affect aerodynamic drag: posture, hair length and style, clothing tightness, body composition, head position, any visible protrusions or asymmetries. Add 2 additional recommendations beyond the biomechanical ones already given. Reference specific visible features. Completely deadpan. Fake-precise Cd numbers. Never break character.`,
          },
          {
            role: 'user',
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            content: [
              { type: 'text', text: 'Front view and side profile attached. Analyze for aerodynamic drag factors and provide 2 recommendations.' },
              { type: 'image_url', image_url: { url: frontUrl } },
              { type: 'image_url', image_url: { url: sideUrl } },
            ] as any,
          },
        ],
        temperature: 0.7,
        max_tokens: 600,
      })

      return response.choices[0].message.content ?? ''
    } catch (err) {
      console.warn(`[AeroMaxx] Vision model ${model} failed, trying next:`, err)
    }
  }

  throw new Error('All vision models failed')
}
