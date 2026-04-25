import type { BodyMeasurements, DragResults, GlbMeasurements } from './types'

const REF_FRONTAL_AREA  = 0.60
const REF_CD            = 0.80
const REF_DEPTH_WIDTH   = 0.50

export async function getRecommendations(
  m: BodyMeasurements,
  d: DragResults,
  glb?: GlbMeasurements | null,
): Promise<string> {

  const areaVsRef  = ((d.frontalArea - REF_FRONTAL_AREA) / REF_FRONTAL_AREA * 100).toFixed(1)
  const cdVsRef    = ((d.Cd - REF_CD) / REF_CD * 100).toFixed(1)
  const areaSource = glb ? '3D photogrammetric reconstruction' : '2D landmark estimation'

  const glbSection = glb ? `
3D GEOMETRY:
  Body width: ${glb.realWidth.toFixed(3)} m
  Body depth: ${glb.realDepth.toFixed(3)} m
  Depth-to-width ratio: ${glb.depthToWidthRatio.toFixed(3)} (reference: ${REF_DEPTH_WIDTH})
  Frontal area (3D): ${glb.frontalArea.toFixed(4)} m²` : ''

  const userMsg = `SUBJECT AERODYNAMIC PROFILE

DRAG EQUATION RESULTS:
  Cd: ${d.Cd.toFixed(4)} [${Number(cdVsRef) >= 0 ? '+' : ''}${cdVsRef}% vs ref]
  Frontal area: ${d.frontalArea.toFixed(4)} m² [${Number(areaVsRef) >= 0 ? '+' : ''}${areaVsRef}% vs ref] — ${areaSource}
  Drag force: ${d.dragForce.toFixed(3)} N
${glbSection}
POSTURAL ANALYSIS:
  Hunch score: ${m.hunchScore.toFixed(3)}
  Postural Cd penalty: +${d.posturalPenalty.toFixed(4)}
  Shoulder-to-hip ratio: ${m.shoulderToHipRatio.toFixed(3)}

LIFETIME IMPACT:
  Total drag work: ${d.lifetimeEnergy.toLocaleString('en-US', { maximumFractionDigits: 0 })} J
  Big Mac equivalents: ${d.bigMacs.toFixed(1)}
  Days lost: ${d.daysLost.toFixed(1)}`

  const response = await fetch(
    'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${import.meta.env.VITE_GEMINI_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'gemma-4-31b-it',
        messages: [
          {
            role: 'system',
            content: `You are a biomechanical aerodynamics analyst. Output ONLY the protocol below. No preamble, no reasoning, no markdown, no dollar signs, no LaTeX.

CONSTRAINTS:
- Interventions must be achievable without surgery, medical procedures, or body mass reduction
- No weight loss, fat reduction, or caloric deficit suggestions
- Focus exclusively on: posture correction, specific exercises (with sets/reps/weights), clothing choices, gait changes
- Use only the subject's actual measured numbers in targets

THE PHYSICS:
Drag reduction comes from two levers:
1. FRONTAL AREA (A) — reducing A by 10% reduces drag by exactly 10%.
2. DRAG COEFFICIENT (Cd) — shape and posture factor.
   - Thoracic kyphosis correction: 3–8% Cd reduction
   - Shoulder retraction: 2–5% Cd reduction
   - Core stabilisation: 3–6% Cd improvement
   - Clothing fit: 5–15% Cd difference
   - Depth-to-width ratio 0.45–0.55: 5–10% lower Cd

OUTPUT FORMAT (fill in brackets, output nothing else):
1. [INTERVENTION NAME]
Mechanism: [one sentence — how it reduces F=½ρv²CdA for this subject]
Target: [measured value] → [goal value with units]
Protocol: [specific exercise or action] — [sets x reps x weight OR daily/weekly frequency]
Reduction: [X]%

2. [INTERVENTION NAME]
Mechanism: [one sentence]
Target: [measured value] → [goal value with units]
Protocol: [specific exercise or action] — [sets x reps x weight OR daily/weekly frequency]
Reduction: [X]%

3. [INTERVENTION NAME]
Mechanism: [one sentence]
Target: [measured value] → [goal value with units]
Protocol: [specific exercise or action] — [sets x reps x weight OR daily/weekly frequency]
Reduction: [X]%

4. [INTERVENTION NAME]
Mechanism: [one sentence]
Target: [measured value] → [goal value with units]
Protocol: [specific exercise or action] — [sets x reps x weight OR daily/weekly frequency]
Reduction: [X]%

Ranked highest to lowest Reduction %. No text before or after. No thinking. No explanation.`,
          },
          { role: 'user', content: userMsg },
        ],
        temperature: 0.3,
        max_tokens: 500,
      }),
    }
  )

  if (!response.ok) {
    const errText = await response.text()
    throw new Error(`Gemma API error ${response.status}: ${errText}`)
  }

  const data = await response.json()
  const raw = data.choices?.[0]?.message?.content ?? ''
  // Strip thinking block if present
  return raw.replace(/<thought>[\s\S]*?<\/thought>/g, '').trim()
}