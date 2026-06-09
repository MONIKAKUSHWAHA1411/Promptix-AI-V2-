const GROQ_BASE = 'https://api.groq.com/openai/v1/chat/completions'
const MODEL = 'llama-3.3-70b-versatile'

export function stripMarkdownFences(text: string): string {
  return text
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```\s*$/, '')
    .trim()
}

async function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

export async function groqChat(
  apiKey: string,
  messages: { role: 'system' | 'user'; content: string }[],
  maxTokens = 4096,
  attempt = 0
): Promise<string> {
  const res = await fetch(GROQ_BASE, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      temperature: 0.2,
      max_tokens: maxTokens,
      response_format: { type: 'json_object' },
    }),
  })

  if (res.status === 429 && attempt < 3) {
    await sleep(Math.pow(2, attempt + 1) * 1000)
    return groqChat(apiKey, messages, maxTokens, attempt + 1)
  }

  if (!res.ok) {
    const err = await res.text()
    throw new Error(`Groq API error ${res.status}: ${err}`)
  }

  const data = await res.json()
  return data.choices[0].message.content as string
}

export async function groqParseJSON<T>(
  apiKey: string,
  messages: { role: 'system' | 'user'; content: string }[],
  maxTokens = 4096
): Promise<T> {
  let raw = await groqChat(apiKey, messages, maxTokens)
  try {
    return JSON.parse(stripMarkdownFences(raw)) as T
  } catch {
    raw = await groqChat(apiKey, messages, maxTokens)
    return JSON.parse(stripMarkdownFences(raw)) as T
  }
}
