import { useState, useCallback, useRef } from 'react'
import { groqParseJSON } from '@/lib/groq'
import type { TestCase, Score, ScoredTestCase, RunSummary, ForgeState, Category } from '@/types'

const THRESHOLD = 0.70

function computeComposite(s: Omit<Score, 'composite'>): number {
  return s.hallucination * 0.4 + s.coverage * 0.4 + s.clarity * 0.2
}

function resolveApiKey(userKey: string): string {
  const envKey = (import.meta.env.VITE_GROQ_API_KEY as string | undefined) ?? ''
  const key = userKey.trim() || envKey.trim()
  if (!key) throw new Error('Groq API key is required. Enter your key in the field above.')
  return key
}

function buildGenerationMessages(scenario: string) {
  return [
    {
      role: 'system' as const,
      content: `You are a senior QA engineer. Given a scenario, generate test cases.
Return ONLY valid JSON with this structure:
{
  "functional": [{ "id": "f1", "title": "", "preconditions": "", "steps": [], "expected_result": "", "priority": "high|medium|low" }],
  "negative": [...],
  "api": [...]
}
Generate 3-5 test cases per category. Make them specific and actionable.`,
    },
    { role: 'user' as const, content: `Scenario: ${scenario}` },
  ]
}

function buildEvaluationMessages(scenario: string, testCase: TestCase) {
  return [
    {
      role: 'system' as const,
      content: `You are a senior QA engineer evaluating a test case.
Score on three dimensions (each 0.0 to 1.0):
- hallucination: are all steps grounded in the scenario? (1.0 = fully grounded, 0.0 = hallucinated)
- coverage: do preconditions + steps + expected_result fully cover the test objective? (1.0 = complete)
- clarity: can a junior tester execute it without ambiguity? (1.0 = crystal clear)
Return ONLY valid JSON: { "hallucination": float, "coverage": float, "clarity": float }`,
    },
    {
      role: 'user' as const,
      content: `Scenario: ${scenario}\n\nTest case:\n${JSON.stringify(testCase, null, 2)}`,
    },
  ]
}

function buildRegenerationMessages(scenario: string, badCase: TestCase & { category: Category }) {
  return [
    {
      role: 'system' as const,
      content: `You are a senior QA engineer. A test case received a low quality score. Rewrite it to be better grounded, more complete, and clearer.
Return ONLY valid JSON with a single test case object: { "id": "", "title": "", "preconditions": "", "steps": [], "expected_result": "", "priority": "high|medium|low" }`,
    },
    {
      role: 'user' as const,
      content: `Scenario: ${scenario}\n\nCategory: ${badCase.category}\n\nOriginal test case:\n${JSON.stringify(badCase, null, 2)}`,
    },
  ]
}

const initialState: ForgeState = {
  step: 1,
  scenario: '',
  groqApiKey: '',
  scoredCases: [],
  summary: null,
  isGenerating: false,
  isEvaluating: false,
  evaluatedCount: 0,
  totalToEvaluate: 0,
  error: null,
  savedRunId: null,
}

export function useForge() {
  const [state, setState] = useState<ForgeState>(initialState)
  const evaluatedRef = useRef(0)

  const setScenario = useCallback((scenario: string) => {
    setState((s) => ({ ...s, scenario }))
  }, [])

  const setGroqApiKey = useCallback((groqApiKey: string) => {
    setState((s) => ({ ...s, groqApiKey }))
  }, [])

  const runGeneration = useCallback(async () => {
    setState((s) => ({ ...s, error: null, isGenerating: true, step: 2, scoredCases: [], summary: null }))

    let apiKey: string
    try {
      apiKey = resolveApiKey(state.groqApiKey)
    } catch (err) {
      setState((s) => ({ ...s, isGenerating: false, step: 1, error: (err as Error).message }))
      return
    }

    try {
      const generated = await groqParseJSON<Record<Category, TestCase[]>>(
        apiKey,
        buildGenerationMessages(state.scenario)
      )

      const allCases: (TestCase & { category: Category })[] = [
        ...(generated.functional ?? []).map((tc) => ({ ...tc, category: 'functional' as Category })),
        ...(generated.negative ?? []).map((tc) => ({ ...tc, category: 'negative' as Category })),
        ...(generated.api ?? []).map((tc) => ({ ...tc, category: 'api' as Category })),
      ]

      evaluatedRef.current = 0
      setState((s) => ({
        ...s,
        isGenerating: false,
        isEvaluating: true,
        step: 3,
        evaluatedCount: 0,
        totalToEvaluate: allCases.length,
      }))

      const scoredCases = await Promise.all(
        allCases.map(async (tc) => {
          const rawScores = await groqParseJSON<Omit<Score, 'composite'>>(
            apiKey,
            buildEvaluationMessages(state.scenario, tc)
          )
          const composite = computeComposite(rawScores)
          evaluatedRef.current += 1
          setState((s) => ({ ...s, evaluatedCount: evaluatedRef.current }))
          return { ...tc, scores: { ...rawScores, composite } } as ScoredTestCase
        })
      )

      const flagged = scoredCases.filter((tc) => tc.scores.composite < THRESHOLD)
      const passed = scoredCases.filter((tc) => tc.scores.composite >= THRESHOLD)

      const regenerated = await Promise.all(
        flagged.map(async (tc) => {
          try {
            const newCase = await groqParseJSON<TestCase>(
              apiKey,
              buildRegenerationMessages(state.scenario, tc)
            )
            const rawScores = await groqParseJSON<Omit<Score, 'composite'>>(
              apiKey,
              buildEvaluationMessages(state.scenario, newCase)
            )
            const composite = computeComposite(rawScores)
            const newScored: ScoredTestCase = { ...newCase, category: tc.category, scores: { ...rawScores, composite } }
            return composite > tc.scores.composite ? newScored : tc
          } catch {
            return tc
          }
        })
      )

      const finalCases = [...passed, ...regenerated]
      const totalPassed = finalCases.filter((tc) => tc.scores.composite >= THRESHOLD).length
      const avgConf = finalCases.reduce((sum, tc) => sum + tc.scores.composite, 0) / finalCases.length

      const summary: RunSummary = {
        total: finalCases.length,
        passed: totalPassed,
        flagged: finalCases.length - totalPassed,
        avg_confidence: Math.round(avgConf * 100) / 100,
      }

      setState((s) => ({
        ...s,
        isEvaluating: false,
        step: 4,
        scoredCases: finalCases,
        summary,
      }))
    } catch (err) {
      setState((s) => ({
        ...s,
        isGenerating: false,
        isEvaluating: false,
        step: 1,
        error: (err as Error).message,
      }))
    }
  }, [state.groqApiKey, state.scenario])

  const setSavedRunId = useCallback((id: string) => {
    setState((s) => ({ ...s, savedRunId: id }))
  }, [])

  const reset = useCallback(() => {
    setState(initialState)
  }, [])

  return { state, setScenario, setGroqApiKey, runGeneration, setSavedRunId, reset }
}
