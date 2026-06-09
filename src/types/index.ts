export type Priority = 'high' | 'medium' | 'low'
export type Category = 'functional' | 'negative' | 'api'

export interface TestCase {
  id: string
  title: string
  preconditions: string
  steps: string[]
  expected_result: string
  priority: Priority
}

export interface Score {
  hallucination: number
  coverage: number
  clarity: number
  composite: number
}

export interface ScoredTestCase extends TestCase {
  category: Category
  scores: Score
}

export interface RunSummary {
  total: number
  passed: number
  flagged: number
  avg_confidence: number
}

export interface RunRow {
  id: string
  user_id: string
  scenario: string
  summary: RunSummary
  created_at: string
}

export interface TestCaseRow {
  id: string
  run_id: string
  category: Category
  title: string
  preconditions: string | null
  steps: string[]
  expected_result: string | null
  priority: Priority
  scores: Score | null
  created_at: string
}

export interface FullRun extends RunRow {
  test_cases: TestCaseRow[]
}

export type ForgeStep = 1 | 2 | 3 | 4

export interface ForgeState {
  step: ForgeStep
  scenario: string
  groqApiKey: string
  scoredCases: ScoredTestCase[]
  summary: RunSummary | null
  isGenerating: boolean
  isEvaluating: boolean
  evaluatedCount: number
  totalToEvaluate: number
  error: string | null
  savedRunId: string | null
}
