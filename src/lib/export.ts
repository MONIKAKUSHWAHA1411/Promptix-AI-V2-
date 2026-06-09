import type { ScoredTestCase } from '@/types'

function downloadBlob(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

export function downloadJSON(cases: ScoredTestCase[], scenario: string) {
  const payload = { scenario, test_cases: cases, exported_at: new Date().toISOString() }
  downloadBlob(JSON.stringify(payload, null, 2), 'qualityforge-test-suite.json', 'application/json')
}

export function downloadCSV(cases: ScoredTestCase[]) {
  const headers = [
    'ID', 'Category', 'Title', 'Priority',
    'Preconditions', 'Steps', 'Expected Result',
    'Hallucination', 'Coverage', 'Clarity', 'Composite',
  ]
  const rows = cases.map((tc) => [
    tc.id,
    tc.category,
    `"${tc.title.replace(/"/g, '""')}"`,
    tc.priority,
    `"${(tc.preconditions || '').replace(/"/g, '""')}"`,
    `"${tc.steps.join(' | ').replace(/"/g, '""')}"`,
    `"${(tc.expected_result || '').replace(/"/g, '""')}"`,
    tc.scores.hallucination.toFixed(2),
    tc.scores.coverage.toFixed(2),
    tc.scores.clarity.toFixed(2),
    tc.scores.composite.toFixed(2),
  ])
  const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n')
  downloadBlob(csv, 'qualityforge-test-suite.csv', 'text/csv')
}
