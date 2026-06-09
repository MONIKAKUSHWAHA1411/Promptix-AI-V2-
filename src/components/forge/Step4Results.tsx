import { useState } from 'react'
import { Check, RotateCcw, Save } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SummaryBar } from '@/components/results/SummaryBar'
import { TestCaseCard } from '@/components/results/TestCaseCard'
import { ExportButtons } from '@/components/results/ExportButtons'
import { Separator } from '@/components/ui/separator'
import type { ScoredTestCase, RunSummary, Category } from '@/types'
import { cn } from '@/lib/utils'

const TABS: { label: string; value: Category | 'all' }[] = [
  { label: 'All', value: 'all' },
  { label: 'Functional', value: 'functional' },
  { label: 'Negative', value: 'negative' },
  { label: 'API', value: 'api' },
]

interface Props {
  scoredCases: ScoredTestCase[]
  summary: RunSummary
  scenario: string
  onSave: () => Promise<void>
  onReset: () => void
  isSaving: boolean
  savedRunId: string | null
  readOnly?: boolean
}

export function Step4Results({ scoredCases, summary, scenario, onSave, onReset, isSaving, savedRunId, readOnly = false }: Props) {
  const [activeTab, setActiveTab] = useState<Category | 'all'>('all')

  const sorted = [...scoredCases].sort((a, b) => a.scores.composite - b.scores.composite)
  const filtered = activeTab === 'all' ? sorted : sorted.filter((tc) => tc.category === activeTab)

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-xl font-bold text-navy">Test Suite Results</h2>
          <p className="text-sm text-muted mt-0.5">Sorted by score (lowest first)</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <ExportButtons cases={scoredCases} scenario={scenario} />
          {!readOnly && (
            <>
              {savedRunId ? (
                <Button variant="secondary" size="sm" className="gap-2 text-success" disabled>
                  <Check className="h-4 w-4" />
                  Saved
                </Button>
              ) : (
                <Button size="sm" className="gap-2" onClick={onSave} disabled={isSaving}>
                  <Save className="h-4 w-4" />
                  {isSaving ? 'Saving...' : 'Save to History'}
                </Button>
              )}
              <Button variant="outline" size="sm" className="gap-2" onClick={onReset}>
                <RotateCcw className="h-4 w-4" />
                Start Over
              </Button>
            </>
          )}
        </div>
      </div>

      <SummaryBar summary={summary} />

      <div className="flex gap-1 border-b border-border">
        {TABS.map((tab) => {
          const count = tab.value === 'all' ? scoredCases.length : scoredCases.filter((tc) => tc.category === tab.value).length
          return (
            <button
              key={tab.value}
              onClick={() => setActiveTab(tab.value)}
              className={cn(
                'px-4 py-2 text-sm font-medium border-b-2 -mb-px transition-colors',
                activeTab === tab.value ? 'border-brand text-brand' : 'border-transparent text-muted hover:text-navy'
              )}
            >
              {tab.label} <span className="ml-1 text-xs opacity-70">({count})</span>
            </button>
          )
        })}
      </div>

      <Separator className="hidden" />

      <div className="space-y-3">
        {filtered.map((tc) => (
          <TestCaseCard key={tc.id} testCase={tc} />
        ))}
        {filtered.length === 0 && (
          <p className="text-sm text-muted text-center py-8">No test cases in this category.</p>
        )}
      </div>
    </div>
  )
}
