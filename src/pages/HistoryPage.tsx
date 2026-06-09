import { useEffect, useState } from 'react'
import { useRuns } from '@/hooks/useRuns'
import { RunList } from '@/components/history/RunList'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Step4Results } from '@/components/forge/Step4Results'
import { Skeleton } from '@/components/ui/skeleton'
import type { FullRun, ScoredTestCase } from '@/types'

function fullRunToScoredCases(run: FullRun): ScoredTestCase[] {
  return run.test_cases.map((tc) => ({
    id: tc.id,
    title: tc.title,
    preconditions: tc.preconditions ?? '',
    steps: tc.steps,
    expected_result: tc.expected_result ?? '',
    priority: tc.priority,
    category: tc.category,
    scores: tc.scores ?? { hallucination: 0, coverage: 0, clarity: 0, composite: 0 },
  }))
}

export function HistoryPage() {
  const { runs, loading, fetchRuns, fetchFullRun, deleteRun } = useRuns()
  const [selectedRun, setSelectedRun] = useState<FullRun | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [loadingRun, setLoadingRun] = useState(false)

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  const handleSelect = async (id: string) => {
    setDialogOpen(true)
    setLoadingRun(true)
    const run = await fetchFullRun(id)
    setSelectedRun(run)
    setLoadingRun(false)
  }

  const handleDelete = async (id: string) => {
    await deleteRun(id)
  }

  return (
    <div>
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-navy">Run History</h1>
        <p className="text-sm text-muted mt-1">Click any run to view the full test suite.</p>
      </div>

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => <Skeleton key={i} className="h-20 w-full" />)}
        </div>
      ) : (
        <RunList runs={runs} onDelete={handleDelete} onSelect={handleSelect} />
      )}

      <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
        <DialogContent className="p-6">
          <DialogHeader>
            <DialogTitle>Run Details</DialogTitle>
          </DialogHeader>
          {loadingRun ? (
            <div className="space-y-3 p-6">
              {[1, 2, 3].map((i) => <Skeleton key={i} className="h-16 w-full" />)}
            </div>
          ) : selectedRun && selectedRun.summary ? (
            <div className="p-6 pt-4">
              <Step4Results
                scoredCases={fullRunToScoredCases(selectedRun)}
                summary={selectedRun.summary}
                scenario={selectedRun.scenario}
                onSave={async () => {}}
                onReset={() => setDialogOpen(false)}
                isSaving={false}
                savedRunId={selectedRun.id}
                readOnly
              />
            </div>
          ) : (
            <p className="p-6 text-sm text-muted">Could not load run details.</p>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}
