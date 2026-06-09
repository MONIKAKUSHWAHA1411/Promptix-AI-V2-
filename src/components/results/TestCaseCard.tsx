import { useState } from 'react'
import { ChevronDown, ChevronUp } from 'lucide-react'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { ScoreBreakdown } from './ScoreBreakdown'
import type { ScoredTestCase } from '@/types'
import { cn } from '@/lib/utils'

function scoreIndicator(composite: number): string {
  if (composite >= 0.85) return 'bg-success'
  if (composite >= 0.70) return 'bg-warning'
  return 'bg-danger'
}

function priorityVariant(p: string): 'danger' | 'warning' | 'secondary' {
  if (p === 'high') return 'danger'
  if (p === 'medium') return 'warning'
  return 'secondary'
}

export function TestCaseCard({ testCase, defaultExpanded = false }: { testCase: ScoredTestCase; defaultExpanded?: boolean }) {
  const [expanded, setExpanded] = useState(defaultExpanded)

  return (
    <Card className="overflow-hidden">
      <button
        className="w-full text-left"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        <div className="flex items-center gap-3 p-4">
          <div className={cn('w-2.5 h-2.5 rounded-full shrink-0', scoreIndicator(testCase.scores.composite))} />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-navy truncate">{testCase.title}</p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Badge variant="outline" className="text-xs capitalize">{testCase.category}</Badge>
            <Badge variant={priorityVariant(testCase.priority)} className="text-xs capitalize">{testCase.priority}</Badge>
            <span className="text-xs font-mono font-semibold text-muted w-12 text-right">
              {(testCase.scores.composite * 100).toFixed(0)}%
            </span>
            {expanded ? <ChevronUp className="h-4 w-4 text-muted" /> : <ChevronDown className="h-4 w-4 text-muted" />}
          </div>
        </div>
      </button>

      {expanded && (
        <CardContent className="pt-0 pb-4 px-4 space-y-4">
          <Separator />
          {testCase.preconditions && (
            <div>
              <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-1">Preconditions</p>
              <p className="text-sm text-navy">{testCase.preconditions}</p>
            </div>
          )}
          <div>
            <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">Steps</p>
            <ol className="space-y-1.5">
              {testCase.steps.map((step, i) => (
                <li key={i} className="flex gap-2 text-sm text-navy">
                  <span className="text-muted shrink-0 font-mono">{i + 1}.</span>
                  <span>{step}</span>
                </li>
              ))}
            </ol>
          </div>
          {testCase.expected_result && (
            <div>
              <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-1">Expected Result</p>
              <p className="text-sm text-navy">{testCase.expected_result}</p>
            </div>
          )}
          <div>
            <p className="text-xs font-semibold text-muted uppercase tracking-wide mb-2">Quality Score</p>
            <ScoreBreakdown scores={testCase.scores} />
          </div>
        </CardContent>
      )}
    </Card>
  )
}
