import { FlaskConical } from 'lucide-react'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'

interface Props {
  evaluatedCount: number
  totalToEvaluate: number
}

export function Step3Evaluating({ evaluatedCount, totalToEvaluate }: Props) {
  const pct = totalToEvaluate > 0 ? Math.round((evaluatedCount / totalToEvaluate) * 100) : 0

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <FlaskConical className="h-5 w-5 text-brand" />
        <div>
          <h2 className="text-lg font-semibold text-navy">Running LLM-as-Judge evaluation...</h2>
          <p className="text-sm text-muted">Scoring each test case on hallucination, coverage, and clarity</p>
        </div>
      </div>

      <div className="space-y-3 rounded-lg border border-border bg-surface p-6">
        <div className="flex items-center justify-between text-sm">
          <span className="text-navy font-medium">{evaluatedCount} of {totalToEvaluate} test cases scored</span>
          <span className="text-muted">{pct}%</span>
        </div>
        <Progress value={pct} className="h-3" />
        <div className="flex items-center gap-2 pt-1">
          <Badge variant="secondary" className="text-xs gap-1">
            <span className="w-2 h-2 rounded-full bg-brand inline-block animate-pulse" />
            Powered by Groq llama-3.3-70b
          </Badge>
          {evaluatedCount > 0 && (
            <span className="text-xs text-muted">Auto-regenerating flagged cases after scoring completes</span>
          )}
        </div>
      </div>
    </div>
  )
}
