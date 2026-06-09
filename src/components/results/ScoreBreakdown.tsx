import { Progress } from '@/components/ui/progress'
import type { Score } from '@/types'
import { cn } from '@/lib/utils'

function scoreColor(v: number) {
  if (v >= 0.85) return 'bg-success'
  if (v >= 0.70) return 'bg-warning'
  return 'bg-danger'
}

interface RowProps {
  label: string
  value: number
  weight?: string
  bold?: boolean
}

function ScoreRow({ label, value, weight, bold }: RowProps) {
  return (
    <div className={cn('flex items-center gap-3', bold && 'pt-2 border-t border-border mt-1')}>
      <span className={cn('w-28 text-xs shrink-0', bold ? 'font-semibold text-navy' : 'text-muted')}>
        {label} {weight && <span className="text-muted font-normal">({weight})</span>}
      </span>
      <Progress value={value * 100} className={cn('h-2', bold && 'h-3')} indicatorClassName={scoreColor(value)} />
      <span className={cn('w-10 text-xs text-right shrink-0', bold ? 'font-semibold text-navy' : 'text-muted')}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  )
}

export function ScoreBreakdown({ scores }: { scores: Score }) {
  return (
    <div className="space-y-2 p-3 rounded-md bg-surface">
      <ScoreRow label="Hallucination" value={scores.hallucination} weight="40%" />
      <ScoreRow label="Coverage" value={scores.coverage} weight="40%" />
      <ScoreRow label="Clarity" value={scores.clarity} weight="20%" />
      <ScoreRow label="Composite" value={scores.composite} bold />
    </div>
  )
}
