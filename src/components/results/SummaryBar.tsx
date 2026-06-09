import { Card, CardContent } from '@/components/ui/card'
import type { RunSummary } from '@/types'
import { cn } from '@/lib/utils'

interface Props {
  summary: RunSummary
}

export function SummaryBar({ summary }: Props) {
  const items = [
    { label: 'Total Cases', value: summary.total, accent: 'border-border' },
    { label: 'Passed', value: summary.passed, accent: 'border-success' },
    { label: 'Flagged', value: summary.flagged, accent: summary.flagged > 0 ? 'border-danger' : 'border-border' },
    { label: 'Avg Confidence', value: `${(summary.avg_confidence * 100).toFixed(0)}%`, accent: summary.avg_confidence >= 0.85 ? 'border-success' : summary.avg_confidence >= 0.70 ? 'border-warning' : 'border-danger' },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {items.map((item) => (
        <Card key={item.label} className={cn('border-l-4', item.accent)}>
          <CardContent className="p-4">
            <p className="text-2xl font-bold text-navy">{item.value}</p>
            <p className="text-xs text-muted mt-0.5">{item.label}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
