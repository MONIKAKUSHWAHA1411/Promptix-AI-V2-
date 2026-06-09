import { Trash2, ChevronRight } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import type { RunRow } from '@/types'

function relativeTime(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  if (days < 30) return `${days}d ago`
  return new Date(dateStr).toLocaleDateString()
}

interface Props {
  run: RunRow
  onDelete: (id: string) => void
  onClick: (id: string) => void
}

export function RunCard({ run, onDelete, onClick }: Props) {
  const summary = run.summary
  const avgPct = summary ? `${(summary.avg_confidence * 100).toFixed(0)}%` : '—'

  return (
    <Card className="group flex items-center gap-4 p-4 hover:border-brand/30 transition-colors cursor-pointer" onClick={() => onClick(run.id)}>
      <div className="flex-1 min-w-0">
        <p className="text-sm text-navy font-medium line-clamp-2 leading-snug">{run.scenario}</p>
        <div className="flex items-center gap-3 mt-1.5 text-xs text-muted">
          <span>{relativeTime(run.created_at)}</span>
          {summary && (
            <>
              <span>·</span>
              <span>{summary.total} cases</span>
              <span>·</span>
              <span>{summary.passed} passed</span>
              <span>·</span>
              <span>avg {avgPct}</span>
            </>
          )}
        </div>
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-muted opacity-0 group-hover:opacity-100 hover:text-danger"
          onClick={(e) => { e.stopPropagation(); onDelete(run.id) }}
          title="Delete run"
        >
          <Trash2 className="h-4 w-4" />
        </Button>
        <ChevronRight className="h-4 w-4 text-muted" />
      </div>
    </Card>
  )
}
