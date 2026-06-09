import { Zap } from 'lucide-react'
import { Link } from 'react-router-dom'
import { RunCard } from './RunCard'
import { Button } from '@/components/ui/button'
import type { RunRow } from '@/types'

interface Props {
  runs: RunRow[]
  onDelete: (id: string) => void
  onSelect: (id: string) => void
}

export function RunList({ runs, onDelete, onSelect }: Props) {
  if (runs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <div className="w-12 h-12 rounded-full bg-surface flex items-center justify-center mb-4">
          <Zap className="h-6 w-6 text-muted" />
        </div>
        <h3 className="text-base font-semibold text-navy mb-1">No runs yet</h3>
        <p className="text-sm text-muted mb-4">Go to Forge to generate your first test suite.</p>
        <Button asChild>
          <Link to="/forge">Start Forging</Link>
        </Button>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {runs.map((run) => (
        <RunCard key={run.id} run={run} onDelete={onDelete} onClick={onSelect} />
      ))}
    </div>
  )
}
