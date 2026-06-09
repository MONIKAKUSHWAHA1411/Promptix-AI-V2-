import { Download, Table } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { downloadJSON, downloadCSV } from '@/lib/export'
import type { ScoredTestCase } from '@/types'

interface Props {
  cases: ScoredTestCase[]
  scenario: string
}

export function ExportButtons({ cases, scenario }: Props) {
  return (
    <div className="flex items-center gap-2">
      <Button variant="outline" size="sm" className="gap-2" onClick={() => downloadJSON(cases, scenario)}>
        <Download className="h-4 w-4" />
        Export JSON
      </Button>
      <Button variant="outline" size="sm" className="gap-2" onClick={() => downloadCSV(cases)}>
        <Table className="h-4 w-4" />
        Export CSV
      </Button>
    </div>
  )
}
