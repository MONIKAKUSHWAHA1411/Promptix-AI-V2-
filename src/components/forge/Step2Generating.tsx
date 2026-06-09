import { Loader2 } from 'lucide-react'
import { Skeleton } from '@/components/ui/skeleton'
import { Card } from '@/components/ui/card'

export function Step2Generating({ scenario }: { scenario: string }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <Loader2 className="h-5 w-5 text-brand animate-spin" />
        <div>
          <h2 className="text-lg font-semibold text-navy">Generating test cases...</h2>
          <p className="text-sm text-muted">Analyzing your scenario with Groq llama-3.3-70b</p>
        </div>
      </div>

      <div className="rounded-md border border-border bg-surface p-3 text-sm text-muted italic line-clamp-3">
        {scenario}
      </div>

      <div className="space-y-3">
        {['Functional', 'Negative', 'API'].map((category) => (
          <div key={category}>
            <Skeleton className="h-4 w-24 mb-3" />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[1, 2].map((i) => (
                <Card key={i} className="p-4 space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-3 w-1/2" />
                  <Skeleton className="h-3 w-full" />
                  <Skeleton className="h-3 w-2/3" />
                </Card>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
