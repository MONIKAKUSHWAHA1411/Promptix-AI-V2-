import { useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '@/hooks/useAuth'
import { useRuns } from '@/hooks/useRuns'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { RunCard } from '@/components/history/RunCard'
import { Skeleton } from '@/components/ui/skeleton'

export function ProfilePage() {
  const { user } = useAuth()
  const { runs, loading, fetchRuns } = useRuns()
  const navigate = useNavigate()

  useEffect(() => {
    fetchRuns()
  }, [fetchRuns])

  const totalCases = runs.reduce((sum, r) => sum + (r.summary?.total ?? 0), 0)
  const avgConf = runs.length > 0
    ? runs.reduce((sum, r) => sum + (r.summary?.avg_confidence ?? 0), 0) / runs.length
    : 0

  const recent = runs.slice(0, 3)

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-navy">Profile</h1>

      <Card>
        <CardHeader><CardTitle>Account</CardTitle></CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted">Email</span>
            <span className="text-navy font-medium">{user?.email}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted">Provider</span>
            <span className="text-navy font-medium capitalize">{user?.app_metadata?.provider ?? 'email'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted">Member since</span>
            <span className="text-navy font-medium">{user?.created_at ? new Date(user.created_at).toLocaleDateString() : '—'}</span>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader><CardTitle>Statistics</CardTitle></CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          ) : (
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-2xl font-bold text-navy">{runs.length}</p>
                <p className="text-xs text-muted">Total Runs</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-navy">{totalCases}</p>
                <p className="text-xs text-muted">Test Cases</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-navy">{runs.length > 0 ? `${(avgConf * 100).toFixed(0)}%` : '—'}</p>
                <p className="text-xs text-muted">Avg Confidence</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-base font-semibold text-navy">Recent Runs</h2>
          <Button variant="link" asChild className="text-sm">
            <Link to="/history">View all</Link>
          </Button>
        </div>
        {loading ? (
          <div className="space-y-2">
            {[1, 2].map((i) => <Skeleton key={i} className="h-20 w-full" />)}
          </div>
        ) : recent.length > 0 ? (
          <div className="space-y-2">
            {recent.map((run) => (
              <RunCard
                key={run.id}
                run={run}
                onDelete={() => {}}
                onClick={() => navigate('/history')}
              />
            ))}
          </div>
        ) : (
          <p className="text-sm text-muted">No runs yet.</p>
        )}
      </div>
    </div>
  )
}
