import { useState, useCallback } from 'react'
import { supabase } from '@/lib/supabase'
import { useAuth } from './useAuth'
import type { RunRow, FullRun, RunSummary, ScoredTestCase } from '@/types'

export function useRuns() {
  const { user } = useAuth()
  const [runs, setRuns] = useState<RunRow[]>([])
  const [loading, setLoading] = useState(false)

  const fetchRuns = useCallback(async () => {
    if (!user) return
    setLoading(true)
    const { data, error } = await supabase
      .from('runs')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })
    setLoading(false)
    if (error) throw error
    setRuns((data as RunRow[]) ?? [])
  }, [user])

  const fetchFullRun = useCallback(async (runId: string): Promise<FullRun | null> => {
    const { data: runData, error: runError } = await supabase
      .from('runs')
      .select('*')
      .eq('id', runId)
      .single()
    if (runError || !runData) return null

    const { data: casesData, error: casesError } = await supabase
      .from('test_cases')
      .select('*')
      .eq('run_id', runId)
      .order('created_at', { ascending: true })
    if (casesError) return null

    return { ...(runData as RunRow), test_cases: casesData ?? [] } as FullRun
  }, [])

  const saveRun = useCallback(async (
    scenario: string,
    summary: RunSummary,
    cases: ScoredTestCase[]
  ): Promise<string> => {
    if (!user) throw new Error('Not authenticated')

    const { data: run, error: runError } = await supabase
      .from('runs')
      .insert({ user_id: user.id, scenario, summary })
      .select('id')
      .single()
    if (runError) throw runError

    const rows = cases.map((tc) => ({
      run_id: run.id,
      category: tc.category,
      title: tc.title,
      preconditions: tc.preconditions || null,
      steps: tc.steps,
      expected_result: tc.expected_result || null,
      priority: tc.priority,
      scores: tc.scores,
    }))

    const { error: casesError } = await supabase.from('test_cases').insert(rows)
    if (casesError) throw casesError

    return run.id as string
  }, [user])

  const deleteRun = useCallback(async (runId: string) => {
    const { error } = await supabase.from('runs').delete().eq('id', runId)
    if (error) throw error
    setRuns((prev) => prev.filter((r) => r.id !== runId))
  }, [])

  return { runs, loading, fetchRuns, fetchFullRun, saveRun, deleteRun }
}
