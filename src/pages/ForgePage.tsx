import { useState } from 'react'
import { useForge } from '@/hooks/useForge'
import { useRuns } from '@/hooks/useRuns'
import { StepIndicator } from '@/components/forge/StepIndicator'
import { Step1ScenarioInput } from '@/components/forge/Step1ScenarioInput'
import { Step2Generating } from '@/components/forge/Step2Generating'
import { Step3Evaluating } from '@/components/forge/Step3Evaluating'
import { Step4Results } from '@/components/forge/Step4Results'

export function ForgePage() {
  const { state, setScenario, setGroqApiKey, runGeneration, setSavedRunId, reset } = useForge()
  const { saveRun } = useRuns()
  const [isSaving, setIsSaving] = useState(false)

  const handleSave = async () => {
    if (!state.summary) return
    setIsSaving(true)
    try {
      const id = await saveRun(state.scenario, state.summary, state.scoredCases)
      setSavedRunId(id)
    } catch (err) {
      console.error('Failed to save run:', err)
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <div>
      <StepIndicator currentStep={state.step} />

      {state.step === 1 && (
        <Step1ScenarioInput
          scenario={state.scenario}
          onScenarioChange={setScenario}
          groqApiKey={state.groqApiKey}
          onGroqApiKeyChange={setGroqApiKey}
          onStart={runGeneration}
          error={state.error}
        />
      )}

      {state.step === 2 && <Step2Generating scenario={state.scenario} />}

      {state.step === 3 && (
        <Step3Evaluating
          evaluatedCount={state.evaluatedCount}
          totalToEvaluate={state.totalToEvaluate}
        />
      )}

      {state.step === 4 && state.summary && (
        <Step4Results
          scoredCases={state.scoredCases}
          summary={state.summary}
          scenario={state.scenario}
          onSave={handleSave}
          onReset={reset}
          isSaving={isSaving}
          savedRunId={state.savedRunId}
        />
      )}
    </div>
  )
}
