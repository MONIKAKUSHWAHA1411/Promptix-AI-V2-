import { KeyRound, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { EXAMPLE_SCENARIOS } from '@/constants/examples'

interface Props {
  scenario: string
  onScenarioChange: (v: string) => void
  groqApiKey: string
  onGroqApiKeyChange: (v: string) => void
  onStart: () => void
  error: string | null
}

export function Step1ScenarioInput({ scenario, onScenarioChange, groqApiKey, onGroqApiKeyChange, onStart, error }: Props) {
  const hasEnvKey = !!(import.meta.env.VITE_GROQ_API_KEY as string | undefined)

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-navy">Describe your scenario</h1>
        <p className="text-muted mt-1 text-sm">Paste any user flow, feature description, or API spec — QualityForge will generate a graded test suite.</p>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <label className="text-sm font-semibold text-navy">Scenario</label>
          <Select onValueChange={(val) => onScenarioChange(val)}>
            <SelectTrigger className="w-48 h-8 text-xs">
              <SelectValue placeholder="Load example..." />
            </SelectTrigger>
            <SelectContent>
              {EXAMPLE_SCENARIOS.map((ex) => (
                <SelectItem key={ex.label} value={ex.value}>{ex.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Textarea
          className="min-h-44 text-sm leading-relaxed"
          placeholder="e.g. User adds items to cart, proceeds to checkout, enters payment details, applies a discount code, and places an order..."
          value={scenario}
          onChange={(e) => onScenarioChange(e.target.value)}
        />
      </div>

      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <KeyRound className="h-4 w-4 text-muted" />
          <label className="text-sm font-semibold text-navy">Groq API Key</label>
          {hasEnvKey && <span className="text-xs text-success bg-green-50 border border-green-200 rounded-full px-2 py-0.5">Configured via env</span>}
        </div>
        <Input
          type="password"
          placeholder={hasEnvKey ? 'Using VITE_GROQ_API_KEY env variable' : 'gsk_...'}
          value={groqApiKey}
          onChange={(e) => onGroqApiKeyChange(e.target.value)}
          disabled={hasEnvKey}
        />
        <p className="text-xs text-muted">Your key is used only in-browser and never sent to our servers. Get one at <a href="https://console.groq.com" target="_blank" rel="noopener noreferrer" className="text-brand hover:underline">console.groq.com</a>.</p>
      </div>

      {error && (
        <div className="rounded-md bg-red-50 border border-red-200 p-3 text-sm text-danger">{error}</div>
      )}

      <Button
        className="w-full h-12 text-base font-semibold gap-2"
        onClick={onStart}
        disabled={!scenario.trim()}
      >
        <Zap className="h-5 w-5" />
        Generate Test Cases
      </Button>
    </div>
  )
}
