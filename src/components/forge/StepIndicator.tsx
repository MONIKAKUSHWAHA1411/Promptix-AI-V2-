import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { ForgeStep } from '@/types'

const STEPS = [
  { n: 1 as ForgeStep, label: 'Scenario' },
  { n: 2 as ForgeStep, label: 'Generating' },
  { n: 3 as ForgeStep, label: 'Evaluating' },
  { n: 4 as ForgeStep, label: 'Results' },
]

export function StepIndicator({ currentStep }: { currentStep: ForgeStep }) {
  return (
    <div className="flex items-center justify-between mb-8">
      <div className="sm:hidden text-sm text-muted font-medium">
        Step {currentStep} of {STEPS.length} — {STEPS[currentStep - 1].label}
      </div>
      <div className="hidden sm:flex items-center gap-0 w-full">
        {STEPS.map((step, i) => {
          const done = currentStep > step.n
          const active = currentStep === step.n
          return (
            <div key={step.n} className="flex items-center flex-1 last:flex-none">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={cn(
                    'w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold border-2 transition-colors',
                    done && 'bg-brand border-brand text-white',
                    active && 'bg-white border-brand text-brand',
                    !done && !active && 'bg-white border-border text-muted'
                  )}
                >
                  {done ? <Check className="h-4 w-4" /> : step.n}
                </div>
                <span className={cn('text-xs font-medium', active ? 'text-navy' : 'text-muted')}>{step.label}</span>
              </div>
              {i < STEPS.length - 1 && (
                <div className={cn('flex-1 h-0.5 mx-2 mb-5 transition-colors', done ? 'bg-brand' : 'bg-border')} />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
