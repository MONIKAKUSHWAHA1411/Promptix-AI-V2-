import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '@/hooks/useAuth'
import { GoogleButton } from '@/components/auth/GoogleButton'
import { LoginForm } from '@/components/auth/LoginForm'
import { SignupForm } from '@/components/auth/SignupForm'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'

export function LoginPage() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [tab, setTab] = useState<'login' | 'signup'>('login')

  useEffect(() => {
    if (user) navigate('/forge', { replace: true })
  }, [user, navigate])

  return (
    <div className="min-h-screen bg-white flex items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-brand text-white font-bold text-lg mb-4">QF</div>
          <h1 className="text-2xl font-bold text-navy">QualityForge</h1>
          <p className="text-sm text-muted mt-1">AI-powered QA test suite generator</p>
        </div>

        <div className="rounded-xl border border-border bg-white shadow-sm p-6 space-y-5">
          <GoogleButton />

          <div className="flex items-center gap-3">
            <Separator className="flex-1" />
            <span className="text-xs text-muted">or continue with email</span>
            <Separator className="flex-1" />
          </div>

          <div className="flex border border-border rounded-lg overflow-hidden">
            {(['login', 'signup'] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={cn(
                  'flex-1 py-2 text-sm font-medium transition-colors',
                  tab === t ? 'bg-navy text-white' : 'bg-white text-muted hover:text-navy'
                )}
              >
                {t === 'login' ? 'Log in' : 'Sign up'}
              </button>
            ))}
          </div>

          {tab === 'login' ? <LoginForm /> : <SignupForm />}
        </div>
      </div>
    </div>
  )
}
