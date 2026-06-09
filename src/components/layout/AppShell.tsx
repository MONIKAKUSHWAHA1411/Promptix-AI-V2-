import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { LogOut, User, Zap } from 'lucide-react'
import { useAuth } from '@/hooks/useAuth'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export function AppShell() {
  const { user, signOut } = useAuth()
  const navigate = useNavigate()

  const handleSignOut = async () => {
    await signOut()
    navigate('/login')
  }

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b border-border bg-white sticky top-0 z-40">
        <div className="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <NavLink to="/forge" className="flex items-center gap-2 font-bold text-navy">
              <span className="bg-brand text-white rounded-md px-1.5 py-0.5 text-sm font-bold">QF</span>
              <span className="hidden sm:block">QualityForge</span>
            </NavLink>
            <nav className="flex items-center gap-1">
              <NavLink
                to="/forge"
                className={({ isActive }) =>
                  cn('px-3 py-1.5 rounded-md text-sm font-medium transition-colors', isActive ? 'bg-surface text-navy' : 'text-muted hover:text-navy hover:bg-surface')
                }
              >
                <span className="flex items-center gap-1.5"><Zap className="h-3.5 w-3.5" />Forge</span>
              </NavLink>
              <NavLink
                to="/history"
                className={({ isActive }) =>
                  cn('px-3 py-1.5 rounded-md text-sm font-medium transition-colors', isActive ? 'bg-surface text-navy' : 'text-muted hover:text-navy hover:bg-surface')
                }
              >
                History
              </NavLink>
            </nav>
          </div>

          <div className="flex items-center gap-2">
            <NavLink
              to="/profile"
              className={({ isActive }) =>
                cn('flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm text-muted hover:text-navy hover:bg-surface transition-colors', isActive && 'bg-surface text-navy')
              }
            >
              <User className="h-3.5 w-3.5" />
              <span className="hidden sm:block max-w-[120px] truncate">{user?.email}</span>
            </NavLink>
            <Button variant="ghost" size="icon" onClick={handleSignOut} title="Sign out">
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-4 py-8">
        <Outlet />
      </main>
    </div>
  )
}
