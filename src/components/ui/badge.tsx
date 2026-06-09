import * as React from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-navy text-white',
        secondary: 'border-transparent bg-surface text-navy',
        outline: 'text-navy border-border',
        success: 'border-transparent bg-green-100 text-success',
        warning: 'border-transparent bg-amber-100 text-warning',
        danger: 'border-transparent bg-red-100 text-danger',
        brand: 'border-transparent bg-orange-100 text-brand',
      },
    },
    defaultVariants: { variant: 'default' },
  }
)

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
