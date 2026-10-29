'use client'

import React from 'react'

interface LensTitleProps {
  text?: string
  children?: React.ReactNode
  className?: string
}

export function LensTitle({ text, children, className = '' }: LensTitleProps) {
  return (
    <h1 className={className}>
      {text || children || 'Quanta'}
    </h1>
  )
}

export default LensTitle