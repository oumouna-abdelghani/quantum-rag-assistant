import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function uid(): string {
  return Math.random().toString(36).substring(2, 9) + Date.now().toString(36)
}

export function extractCitations(content: string, maxSources: number = 20): number[] {
  if (!content) return []
  const matches = content.match(/\[(?:Source\s*)?(\d+)\]/gi)
  if (!matches) return []

  const citations = new Set<number>()
  for (const match of matches) {
    const strMatch = String(match)
    const num = parseInt(strMatch.replace(/\D/g, ''), 10)
    if (!isNaN(num) && num > 0 && num <= maxSources) {
      citations.add(num)
    }
  }

  return Array.from(citations).sort((a, b) => a - b)
}

export default uid