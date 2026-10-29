'use client'

import { useState, useEffect } from 'react'

const GREETINGS = [
  "Ask me anything about Quantum Computing...",
  "How do I construct a Bell state in Qiskit?",
  "Explain Shor's algorithm step-by-step.",
  "What is quantum teleportation?",
  "How do quantum logic gates work?"
]

export function GreetingLoop() {
  const [index, setIndex] = useState(0)
  const [subIndex, setSubIndex] = useState(0)
  const [reverse, setReverse] = useState(false)

  useEffect(() => {
    if (subIndex === GREETINGS[index].length + 1 && !reverse) {
      const timeout = setTimeout(() => setReverse(true), 2500)
      return () => clearTimeout(timeout)
    }

    if (subIndex === 0 && reverse) {
      setReverse(false)
      setIndex((prev) => (prev + 1) % GREETINGS.length)
      return
    }

    const timeout = setTimeout(() => {
      setSubIndex((prev) => prev + (reverse ? -1 : 1))
    }, reverse ? 30 : 60)

    return () => clearTimeout(timeout)
  }, [subIndex, index, reverse])

  return (
    <span className="text-zinc-400 font-normal">
      {GREETINGS[index].substring(0, subIndex)}
      <span className="animate-pulse text-cyan-400">|</span>
    </span>
  )
}

export default GreetingLoop