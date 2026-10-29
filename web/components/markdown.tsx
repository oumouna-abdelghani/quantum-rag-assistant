'use client'

import React, { useMemo, useState } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Check, Copy } from 'lucide-react'
import 'katex/dist/katex.min.css'
import type { Source } from '@/lib/types'

interface MarkdownProps {
  content: string
  sources?: Source[]
  onCite?: (index: number) => void
}

export function CopyButton({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs text-zinc-400 transition hover:bg-white/[0.06] hover:text-zinc-200"
    >
      {copied ? <Check className="size-3 text-emerald-400" /> : <Copy className="size-3" />}
      <span>{copied ? 'Copied!' : (label ?? 'Copy')}</span>
    </button>
  )
}

/** Turn [1], [2][3] and [1, 2] into links like [1](#cite-1) so the renderer can style them.
 *  Fenced code blocks (even unclosed ones, while streaming) are left untouched. */
function withCitations(text: string): string {
  return text
    .split(/(```[\s\S]*?(?:```|$))/g)
    .map((segment, i) =>
      i % 2 === 1
        ? segment
        : segment.replace(/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g, (_, nums: string) =>
            nums
              .split(/\s*,\s*/)
              .map((n) => `[${n}](#cite-${n})`)
              .join('')
          )
    )
    .join('')
}

export function Markdown({ content, onCite }: MarkdownProps) {
  const components = useMemo<Components>(
    () => ({
      h1: ({ children }) => <h1 className="mb-3 mt-7 text-2xl font-semibold tracking-tight text-white first:mt-0">{children}</h1>,
      h2: ({ children }) => <h2 className="mb-2 mt-7 border-b border-white/[0.07] pb-1.5 text-xl font-semibold tracking-tight text-white first:mt-0">{children}</h2>,
      h3: ({ children }) => <h3 className="mb-1.5 mt-5 text-base font-semibold text-sky-200 first:mt-0">{children}</h3>,
      h4: ({ children }) => <h4 className="mb-1 mt-4 text-sm font-semibold uppercase tracking-wide text-zinc-300">{children}</h4>,
      p: ({ children }) => <p className="my-3 first:mt-0 last:mb-0">{children}</p>,
      strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
      em: ({ children }) => <em className="italic text-zinc-100">{children}</em>,
      ul: ({ children }) => <ul className="my-3 list-disc space-y-1.5 pl-6 marker:text-sky-300/60">{children}</ul>,
      ol: ({ children }) => <ol className="my-3 list-decimal space-y-1.5 pl-6 marker:text-sky-300/60">{children}</ol>,
      li: ({ children }) => <li className="pl-1">{children}</li>,
      hr: () => <hr className="my-6 border-white/[0.08]" />,
      blockquote: ({ children }) => <blockquote className="my-4 border-l-2 border-sky-300/40 pl-4 text-zinc-400">{children}</blockquote>,
      table: ({ children }) => <div className="my-4 overflow-x-auto"><table className="w-full border-collapse text-sm">{children}</table></div>,
      th: ({ children }) => <th className="border border-white/10 bg-white/[0.04] px-3 py-1.5 text-left font-semibold">{children}</th>,
      td: ({ children }) => <td className="border border-white/10 px-3 py-1.5">{children}</td>,
      a: ({ href, children }) => {
        const cite = /^#cite-(\d+)$/.exec(href ?? '')
        if (cite) {
          const index = parseInt(cite[1], 10)
          if (!onCite) return <span className="text-xs font-semibold text-sky-300">[{index}]</span>
          return (
            <button
              type="button"
              onClick={() => onCite(index)}
              className="mx-0.5 inline-flex -translate-y-px items-center justify-center rounded-md border border-sky-400/30 bg-sky-400/10 px-1.5 py-0.5 align-baseline text-[11px] font-semibold leading-none text-sky-300 transition hover:bg-sky-400/20"
            >
              {index}
            </button>
          )
        }
        return <a href={href} target="_blank" rel="noopener noreferrer" className="text-sky-300 underline decoration-sky-300/30 underline-offset-2 hover:decoration-sky-300">{children}</a>
      },
      // Inline code only: fenced blocks are handled by `pre` below
      code: ({ children }) => <code className="rounded-md bg-white/[0.08] px-1.5 py-0.5 font-mono text-[0.85em] text-cyan-200">{children}</code>,
      pre: ({ children }) => {
        const child = React.Children.toArray(children)[0] as React.ReactElement<{ className?: string; children?: React.ReactNode }> | undefined
        const code = String(child?.props?.children ?? '').replace(/\n$/, '')
        const language = /language-([\w-]+)/.exec(child?.props?.className ?? '')?.[1]
        return (
          <div className="my-4 overflow-hidden rounded-2xl border border-white/[0.08] bg-zinc-900/90 shadow-lg">
            <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-1.5">
              <span className="font-mono text-[11px] uppercase tracking-wider text-zinc-500">{language ?? 'code'}</span>
              <CopyButton text={code} />
            </div>
            <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-cyan-100"><code>{code}</code></pre>
          </div>
        )
      },
    }),
    [onCite]
  )

  const processed = useMemo(() => withCitations(content), [content])
  if (!content) return null

  return (
    <div className="text-[15px] leading-7 text-zinc-200 [&_.katex-display]:my-4 [&_.katex-display]:overflow-x-auto [&_.katex-display]:overflow-y-hidden [&_.katex]:text-[1.05em]">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]} components={components}>
        {processed}
      </ReactMarkdown>
    </div>
  )
}

export default Markdown