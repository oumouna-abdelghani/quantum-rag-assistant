import './globals.css'

export const metadata = {
  title: 'Quanta RAG',
  description: 'Quantum RAG Assistant',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-white min-h-screen antialiased">
        {children}
      </body>
    </html>
  )
}