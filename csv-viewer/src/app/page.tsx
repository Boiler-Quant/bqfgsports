'use client'

import CSVViewer from '@/components/csv-viewer'

export default function Home() {
  return (
    <main className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">BQFG Sports</h1>
      <CSVViewer url="https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv" />
    </main>
  )
}