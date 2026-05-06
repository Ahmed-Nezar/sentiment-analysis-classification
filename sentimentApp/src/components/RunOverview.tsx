import type { DashboardSection, RunSummary } from '../types/runSummary'
import { formatLabel } from '../utils/runSummary'

type RunOverviewProps = {
  runs: RunSummary[]
  section: DashboardSection
  totalRuns: number
}

export function RunOverview({ runs, section, totalRuns }: RunOverviewProps) {
  const groupCounts = runs.reduce<Record<string, number>>((counts, run) => {
    counts[run.family] = (counts[run.family] ?? 0) + 1
    return counts
  }, {})

  return (
    <section className="overview-band">
      <div>
        <span className="metric-value">{runs.length}</span>
        <span className="metric-label">visible {section}</span>
      </div>
      <div>
        <span className="metric-value">{totalRuns}</span>
        <span className="metric-label">section total</span>
      </div>
      {Object.entries(groupCounts).map(([group, count]) => (
        <div key={group}>
          <span className="metric-value">{count}</span>
          <span className="metric-label">{formatLabel(group)}</span>
        </div>
      ))}
    </section>
  )
}
