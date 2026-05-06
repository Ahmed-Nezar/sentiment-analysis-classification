type EmptyStateProps = {
  title: string
  detail?: string
}

export function EmptyState({ title, detail }: EmptyStateProps) {
  return (
    <section className="empty-state">
      <h2>{title}</h2>
      {detail && <p>{detail}</p>}
    </section>
  )
}
