import type { HealthResponse, ModelInfo, SourceSummary, Stats } from "../types";

interface Props {
  sources: SourceSummary[];
  stats: Stats | null;
  health: HealthResponse | null;
  model: ModelInfo | null;
  activeFilter: string | null;
  onFilterChange: (source: string | null) => void;
}

export function Sidebar({ sources, stats, health, model, activeFilter, onFilterChange }: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__brand-mark">RAG</span>
        <div>
          <div className="sidebar__title">Yerel Kanıt Asistanı</div>
          <div className="sidebar__subtitle">
            {model ? model.alias : "model yükleniyor…"}
          </div>
        </div>
      </div>

      <div className="sidebar__section">
        <div className="sidebar__label">Durum</div>
        <div className="status-row">
          <span className="status-dot status-dot--ok" />
          <span>Sistem hazır</span>
        </div>
        <div className="status-row">
          <span className="mono" style={{ color: "var(--text-dim)" }}>
            {stats ? `${stats.total_chunks} parça · ${stats.source_count} kaynak` : "…"}
          </span>
        </div>
      </div>

      <div className="sidebar__section">
        <div className="sidebar__label">Kaynaklar</div>
        <div className="source-list">
          <button
            type="button"
            className={`source-item ${activeFilter === null ? "source-item--active" : ""}`}
            onClick={() => onFilterChange(null)}
          >
            <span className="source-item__name">Tüm kaynaklar</span>
            <span className="source-item__count">{sources.length}</span>
          </button>
          {sources.map((source) => (
            <button
              type="button"
              key={source.source_name}
              className={`source-item ${
                activeFilter === source.source_name ? "source-item--active" : ""
              }`}
              onClick={() =>
                onFilterChange(activeFilter === source.source_name ? null : source.source_name)
              }
              title={source.source_name}
            >
              <span className="source-item__name">{source.source_name}</span>
              <span className="source-item__count">{source.chunk_count}</span>
            </button>
          ))}
          {sources.length === 0 && <div className="empty-hint">Henüz doküman yok</div>}
        </div>
      </div>
    </aside>
  );
}
