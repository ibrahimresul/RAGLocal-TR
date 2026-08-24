import { useRef, useState } from "react";
import { deleteDocument, exportSession, reindex, uploadDocument } from "../api/client";
import type { SourceSummary } from "../types";

interface Props {
  sources: SourceSummary[];
  onDocumentsChanged: () => void;
  onOpenSettings: () => void;
}

export function TopBar({ sources, onDocumentsChanged, onOpenSettings }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  const handleUpload = async (file: File) => {
    setBusy("upload");
    try {
      await uploadDocument(file);
      onDocumentsChanged();
    } catch (error) {
      window.alert((error as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const handleRemove = async (sourceName: string) => {
    if (!window.confirm(`${sourceName} silinsin mi?`)) return;
    setBusy("remove");
    try {
      await deleteDocument(sourceName);
      onDocumentsChanged();
    } catch (error) {
      window.alert((error as Error).message);
    } finally {
      setBusy(null);
    }
  };

  const handleReindex = async () => {
    setBusy("reindex");
    try {
      const result = await reindex();
      onDocumentsChanged();
      window.alert(`İndeks güncellendi: ${result.chunk_count} parça.`);
    } catch (error) {
      window.alert((error as Error).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="topbar">
      <div style={{ position: "relative" }}>
        <button type="button" className="icon-btn" onClick={() => setMenuOpen((open) => !open)}>
          Dokümanlar ({sources.length})
        </button>
        {menuOpen && (
          <div
            className="drawer"
            style={{
              position: "absolute",
              top: 34,
              left: 0,
              bottom: "auto",
              width: 320,
              maxHeight: 360,
              borderRadius: 10,
              boxShadow: "var(--shadow)",
            }}
          >
            <div className="drawer__body">
              <label className="upload-dropzone">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".txt,.pdf,.docx"
                  style={{ display: "none" }}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) handleUpload(file);
                    event.target.value = "";
                  }}
                />
                {busy === "upload" ? "Yükleniyor…" : "TXT / PDF / DOCX yükle"}
              </label>
              {sources.map((source) => (
                <div className="doc-row" key={source.source_name}>
                  <span title={source.source_name}>{source.source_name}</span>
                  <button
                    type="button"
                    className="doc-row__remove"
                    onClick={() => handleRemove(source.source_name)}
                  >
                    sil
                  </button>
                </div>
              ))}
              {sources.length === 0 && <div className="empty-hint">Henüz doküman yok</div>}
            </div>
          </div>
        )}
      </div>

      <div className="topbar__actions">
        <button type="button" className="icon-btn" onClick={handleReindex} disabled={!!busy}>
          {busy === "reindex" ? "İndeksleniyor…" : "Reindex"}
        </button>
        <button type="button" className="icon-btn" onClick={() => exportSession("markdown")}>
          Export MD
        </button>
        <button type="button" className="icon-btn" onClick={() => exportSession("json")}>
          Export JSON
        </button>
        <button type="button" className="icon-btn" onClick={onOpenSettings}>
          Ayarlar
        </button>
      </div>
    </div>
  );
}
