import { useCallback, useEffect, useState } from "react";
import { getFilter, getHealth, getModel, getSources, getStats, setFilter } from "./api/client";
import { Composer } from "./components/Composer";
import { MessageCard } from "./components/MessageCard";
import { SettingsDrawer } from "./components/SettingsDrawer";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { useChat } from "./hooks/useChat";
import type { HealthResponse, ModelInfo, SourceSummary, Stats } from "./types";

export default function App() {
  const [sources, setSources] = useState<SourceSummary[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [activeFilter, setActiveFilter] = useState<string | null>(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const { messages, ask, cancel } = useChat();

  const refreshIndex = useCallback(() => {
    getSources().then((response) => setSources(response.sources));
    getStats().then(setStats);
    getHealth().then(setHealth);
  }, []);

  useEffect(() => {
    refreshIndex();
    getModel().then(setModel);
    getFilter().then((response) => setActiveFilter(response.source));
  }, [refreshIndex]);

  const handleFilterChange = async (source: string | null) => {
    setActiveFilter(source);
    await setFilter(source);
  };

  const isStreaming = messages.length > 0 && messages[messages.length - 1].status === "streaming";

  return (
    <div className="app-shell">
      <Sidebar
        sources={sources}
        stats={stats}
        health={health}
        model={model}
        activeFilter={activeFilter}
        onFilterChange={handleFilterChange}
      />
      <div className="main">
        <TopBar
          sources={sources}
          onDocumentsChanged={refreshIndex}
          onOpenSettings={() => setSettingsOpen(true)}
        />

        <div className="chat-scroll">
          {messages.length === 0 ? (
            <div className="chat-empty">
              <div className="chat-empty__mark">⌗</div>
              <p>
                Sol taraftaki dokümanlarda anlamsal arama yapan, bulduğu kanıta
                dayanarak cevap veren bir asistan. Kanıtı olmayan soruya cevap
                uydurmaz.
              </p>
            </div>
          ) : (
            <div className="message-thread">
              {messages.map((message) => (
                <MessageCard key={message.id} message={message} />
              ))}
            </div>
          )}
        </div>

        <Composer
          activeFilter={activeFilter}
          isStreaming={isStreaming}
          onSend={(question) => ask(question, activeFilter)}
          onCancel={cancel}
        />
      </div>

      {settingsOpen && (
        <SettingsDrawer
          health={health}
          model={model}
          onClose={() => setSettingsOpen(false)}
          onRepeat={(question, source) => ask(question, source)}
        />
      )}
    </div>
  );
}
