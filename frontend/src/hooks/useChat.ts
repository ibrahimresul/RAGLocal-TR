import { useCallback, useRef, useState } from "react";
import { askStream } from "../api/client";
import type { AskDone, ChatMessage } from "../types";

let nextId = 1;

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const abortRef = useRef<AbortController | null>(null);

  const patch = useCallback((id: string, changes: Partial<ChatMessage>) => {
    setMessages((prev) =>
      prev.map((message) => (message.id === id ? { ...message, ...changes } : message)),
    );
  }, []);

  const ask = useCallback(
    async (question: string, source: string | null) => {
      const id = String(nextId++);
      const message: ChatMessage = {
        id,
        question,
        askedAt: new Date().toISOString(),
        status: "streaming",
        stage: null,
      };
      setMessages((prev) => [...prev, message]);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        for await (const event of askStream(question, source, controller.signal)) {
          switch (event.event) {
            case "index_warning":
              patch(id, {
                indexWarning: event.data as ChatMessage["indexWarning"],
              });
              break;
            case "query_rewrite": {
              const data = event.data as { added_terms: string[] };
              patch(id, { addedTerms: data.added_terms });
              break;
            }
            case "stage": {
              const data = event.data as { stage: string; state: string };
              patch(id, {
                stage:
                  data.state === "start"
                    ? (data.stage as ChatMessage["stage"])
                    : null,
              });
              break;
            }
            case "token": {
              const data = event.data as { text: string };
              patch(id, { liveText: data.text });
              break;
            }
            case "done":
              patch(id, {
                status: "done",
                stage: null,
                result: event.data as AskDone,
              });
              break;
            case "error": {
              const data = event.data as { message: string };
              patch(id, { status: "error", stage: null, errorMessage: data.message });
              break;
            }
          }
        }
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          patch(id, {
            status: "error",
            stage: null,
            errorMessage: (error as Error).message,
          });
        }
      } finally {
        abortRef.current = null;
      }
    },
    [patch],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  return { messages, ask, cancel };
}
