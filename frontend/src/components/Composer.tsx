import { useRef, useState, type KeyboardEvent } from "react";

interface Props {
  activeFilter: string | null;
  isStreaming: boolean;
  onSend: (question: string) => void;
  onCancel: () => void;
}

export function Composer({ activeFilter, isStreaming, onSend, onCancel }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const question = value.trim();
    if (!question || isStreaming) return;
    onSend(question);
    setValue("");
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <div className="composer">
      {activeFilter && (
        <div className="composer__filter-note">kaynak filtresi: {activeFilter}</div>
      )}
      <div className="composer__form">
        <textarea
          ref={textareaRef}
          className="composer__input"
          placeholder="Dokümanlara bir soru sor…"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        {isStreaming ? (
          <button type="button" className="composer__cancel" onClick={onCancel}>
            İptal
          </button>
        ) : (
          <button
            type="button"
            className="composer__send"
            onClick={submit}
            disabled={!value.trim()}
          >
            Sor
          </button>
        )}
      </div>
    </div>
  );
}
