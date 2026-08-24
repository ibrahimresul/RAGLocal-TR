

from app.retrieval import (
    extract_question_terms,
    terms_match,
    tokenize,
)










FOLLOW_UP_MARKERS = frozenset({
    "peki", "ya", "yani", "o", "onu", "onun", "ona", "onda", "ondan",
    "bu", "bunu", "bunun", "buna", "bunda", "bundan", "bunlar", "bunları",
    "şu", "şunu", "şunun", "şuna", "şunlar",
    "aynı", "aynısı", "orada", "oradaki", "burada", "buradaki",
    "hepsi", "hangisi",
})





FOLLOW_UP_MAX_TERMS_WITH_MARKER = 1




MAX_CARRIED_TERMS = 3


def content_terms(question):

    return [
        term
        for term in extract_question_terms(question)
        if term not in FOLLOW_UP_MARKERS
    ]


def has_follow_up_marker(question):
    return any(token in FOLLOW_UP_MARKERS for token in tokenize(question))


def is_follow_up(question):
    terms = content_terms(question)

    if not terms:
        return True

    return (
        has_follow_up_marker(question)
        and len(terms) <= FOLLOW_UP_MAX_TERMS_WITH_MARKER
    )


def carried_terms(question, topic, max_terms=MAX_CARRIED_TERMS):

    existing = content_terms(question)
    carried = []

    for term in content_terms(topic):
        if any(terms_match(term, present) for present in existing):
            continue

        if term in carried:
            continue

        carried.append(term)

        if len(carried) >= max_terms:
            break

    return tuple(carried)


class FollowUpContext:


    def __init__(self, max_terms=MAX_CARRIED_TERMS):
        self.max_terms = max_terms
        self.topic = None

    def clear(self):
        self.topic = None

    def remember(self, question):
        clean = question.strip()

        if clean:
            self.topic = clean

    def resolve(self, question):

        clean = question.strip()

        if not clean or not self.topic or not is_follow_up(clean):
            return clean, ()

        carried = carried_terms(clean, self.topic, max_terms=self.max_terms)

        if not carried:
            return clean, ()

        return f"{clean} {' '.join(carried)}", carried

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class SessionExportError(ValueError):
    pass


@dataclass(frozen=True)
class SessionEntry:
    id: int
    created_at: str
    question: str
    answer: str
    mode: str
    best_score: float
    source_filter: str | None
    sources: tuple[dict, ...]
    timings: dict

    @classmethod
    def from_result(cls, entry_id, result, created_at):
        sources = tuple(
            {
                "id": source.id,
                "source_name": source.source_name,
                "source_type": source.source_type,
                "page_number": source.page_number,
                "chunk_index": source.chunk_index,
                "score": source.score,
                "context_role": source.context_role,
            }
            for source in result.sources
        )
        return cls(
            id=entry_id,
            created_at=created_at.isoformat(),
            question=result.question,
            answer=result.answer,
            mode=result.mode,
            best_score=result.best_score,
            source_filter=result.source_filter,
            sources=sources,
            timings=asdict(result.timings),
        )


class SessionHistory:
    def __init__(self, now_factory=None):
        self.now_factory = now_factory or (lambda: datetime.now(timezone.utc))
        self.entries = []

    def add_result(self, result):
        entry = SessionEntry.from_result(
            len(self.entries) + 1,
            result,
            self.now_factory(),
        )
        self.entries.append(entry)
        return entry

    def clear(self):
        self.entries.clear()

    def ids(self):
        return [entry.id for entry in self.entries]

    def get(self, entry_id=None):
        if not self.entries:
            return None
        if entry_id is None:
            return self.entries[-1]
        return next((entry for entry in self.entries if entry.id == entry_id), None)

    def export(self, export_format, export_dir, output_path=None):
        if not self.entries:
            raise SessionExportError("There are no session entries to export.")

        normalized_format = export_format.casefold()
        if normalized_format == "md":
            normalized_format = "markdown"
        if normalized_format not in {"markdown", "json"}:
            raise SessionExportError("Format must be markdown or json.")

        extension = ".md" if normalized_format == "markdown" else ".json"
        exported_at = self.now_factory()
        destination = self._resolve_destination(
            export_dir,
            output_path,
            extension,
            exported_at,
        )
        content = (
            self._to_markdown(exported_at)
            if normalized_format == "markdown"
            else self._to_json(exported_at)
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("x", encoding="utf-8") as output_file:
                output_file.write(content)
        except FileExistsError as error:
            raise SessionExportError(
                f"File already exists and was not overwritten: {destination}"
            ) from error
        except OSError as error:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                pass
            raise SessionExportError(
                f"Session file could not be written: {destination}"
            ) from error

        return destination

    @staticmethod
    def _resolve_destination(export_dir, output_path, extension, exported_at):
        if output_path is None:
            filename = f"session-{exported_at:%Y%m%d-%H%M%S}{extension}"
            return Path(export_dir) / filename

        destination = Path(output_path).expanduser()
        if not destination.is_absolute():
            destination = Path.cwd() / destination
        if destination.suffix:
            if destination.suffix.casefold() != extension:
                raise SessionExportError(
                    f"File extension must be {extension}."
                )
        else:
            destination = destination.with_suffix(extension)
        return destination.resolve()

    def _to_json(self, exported_at):
        payload = {
            "exported_at": exported_at.isoformat(),
            "entries": [asdict(entry) for entry in self.entries],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    def _to_markdown(self, exported_at):
        lines = [
            "# Local RAG Session",
            "",
            f"Exported at: {exported_at.isoformat()}",
        ]

        for entry in self.entries:
            lines.extend([
                "",
                f"## {entry.id}. Question",
                "",
                entry.question,
                "",
                f"- Mode: `{entry.mode}`",
                f"- Score: `{entry.best_score:.4f}`",
                f"- Source filter: `{entry.source_filter or 'none'}`",
                "",
                "### Answer",
                "",
                entry.answer,
            ])

            if entry.sources:
                lines.extend(["", "### Sources", ""])
                for source in entry.sources:
                    metadata = [source["source_name"]]
                    if source.get("page_number") is not None:
                        metadata.append(f"page {source['page_number']}")
                    if source.get("chunk_index") is not None:
                        metadata.append(f"chunk {source['chunk_index']}")
                    metadata.append(f"ID {source['id']}")
                    metadata.append(f"score {source['score']:.4f}")
                    metadata.append(
                        "neighbor context"
                        if source.get("context_role") == "neighbor"
                        else "match"
                    )
                    lines.append(f"- {' · '.join(metadata)}")

        return "\n".join(lines) + "\n"
