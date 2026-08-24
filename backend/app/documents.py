

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from app import database
from app.database import init_db, replace_chunks
from app.models import embed_texts, get_embedding_tokenizer


SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".docx"}
HASH_BLOCK_SIZE = 1024 * 1024








@dataclass(frozen=True)
class IndexFreshness:
    status: str
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    error: str | None = None

    @property
    def is_current(self):
        return self.status == "current"

    def change_summary(self):
        parts = []

        if self.added:
            parts.append(f"Added: {', '.join(self.added)}")

        if self.modified:
            parts.append(f"Modified: {', '.join(self.modified)}")

        if self.deleted:
            parts.append(f"Deleted: {', '.join(self.deleted)}")

        return " · ".join(parts)

    def display_status(self):
        if self.status == "current":
            return "current"

        if self.status == "stale":
            return f"stale · {self.change_summary()}"

        if self.status == "untracked":
            return "untracked · reindex required"

        if self.status == "missing":
            return "index not found"

        return f"check failed · {self.error}"


def list_document_paths(docs_dir):
    directory = Path(docs_dir)

    if not directory.is_dir():
        return []

    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
        ),
        key=lambda path: path.name,
    )


def hash_file(file_path):
    digest = hashlib.sha256()

    with file_path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(HASH_BLOCK_SIZE), b""):
            digest.update(block)

    return digest.hexdigest()


def build_source_manifest(docs_dir):
    return [
        {
            "source_name": file_path.name,
            "source_type": file_path.suffix.lower().lstrip("."),
            "file_size": file_path.stat().st_size,
            "sha256": hash_file(file_path),
        }
        for file_path in list_document_paths(docs_dir)
    ]


def get_index_freshness(docs_dir, db_path=None):
    path = Path(db_path) if db_path is not None else database.DB_PATH

    if not path.exists():
        return IndexFreshness(status="missing")

    try:
        current_manifest = build_source_manifest(docs_dir)
        stored_manifest = database.get_source_manifest(db_path=path)
    except Exception as error:
        return IndexFreshness(status="error", error=str(error))

    if not stored_manifest:
        return IndexFreshness(status="untracked")

    current_by_name = {
        source["source_name"]: source
        for source in current_manifest
    }
    stored_by_name = {
        source["source_name"]: source
        for source in stored_manifest
    }

    current_names = set(current_by_name)
    stored_names = set(stored_by_name)
    added = tuple(sorted(current_names - stored_names))
    deleted = tuple(sorted(stored_names - current_names))
    modified = tuple(sorted(
        source_name
        for source_name in current_names & stored_names
        if current_by_name[source_name] != stored_by_name[source_name]
    ))

    if added or modified or deleted:
        return IndexFreshness(
            status="stale",
            added=added,
            modified=modified,
            deleted=deleted,
        )

    return IndexFreshness(status="current")







DOCS_DIR = Path("docs")






CHUNK_SIZE = 128
CHUNK_OVERLAP = 20
SENTENCE_END_PATTERN = re.compile(r"[.!?](?=\s|$)")


class IgnoredPdfObjectFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Ignoring wrong pointing object")


logging.getLogger("pypdf._reader").addFilter(IgnoredPdfObjectFilter())


def read_txt_file(file_path):
    text = file_path.read_text(encoding="utf-8")

    return {
        "source_name": file_path.name,
        "source_type": "txt",
        "page_number": None,
        "text": text
    }


def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    documents = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        clean_text = text.strip()

        if not clean_text:
            continue

        documents.append({
            "source_name": file_path.name,
            "source_type": "pdf",
            "page_number": page_index,
            "text": clean_text
        })

    return documents


def read_docx_file(file_path):
    document = DocxDocument(file_path)
    text = "\n\n".join(
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    )

    return {
        "source_name": file_path.name,
        "source_type": "docx",
        "page_number": None,
        "text": text
    }


def read_documents():
    documents = []

    for file_path in list_document_paths(DOCS_DIR):
        if file_path.suffix.lower() == ".txt":
            documents.append(read_txt_file(file_path))
        elif file_path.suffix.lower() == ".pdf":
            documents.extend(read_pdf_file(file_path))
        elif file_path.suffix.lower() == ".docx":
            documents.append(read_docx_file(file_path))

    return documents


def split_text_into_chunks(text, tokenizer=None):
    active_tokenizer = tokenizer or get_embedding_tokenizer()
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        clean_paragraph = paragraph.strip()

        if clean_paragraph:
            chunks.extend(
                split_long_text(clean_paragraph, tokenizer=active_tokenizer)
            )

    return chunks


def split_long_text(
    text,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    tokenizer=None,
):
    clean_text = " ".join(text.split())

    if not clean_text:
        return []

    active_tokenizer = tokenizer or get_embedding_tokenizer()
    special_token_count = active_tokenizer.num_special_tokens_to_add(pair=False)
    content_limit = chunk_size - special_token_count

    if content_limit < 1:
        raise ValueError("Chunk size must exceed the model's special-token count.")

    if chunk_overlap < 0 or chunk_overlap >= content_limit:
        raise ValueError("Chunk overlap must be smaller than the usable token limit.")

    encoded = active_tokenizer(
        clean_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        verbose=False,
    )
    offsets = [
        tuple(offset)
        for offset in encoded["offset_mapping"]
        if offset[1] > offset[0]
    ]

    if len(offsets) <= content_limit:
        return [clean_text]

    chunks = []
    start_token = 0

    while start_token < len(offsets):
        hard_end_token = min(start_token + content_limit, len(offsets))
        end_token = hard_end_token

        if hard_end_token < len(offsets):
            midpoint_token = min(
                start_token + max(1, content_limit // 3),
                hard_end_token - 1,
            )
            search_start = offsets[midpoint_token][0]
            search_end = offsets[hard_end_token - 1][1]
            sentence_ends = list(
                SENTENCE_END_PATTERN.finditer(clean_text, search_start, search_end)
            )

            if sentence_ends:
                sentence_end = sentence_ends[-1].end()
                aligned_end = hard_end_token
                while (
                    aligned_end > start_token + 1
                    and offsets[aligned_end - 1][1] > sentence_end
                ):
                    aligned_end -= 1
                end_token = aligned_end
            else:
                end_token = align_token_end(
                    clean_text,
                    offsets,
                    start_token,
                    hard_end_token,
                )

        start_char = offsets[start_token][0]
        end_char = offsets[end_token - 1][1]
        chunk = clean_text[start_char:end_char].strip()

        if chunk:
            chunks.append(chunk)

        if end_token >= len(offsets):
            break

        desired_start = max(end_token - chunk_overlap, start_token + 1)
        next_start = align_token_start(
            clean_text,
            offsets,
            desired_start,
            end_token,
        )
        start_token = max(next_start, start_token + 1)

    return chunks


def align_token_end(text, offsets, start_token, hard_end_token):
    minimum_end = start_token + max(1, (hard_end_token - start_token) // 2)
    end_token = hard_end_token

    while end_token > minimum_end:
        end_char = offsets[end_token - 1][1]
        if end_char >= len(text) or text[end_char].isspace():
            break
        end_token -= 1

    return max(end_token, start_token + 1)


def align_token_start(text, offsets, desired_start, previous_end):
    previous_end_char = offsets[previous_end - 1][1]
    if text[previous_end_char - 1] in ".!?":
        return previous_end

    desired_char = offsets[desired_start][0]
    sentence_end = SENTENCE_END_PATTERN.search(
        text,
        desired_char,
        previous_end_char,
    )

    if sentence_end is not None:
        aligned_char = sentence_end.end()
        next_token = desired_start
        while (
            next_token < previous_end
            and offsets[next_token][0] < aligned_char
        ):
            next_token += 1

        if next_token < previous_end:
            return next_token

    next_token = desired_start
    while next_token < previous_end:
        start_char = offsets[next_token][0]
        if start_char == 0 or not text[start_char - 1].isalnum():
            return next_token
        next_token += 1

    return desired_start


def ingest_documents():
    init_db()
    source_manifest = build_source_manifest(DOCS_DIR)
    documents = read_documents()
    tokenizer = get_embedding_tokenizer()
    indexed_chunks = []

    for document in documents:
        chunks = split_text_into_chunks(document["text"], tokenizer=tokenizer)

        if not chunks:
            continue

        embeddings = embed_texts(chunks)

        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
            indexed_chunks.append({
                "source_name": document["source_name"],
                "source_type": document["source_type"],
                "page_number": document["page_number"],
                "chunk_index": chunk_index,
                "chunk_text": chunk,
                "embedding": embedding,
            })

    if not indexed_chunks:
        raise ValueError("No indexable text was found; the existing index was preserved.")

    final_manifest = build_source_manifest(DOCS_DIR)

    if final_manifest != source_manifest:
        raise RuntimeError(
            "Documents changed during indexing; the existing index was preserved."
        )

    replace_chunks(indexed_chunks, source_manifest=source_manifest)
    return len(indexed_chunks)








class DocumentManagementError(ValueError):
    pass


def validate_document(file_path):
    path = Path(file_path).expanduser()

    if not path.exists():
        raise DocumentManagementError(f"File not found: {path}")

    if not path.is_file():
        raise DocumentManagementError(f"Path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise DocumentManagementError("Only TXT, PDF, and DOCX files are supported.")

    try:
        if path.suffix.lower() == ".txt":
            has_text = bool(read_txt_file(path)["text"].strip())
        elif path.suffix.lower() == ".docx":
            has_text = bool(read_docx_file(path)["text"].strip())
        else:
            has_text = bool(read_pdf_file(path))
    except UnicodeDecodeError as error:
        raise DocumentManagementError("TXT file cannot be decoded as UTF-8.") from error
    except Exception as error:
        raise DocumentManagementError(f"File could not be read: {path.name}") from error

    if not has_text:
        raise DocumentManagementError("The file contains no indexable text.")

    return path


def add_document(source_path, docs_dir):
    source = validate_document(source_path)
    directory = Path(docs_dir)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name

    if source.resolve() == destination.resolve():
        raise DocumentManagementError(f"{source.name} is already inside the docs directory.")

    if destination.exists():
        raise DocumentManagementError(
            f"{source.name} already exists in the docs directory; the existing file was not overwritten."
        )

    try:
        with source.open("rb") as source_file, destination.open("xb") as destination_file:
            shutil.copyfileobj(source_file, destination_file)
        validate_document(destination)
    except FileExistsError as error:
        raise DocumentManagementError(
            f"{source.name} already exists in the docs directory; the existing file was not overwritten."
        ) from error
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return destination


def resolve_managed_document(source_name, docs_dir):
    name = str(source_name).strip()
    candidate = Path(name)

    if not name or candidate.name != name or candidate.is_absolute():
        raise DocumentManagementError(
            "Provide only the name of a file inside the docs directory."
        )

    if candidate.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise DocumentManagementError("Only TXT, PDF, and DOCX files can be managed.")

    destination = Path(docs_dir) / name

    if not destination.is_file():
        raise DocumentManagementError(f"{name} was not found in the docs directory.")

    return destination


def remove_document(source_name, docs_dir):
    destination = resolve_managed_document(source_name, docs_dir)
    destination.unlink()
    return destination
