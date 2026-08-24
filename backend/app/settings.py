



SIMILARITY_THRESHOLD = 0.20


CONTEXT_SCORE_THRESHOLD = 0.35
CONTEXT_RELATIVE_SCORE_MARGIN = 0.20
TOP_K = 3
NEIGHBOR_CHUNK_RADIUS = 1
MAX_CONTEXT_CHUNKS = 5

NO_EVIDENCE_ANSWER = "The requested information is not available in the provided documents."





TERM_EVIDENCE_THRESHOLD = 0.12
TERM_EVIDENCE_MIN_PREFIX = 5
TERM_EVIDENCE_MIN_SHORT_ROOT = 3
TERM_EVIDENCE_MIN_TERM_LENGTH = 3




USE_HYBRID_SEARCH = True



CONTEXT_TERM_EVIDENCE_MIN = 0.30


BM25_K1 = 1.5
BM25_B = 0.75



RRF_K = 2




RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_MAX_LENGTH = 512
RERANK_CANDIDATE_POOL = 15
USE_RERANKER = False



GROUNDEDNESS_THRESHOLD = 0.50
GROUNDEDNESS_SENTENCE_SUPPORT = 0.60



GROUNDEDNESS_MIN_SENTENCE_TERMS = 2

USE_EXTRACTIVE_FALLBACK = True



EXTRACTIVE_SCORE_THRESHOLD = 0.50




EXTRACTIVE_TERM_EVIDENCE_MIN = 0.675
MAX_EXTRACTIVE_CHARS = 500

MIN_GENERATIVE_ANSWER_CHARS = 30

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ENV_VAR = "LOCAL_RAG_HOME"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    docs_dir: Path
    db_path: Path
    history_path: Path
    benchmark_report_path: Path
    session_export_dir: Path


def resolve_project_root(explicit_path=None, environ=None):
    environment = os.environ if environ is None else environ
    configured_path = explicit_path

    if configured_path is None:
        configured_path = environment.get(PROJECT_ENV_VAR)

    if configured_path is None or not str(configured_path).strip():
        return DEFAULT_PROJECT_ROOT.resolve()

    candidate = Path(str(configured_path).strip()).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    candidate = candidate.resolve()

    if not candidate.exists():
        raise ProjectConfigurationError(
            f"Project directory not found: {candidate}"
        )

    if not candidate.is_dir():
        raise ProjectConfigurationError(
            f"Project path must be a directory: {candidate}"
        )

    return candidate


def get_project_paths(explicit_path=None, environ=None):
    root = resolve_project_root(explicit_path, environ=environ)
    return ProjectPaths(
        root=root,
        docs_dir=root / "docs",
        db_path=root / "data" / "rag.db",
        history_path=root / "data" / "cli_history",
        benchmark_report_path=root / "data" / "model_benchmark.json",
        session_export_dir=root / "data" / "exports",
    )
