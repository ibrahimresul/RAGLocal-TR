import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from app import database
from app.settings import RERANKER_MODEL, USE_RERANKER
from app.documents import DOCS_DIR, SUPPORTED_DOCUMENT_EXTENSIONS, get_index_freshness
from app.models import MODEL_ALIAS


EXPECTED_EMBEDDING_DIMENSION = 384


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str
    message: str
    solution: str | None = None


def check_documents(docs_dir=None):
    directory = Path(docs_dir) if docs_dir is not None else DOCS_DIR

    if not directory.is_dir():
        return HealthCheck(
            name="Documents",
            status="error",
            message=f"{directory} directory was not found.",
            solution=f"{directory} Create the directory and add a TXT, PDF, or DOCX file.",
        )

    documents = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS
    ]

    if not documents:
        return HealthCheck(
            name="Documents",
            status="warning",
            message="No indexable TXT, PDF, or DOCX files were found.",
            solution=f"{directory} Add at least one TXT, PDF, or DOCX file to the directory.",
        )

    return HealthCheck(
        name="Documents",
        status="ok",
        message=f"{len(documents)} supported files are ready.",
    )


def check_index(db_path=None):
    path = Path(db_path) if db_path is not None else database.DB_PATH

    if not path.exists():
        missing_database = HealthCheck(
            name="Database",
            status="warning",
            message=f"{path} has not been created yet.",
            solution="Run /reindex.",
        )
        missing_index = HealthCheck(
            name="Embedding index",
            status="warning",
            message="No index is available to check.",
            solution="Run /reindex.",
        )
        return [missing_database, missing_index]

    try:
        chunks = database.get_all_chunks()
    except Exception as error:
        return [
            HealthCheck(
                name="Database",
                status="error",
                message=f"Index could not be read: {error}",
                solution="Run /reindex first; if the issue persists, inspect data/rag.db.",
            ),
            HealthCheck(
                name="Embedding index",
                status="warning",
                message="The database error prevented this check.",
                solution="Resolve the database error and run /doctor again.",
            ),
        ]

    if not chunks:
        return [
            HealthCheck(
                name="Database",
                status="ok",
                message=f"{path} is readable.",
            ),
            HealthCheck(
                name="Embedding index",
                status="warning",
                message="The index is empty.",
                solution="Run /reindex.",
            ),
        ]

    source_count = len({chunk["source_name"] for chunk in chunks})
    database_check = HealthCheck(
        name="Database",
        status="ok",
        message=f"{source_count} sources and {len(chunks)} chunk is readable.",
    )

    for chunk in chunks:
        embedding = chunk.get("embedding")

        if not isinstance(embedding, list) or len(embedding) != EXPECTED_EMBEDDING_DIMENSION:
            return [
                database_check,
                HealthCheck(
                    name="Embedding index",
                    status="error",
                    message=f"chunk_id={chunk['id']} has an invalid embedding dimension.",
                    solution="Run /reindex.",
                ),
            ]

        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
            return [
                database_check,
                HealthCheck(
                    name="Embedding index",
                    status="error",
                    message=f"chunk_id={chunk['id']} contains an invalid embedding value.",
                    solution="Run /reindex.",
                ),
            ]

    return [
        database_check,
        HealthCheck(
            name="Embedding index",
            status="ok",
            message=(
                f"{len(chunks)} healthy embeddings "
                f"({EXPECTED_EMBEDDING_DIMENSION} dimensions)."
            ),
        ),
    ]


def check_index_freshness(docs_dir=None, db_path=None):
    directory = Path(docs_dir) if docs_dir is not None else DOCS_DIR
    path = Path(db_path) if db_path is not None else database.DB_PATH
    freshness = get_index_freshness(directory, path)

    if freshness.status == "current":
        return HealthCheck(
            name="Index freshness",
            status="ok",
            message="Documents match the index.",
        )

    if freshness.status == "stale":
        return HealthCheck(
            name="Index freshness",
            status="warning",
            message=f"The index is stale. {freshness.change_summary()}",
            solution="Run /reindex or local-rag reindex.",
        )

    if freshness.status == "untracked":
        return HealthCheck(
            name="Index freshness",
            status="warning",
            message="The index source manifest is unavailable.",
            solution="Run /reindex or local-rag reindex.",
        )

    if freshness.status == "missing":
        return HealthCheck(
            name="Index freshness",
            status="warning",
            message="No index is available to check.",
            solution="Run /reindex or local-rag reindex.",
        )

    return HealthCheck(
        name="Index freshness",
        status="error",
        message=f"Document changes could not be checked: {freshness.error}",
        solution="Check file permissions and run /doctor again.",
    )


def check_foundry(
    foundry_home=None,
    executable_finder=shutil.which,
    model_alias=None,
):
    active_model_alias = model_alias or MODEL_ALIAS

    if executable_finder("foundry") is None:
        return [
            HealthCheck(
                name="Foundry Local",
                status="error",
                message="The foundry command was not found.",
                solution="Check the Foundry Local installation.",
            ),
            HealthCheck(
                name="LLM model",
                status="warning",
                message=f"{active_model_alias} cache status could not be checked.",
                solution="Complete the Foundry Local installation first.",
            ),
        ]

    home = Path(foundry_home) if foundry_home is not None else Path.home() / ".foundry"
    config_path = home / "foundry.config.json"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        cache_directory = config["serviceSettings"]["cacheDirectoryPath"]
        cache_path = Path(cache_directory).expanduser()
    except Exception as error:
        return [
            HealthCheck(
                name="Foundry Local",
                status="error",
                message=f"Cache configuration could not be read: {error}",
                solution="Check the Foundry Local installation.",
            ),
            HealthCheck(
                name="LLM model",
                status="warning",
                message=f"{active_model_alias} cache status could not be checked.",
                solution="Resolve the Foundry Local issue and run /doctor again.",
            ),
        ]

    if not cache_path.is_dir():
        return [
            HealthCheck(
                name="Foundry Local",
                status="error",
                message=f"Model cache directory was not found: {cache_path}",
                solution="Check the Foundry Local installation.",
            ),
            HealthCheck(
                name="LLM model",
                status="warning",
                message=f"{active_model_alias} cache status could not be checked.",
                solution=(
                    f"Run foundry model download {active_model_alias} if needed."
                ),
            ),
        ]

    foundry_check = HealthCheck(
        name="Foundry Local",
        status="ok",
        message="The CLI and model cache directory are ready.",
    )
    model_name = active_model_alias.lower()
    model_cached = any(
        model_name in str(metadata_path.parent).lower()
        and any(
            model_file.is_file() and model_file.stat().st_size > 0
            for model_file in metadata_path.parent.glob("model.onnx*")
        )
        for metadata_path in cache_path.rglob("inference_model.json")
    )

    if not model_cached:
        return [
            foundry_check,
            HealthCheck(
                name="LLM model",
                status="error",
                message=f"{active_model_alias} was not found in the local cache.",
                solution=f"foundry model download {active_model_alias} if needed.",
            ),
        ]

    return [
        foundry_check,
        HealthCheck(
            name="LLM model",
            status="ok",
            message=f"{active_model_alias} is available in the cache.",
        ),
    ]


def check_reranker(cache_dir=None):





    if not USE_RERANKER:
        return HealthCheck(
            name="Reranking",
            status="ok",
            message="Disabled; ranking uses the hybrid search result.",
        )

    root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = root / ("models--" + RERANKER_MODEL.replace("/", "--"))

    if not model_dir.is_dir():
        return HealthCheck(
            name="Reranking",
            status="warning",
            message=f"{RERANKER_MODEL} is not downloaded; ranking falls back to the hybrid result.",
            solution="Submit a question while online to download the model once.",
        )

    return HealthCheck(
        name="Reranking",
        status="ok",
        message=f"{RERANKER_MODEL} is available in the cache.",
    )


def run_health_checks(
    docs_dir=None,
    db_path=None,
    foundry_home=None,
    executable_finder=shutil.which,
    reranker_cache_dir=None,
):
    checks = [check_documents(docs_dir=docs_dir)]
    checks.append(check_index_freshness(docs_dir=docs_dir, db_path=db_path))
    checks.extend(check_index(db_path=db_path))
    checks.extend(
        check_foundry(
            foundry_home=foundry_home,
            executable_finder=executable_finder,
        )
    )
    checks.append(check_reranker(cache_dir=reranker_cache_dir))
    return checks
