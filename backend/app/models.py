from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedding_model = None


def get_local_model_path():
    try:
        return snapshot_download(repo_id=MODEL_NAME, local_files_only=True)
    except LocalEntryNotFoundError:
        return None


def is_embedding_model_loaded():
    return _embedding_model is not None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        model_source = get_local_model_path() or MODEL_NAME
        _embedding_model = SentenceTransformer(model_source)

    return _embedding_model


def get_embedding_tokenizer():
    return get_embedding_model().tokenizer


def embed_text(text):
    model = get_embedding_model()

    return model.encode(text).tolist()


def embed_texts(texts):
    model = get_embedding_model()

    return model.encode(texts).tolist()



from app.settings import (
    RERANK_MAX_LENGTH,
    RERANKER_MODEL,
)


class RerankerUnavailableError(RuntimeError):
    pass


_cross_encoder = None
_load_error = None


def load_cross_encoder(model_name=RERANKER_MODEL, max_length=RERANK_MAX_LENGTH):

    global _cross_encoder, _load_error

    if _cross_encoder is not None:
        return _cross_encoder

    if _load_error is not None:
        raise RerankerUnavailableError(_load_error)

    try:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder(model_name, max_length=max_length)
    except Exception as error:
        _load_error = f"Reranking model could not be loaded: {error}"
        raise RerankerUnavailableError(_load_error) from error

    return _cross_encoder


def reset_cross_encoder():

    global _cross_encoder, _load_error

    _cross_encoder = None
    _load_error = None


def score_pairs(question, texts, model=None):
    encoder = model or load_cross_encoder()
    pairs = [(question, text) for text in texts]

    return [float(score) for score in encoder.predict(pairs)]


def rerank(question, results, score_func=None):

    if not results:
        return []

    scorer = score_func or score_pairs
    scores = scorer(question, [result["chunk_text"] for result in results])

    if len(scores) != len(results):
        raise ValueError("Reranking score count does not match candidate count.")

    scored = [
        (dict(result, rerank_score=score), order)
        for order, (result, score) in enumerate(zip(results, scores))
    ]
    scored.sort(key=lambda item: (-item[0]["rerank_score"], item[1]))

    return [result for result, _order in scored]

import os
import re
import subprocess
import time
from collections import Counter
from contextlib import contextmanager

import foundry_local.api as foundry_api
import openai
from foundry_local import FoundryLocalManager

from app.settings import MIN_GENERATIVE_ANSWER_CHARS, NO_EVIDENCE_ANSWER


DEFAULT_MODEL_ALIAS = "phi-4-mini"
MODEL_ALIAS_ENV_VAR = "LOCAL_RAG_MODEL"


def get_model_alias(environ=None):
    environment = os.environ if environ is None else environ
    configured_alias = environment.get(MODEL_ALIAS_ENV_VAR, "").strip()
    return configured_alias or DEFAULT_MODEL_ALIAS


def get_model_alias_source(environ=None):
    environment = os.environ if environ is None else environ
    return (
        MODEL_ALIAS_ENV_VAR
        if environment.get(MODEL_ALIAS_ENV_VAR, "").strip()
        else "default"
    )


MODEL_ALIAS = get_model_alias()




MIN_REPETITION_WORDS = 12
REPEATED_TRIGRAM_LIMIT = 3

FOUNDRY_START_ATTEMPTS = 100
FOUNDRY_START_INTERVAL_SECONDS = 0.1
FOUNDRY_STATUS_TIMEOUT_SECONDS = 15
FOUNDRY_HTTP_TIMEOUT_SECONDS = 120

ANSWER_STOP_MARKERS = [
    "Kaynak:",
    "kaynak:",
    "KAYNAK:",
    "Source:",
    "source:"
]

ANSWER_PREFIXES = [
    "Cevap:",
    "cevap:",
    "Answer:",
    "answer:"
]

CITATION_BODY_PATTERN = (
    r"(?:Parça|Parca)\s+\d+"
    r"(?:(?:\s*[-–,]\s*|\s+ve\s+)(?:(?:Parça|Parca)\s+)?\d+)*"
)
CITATION_PATTERN = rf"(?:\[{CITATION_BODY_PATTERN}\]|\({CITATION_BODY_PATTERN}\))"
TRAILING_CITATION_PATTERN = (
    rf"(?:^|\s+){CITATION_BODY_PATTERN}\s*[.!?]?\s*$"
)


def remove_answer_prefix(text):
    cleaned = text.strip()

    prefix_removed = True

    while prefix_removed:
        prefix_removed = False

        for prefix in ANSWER_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                prefix_removed = True

    return cleaned


def remove_citations(text):
    cleaned = re.sub(CITATION_PATTERN, "", text, flags=re.IGNORECASE)
    cleaned = re.sub(
        TRAILING_CITATION_PATTERN,
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_answer(answer):
    original_answer = answer.strip()
    filtered_lines = []

    for line in original_answer.splitlines():
        stripped = remove_answer_prefix(line)

        if not stripped:
            continue

        if stripped.lower().startswith(("kaynak:", "source:")):
            continue

        filtered_lines.append(stripped)

    cleaned = "\n".join(filtered_lines).strip()
    cleaned = remove_citations(cleaned)

    for marker in ANSWER_STOP_MARKERS:
        marker_index = cleaned.find(marker)

        if marker_index > 0:
            cleaned = cleaned[:marker_index].strip()

    return cleaned or original_answer


def clean_streaming_preview(answer):
    cleaned = answer.strip()
    if not cleaned:
        return ""

    if any(
        prefix.casefold().startswith(cleaned.casefold())
        for prefix in ANSWER_PREFIXES
    ):
        return ""

    filtered_lines = []
    for line in cleaned.splitlines():
        stripped = remove_answer_prefix(line)
        if stripped.lower().startswith(("kaynak:", "source:")):
            break
        if stripped:
            filtered_lines.append(stripped)

    preview = remove_citations("\n".join(filtered_lines))
    for marker in ANSWER_STOP_MARKERS:
        marker_index = preview.find(marker)
        if marker_index >= 0:
            preview = preview[:marker_index].strip()

    return preview


def has_repeating_trigram(answer, minimum=REPEATED_TRIGRAM_LIMIT):

    words = re.findall(r"\b\w+\b", answer.casefold(), flags=re.UNICODE)

    if len(words) < MIN_REPETITION_WORDS:
        return False

    trigram_counts = Counter(zip(words, words[1:], words[2:]))

    return bool(trigram_counts and trigram_counts.most_common(1)[0][1] >= minimum)


def has_excessive_repetition(answer):
    words = re.findall(r"\b\w+\b", answer.casefold(), flags=re.UNICODE)

    if len(words) < MIN_REPETITION_WORDS:
        return False

    word_counts = Counter(words)
    most_common_count = word_counts.most_common(1)[0][1]

    if most_common_count >= 8 and most_common_count / len(words) > 0.25:
        return True

    return has_repeating_trigram(answer)


def get_answer_validation_error(answer):
    if not answer:
        return "empty"

    cleaned = answer.strip()











    if len(cleaned) < MIN_GENERATIVE_ANSWER_CHARS:
        return "too_short"

    if cleaned.lower().startswith(("kaynak:", "source:")):
        return "source_label"

    without_citations = remove_citations(cleaned)
    without_prefix = remove_answer_prefix(without_citations)

    if has_excessive_repetition(without_prefix):
        return "repetition"

    if len(without_prefix) < MIN_GENERATIVE_ANSWER_CHARS:
        return "too_short"

    return None


def is_valid_answer(answer):
    return get_answer_validation_error(answer) is None


def get_foundry_service_uri():
    try:
        result = subprocess.run(
            ["foundry", "service", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=FOUNDRY_STATUS_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Foundry Local service status timed out."
        ) from error

    match = re.search(
        r"http://(?:[a-zA-Z0-9.-]+|\d{1,3}(?:\.\d{1,3}){3}):\d+",
        result.stdout,
    )
    return match.group(0) if match else None


@contextmanager
def safe_foundry_service_lookup():
    original_lookup = foundry_api.get_service_uri
    foundry_api.get_service_uri = get_foundry_service_uri

    try:
        yield
    finally:
        foundry_api.get_service_uri = original_lookup


def create_foundry_manager(show_startup_output=False):
    with safe_foundry_service_lookup():
        manager = FoundryLocalManager(
            bootstrap=False,
            timeout=FOUNDRY_HTTP_TIMEOUT_SECONDS,
        )

        if manager.is_service_running():
            return manager

        process_options = {}

        if not show_startup_output:
            process_options = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }

        with subprocess.Popen(
            ["foundry", "service", "start"],
            **process_options,
        ):
            for _ in range(FOUNDRY_START_ATTEMPTS):
                if manager.is_service_running():
                    return manager

                time.sleep(FOUNDRY_START_INTERVAL_SECONDS)

    raise RuntimeError("Foundry Local service startup timed out.")


class LocalLLM:
    def __init__(self, show_startup_output=False, model_alias=None):
        self.model_alias = model_alias or MODEL_ALIAS
        self.manager = create_foundry_manager(show_startup_output)

        self.model_info = self.manager.load_model(self.model_alias)

        self.client = openai.OpenAI(
            base_url=self.manager.endpoint,
            api_key=self.manager.api_key,
            timeout=FOUNDRY_HTTP_TIMEOUT_SECONDS,
        )

    def generate_answer(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_info.id,
            messages=messages,
            temperature=0.1,
            max_tokens=220
        )

        raw_answer = response.choices[0].message.content

        return clean_answer(raw_answer)

    def generate_answer_stream(self, messages, on_update=None):
        response = self.client.chat.completions.create(
            model=self.model_info.id,
            messages=messages,
            temperature=0.1,
            max_tokens=220,
            stream=True,
        )
        raw_parts = []
        last_preview = ""

        try:
            for chunk in response:
                if not chunk.choices:
                    continue

                content = chunk.choices[0].delta.content or ""
                if not content:
                    continue

                raw_parts.append(content)
                preview = clean_streaming_preview("".join(raw_parts))
                if on_update is not None and preview and preview != last_preview:
                    on_update(preview)
                    last_preview = preview





                if has_repeating_trigram(preview):
                    break
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

        return clean_answer("".join(raw_parts))
