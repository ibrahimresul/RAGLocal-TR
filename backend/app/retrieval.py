import math
import re

import numpy as np
from sklearn.preprocessing import normalize

from app.settings import (
    BM25_B,
    BM25_K1,
    RERANK_CANDIDATE_POOL,
    RRF_K,
    TERM_EVIDENCE_MIN_PREFIX,
    TERM_EVIDENCE_MIN_SHORT_ROOT,
    TERM_EVIDENCE_MIN_TERM_LENGTH,
    TERM_EVIDENCE_THRESHOLD,
    USE_HYBRID_SEARCH,
    USE_RERANKER,
)
from app.database import get_all_chunks
from app.models import embed_texts
from app.stopwords import QUESTION_STOPWORDS






















TURKISH_LOWER_MAP = str.maketrans({"İ": "i", "I": "ı"})
COMBINING_DOT_ABOVE = "̇"

WORD_PATTERN = re.compile(r"[0-9a-zçğıöşü\-]+")



CONSONANT_MUTATIONS = {"p": "b", "ç": "c", "t": "d", "k": "ğ"}








def normalize_text(text):
    lowered = str(text).translate(TURKISH_LOWER_MAP).casefold()
    lowered = lowered.replace(COMBINING_DOT_ABOVE, "")

    return " ".join(lowered.split())


def tokenize(text):
    return WORD_PATTERN.findall(normalize_text(text))


def extract_question_terms(question, min_length=TERM_EVIDENCE_MIN_TERM_LENGTH):

    seen = []

    for token in tokenize(question):
        if len(token) < min_length:
            continue

        if token in QUESTION_STOPWORDS:
            continue

        if token not in seen:
            seen.append(token)

    return seen


def common_prefix_length(first, second):

    length = 0

    for left, right in zip(first, second):
        equivalent = (
            left == right
            or CONSONANT_MUTATIONS.get(left) == right
            or CONSONANT_MUTATIONS.get(right) == left
        )

        if not equivalent:
            break

        length += 1

    return length


def terms_match(
    term,
    context_term,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
    min_short=TERM_EVIDENCE_MIN_SHORT_ROOT,
):

    if term == context_term:
        return True

    length = common_prefix_length(term, context_term)

    if length >= min_prefix:
        return True

    shorter = min(len(term), len(context_term))

    return shorter >= min_short and length == shorter


def build_context_terms(chunks):
    context_text = "\n".join(chunk["chunk_text"] for chunk in chunks)

    return set(tokenize(context_text))


def term_coverage(
    question,
    chunks,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
    weights=None,
):

    terms = extract_question_terms(question)

    if not terms:
        return None

    def weight_of(term):
        if weights is None:
            return 1.0

        return weights.get(term, 1.0)

    total_weight = sum(weight_of(term) for term in terms)

    if total_weight <= 0:
        return None

    context_terms = build_context_terms(chunks)
    matched_weight = sum(
        weight_of(term)
        for term in terms
        if any(
            terms_match(term, context_term, min_prefix=min_prefix)
            for context_term in context_terms
        )
    )

    return matched_weight / total_weight


def has_term_evidence(
    question,
    chunks,
    threshold=TERM_EVIDENCE_THRESHOLD,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
    weights=None,
):
    if not chunks:
        return False

    coverage = term_coverage(
        question,
        chunks,
        min_prefix=min_prefix,
        weights=weights,
    )

    if coverage is None:
        return True

    return coverage >= threshold

















def build_document_terms(texts):
    return [tokenize(text) for text in texts]


def term_frequency(term, document_terms, min_prefix):

    return sum(
        1
        for document_term in document_terms
        if terms_match(term, document_term, min_prefix=min_prefix)
    )


def inverse_document_frequency(document_count, matching_document_count):

    numerator = document_count - matching_document_count + 0.5
    denominator = matching_document_count + 0.5

    return math.log(1 + numerator / denominator)


def corpus_term_weights(
    question,
    document_terms,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
):

    terms = extract_question_terms(question)
    document_count = len(document_terms)

    if not document_count:
        return {term: 1.0 for term in terms}

    weights = {}

    for term in terms:
        matching_document_count = sum(
            1
            for terms_in_document in document_terms
            if term_frequency(term, terms_in_document, min_prefix)
        )
        weights[term] = inverse_document_frequency(
            document_count,
            matching_document_count,
        )

    return weights


def bm25_scores(
    question,
    document_terms,
    k1=BM25_K1,
    b=BM25_B,
    min_prefix=TERM_EVIDENCE_MIN_PREFIX,
):

    document_count = len(document_terms)

    if not document_count:
        return []

    query_terms = extract_question_terms(question)
    scores = [0.0] * document_count

    if not query_terms:
        return scores

    lengths = [len(terms) for terms in document_terms]
    average_length = sum(lengths) / document_count

    if average_length == 0:
        return scores

    for term in query_terms:
        frequencies = [
            term_frequency(term, terms, min_prefix)
            for terms in document_terms
        ]
        matching_document_count = sum(1 for frequency in frequencies if frequency)

        if not matching_document_count:
            continue

        idf = inverse_document_frequency(document_count, matching_document_count)

        for index, frequency in enumerate(frequencies):
            if not frequency:
                continue

            normalization = 1 - b + b * lengths[index] / average_length
            scores[index] += idf * (
                frequency * (k1 + 1) / (frequency + k1 * normalization)
            )

    return scores








def calculate_cosine_similarities(question_embedding, chunk_embeddings):
    normalized_question = normalize(question_embedding, norm="l2")
    normalized_chunks = normalize(chunk_embeddings, norm="l2")
    return np.einsum(
        "ij,kj->ik",
        normalized_question,
        normalized_chunks,
    )[0]


def rank_positions(scores, only_positive=False):

    order = sorted(
        range(len(scores)),
        key=lambda index: (-scores[index], index),
    )

    positions = [None] * len(scores)

    for rank, index in enumerate(order, start=1):
        if only_positive and scores[index] <= 0:
            continue

        positions[index] = rank

    return positions


def reciprocal_rank_fusion(dense_scores, sparse_scores, rrf_k=RRF_K):

    dense_positions = rank_positions(dense_scores)
    sparse_positions = rank_positions(sparse_scores, only_positive=True)

    fused = []

    for dense_rank, sparse_rank in zip(dense_positions, sparse_positions):
        score = 1 / (rrf_k + dense_rank)

        if sparse_rank is not None:
            score += 1 / (rrf_k + sparse_rank)

        fused.append(score)

    return fused


def gate_score(results):

    if not results:
        return 0.0

    return max(
        float(result.get("dense_best_score", result["score"]))
        for result in results
    )


def document_order_key(chunk):
    page_number = chunk.get("page_number")
    chunk_index = chunk.get("chunk_index")
    return (
        chunk["source_name"].casefold(),
        page_number if page_number is not None else 0,
        chunk_index if chunk_index is not None else chunk["id"],
        chunk["id"],
    )


def attach_neighbor_chunks(ranked_results, selected_results, radius=1):
    if radius <= 0:
        return [dict(result, neighbors=[]) for result in selected_results]

    chunks_by_source = {}
    for result in ranked_results:
        chunks_by_source.setdefault(result["source_name"], []).append(result)

    positions = {}
    for source_chunks in chunks_by_source.values():
        source_chunks.sort(key=document_order_key)
        positions.update({chunk["id"]: index for index, chunk in enumerate(source_chunks)})

    enriched_results = []
    for result in selected_results:
        source_chunks = chunks_by_source[result["source_name"]]
        position = positions[result["id"]]
        start = max(0, position - radius)
        end = min(len(source_chunks), position + radius + 1)
        neighbors = [
            dict(chunk)
            for chunk in source_chunks[start:end]
            if chunk["id"] != result["id"]
        ]
        enriched_results.append(dict(result, neighbors=neighbors))

    return enriched_results


def apply_reranking(
    question,
    ranked_results,
    top_k,
    use_reranker=USE_RERANKER,
    candidate_pool=RERANK_CANDIDATE_POOL,
    rerank_func=None,
):

    if not use_reranker or not ranked_results:
        return ranked_results[:top_k]

    pool = ranked_results[:max(top_k, candidate_pool)]

    if rerank_func is None:
        from app.models import RerankerUnavailableError, rerank

        try:
            reranked = rerank(question, pool)
        except RerankerUnavailableError:
            return ranked_results[:top_k]
    else:
        reranked = rerank_func(question, pool)

    return reranked[:top_k]


def rank_chunks(
    question,
    chunks,
    top_k=3,
    neighbor_radius=1,
    use_hybrid=USE_HYBRID_SEARCH,
    rrf_k=RRF_K,
    bm25_k1=BM25_K1,
    bm25_b=BM25_B,
    use_reranker=USE_RERANKER,
    candidate_pool=RERANK_CANDIDATE_POOL,
    rerank_func=None,
):

    if not chunks:
        return []

    question_embedding = np.asarray(embed_texts([question]), dtype=np.float32)

    if not np.isfinite(question_embedding).all():
        return []

    chunk_embeddings = []
    valid_chunks = []

    for chunk in chunks:
        embedding = np.asarray(chunk["embedding"], dtype=np.float32)

        if embedding.ndim != 1:
            continue

        if not np.isfinite(embedding).all():
            continue

        chunk_embeddings.append(embedding)
        valid_chunks.append(chunk)

    if not chunk_embeddings:
        return []

    chunk_embeddings = np.vstack(chunk_embeddings)
    similarities = calculate_cosine_similarities(
        question_embedding,
        chunk_embeddings,
    )
    similarities = np.nan_to_num(similarities, nan=-1.0, posinf=-1.0, neginf=-1.0)

    dense_scores = [float(score) for score in similarities]
    document_terms = build_document_terms(
        chunk["chunk_text"] for chunk in valid_chunks
    )




    question_term_weights = corpus_term_weights(question, document_terms)

    if use_hybrid:
        sparse_scores = bm25_scores(question, document_terms, k1=bm25_k1, b=bm25_b)
        fusion_scores = reciprocal_rank_fusion(
            dense_scores,
            sparse_scores,
            rrf_k=rrf_k,
        )
    else:
        sparse_scores = [0.0] * len(valid_chunks)
        fusion_scores = list(dense_scores)

    dense_best_score = max(dense_scores)
    results = []

    for index, chunk in enumerate(valid_chunks):
        results.append({
            "id": chunk["id"],
            "source_name": chunk["source_name"],
            "source_type": chunk["source_type"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "chunk_text": chunk["chunk_text"],
            "score": dense_scores[index],
            "sparse_score": sparse_scores[index],
            "fusion_score": fusion_scores[index],
            "dense_best_score": dense_best_score,
            "question_term_weights": question_term_weights,
        })



    results.sort(
        key=lambda item: (item["fusion_score"], item["score"]),
        reverse=True,
    )
    selected_results = apply_reranking(
        question,
        results,
        top_k=top_k,
        use_reranker=use_reranker,
        candidate_pool=candidate_pool,
        rerank_func=rerank_func,
    )

    return attach_neighbor_chunks(
        results,
        selected_results,
        radius=neighbor_radius,
    )


def get_top_chunks(
    question,
    top_k=3,
    source_name=None,
    neighbor_radius=1,
    use_hybrid=USE_HYBRID_SEARCH,
    rrf_k=RRF_K,
    bm25_k1=BM25_K1,
    bm25_b=BM25_B,
    use_reranker=USE_RERANKER,
    candidate_pool=RERANK_CANDIDATE_POOL,
    rerank_func=None,
):
    return rank_chunks(
        question,
        get_all_chunks(source_name=source_name),
        top_k=top_k,
        neighbor_radius=neighbor_radius,
        use_hybrid=use_hybrid,
        rrf_k=rrf_k,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        use_reranker=use_reranker,
        candidate_pool=candidate_pool,
        rerank_func=rerank_func,
    )
