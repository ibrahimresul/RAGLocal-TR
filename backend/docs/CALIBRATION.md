# Retrieval and Generation Calibration

This document records the rationale for thresholds in `app/settings.py`.
Values should only be changed together with retrieval and groundedness tests.

## Retrieval Gate

`SIMILARITY_THRESHOLD = 0.20` rejects only completely unrelated indexes.
Cosine similarity measures topic proximity rather than answer availability, so
it is intentionally not used as the final evidence decision.

`TERM_EVIDENCE_THRESHOLD = 0.12` is the weighted question-term coverage gate.
It was reduced from `0.21` after valid questions were rejected too frequently.
Groundedness validation remains the final protection against unsupported
generated answers.

## Context Selection

- `TOP_K = 3`
- `CONTEXT_SCORE_THRESHOLD = 0.35`
- `CONTEXT_RELATIVE_SCORE_MARGIN = 0.20`
- `CONTEXT_TERM_EVIDENCE_MIN = 0.30`
- `NEIGHBOR_CHUNK_RADIUS = 1`
- `MAX_CONTEXT_CHUNKS = 5`

The first ranked chunk always enters context. Lower-ranked chunks must satisfy
both score proximity and weighted term-evidence requirements. This prevents
topically similar but unsupported chunks from expanding the prompt.

## Turkish Stopwords and Matching

The base stopword set in `app/stopwords.py` comes from the official NLTK
`stopwords` corpus. A small `QUERY_NOISE_TERMS` set contains interrogative and
instructional forms that are specific to this retrieval pipeline.

- `TERM_EVIDENCE_MIN_PREFIX = 5`
- `TERM_EVIDENCE_MIN_SHORT_ROOT = 3`
- `TERM_EVIDENCE_MIN_TERM_LENGTH = 3`

Turkish suffix matching uses common prefixes and consonant-mutation tolerance.
The language data remains local so runtime retrieval never depends on network
availability.

## Hybrid Search

- `USE_HYBRID_SEARCH = True`
- `BM25_K1 = 1.5`
- `BM25_B = 0.75`
- `RRF_K = 2`

Dense retrieval is combined with BM25 through Reciprocal Rank Fusion. The
calibrated corpus favored a small RRF smoothing constant because exact term
matches often identify the correct chunk already present in the dense pool.

## Reranking

- `USE_RERANKER = False`
- `RERANKER_MODEL = BAAI/bge-reranker-base`
- `RERANK_CANDIDATE_POOL = 15`

Cross-encoder reranking is disabled because it reduced measured quality on the
short 128-token chunks. The implementation remains available as an optional
feature and falls back to hybrid ranking when the model is unavailable.

## Groundedness

- `GROUNDEDNESS_THRESHOLD = 0.50`
- `GROUNDEDNESS_SENTENCE_SUPPORT = 0.60`
- `GROUNDEDNESS_MIN_SENTENCE_TERMS = 2`

Generated answers are evaluated sentence by sentence against context terms.
At least half of measurable sentences must meet the support threshold.

## Extractive Fallback

- `USE_EXTRACTIVE_FALLBACK = True`
- `EXTRACTIVE_SCORE_THRESHOLD = 0.50`
- `EXTRACTIVE_TERM_EVIDENCE_MIN = 0.675`
- `MAX_EXTRACTIVE_CHARS = 500`

The extractive path makes a stronger relevance claim because it bypasses model
generation and groundedness checks. It therefore keeps a stricter term-evidence
threshold than the general retrieval gate.
