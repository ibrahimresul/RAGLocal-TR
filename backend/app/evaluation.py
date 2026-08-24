from app.retrieval import normalize_text
def chunk_matches_signature(chunk_text, signature):
    normalized_chunk = normalize_text(chunk_text)

    return all(
        normalize_text(term) in normalized_chunk
        for term in signature
    )


def find_signature_ranks(results, signatures):

    ranks = []

    for signature in signatures:
        matched_rank = None

        for position, result in enumerate(results, start=1):
            if chunk_matches_signature(result["chunk_text"], signature):
                matched_rank = position
                break

        ranks.append(matched_rank)

    return ranks


def recall_at_k(signature_ranks, k):

    if not signature_ranks:
        return None

    found = sum(
        1
        for rank in signature_ranks
        if rank is not None and rank <= k
    )

    return found / len(signature_ranks)


def reciprocal_rank(signature_ranks):

    found_ranks = [rank for rank in signature_ranks if rank is not None]

    if not found_ranks:
        return 0.0

    return 1.0 / min(found_ranks)


def average(values):
    numeric_values = [value for value in values if value is not None]

    if not numeric_values:
        return None

    return sum(numeric_values) / len(numeric_values)


def summarize_case_metrics(case_metrics, k_values=(1, 3, 5)):

    summary = {"case_count": len(case_metrics)}

    for k in k_values:
        summary[f"recall_at_{k}"] = average(
            [metrics.get(f"recall_at_{k}") for metrics in case_metrics]
        )

    summary["mrr"] = average(
        [metrics.get("reciprocal_rank") for metrics in case_metrics]
    )

    return summary


def build_case_metrics(results, signatures, k_values=(1, 3, 5)):
    signature_ranks = find_signature_ranks(results, signatures)
    metrics = {
        "signature_ranks": signature_ranks,
        "reciprocal_rank": reciprocal_rank(signature_ranks),
    }

    for k in k_values:
        metrics[f"recall_at_{k}"] = recall_at_k(signature_ranks, k)

    return metrics


def find_unmatched_signatures(all_chunks, signatures):

    unmatched = []

    for signature in signatures:
        matched = any(
            chunk_matches_signature(chunk["chunk_text"], signature)
            for chunk in all_chunks
        )

        if not matched:
            unmatched.append(signature)

    return unmatched


def format_metric(value):
    if value is None:
        return "-"

    return f"{value:.4f}"


def compare_summaries(baseline, current, k_values=(1, 3, 5)):

    metric_names = [f"recall_at_{k}" for k in k_values] + ["mrr"]
    comparisons = []

    for name in metric_names:
        baseline_value = baseline.get(name)
        current_value = current.get(name)

        if baseline_value is None or current_value is None:
            delta = None
        else:
            delta = current_value - baseline_value

        comparisons.append({
            "name": name,
            "baseline": baseline_value,
            "current": current_value,
            "delta": delta,
        })

    return comparisons

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.settings import CONTEXT_SCORE_THRESHOLD, SIMILARITY_THRESHOLD, TOP_K
from app.models import LocalLLM, MODEL_ALIAS, is_valid_answer
from app.rag import build_rag_messages
from app.retrieval import get_top_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CASES_PATH = PROJECT_ROOT / "benchmark_cases.json"
DEFAULT_REPORT_PATH = Path("data/model_benchmark.json")


class BenchmarkPreparationError(RuntimeError):
    pass


def load_benchmark_cases(cases_path=BENCHMARK_CASES_PATH):
    path = Path(cases_path)
    cases = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(cases, list) or not cases:
        raise BenchmarkPreparationError("Benchmark case set is empty or invalid.")

    return cases


def normalize_model_aliases(model_aliases):
    aliases = model_aliases or [MODEL_ALIAS]
    normalized = []

    for alias in aliases:
        clean_alias = str(alias).strip()

        if clean_alias and clean_alias not in normalized:
            normalized.append(clean_alias)

    if not normalized:
        raise BenchmarkPreparationError("At least one model alias is required.")

    return normalized


def prepare_benchmark_cases(cases, retrieval_func=get_top_chunks):
    prepared_cases = []

    for case in cases:
        results = retrieval_func(case["question"], top_k=TOP_K)

        if not results:
            raise BenchmarkPreparationError(
                f"{case['name']} returned no retrieval results."
            )

        best_result = results[0]
        expected_source = case["expected_source"]
        min_score = case.get("min_score", SIMILARITY_THRESHOLD)

        if best_result["source_name"] != expected_source:
            raise BenchmarkPreparationError(
                f"{case['name']} retrieved the wrong source: "
                f"{best_result['source_name']} (expected {expected_source})."
            )

        if best_result["score"] < min_score:
            raise BenchmarkPreparationError(
                f"{case['name']} retrieval score is too low: "
                f"{best_result['score']:.4f} (minimum {min_score:.4f})."
            )

        context_chunks = [
            chunk
            for chunk in results
            if chunk["score"] >= CONTEXT_SCORE_THRESHOLD
        ] or [best_result]

        prepared_cases.append({
            "name": case["name"],
            "question": case["question"],
            "expected_source": expected_source,
            "expected_terms": case.get("expected_terms", []),
            "retrieval_source": best_result["source_name"],
            "retrieval_score": best_result["score"],
            "messages": build_rag_messages(case["question"], context_chunks),
        })

    return prepared_cases


def evaluate_expected_terms(answer, expected_terms):
    if not expected_terms:
        return 1.0, []

    normalized_answer = answer.casefold()
    missing_terms = [
        term
        for term in expected_terms
        if term.casefold() not in normalized_answer
    ]
    coverage = (len(expected_terms) - len(missing_terms)) / len(expected_terms)
    return coverage, missing_terms


def timed_generation(llm, messages, timer=time.perf_counter):
    start_time = timer()

    try:
        answer = llm.generate_answer(messages)
        error = None
    except Exception as generation_error:
        answer = ""
        error = str(generation_error)

    elapsed = timer() - start_time
    return {
        "seconds": elapsed,
        "answer": answer,
        "error": error,
    }


def benchmark_model(
    model_alias,
    prepared_cases,
    llm_factory=LocalLLM,
    timer=time.perf_counter,
):
    load_start = timer()

    try:
        llm = llm_factory(model_alias=model_alias, show_startup_output=False)
    except Exception as error:
        return {
            "model": model_alias,
            "status": "error",
            "load_seconds": timer() - load_start,
            "error": str(error),
            "cases": [],
            "summary": None,
        }

    load_seconds = timer() - load_start
    case_results = []

    for case_index, case in enumerate(prepared_cases):
        run_labels = ["cold", "warm"] if case_index == 0 else ["measured"]
        runs = []

        for label in run_labels:
            run = timed_generation(llm, case["messages"], timer=timer)
            run["label"] = label
            runs.append(run)

        selected_answer = runs[-1]["answer"]
        coverage, missing_terms = evaluate_expected_terms(
            selected_answer,
            case["expected_terms"],
        )
        case_results.append({
            "name": case["name"],
            "question": case["question"],
            "retrieval_source": case["retrieval_source"],
            "retrieval_score": case["retrieval_score"],
            "expected_terms": case["expected_terms"],
            "valid_answer": is_valid_answer(selected_answer),
            "term_coverage": coverage,
            "missing_terms": missing_terms,
            "answer": selected_answer,
            "runs": runs,
        })

    successful_cases = [
        case
        for case in case_results
        if case["valid_answer"] and not case["runs"][-1]["error"]
    ]
    selected_times = [case["runs"][-1]["seconds"] for case in case_results]
    average_coverage = sum(case["term_coverage"] for case in case_results) / len(
        case_results
    )
    first_runs = case_results[0]["runs"]
    has_generation_error = any(
        run["error"]
        for case in case_results
        for run in case["runs"]
    )

    return {
        "model": model_alias,
        "status": "partial" if has_generation_error else "ok",
        "load_seconds": load_seconds,
        "error": None,
        "cases": case_results,
        "summary": {
            "case_count": len(case_results),
            "valid_case_count": len(successful_cases),
            "average_term_coverage": average_coverage,
            "cold_generation_seconds": first_runs[0]["seconds"],
            "warm_generation_seconds": first_runs[1]["seconds"],
            "average_selected_generation_seconds": (
                sum(selected_times) / len(selected_times)
            ),
        },
    }


def write_benchmark_report(report, report_path=DEFAULT_REPORT_PATH):
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def run_model_benchmark(
    model_aliases=None,
    cases_path=BENCHMARK_CASES_PATH,
    report_path=DEFAULT_REPORT_PATH,
    llm_factory=LocalLLM,
    retrieval_func=get_top_chunks,
    timer=time.perf_counter,
):
    aliases = normalize_model_aliases(model_aliases)
    cases = load_benchmark_cases(cases_path)
    prepared_cases = prepare_benchmark_cases(cases, retrieval_func=retrieval_func)
    model_results = [
        benchmark_model(
            alias,
            prepared_cases,
            llm_factory=llm_factory,
            timer=timer,
        )
        for alias in aliases
    ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(prepared_cases),
        "models": model_results,
    }
    path = write_benchmark_report(report, report_path=report_path)
    return report, path
