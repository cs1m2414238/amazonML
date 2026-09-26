"""Reusable benchmark harness comparing baseline and optimized retrieval on identical samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.evaluation.candidate_recall import (
    CandidateRecallEvaluator,
)
from business_entity_resolution.src.evaluation.sampling import (
    load_validation_sample,
)
from business_entity_resolution.src.retrieval.candidate_generator import (
    CandidateGenerator,
)
from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run_retrieval_benchmark(
    sample_records: list[dict[str, Any]],
    retriever_name: str,
    retriever_instance: Any,
    budgets: list[int],
) -> dict[str, Any]:
    """Run candidate retrieval across queries and budgets, measuring exact latency and recall."""
    max_budget = max(budgets)
    query_count = len(sample_records)
    logging.info(
        f"Running {retriever_name} on {query_count} queries with max_budget={max_budget}..."
    )

    all_candidates: dict[str, list[str]] = {}
    latencies: list[float] = []

    t_start = time.perf_counter()
    for idx, rec in enumerate(sample_records):
        s1_id = rec["entity_id"]
        name = rec.get("business_name", "")
        addr = rec.get("business_address", "")
        country = rec.get("country", "")

        q0 = time.perf_counter()
        cands = retriever_instance.get_candidates(
            name, addr, country, budget=max_budget
        )
        q1 = time.perf_counter()

        latencies.append(q1 - q0)
        all_candidates[s1_id] = cands

        if (idx + 1) % 100 == 0 or (idx + 1) == query_count:
            logging.info(f"[{retriever_name}] Processed {idx + 1}/{query_count} queries...")

    t_total = time.perf_counter() - t_start
    peak_rss = get_peak_memory_mb()

    # Build ground truth map from sample records
    gt_map = {r["entity_id"]: r["ground_truth_matches"] for r in sample_records}
    evaluator = CandidateRecallEvaluator(gt_map=gt_map)

    results_by_budget: dict[str, Any] = {}
    for b in budgets:
        sliced_cands = {s1_id: cands[:b] for s1_id, cands in all_candidates.items()}
        budget_res = evaluator.evaluate(
            retrieved_candidates=sliced_cands,
            sample_metadata=sample_records,
            runtime_seconds=t_total,
            latencies=latencies,
            peak_memory_mb=peak_rss,
        )
        results_by_budget[f"B_{b}"] = budget_res

    return {
        "retriever": retriever_name,
        "query_count": query_count,
        "runtime_seconds": round(t_total, 2),
        "queries_per_second": round(query_count / t_total, 2) if t_total > 0 else 0.0,
        "peak_rss_mb": round(peak_rss, 2),
        "budgets": results_by_budget,
    }


def generate_markdown_report(benchmark_data: dict[str, Any]) -> str:
    """Format benchmark results into GitHub-flavored Markdown comparison tables."""
    lines = [
        "# Retrieval Benchmark Comparison Report",
        "",
        "> [!IMPORTANT]",
        "> **Methodological safeguard**: Candidate recall measures search space coverage and blocking efficiency. "
        "It is **NOT** a measurement of final matching accuracy or macro-F0.5. True macro-F0.5 can only be evaluated "
        "once a matching model (e.g. LightGBM) produces final predicted match sets.",
        "",
        "## Evaluation Metadata",
        "",
        f"- **Validation Sample**: {benchmark_data['metadata']['sample_name']}",
        f"- **Sample Size**: {benchmark_data['metadata']['sample_size']} S1 entities",
        f"- **Sample Entities SHA-256**: {benchmark_data['metadata']['entity_ids_sha256']}",
        f"- **Platform**: {benchmark_data['metadata']['platform']}",
        f"- **Python**: {benchmark_data['metadata']['python_version']}",
        f"- **CPU**: {benchmark_data['metadata']['processor']} ({benchmark_data['metadata']['cpu_count']} cores)",
        f"- **Index Path**: {benchmark_data['metadata']['index_path']}",
        "",
        "## Recall vs. Candidate Count vs. Speed Comparison",
        "",
    ]

    retrievers = list(benchmark_data.get("results", {}).keys())
    budgets = benchmark_data.get("budgets_evaluated", [16, 32, 64, 128, 256])

    header = "| Budget | Metric | " + " | ".join(retrievers) + " |"
    separator = "| ---: | --- | " + " | ".join(["---:"] * len(retrievers)) + " |"
    lines.append(header)
    lines.append(separator)

    metrics_to_show = [
        ("Pair Recall", lambda r: f"{r['representative_metrics']['pair_level_recall'] * 100:.2f}%"),
        ("Complete Match Recall", lambda r: f"{r['representative_metrics']['complete_match_set_recall'] * 100:.2f}%"),
        ("Source 2 Recall", lambda r: f"{r['representative_metrics']['source2_recall'] * 100:.2f}%"),
        ("Source 3 Recall", lambda r: f"{r['representative_metrics']['source3_recall'] * 100:.2f}%"),
        ("Average Candidates", lambda r: f"{r['representative_metrics']['candidate_counts']['mean']:.1f}"),
        ("P95 Candidates", lambda r: f"{r['representative_metrics']['candidate_counts']['p95']}"),
        ("Reduction Ratio", lambda r: f"{r['representative_metrics']['reduction_ratio']:.6f}"),
    ]

    for b in budgets:
        b_key = f"B_{b}"
        for label, extractor in metrics_to_show:
            row = [f"B={b}", label]
            for ret in retrievers:
                b_res = (
                    benchmark_data["results"][ret]["budgets"].get(b_key, {})
                )
                if b_res:
                    row.append(extractor(b_res))
                else:
                    row.append("N/A")
            lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Runtime Efficiency and Resource Utilization",
        "",
        "| Retriever | Total Runtime (s) | QPS | Median Latency (ms) | P95 Latency (ms) | Peak RSS (MiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])

    for ret in retrievers:
        res = benchmark_data["results"][ret]
        runtime = res.get("runtime_seconds", 0.0)
        qps = res.get("queries_per_second", 0.0)
        peak_rss = res.get("peak_rss_mb", 0.0)

        # Grab latency from highest budget
        max_b_key = f"B_{max(budgets)}"
        b_data = res["budgets"].get(max_b_key, {})
        lat_stats = b_data.get("representative_metrics", {}).get("efficiency", {}).get("latency_stats", {})

        med_lat = f"{lat_stats.get('median_seconds', 0.0) * 1000:.1f}" if lat_stats else "N/A"
        p95_lat = f"{lat_stats.get('p95_seconds', 0.0) * 1000:.1f}" if lat_stats else "N/A"

        lines.append(f"| {ret} | {runtime:.2f} | {qps:.2f} | {med_lat} | {p95_lat} | {peak_rss:.1f} |")

    # Strata breakdown table for recommended default budget B=64
    ref_budget = 64 if 64 in budgets else budgets[0]
    ref_b_key = f"B_{ref_budget}"
    primary_retriever = retrievers[-1]  # Usually optimized
    primary_res = benchmark_data["results"][primary_retriever]["budgets"].get(ref_b_key, {})
    diag_strata = primary_res.get("diagnostic_strata", {})

    if diag_strata:
        lines.extend([
            "",
            f"## Diagnostic Strata Breakdown ({primary_retriever} @ B={ref_budget})",
            "",
            "> [!NOTE]",
            "> Diagnostic strata are overlapping categories designed to test difficult subsets. "
            "These metrics are reported separately from the population-representative results.",
            "",
            "| Stratum | Entities | True Pairs | Pair Recall | Complete Match Recall | Avg Candidates |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for s_name, s_data in sorted(diag_strata.items()):
            lines.append(
                f"| {s_name} | {s_data['entity_count']} | {s_data['total_true_pairs']} | "
                f"{s_data['pair_level_recall'] * 100:.2f}% | "
                f"{s_data['complete_match_set_recall'] * 100:.2f}% | "
                f"{s_data['average_candidates']:.1f} |"
            )

    return "\n".join(lines)


def run_comparison(
    sample_path: str | Path,
    retrievers_to_run: list[str],
    budgets: list[int],
    output_dir: str | Path,
    index_path: str | Path | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Run full comparison harness and write JSON/Markdown reports."""
    sample_data = load_validation_sample(sample_path)
    sample_records = sample_data["records"]
    sample_metadata = sample_data["metadata"]
    if max_queries and max_queries < len(sample_records):
        sample_records = sample_records[:max_queries]

    config = load_config()
    db_path = (
        Path(index_path).resolve()
        if index_path
        else (Path(config["output"]["directory"]) / "step5/indices.db").resolve()
    )
    if not db_path.exists():
        # Check fallback path
        fallback = Path(r"H:/Amazon_ML_Work\output/step5/indices.db")
        if fallback.exists():
            db_path = fallback

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_summary: dict[str, Any] = {
        "metadata": {
            "sample_name": sample_metadata.get("sample_name", Path(sample_path).stem),
            "sample_size": len(sample_records),
            "entity_ids_sha256": sample_metadata.get("entity_ids_sha256", ""),
            "index_path": str(db_path),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "budgets_evaluated": budgets,
        "results": {},
    }

    high_df = config.get("retrieval", {}).get("high_df_threshold", 50000)

    for ret_name in retrievers_to_run:
        if ret_name == "baseline":
            logging.info("Initializing Baseline CandidateGenerator...")
            retriever = CandidateGenerator(str(db_path), high_df_threshold=high_df)
            res = run_retrieval_benchmark(
                sample_records=sample_records,
                retriever_name="Baseline (CandidateGenerator)",
                retriever_instance=retriever,
                budgets=budgets,
            )
            benchmark_summary["results"]["baseline"] = res
            retriever.close()

        elif ret_name == "optimized":
            logging.info("Initializing OptimizedCandidateGenerator...")
            retriever = OptimizedCandidateGenerator(
                db_path=db_path,
                high_df_threshold=high_df,
                posting_cache_mb=1024,
                sqlite_cache_mb=512,
                mmap_mb=8192,
            )
            res = run_retrieval_benchmark(
                sample_records=sample_records,
                retriever_name="Optimized (OptimizedCandidateGenerator)",
                retriever_instance=retriever,
                budgets=budgets,
            )
            benchmark_summary["results"]["optimized"] = res
            retriever.close()

    # Save outputs
    json_path = output_dir / "benchmark_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)
    logging.info(f"Wrote benchmark JSON report to {json_path}")

    md_content = generate_markdown_report(benchmark_summary)
    md_path = output_dir / "benchmark_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logging.info(f"Wrote benchmark Markdown report to {md_path}")

    return benchmark_summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark harness for entity resolution retrieval")
    parser.add_argument(
        "--sample",
        type=str,
        default="output/validation/validation_smoke_500.json",
        help="Path to validation sample JSON",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="optimized",
        choices=["baseline", "optimized", "both"],
        help="Which retriever(s) to benchmark",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="16,32,64,128,256",
        help="Comma-separated candidate budgets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/validation",
        help="Directory to save comparison reports",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="Explicit path to indices.db",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum queries to evaluate from the sample",
    )
    args = parser.parse_args()

    budgets = [int(b.strip()) for b in args.budgets.split(",") if b.strip()]
    retrievers = ["baseline", "optimized"] if args.retriever == "both" else [args.retriever]

    run_comparison(
        sample_path=args.sample,
        retrievers_to_run=retrievers,
        budgets=budgets,
        output_dir=args.output_dir,
        index_path=args.index_path,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
