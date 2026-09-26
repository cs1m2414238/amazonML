"""Held-out validation and threshold search for business entity matching."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.evaluation.f05 import MacroF05Evaluator
from business_entity_resolution.src.evaluation.sampling import load_validation_sample
from business_entity_resolution.src.evaluation.threshold_search import search_best_threshold
from business_entity_resolution.src.features.generate_training_data import hydrate_target_records
from business_entity_resolution.src.models.matcher_inference import BatchMatcher
from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def evaluate_matching_pipeline(
    model_path: str | Path,
    sample_path: str | Path = "output/validation/validation_smoke_500.json",
    retrieval_budget: int = 64,
    output_dir: str | Path = "output/models",
    db_path: str | Path | None = None,
    max_queries: int | None = None,
) -> dict[str, Any]:
    """Run real candidate retrieval, score pairs with LightGBM, and optimize threshold for Macro-F0.5."""
    config = load_config()
    data_root = Path(config["data"]["root"])
    s2_path = data_root / "dataset/train/train_source2.tsv"
    s3_path = data_root / "dataset/train/train_source3.tsv"

    if db_path is None:
        db_path = Path(config["output"]["directory"]) / "step5/indices.db"
        if not db_path.exists():
            db_path = Path(r"H:/Amazon_ML_Work/output/step5/indices.db")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation sample
    logging.info(f"Loading held-out validation sample from {sample_path}...")
    sample_data = load_validation_sample(sample_path)
    sample_records = sample_data["records"]
    if max_queries and max_queries < len(sample_records):
        sample_records = sample_records[:max_queries]

    query_count = len(sample_records)
    logging.info(f"Evaluating {query_count} validation S1 queries (retrieval budget B={retrieval_budget})...")

    # 1. Candidate Retrieval (real retrieval, ZERO ground-truth injection)
    high_df = config.get("retrieval", {}).get("high_df_threshold", 50000)
    retriever = OptimizedCandidateGenerator(
        db_path=db_path,
        high_df_threshold=high_df,
        posting_cache_mb=1024,
        sqlite_cache_mb=512,
        mmap_mb=8192,
    )

    t0_retr = time.perf_counter()
    candidates_by_s1: dict[str, list[str]] = {}
    all_target_ids: set[str] = set()

    for idx, s1 in enumerate(sample_records):
        s1_id = s1["entity_id"]
        cands = retriever.get_candidates(
            s1["business_name"], s1["business_address"], s1["country"], budget=retrieval_budget
        )
        candidates_by_s1[s1_id] = cands
        all_target_ids.update(cands)

        if (idx + 1) % 100 == 0 or (idx + 1) == query_count:
            logging.info(f"Retrieved candidates for {idx + 1}/{query_count} validation queries...")

    retrieval_time = time.perf_counter() - t0_retr
    retriever.close()
    logging.info(f"Retrieval finished in {retrieval_time:.2f}s ({query_count / retrieval_time:.2f} QPS). Unique targets: {len(all_target_ids)}")

    # 2. Candidate Recall Ceiling Check
    gt_map = {r["entity_id"]: r["ground_truth_matches"] for r in sample_records}
    total_gt_pairs = sum(len(t) for t in gt_map.values())
    found_gt_pairs = 0
    for s1_id, targets in gt_map.items():
        cands = set(candidates_by_s1.get(s1_id, []))
        found_gt_pairs += len(cands & set(targets))
    candidate_recall_ceiling = (found_gt_pairs / total_gt_pairs) if total_gt_pairs > 0 else 1.0
    logging.info(f"Candidate Recall Ceiling: {candidate_recall_ceiling * 100:.2f}% ({found_gt_pairs}/{total_gt_pairs} pairs)")

    # 3. Hydrate candidate target records
    cache_path = output_dir / "cached_val_target_records.json"
    target_records = hydrate_target_records(
        target_ids=all_target_ids,
        s2_path=s2_path,
        s3_path=s3_path,
        cache_path=cache_path,
    )

    # 4. Batch Scoring with Matcher
    matcher = BatchMatcher(model_path=model_path)
    s1_formatted = [
        {
            "id": r["entity_id"],
            "name": r["business_name"],
            "addr": r["business_address"],
            "country": r["country"],
        }
        for r in sample_records
    ]

    t0_score = time.perf_counter()
    scored_by_s1 = matcher.score_pairs_batch(
        s1_records=s1_formatted,
        candidate_records=target_records,
        candidates_by_s1=candidates_by_s1,
    )
    scoring_time = time.perf_counter() - t0_score
    logging.info(f"Scored candidate pairs in {scoring_time:.2f}s.")

    # Flatten for threshold tuning
    scored_pairs_list: list[tuple[str, str, float]] = []
    for s1_id, pair_list in scored_by_s1.items():
        for cand_id, prob in pair_list:
            scored_pairs_list.append((s1_id, cand_id, prob))

    # 5. Threshold Search maximizing Macro-F0.5
    thresholds_to_test = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99]
    search_result = search_best_threshold(
        scored_pairs=scored_pairs_list,
        gt_map=gt_map,
        thresholds=thresholds_to_test,
        sample_metadata=sample_records,
    )

    best_thresh = search_result["best_threshold"]
    best_macro_f05 = search_result["best_macro_f05"]
    eval_at_best = search_result["evaluation_at_best"]
    rep_metrics = eval_at_best["representative_metrics"]
    diag_strata = eval_at_best.get("diagnostic_strata", {})

    logging.info(f"Optimal Threshold: {best_thresh} -> Held-out Macro-F0.5: {best_macro_f05:.4f}")
    logging.info(f"Pair Precision: {rep_metrics['pair_precision']:.4f} | Pair Recall: {rep_metrics['pair_recall']:.4f}")
    logging.info(f"Singleton Accuracy: {rep_metrics['singleton_accuracy']:.4f}")

    # Generate final predictions at best threshold
    final_preds = matcher.predict_batch(
        s1_records=s1_formatted,
        candidate_records=target_records,
        candidates_by_s1=candidates_by_s1,
        threshold=best_thresh,
    )

    # Average predicted matches per entity
    pred_counts = [len(p) for p in final_preds.values()]
    avg_pred_matches = float(np.mean(pred_counts))

    # Save predictions
    preds_path = output_dir / "validation_predictions.json"
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(final_preds, f, indent=2)

    # TSV format
    tsv_path = output_dir / "validation_matching_results.tsv"
    BatchMatcher.format_tsv(final_preds, tsv_path)

    # Build report
    report_data = {
        "metadata": {
            "model_path": str(Path(model_path).resolve()),
            "sample_path": str(Path(sample_path).resolve()),
            "sample_size": query_count,
            "retrieval_budget": retrieval_budget,
            "optimal_threshold": best_thresh,
            "candidate_recall_ceiling": round(candidate_recall_ceiling, 4),
            "retrieval_time_seconds": round(retrieval_time, 2),
            "scoring_time_seconds": round(scoring_time, 2),
            "prediction_qps": round(query_count / (retrieval_time + scoring_time), 2),
        },
        "performance_metrics": {
            "macro_f05": rep_metrics["macro_f05"],
            "singleton_accuracy": rep_metrics["singleton_accuracy"],
            "pair_precision": rep_metrics["pair_precision"],
            "pair_recall": rep_metrics["pair_recall"],
            "pair_f05": rep_metrics["pair_f05"],
            "average_predicted_matches": round(avg_pred_matches, 2),
            "total_queries": rep_metrics["total_queries"],
            "singleton_queries": rep_metrics["singleton_queries"],
            "non_singleton_queries": rep_metrics["non_singleton_queries"],
        },
        "validation_checks": rep_metrics["validation_checks"],
        "diagnostic_strata": diag_strata,
        "threshold_curve": search_result["curve"],
    }

    report_json_path = output_dir / "matching_validation_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logging.info(f"Saved matching validation JSON report to {report_json_path}")

    # Build Markdown report
    md_lines = [
        "# LightGBM Business Entity Matcher ? Validation Report",
        "",
        "> [!IMPORTANT]",
        "> **Official Scoring**: Evaluated strictly on held-out validation entities using real retrieved candidates. "
        "Macro-{0.5}$ weights precision over recall ({0.5} = \frac{5 TP}{5 TP + 4 FP + FN}$ for non-singletons, "
        ".0$ for correctly empty singletons, .0$ for false-positive singletons).",
        "",
        "## Summary Metrics",
        "",
        f"- **Held-Out Macro-{0.5}$**: **{rep_metrics['macro_f05'] * 100:.2f}%**",
        f"- **Optimal Decision Threshold**: **{best_thresh}**",
        f"- **Pair Precision**: {rep_metrics['pair_precision'] * 100:.2f}%",
        f"- **Pair Recall**: {rep_metrics['pair_recall'] * 100:.2f}%",
        f"- **Singleton Accuracy**: {rep_metrics['singleton_accuracy'] * 100:.2f}%",
        f"- **Candidate Recall Ceiling (Blocking)**: {candidate_recall_ceiling * 100:.2f}%",
        f"- **Average Predicted Matches / Query**: {avg_pred_matches:.2f}",
        f"- **Candidate Subset Violations**: {rep_metrics['validation_checks']['candidate_subset_violations_count']}",
        "",
        "## Performance Across Diagnostic Strata",
        "",
        "| Stratum | Entities | Macro-F0.5 | Pair Precision | Pair Recall | Singleton Acc |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for s_name, s_data in sorted(diag_strata.items()):
        md_lines.append(
            f"| {s_name} | {s_data['entity_count']} | {s_data['macro_f05'] * 100:.2f}% | "
            f"{s_data['pair_precision'] * 100:.2f}% | {s_data['pair_recall'] * 100:.2f}% | "
            f"{s_data['singleton_accuracy'] * 100:.2f}% |"
        )

    md_lines.extend([
        "",
        "## Threshold Optimization Curve",
        "",
        "| Threshold | Macro-F0.5 | Pair Precision | Pair Recall | Singleton Acc |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ])
    for pt in search_result["curve"]:
        marker = " **(Optimal)**" if pt["threshold"] == best_thresh else ""
        md_lines.append(
            f"| {pt['threshold']:.2f}{marker} | {pt['macro_f05'] * 100:.2f}% | "
            f"{pt['pair_precision'] * 100:.2f}% | {pt['pair_recall'] * 100:.2f}% | "
            f"{pt['singleton_accuracy'] * 100:.2f}% |"
        )

    report_md_path = output_dir / "matching_validation_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logging.info(f"Saved matching validation Markdown report to {report_md_path}")

    return report_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate LightGBM business entity matcher on held-out validation sample")
    parser.add_argument("--model", type=str, default="output/models/lightgbm_matcher.txt")
    parser.add_argument("--sample", type=str, default="output/validation/validation_smoke_500.json")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="output/models")
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    evaluate_matching_pipeline(
        model_path=args.model,
        sample_path=args.sample,
        retrieval_budget=args.budget,
        output_dir=args.output_dir,
        max_queries=args.max_queries,
    )


if __name__ == "__main__":
    main()
