"""Benchmark optimized retrieval on the exact baseline query set."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import time
from pathlib import Path

import numpy as np

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.evaluation.recall_evaluator import RecallEvaluator
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.retrieval.candidate_generator import CandidateGenerator
from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--equivalence-sample", type=int, default=20)
    parser.add_argument("--posting-cache-mb", type=int)
    return parser.parse_args()


def percentile(values, q):
    return round(float(np.percentile(values, q)), 6) if values else 0.0


def main():
    args = parse_args()
    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    retrieval_config = config.get("retrieval", {})
    optimized_config = retrieval_config.get("optimized", {})

    budgets = retrieval_config.get("budgets", [16, 32, 64, 128, 256])
    sample_size = args.sample_size or retrieval_config.get("experiment_sample_size", 2000)
    high_df_threshold = retrieval_config.get("high_df_threshold", 50_000)
    posting_cache_mb = args.posting_cache_mb or optimized_config.get(
        "posting_cache_mb", 1024
    )
    sqlite_cache_mb = optimized_config.get("sqlite_cache_mb", 512)
    mmap_mb = optimized_config.get("mmap_mb", 8192)

    db_path = output_root / "step5/indices.db"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    s1_path = data_root / "dataset/train/train_source1.tsv"

    s1_records = []
    for chunk in read_tsv_chunks(s1_path, "source1", sample_size):
        for row in chunk.itertuples(index=False):
            s1_records.append(
                (row.entity_id, row.business_name, row.business_address, row.country)
            )
            if len(s1_records) >= sample_size:
                break
        if len(s1_records) >= sample_size:
            break

    sample_hash = hashlib.sha256(
        "\n".join(record[0] for record in s1_records).encode("utf-8")
    ).hexdigest()

    logging.info("Checking optimized/baseline candidate equivalence...")
    with CandidateGenerator(str(db_path), high_df_threshold) as baseline, \
         OptimizedCandidateGenerator(
            db_path,
            high_df_threshold=high_df_threshold,
            posting_cache_mb=posting_cache_mb,
            sqlite_cache_mb=sqlite_cache_mb,
            mmap_mb=mmap_mb,
        ) as optimized:
        equivalence_failures = []
        for record in s1_records[: args.equivalence_sample]:
            entity_id, name, address, country = record
            expected = baseline.get_candidates(name, address, country, max(budgets))
            actual = optimized.get_candidates(name, address, country, max(budgets))
            if expected != actual:
                equivalence_failures.append(entity_id)
        if equivalence_failures:
            raise RuntimeError(
                "Candidate equivalence failed for: "
                + ", ".join(equivalence_failures[:10])
            )
        logging.info("Candidate equivalence passed.")

    # Measure a new, cold retriever. Equivalence queries must not pre-warm the
    # posting cache or SQLite page cache used by the timed benchmark.
    del baseline, optimized
    with OptimizedCandidateGenerator(
        db_path,
        high_df_threshold=high_df_threshold,
        posting_cache_mb=posting_cache_mb,
        sqlite_cache_mb=sqlite_cache_mb,
        mmap_mb=mmap_mb,
    ) as optimized:
        master_results = {}
        latencies = []
        start = time.perf_counter()
        for index, (entity_id, name, address, country) in enumerate(s1_records, 1):
            query_start = time.perf_counter()
            master_results[entity_id] = optimized.get_candidates(
                name, address, country, max(budgets)
            )
            latencies.append(time.perf_counter() - query_start)
            if index % 200 == 0:
                elapsed = time.perf_counter() - start
                logging.info(
                    f"Processed {index}/{sample_size} queries "
                    f"({index / elapsed:.2f} QPS)"
                )
        runtime = time.perf_counter() - start
        profile = optimized.profile_snapshot()

    evaluator = RecallEvaluator(str(gt_path))
    report = {
        "metadata": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "sample_size": len(s1_records),
            "sample_first_id": s1_records[0][0] if s1_records else None,
            "sample_last_id": s1_records[-1][0] if s1_records else None,
            "sample_id_sha256": sample_hash,
            "equivalence_sample_size": args.equivalence_sample,
            "equivalence_passed": True,
            "posting_cache_limit_mb": posting_cache_mb,
            "sqlite_cache_mb": sqlite_cache_mb,
            "mmap_mb": mmap_mb,
            "index_path": str(db_path.resolve()),
        },
        "performance": {
            "runtime_seconds": round(runtime, 2),
            "queries_per_second": round(len(s1_records) / runtime, 2),
            "peak_rss_mb": round(get_peak_memory_mb(), 2),
            "latency_seconds": {
                "mean": round(float(np.mean(latencies)), 6),
                "median": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "p99": percentile(latencies, 99),
                "max": round(max(latencies), 6) if latencies else 0.0,
            },
        },
        "profile": profile,
        "budgets": {},
    }

    for budget in budgets:
        sliced = {
            entity_id: candidates[:budget]
            for entity_id, candidates in master_results.items()
        }
        counts = [len(candidates) for candidates in sliced.values()]
        metrics = evaluator.evaluate(sliced)
        metrics.update(
            {
                "average_candidates": round(float(np.mean(counts)), 2),
                "p95_candidates": int(np.percentile(counts, 95)),
                "p99_candidates": int(np.percentile(counts, 99)),
                "max_candidates": int(np.max(counts)),
            }
        )
        report["budgets"][f"B_{budget}"] = metrics

    output_dir = output_root / "step6"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "optimized_budget_experiment_results.json"
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    logging.info(f"Optimized benchmark saved to {report_path}")


if __name__ == "__main__":
    main()
