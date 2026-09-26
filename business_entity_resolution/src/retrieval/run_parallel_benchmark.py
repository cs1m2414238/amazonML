"""Benchmark B=512 retrieval with one, two, and four worker processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.evaluation.recall_evaluator import RecallEvaluator
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)
from business_entity_resolution.src.retrieval.parallel_resumable_retrieval import (
    ParallelResumableRetriever,
    RunSettings,
    WorkerSettings,
    latency_summary,
    make_record_batches,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--workers", default="1,2,4")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--posting-cache-mb", type=int, default=1024)
    parser.add_argument("--sqlite-cache-mb", type=int, default=512)
    parser.add_argument("--mmap-mb", type=int, default=8192)
    parser.add_argument("--direct-equivalence-sample", type=int, default=100)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def load_records(path: Path, sample_size: int):
    records = []
    for chunk in read_tsv_chunks(path, "source1", sample_size):
        for row in chunk.itertuples(index=False):
            records.append(
                (row.entity_id, row.business_name, row.business_address, row.country)
            )
            if len(records) >= sample_size:
                return records
    return records


def safe_remove_benchmark_directory(path: Path, root: Path) -> None:
    resolved = path.resolve()
    expected_root = root.resolve()
    if expected_root not in resolved.parents or resolved == expected_root:
        raise ValueError(f"Refusing to remove unexpected path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def result_map(runner: ParallelResumableRetriever):
    return {
        row["source1_entity_id"]: row["candidates"]
        for row in runner.iter_results()
    }


def main():
    args = parse_args()
    worker_counts = [int(value) for value in args.workers.split(",")]
    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    s1_path = data_root / "dataset/train/train_source1.tsv"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    db_path = (output_root / "step5/indices.db").resolve()
    benchmark_root = (output_root / "step6/parallel_b512_benchmark").resolve()
    if args.fresh:
        safe_remove_benchmark_directory(benchmark_root, output_root)
    benchmark_root.mkdir(parents=True, exist_ok=True)

    records = load_records(s1_path, args.sample_size)
    id_hash = hashlib.sha256(
        "\n".join(record[0] for record in records).encode("utf-8")
    ).hexdigest()
    dataset_signature = f"first-{len(records)}:{id_hash}"
    index_before = db_path.stat()
    report = {
        "metadata": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "sample_size": len(records),
            "sample_first_id": records[0][0],
            "sample_last_id": records[-1][0],
            "sample_id_sha256": id_hash,
            "budget": args.budget,
            "batch_size": args.batch_size,
            "index_path": str(db_path),
            "index_size_bytes_before": index_before.st_size,
            "index_mtime_ns_before": index_before.st_mtime_ns,
            "memory_note": (
                "aggregate_peak_upper_bound_mb is parent peak plus each worker's "
                "individual peak; it is conservative and not a simultaneous RSS sample"
            ),
        },
        "workers": {},
    }

    runners = {}
    for workers in worker_counts:
        runner = ParallelResumableRetriever(benchmark_root / f"workers_{workers}")
        runners[workers] = runner
        worker_settings = WorkerSettings(
            db_path=str(db_path),
            high_df_threshold=config["retrieval"].get("high_df_threshold", 50_000),
            posting_cache_mb=args.posting_cache_mb,
            sqlite_cache_mb=args.sqlite_cache_mb,
            mmap_mb=args.mmap_mb,
        )
        settings = RunSettings(
            budget=args.budget,
            workers=workers,
            batch_size=args.batch_size,
            dataset_signature=dataset_signature,
            total_records=len(records),
            total_batches=math.ceil(len(records) / args.batch_size),
            worker=worker_settings,
        )
        started = time.perf_counter()
        result = runner.run(make_record_batches(records, args.batch_size), settings)
        measured_wall = time.perf_counter() - started
        candidate_counts = [
            len(row["candidates"]) for row in runner.iter_results()
        ]
        report["workers"][str(workers)] = {
            **{k: v for k, v in result.items() if k != "latencies_seconds"},
            "measured_wall_seconds": measured_wall,
            "end_to_end_qps": len(records) / measured_wall,
            "latency_seconds": latency_summary(result["latencies_seconds"]),
            "average_candidates": float(np.mean(candidate_counts)),
            "p95_candidates": float(np.percentile(candidate_counts, 95)),
            "max_candidates": max(candidate_counts),
            "result_sha256": runner.result_digest(),
            "projected_full_test_hours_processing_qps": (
                1_732_544 / result["processing_qps"] / 3600
                if result["processing_qps"]
                else None
            ),
        }

    reference_workers = worker_counts[0]
    reference_digest = report["workers"][str(reference_workers)]["result_sha256"]
    reference_results = result_map(runners[reference_workers])
    evaluator = RecallEvaluator(str(gt_path))
    for workers in worker_counts:
        candidates = result_map(runners[workers])
        exact_equal = candidates == reference_results
        metrics = evaluator.evaluate(candidates)
        report["workers"][str(workers)]["equivalent_to_reference"] = exact_equal
        report["workers"][str(workers)]["digest_matches_reference"] = (
            report["workers"][str(workers)]["result_sha256"] == reference_digest
        )
        report["workers"][str(workers)]["recall"] = metrics
        if not exact_equal:
            raise RuntimeError(
                f"Candidate output from {workers} workers differs from "
                f"{reference_workers}-worker reference"
            )

    direct_count = min(args.direct_equivalence_sample, len(records))
    direct_failures = []
    with OptimizedCandidateGenerator(
        db_path,
        high_df_threshold=config["retrieval"].get("high_df_threshold", 50_000),
        posting_cache_mb=args.posting_cache_mb,
        sqlite_cache_mb=args.sqlite_cache_mb,
        mmap_mb=args.mmap_mb,
    ) as direct:
        for entity_id, name, address, country in records[:direct_count]:
            expected = direct.get_candidates(name, address, country, args.budget)
            if expected != reference_results[entity_id]:
                direct_failures.append(entity_id)
    report["direct_equivalence"] = {
        "sample_size": direct_count,
        "passed": not direct_failures,
        "failure_ids": direct_failures,
    }
    if direct_failures:
        raise RuntimeError(
            "Parallel output differs from direct optimized retrieval for: "
            + ", ".join(direct_failures[:10])
        )

    index_after = db_path.stat()
    report["metadata"].update(
        {
            "index_size_bytes_after": index_after.st_size,
            "index_mtime_ns_after": index_after.st_mtime_ns,
            "index_unchanged": (
                index_after.st_size == index_before.st_size
                and index_after.st_mtime_ns == index_before.st_mtime_ns
            ),
        }
    )
    report_path = benchmark_root / "parallel_b512_results.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark report to {report_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
