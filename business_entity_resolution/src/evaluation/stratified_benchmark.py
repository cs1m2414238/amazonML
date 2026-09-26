"""Resumable B=256/B=512 benchmark on the fixed stratified sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sqlite3
import time
from pathlib import Path
from typing import Any

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.evaluation.candidate_recall import CandidateRecallEvaluator
from business_entity_resolution.src.evaluation.sampling import load_validation_sample
from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
DEFAULT_BUDGETS = (256, 512)


def entity_ids_sha256(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        "\n".join(str(record["entity_id"]) for record in records).encode("utf-8")
    ).hexdigest()


class RetrievalCheckpoint:
    """Atomic SQLite checkpoint for candidate lists and measured query latency."""

    def __init__(
        self,
        path: str | Path,
        sample_hash: str,
        max_budget: int,
        index_path: str | Path,
    ):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                position INTEGER PRIMARY KEY,
                entity_id TEXT NOT NULL UNIQUE,
                latency_seconds REAL NOT NULL,
                candidates_json TEXT NOT NULL
            )
            """
        )
        expected = {
            "sample_hash": sample_hash,
            "max_budget": str(max_budget),
            "index_path": str(Path(index_path).resolve()),
            "format_version": "1",
        }
        existing = dict(self.conn.execute("SELECT key, value FROM metadata"))
        if existing:
            mismatches = {
                key: {"expected": value, "found": existing.get(key)}
                for key, value in expected.items()
                if existing.get(key) != value
            }
            if mismatches:
                self.conn.close()
                raise ValueError(f"Checkpoint metadata mismatch: {mismatches}")
        else:
            self.conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items()
            )
            self.conn.execute(
                "INSERT INTO metadata(key, value) VALUES ('end_to_end_seconds', '0.0')"
            )
            self.conn.execute(
                "INSERT INTO metadata(key, value) VALUES ('peak_rss_mb', '0.0')"
            )
            self.conn.commit()

    def completed_positions(self) -> dict[int, str]:
        return dict(self.conn.execute("SELECT position, entity_id FROM results"))

    def write_batch(
        self,
        rows: list[tuple[int, str, float, str]],
        elapsed_seconds: float,
        peak_rss_mb: float,
    ) -> None:
        self.conn.executemany(
            """
            INSERT INTO results(position, entity_id, latency_seconds, candidates_json)
            VALUES (?, ?, ?, ?)
            """,
            rows,
        )
        previous_elapsed = float(
            self.conn.execute(
                "SELECT value FROM metadata WHERE key='end_to_end_seconds'"
            ).fetchone()[0]
        )
        previous_peak = float(
            self.conn.execute(
                "SELECT value FROM metadata WHERE key='peak_rss_mb'"
            ).fetchone()[0]
        )
        self.conn.execute(
            "UPDATE metadata SET value=? WHERE key='end_to_end_seconds'",
            (str(previous_elapsed + elapsed_seconds),),
        )
        self.conn.execute(
            "UPDATE metadata SET value=? WHERE key='peak_rss_mb'",
            (str(max(previous_peak, peak_rss_mb)),),
        )
        self.conn.commit()

    def load_results(self) -> tuple[dict[str, list[str]], list[float], dict[str, float]]:
        candidates: dict[str, list[str]] = {}
        latencies: list[float] = []
        for _, entity_id, latency, candidates_json in self.conn.execute(
            "SELECT position, entity_id, latency_seconds, candidates_json "
            "FROM results ORDER BY position"
        ):
            candidates[entity_id] = json.loads(candidates_json)
            latencies.append(float(latency))
        metadata = dict(self.conn.execute("SELECT key, value FROM metadata"))
        timing = {
            "end_to_end_seconds": float(metadata["end_to_end_seconds"]),
            "retrieval_latency_seconds": sum(latencies),
            "peak_rss_mb": float(metadata["peak_rss_mb"]),
        }
        return candidates, latencies, timing

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "RetrievalCheckpoint":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def validate_completed_positions(
    completed: dict[int, str], records: list[dict[str, Any]]
) -> None:
    for position, entity_id in completed.items():
        if position < 0 or position >= len(records):
            raise ValueError(f"Checkpoint position outside sample: {position}")
        expected_id = str(records[position]["entity_id"])
        if entity_id != expected_id:
            raise ValueError(
                f"Checkpoint/sample mismatch at position {position}: "
                f"{entity_id} != {expected_id}"
            )


def retrieve_with_checkpoint(
    records: list[dict[str, Any]],
    retriever: Any,
    checkpoint: RetrievalCheckpoint,
    max_budget: int,
    checkpoint_every: int = 25,
) -> tuple[dict[str, list[str]], list[float], dict[str, float]]:
    """Retrieve missing sample rows and commit progress in atomic batches."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    completed = checkpoint.completed_positions()
    validate_completed_positions(completed, records)
    logging.info("Resuming with %s/%s queries complete", len(completed), len(records))

    pending_rows: list[tuple[int, str, float, str]] = []
    interval_start = time.perf_counter()
    for position, record in enumerate(records):
        if position in completed:
            continue
        query_start = time.perf_counter()
        candidates = retriever.get_candidates(
            record.get("business_name", ""),
            record.get("business_address", ""),
            record.get("country", ""),
            budget=max_budget,
        )
        latency = time.perf_counter() - query_start
        pending_rows.append(
            (
                position,
                str(record["entity_id"]),
                latency,
                json.dumps(candidates, separators=(",", ":")),
            )
        )
        if len(pending_rows) >= checkpoint_every:
            interval_elapsed = time.perf_counter() - interval_start
            checkpoint.write_batch(
                pending_rows,
                elapsed_seconds=interval_elapsed,
                peak_rss_mb=get_peak_memory_mb(),
            )
            completed_count = len(completed) + len(pending_rows)
            completed.update({row[0]: row[1] for row in pending_rows})
            pending_rows.clear()
            cumulative = checkpoint.load_results()[2]["end_to_end_seconds"]
            logging.info(
                "Checkpointed %s/%s queries (%.2f end-to-end QPS)",
                completed_count,
                len(records),
                completed_count / cumulative if cumulative else 0.0,
            )
            interval_start = time.perf_counter()

    if pending_rows:
        checkpoint.write_batch(
            pending_rows,
            elapsed_seconds=time.perf_counter() - interval_start,
            peak_rss_mb=get_peak_memory_mb(),
        )
    return checkpoint.load_results()


def _sample_metrics(
    records: list[dict[str, Any]],
    all_candidates: dict[str, list[str]],
    latencies: list[float],
    timing: dict[str, float],
    budget: int,
) -> dict[str, Any]:
    gt_map = {
        str(record["entity_id"]): record.get("ground_truth_matches", [])
        for record in records
    }
    sliced = {
        entity_id: candidates[:budget]
        for entity_id, candidates in all_candidates.items()
    }
    evaluator = CandidateRecallEvaluator(gt_map=gt_map)
    evaluated = evaluator.evaluate(
        retrieved_candidates=sliced,
        sample_metadata=records,
        runtime_seconds=timing["end_to_end_seconds"],
        latencies=latencies,
        peak_memory_mb=timing["peak_rss_mb"],
    )
    return {
        "sample_metrics": evaluated["representative_metrics"],
        "diagnostic_strata": evaluated["diagnostic_strata"],
    }


def generate_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stratified blocking benchmark",
        "",
        "Candidate recall is a blocking metric. It is not matcher macro-F0.5.",
        "",
        "## Sample",
        "",
        f"- Entities: {report['metadata']['sample_size']:,}",
        f"- Entity ID SHA-256: `{report['metadata']['entity_ids_sha256']}`",
        f"- Index: `{report['metadata']['index_path']}` (opened read-only)",
        f"- Missing-address training S1 population: {report['metadata']['missing_address_population']:,}",
        "",
        "The sample intentionally over-represents rare match cardinalities. Aggregate values are diagnostic sample metrics.",
        "",
        "## B=256 versus B=512",
        "",
        "| Budget | Pair recall | Complete-set recall | S2 recall | S3 recall | Avg candidates | P95 | Max |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in report["budgets"]:
        metrics = report["budgets"][budget]["sample_metrics"]
        counts = metrics["candidate_counts"]
        lines.append(
            f"| {budget[2:]} | {metrics['pair_level_recall']:.4f} | "
            f"{metrics['complete_match_set_recall']:.4f} | {metrics['source2_recall']:.4f} | "
            f"{metrics['source3_recall']:.4f} | {counts['mean']:.2f} | "
            f"{counts['p95']} | {counts['max']} |"
        )
    performance = report["performance"]
    lines.extend(
        [
            "",
            "## Performance",
            "",
            f"- End-to-end retrieval/checkpoint runtime: {performance['end_to_end_seconds']:.2f} seconds",
            f"- End-to-end throughput: {performance['queries_per_second']:.2f} QPS",
            f"- Retrieval-call throughput: {performance['retrieval_queries_per_second']:.2f} QPS",
            f"- Peak RSS: {performance['peak_rss_mb']:.2f} MiB",
            "",
            "The macro-F0.5 evaluator must be run later on final matcher predictions, including empty predictions for true singletons.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark(
    sample_path: str | Path,
    index_path: str | Path,
    output_dir: str | Path,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    max_queries: int | None = None,
    checkpoint_every: int = 25,
) -> dict[str, Any]:
    sample = load_validation_sample(sample_path)
    records = sample["records"]
    if max_queries is not None:
        records = records[:max_queries]
    if not records:
        raise ValueError("Validation sample contains no records")

    index_path = Path(index_path).resolve()
    if not index_path.exists():
        raise FileNotFoundError(f"Retrieval index not found: {index_path}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_hash = entity_ids_sha256(records)
    max_budget = max(budgets)
    checkpoint_path = output_dir / (
        f"stratified_b{max_budget}_{len(records)}_{sample_hash[:12]}.checkpoint.db"
    )

    config = load_config()
    retrieval = config.get("retrieval", {})
    optimized = retrieval.get("optimized", {})
    with RetrievalCheckpoint(
        checkpoint_path,
        sample_hash=sample_hash,
        max_budget=max_budget,
        index_path=index_path,
    ) as checkpoint, OptimizedCandidateGenerator(
        db_path=index_path,
        high_df_threshold=retrieval.get("high_df_threshold", 50_000),
        posting_cache_mb=optimized.get("posting_cache_mb", 1024),
        sqlite_cache_mb=optimized.get("sqlite_cache_mb", 512),
        mmap_mb=optimized.get("mmap_mb", 8192),
    ) as retriever:
        all_candidates, latencies, timing = retrieve_with_checkpoint(
            records=records,
            retriever=retriever,
            checkpoint=checkpoint,
            max_budget=max_budget,
            checkpoint_every=checkpoint_every,
        )

    if len(all_candidates) != len(records):
        raise RuntimeError(
            f"Checkpoint has {len(all_candidates)} results for {len(records)} sample records"
        )
    query_count = len(records)
    report = {
        "metadata": {
            "sample_path": str(Path(sample_path).resolve()),
            "sample_size": query_count,
            "entity_ids_sha256": sample_hash,
            "sampling_method": sample["metadata"].get("selection_method"),
            "primary_strata": sample["metadata"].get("realized_primary_strata", {}),
            "diagnostic_strata": sample["metadata"].get("diagnostic_strata_counts", {}),
            "missing_address_population": sample["metadata"].get("missing_address_population", 0),
            "index_path": str(index_path),
            "index_open_mode": "read_only_immutable",
            "checkpoint_path": str(checkpoint_path),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "performance": {
            **timing,
            "queries_per_second": round(
                query_count / timing["end_to_end_seconds"], 4
                if timing["end_to_end_seconds"]
                else 0.0
            ),
            "retrieval_queries_per_second": round(
                query_count / timing["retrieval_latency_seconds"], 4
                if timing["retrieval_latency_seconds"]
                else 0.0
            ),
        },
        "budgets": {},
    }
    for budget in budgets:
        report["budgets"][f"B_{budget}"] = _sample_metrics(
            records,
            all_candidates,
            latencies,
            timing,
            budget,
        )

    suffix = f"{query_count}_{sample_hash[:12]}"
    json_path = output_dir / f"stratified_blocking_benchmark_{suffix}.json"
    md_path = output_dir / f"stratified_blocking_benchmark_{suffix}.md"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    with md_path.open("w", encoding="utf-8") as stream:
        stream.write(generate_markdown(report))
    logging.info("Saved benchmark report to %s", json_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resumable stratified blocking benchmark")
    parser.add_argument("--sample", type=str)
    parser.add_argument("--index-path", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--budgets", type=str, default="256,512")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    config = load_config()
    output_root = Path(config["output"]["directory"])
    sample_path = Path(args.sample) if args.sample else (
        output_root / "validation_stratified/validation_stratified_10000.json"
    )
    index_path = Path(args.index_path) if args.index_path else output_root / "step5/indices.db"
    output_dir = Path(args.output_dir) if args.output_dir else output_root / "validation_stratified"
    budgets = tuple(int(item.strip()) for item in args.budgets.split(",") if item.strip())
    run_benchmark(
        sample_path=sample_path,
        index_path=index_path,
        output_dir=output_dir,
        budgets=budgets,
        max_queries=args.max_queries,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
