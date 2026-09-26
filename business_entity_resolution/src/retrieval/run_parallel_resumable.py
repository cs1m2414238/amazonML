"""Run resumable candidate retrieval over train or test Source 1 records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.retrieval.parallel_resumable_retrieval import (
    ParallelResumableRetriever,
    RunSettings,
    WorkerSettings,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("train", "test"), default="test")
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--posting-cache-mb", type=int, default=1024)
    parser.add_argument("--sqlite-cache-mb", type=int, default=512)
    parser.add_argument("--mmap-mb", type=int, default=8192)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--candidate-tsv",
        help="Optionally export the completed exact model-input candidate set",
    )
    return parser.parse_args()


def file_signature(path: Path, sample_size: int | None) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{sample_size}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def iter_batches(path: Path, batch_size: int, limit: int | None):
    batch = []
    batch_id = 0
    seen = 0
    for chunk in read_tsv_chunks(path, "source1"):
        for row in chunk.itertuples(index=False):
            if limit is not None and seen >= limit:
                break
            batch.append(
                (row.entity_id, row.business_name, row.business_address, row.country)
            )
            seen += 1
            if len(batch) == batch_size:
                yield batch_id, batch
                batch_id += 1
                batch = []
        if limit is not None and seen >= limit:
            break
    if batch:
        yield batch_id, batch


def main():
    args = parse_args()
    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    s1_path = data_root / f"dataset/{args.source}/{args.source}_source1.tsv"
    db_path = output_root / "step5/indices.db"

    if args.sample_size is None:
        with s1_path.open("r", encoding="utf-8") as stream:
            total_records = sum(1 for _ in stream) - 1
    else:
        total_records = args.sample_size
    total_batches = math.ceil(total_records / args.batch_size)

    run_name = args.run_name or (
        f"{args.source}_b{args.budget}_w{args.workers}_batch{args.batch_size}"
    )
    runner = ParallelResumableRetriever(
        output_root / "step7_parallel_retrieval" / run_name
    )
    worker_settings = WorkerSettings(
        db_path=str(db_path.resolve()),
        high_df_threshold=config["retrieval"].get("high_df_threshold", 50_000),
        posting_cache_mb=args.posting_cache_mb,
        sqlite_cache_mb=args.sqlite_cache_mb,
        mmap_mb=args.mmap_mb,
    )
    settings = RunSettings(
        budget=args.budget,
        workers=args.workers,
        batch_size=args.batch_size,
        dataset_signature=file_signature(s1_path, args.sample_size),
        total_records=total_records,
        total_batches=total_batches,
        worker=worker_settings,
    )
    result = runner.run(
        iter_batches(s1_path, args.batch_size, args.sample_size),
        settings,
        collect_latencies=False,
    )
    if args.candidate_tsv:
        result["candidate_tsv"] = runner.export_candidate_pairs_tsv(
            args.candidate_tsv
        )
    result.pop("latencies_seconds", None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    mp.freeze_support()
    main()
