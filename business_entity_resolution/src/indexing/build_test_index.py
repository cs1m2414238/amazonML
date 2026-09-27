"""Script to build the test index from test_source2.tsv and test_source3.tsv."""

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.indexing.indexer import SPIMIIndexer
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def build_test_index(
    output_db: str | Path = r"H:\Amazon_ML_Work\output\test_index\indices.db",
    num_shards: int = 64,
    chunk_size: int = 100_000,
):
    config = load_config()
    data_root = Path(config["data"]["root"])

    s2_path = data_root / "dataset/test/test_source2.tsv"
    s3_path = data_root / "dataset/test/test_source3.tsv"

    if not s2_path.exists() or not s3_path.exists():
        raise FileNotFoundError(f"Test files not found: {s2_path}, {s3_path}")

    files_to_index = [
        (s2_path, "source2"),
        (s3_path, "source3"),
    ]

    output_db = Path(output_db).resolve()
    output_db.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Building TEST index at {output_db} (num_shards={num_shards}, chunk_size={chunk_size})...")
    indexer = SPIMIIndexer(db_path=output_db, num_shards=num_shards, chunk_size=chunk_size)

    t0 = time.time()
    for fp, src in files_to_index:
        indexer.map_phase(fp, src)

    reduce_stats = indexer.reduce_phase()

    t1 = time.time()
    build_time = t1 - t0
    records_per_sec = indexer.doc_counter / build_time if build_time > 0 else 0
    db_size_mb = output_db.stat().st_size / (1024 * 1024)
    peak_mem_mb = get_peak_memory_mb()

    report = {
        "index_type": "test",
        "output_db": str(output_db),
        "environment": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
            "chunk_size": chunk_size,
            "num_shards": num_shards,
        },
        "build_time_seconds": round(build_time, 2),
        "total_records_indexed": indexer.doc_counter,
        "records_per_second": round(records_per_sec, 2),
        "peak_rss_mb": round(peak_mem_mb, 2),
        "index_disk_size_mb": round(db_size_mb, 2),
        "unique_key_counts": {
            "name_tokens": reduce_stats["unique_keys"][0],
            "name_ngrams": reduce_stats["unique_keys"][1],
            "address_tokens": reduce_stats["unique_keys"][2],
            "address_ngrams": reduce_stats["unique_keys"][3],
        },
        "missing_value_statistics": indexer.missing_stats,
    }

    report_path = output_db.parent / "test_index_build_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logging.info(f"Test index build complete! Saved report to {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Build test index from test_source2 and test_source3")
    parser.add_argument("--output-db", type=str, default=r"H:\Amazon_ML_Work\output\test_index\indices.db")
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()

    build_test_index(
        output_db=args.output_db,
        num_shards=args.num_shards,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
