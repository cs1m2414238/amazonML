"""End-to-end inference pipeline for business entity resolution.

Integrates Agent 1 candidate retrieval with Agent 3 LightGBM pairwise BatchMatcher.
Supports batch chunking, target hydration, threshold filtering, and official TSV export.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.features.generate_training_data import hydrate_target_records
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.models.matcher_inference import BatchMatcher
from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class EndToEndPipeline:
    """Coordinator running retrieval, pair feature generation, and model prediction."""

    def __init__(
        self,
        model_path: str | Path = "output/models/lightgbm_matcher.txt",
        feature_schema_path: str | Path | None = None,
        threshold: float = 0.60,
        budget: int = 64,
        db_path: str | Path | None = None,
        use_fallback: bool = False,
    ):
        config = load_config()
        output_dir = Path(config["output"]["directory"])

        if db_path is None:
            self.db_path = output_dir / "step5/indices.db"
            if not self.db_path.exists():
                self.db_path = Path(r"H:/Amazon_ML_Work/output/step5/indices.db")
        else:
            self.db_path = Path(db_path).resolve()

        self.budget = int(budget)
        self.threshold = float(threshold)

        model_p = Path(model_path).resolve()
        if use_fallback:
            fallback_p = model_p.parent / "lightgbm_matcher_fallback.txt"
            if fallback_p.exists():
                model_p = fallback_p
                logging.info(f"Using fallback model: {model_p}")
            else:
                logging.warning(f"Fallback model not found at {fallback_p}, using {model_p}")

        self.matcher = BatchMatcher(
            model_path=model_p,
            feature_schema_path=feature_schema_path,
            threshold=self.threshold,
        )

        high_df = config.get("retrieval", {}).get("high_df_threshold", 50000)
        self.retriever = OptimizedCandidateGenerator(
            db_path=self.db_path,
            high_df_threshold=high_df,
            posting_cache_mb=1024,
            sqlite_cache_mb=512,
            mmap_mb=8192,
        )

    def run_inference(
        self,
        s1_records: list[dict[str, Any]],
        s2_path: str | Path,
        s3_path: str | Path,
        cache_path: str | Path | None = None,
        batch_size: int = 10000,
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        """Execute full retrieval and matching over a collection of S1 records."""
        query_count = len(s1_records)
        logging.info(f"Starting end-to-end inference on {query_count} queries (B={self.budget}, tau={self.threshold})...")

        # 1. Candidate Retrieval
        t0_retr = time.perf_counter()
        candidates_by_s1: dict[str, list[str]] = {}
        all_targets: set[str] = set()

        for idx, s1 in enumerate(s1_records):
            s1_id = s1.get("id") or s1.get("entity_id")
            cands = self.retriever.get_candidates(
                s1.get("name") or s1.get("business_name") or "",
                s1.get("addr") or s1.get("business_address") or "",
                s1.get("country") or "",
                budget=self.budget,
            )
            candidates_by_s1[s1_id] = cands
            all_targets.update(cands)

        t_retr = time.perf_counter() - t0_retr
        retr_qps = query_count / t_retr if t_retr > 0 else 0.0
        logging.info(f"Retrieval completed in {t_retr:.2f}s ({retr_qps:.2f} QPS). Total unique candidate targets: {len(all_targets)}")

        # 2. Target Hydration
        t0_hydr = time.perf_counter()
        target_records = hydrate_target_records(
            target_ids=all_targets,
            s2_path=s2_path,
            s3_path=s3_path,
            cache_path=cache_path,
        )
        t_hydr = time.perf_counter() - t0_hydr
        logging.info(f"Hydration completed in {t_hydr:.2f}s ({len(target_records)} hydrated records)")

        # 3. Model Scoring & Threshold Filtering
        t0_score = time.perf_counter()
        predictions = self.matcher.predict_batch(
            s1_records=s1_records,
            candidate_records=target_records,
            candidates_by_s1=candidates_by_s1,
            threshold=self.threshold,
            batch_size=batch_size,
        )
        t_score = time.perf_counter() - t0_score
        score_qps = query_count / t_score if t_score > 0 else 0.0

        total_pairs = sum(len(cands) for cands in candidates_by_s1.values())
        pairs_per_sec = total_pairs / t_score if t_score > 0 else 0.0
        logging.info(f"Scoring completed in {t_score:.2f}s ({score_qps:.2f} query QPS, {pairs_per_sec:.2f} pairs/s)")

        t_total = t_retr + t_hydr + t_score
        end_to_end_qps = query_count / t_total if t_total > 0 else 0.0

        stats = {
            "query_count": query_count,
            "candidate_pairs": total_pairs,
            "retrieval_seconds": t_retr,
            "retrieval_qps": retr_qps,
            "hydration_seconds": t_hydr,
            "scoring_seconds": t_score,
            "scoring_qps": score_qps,
            "scoring_pairs_per_sec": pairs_per_sec,
            "total_seconds": t_total,
            "end_to_end_qps": end_to_end_qps,
            "peak_memory_mb": get_peak_memory_mb(),
            "threshold": self.threshold,
            "budget": self.budget,
        }

        return predictions, stats

    def close(self):
        if hasattr(self, "retriever") and self.retriever is not None:
            self.retriever.close()


def run_benchmark(
    sample_size: int = 100,
    budget: int = 64,
    threshold: float = 0.60,
    source: str = "train",
) -> dict[str, Any]:
    """Run an end-to-end benchmark measuring exact inference throughput."""
    config = load_config()
    data_root = Path(config["data"]["root"])
    s1_path = data_root / f"dataset/{source}/{source}_source1.tsv"
    s2_path = data_root / f"dataset/{source}/{source}_source2.tsv"
    s3_path = data_root / f"dataset/{source}/{source}_source3.tsv"

    logging.info(f"Loading {sample_size} S1 records from {s1_path} for benchmark...")
    records = []
    for chunk in read_tsv_chunks(s1_path, "source1", chunk_size=5000):
        for row in chunk.itertuples(index=False):
            records.append({
                "id": str(row.entity_id).strip(),
                "name": str(row.business_name).strip() if pd.notna(row.business_name) else "",
                "addr": str(row.business_address).strip() if pd.notna(row.business_address) else "",
                "country": str(row.country).strip() if pd.notna(row.country) else "",
            })
            if len(records) >= sample_size:
                break
        if len(records) >= sample_size:
            break

    pipeline = EndToEndPipeline(
        budget=budget,
        threshold=threshold,
    )
    try:
        cache_p = Path("output/models/cached_val_target_records.json")
        preds, stats = pipeline.run_inference(
            s1_records=records,
            s2_path=s2_path,
            s3_path=s3_path,
            cache_path=cache_p if cache_p.exists() else None,
        )
    finally:
        pipeline.close()

    logging.info("=== BENCHMARK RESULTS ===")
    logging.info(f"Queries: {stats['query_count']}")
    logging.info(f"Retrieval QPS: {stats['retrieval_qps']:.2f}")
    logging.info(f"Scoring QPS: {stats['scoring_qps']:.2f} ({stats['scoring_pairs_per_sec']:.2f} pairs/sec)")
    logging.info(f"End-to-End QPS: {stats['end_to_end_qps']:.2f}")
    logging.info(f"Peak RSS: {stats['peak_memory_mb']:.2f} MB")

    return stats


def main():
    parser = argparse.ArgumentParser(description="End-to-end inference and matching pipeline")
    parser.add_argument("--source", choices=("train", "test"), default="test")
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--model-path", type=str, default="output/models/lightgbm_matcher.txt")
    parser.add_argument("--output-tsv", type=str, default="output/matching_results.tsv")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--use-fallback", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        size = args.sample_size or 100
        run_benchmark(sample_size=size, budget=args.budget, threshold=args.threshold, source=args.source)
        return

    config = load_config()
    data_root = Path(config["data"]["root"])
    s1_path = data_root / f"dataset/{args.source}/{args.source}_source1.tsv"
    s2_path = data_root / f"dataset/{args.source}/{args.source}_source2.tsv"
    s3_path = data_root / f"dataset/{args.source}/{args.source}_source3.tsv"

    s1_records = []
    for chunk in read_tsv_chunks(s1_path, "source1", chunk_size=50000):
        for row in chunk.itertuples(index=False):
            s1_records.append({
                "id": str(row.entity_id).strip(),
                "name": str(row.business_name).strip() if pd.notna(row.business_name) else "",
                "addr": str(row.business_address).strip() if pd.notna(row.business_address) else "",
                "country": str(row.country).strip() if pd.notna(row.country) else "",
            })
            if args.sample_size and len(s1_records) >= args.sample_size:
                break
        if args.sample_size and len(s1_records) >= args.sample_size:
            break

    pipeline = EndToEndPipeline(
        model_path=args.model_path,
        threshold=args.threshold,
        budget=args.budget,
        use_fallback=args.use_fallback,
    )

    try:
        preds, stats = pipeline.run_inference(
            s1_records=s1_records,
            s2_path=s2_path,
            s3_path=s3_path,
        )
        BatchMatcher.format_tsv(preds, args.output_tsv)
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
