"""End-to-end production inference pipeline for business entity resolution.

Integrates:
- Multi-process parallel resumable candidate retrieval (ParallelResumableRetriever)
- Instant O(1) indexed SQLite entity hydration
- Vectorized LightGBM pairwise BatchMatcher scoring
- Bounded-memory chunked streaming with full resumption checkpointing
- Synchronized dual-TSV generation (candidate_pairs.tsv and matching_results.tsv)
- Strict validation of submission invariants (output_validator.py)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import pandas as pd

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.features.generate_training_data import hydrate_target_records
from business_entity_resolution.src.indexing.sqlite_hydration import hydrate_records_from_db
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.models.matcher_inference import BatchMatcher
from business_entity_resolution.src.preprocessing import CountryNormalizer
from business_entity_resolution.src.retrieval.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)
from business_entity_resolution.src.retrieval.parallel_resumable_retrieval import (
    ParallelResumableRetriever,
    Record,
    RecordBatch,
    RunSettings,
    WorkerSettings,
)
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb
from business_entity_resolution.src.validation.output_validator import validate_tsv_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def verify_index_isolation(source: str, db_path: str | Path) -> None:
    """Guarantee that test inference strictly isolates from the training index."""
    idx_p = Path(db_path).resolve()
    training_patterns = ["step5", "train_index", "train"]

    if source == "test":
        # Check against known training index path
        if "step5" in idx_p.parts or "step5\\indices.db" in str(idx_p):
            raise ValueError(
                f"CRITICAL SAFETY VIOLATION: Test inference attempted to open the training index at {idx_p}!"
            )
        if not idx_p.exists():
            raise FileNotFoundError(f"Test SQLite index not found at {idx_p}")
    logging.info(f"Verified index isolation for source '{source}': using {idx_p}")


def stream_s1_batches(
    s1_path: str | Path,
    batch_size: int,
    limit: int | None = None,
) -> Iterator[RecordBatch]:
    """Stream S1 records in bounded chunks without loading the full file into RAM."""
    s1_path = Path(s1_path).resolve()
    current_batch: list[Record] = []
    batch_id = 0
    total_yielded = 0

    for chunk in read_tsv_chunks(s1_path, "source1", chunk_size=max(batch_size, 50_000)):
        for row in chunk.itertuples(index=False):
            eid = str(row.entity_id).strip()
            name = str(row.business_name).strip() if pd.notna(row.business_name) else ""
            addr = str(row.business_address).strip() if pd.notna(row.business_address) else ""
            country = str(row.country).strip() if pd.notna(row.country) else ""

            current_batch.append((eid, name, addr, country))
            total_yielded += 1

            if len(current_batch) >= batch_size:
                yield batch_id, current_batch
                batch_id += 1
                current_batch = []

            if limit is not None and total_yielded >= limit:
                break
        if limit is not None and total_yielded >= limit:
            break

    if current_batch:
        yield batch_id, current_batch


class EndToEndPipeline:
    """High-throughput coordinator for parallel candidate retrieval and LightGBM matching."""

    def __init__(
        self,
        model_path: str | Path = "output/models/lightgbm_matcher.txt",
        feature_schema_path: str | Path | None = None,
        threshold: float = 0.60,
        budget: int = 64,
        db_path: str | Path | None = None,
        use_fallback: bool = False,
        workers: int = 4,
        run_dir: str | Path | None = None,
    ):
        config = load_config()
        output_dir = Path(config["output"]["directory"])

        if db_path is None:
            self.db_path = output_dir / "test_index/indices.db"
            if not self.db_path.exists():
                self.db_path = Path(r"H:/Amazon_ML_Work/output/test_index/indices.db")
        else:
            self.db_path = Path(db_path).resolve()

        self.budget = int(budget)
        self.threshold = float(threshold)
        self.workers = int(workers)

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

        if run_dir is None:
            self.run_dir = output_dir / "inference_run"
        else:
            self.run_dir = Path(run_dir).resolve()

        self.retrieval_dir = self.run_dir / "retrieval"
        self.scoring_dir = self.run_dir / "scoring"
        self.score_parts_dir = self.scoring_dir / "parts"

    def close(self) -> None:
        """Cleanup resources."""
        pass

    def run_chunked_inference(
        self,
        s1_path: str | Path,
        total_records: int,
        batch_size: int = 5000,
        candidate_tsv: str | Path = "output/candidate_pairs.tsv",
        matching_tsv: str | Path = "output/matching_results.tsv",
        sample_limit: int | None = None,
        source: str = "test",
    ) -> dict[str, Any]:
        """Execute parallel, resumable end-to-end inference in bounded chunks."""
        verify_index_isolation(source, self.db_path)

        candidate_tsv = Path(candidate_tsv).resolve()
        matching_tsv = Path(matching_tsv).resolve()
        candidate_tsv.parent.mkdir(parents=True, exist_ok=True)
        matching_tsv.parent.mkdir(parents=True, exist_ok=True)

        effective_records = min(total_records, sample_limit) if sample_limit else total_records
        total_batches = math.ceil(effective_records / batch_size)

        logging.info(
            f"Starting chunked end-to-end inference: {effective_records:,} queries in {total_batches} batches "
            f"(B={self.budget}, tau={self.threshold}, workers={self.workers}, batch_size={batch_size})"
        )

        overall_start = time.perf_counter()

        # =====================================================================
        # STAGE 1 & 2: Controlled Concurrent Retrieval (3 workers) & Scoring (1 worker)
        # =====================================================================
        retriever = ParallelResumableRetriever(self.retrieval_dir)
        self.score_parts_dir.mkdir(parents=True, exist_ok=True)

        retrieval_error: list[BaseException] = []
        retrieval_stats: dict[str, Any] = {}

        def _run_retrieval() -> None:
            try:
                worker_settings = WorkerSettings(
                    db_path=str(self.db_path),
                    high_df_threshold=50000,
                    posting_cache_mb=1024,
                    sqlite_cache_mb=512,
                    mmap_mb=8192,
                )
                run_settings = RunSettings(
                    budget=self.budget,
                    workers=self.workers,
                    batch_size=batch_size,
                    dataset_signature=f"{source}_s1_{effective_records}",
                    total_records=effective_records,
                    total_batches=total_batches,
                    worker=worker_settings,
                )
                batches_iter = stream_s1_batches(s1_path, batch_size=batch_size, limit=sample_limit)
                stats = retriever.run(batches_iter, run_settings)
                retrieval_stats.update(stats)
            except BaseException as exc:
                logging.error(f"Retrieval worker error: {exc}", exc_info=True)
                retrieval_error.append(exc)

        # Launch 3-worker retrieval concurrently in background
        retrieval_thread = threading.Thread(
            target=_run_retrieval, name="RetrievalWorkerThread", daemon=True
        )
        retrieval_thread.start()

        # Helper checkpoint checkers
        def is_score_chunk_complete(b_id: int, rec_count: int) -> tuple[bool, int, int]:
            cand_p = self.score_parts_dir / f"score-part-{b_id:06d}.candidates.tsv"
            match_p = self.score_parts_dir / f"score-part-{b_id:06d}.matching.tsv"
            meta_p = self.score_parts_dir / f"score-part-{b_id:06d}.meta.json"
            if cand_p.exists() and match_p.exists() and meta_p.exists():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    if meta.get("batch_id") == b_id and meta.get("record_count") == rec_count:
                        return True, int(meta.get("candidate_total", 0)), int(meta.get("matching_total", 0))
                except Exception:
                    pass
            return False, 0, 0

        def is_retrieval_chunk_complete(b_id: int, rec_count: int) -> bool:
            part_p, meta_p = retriever._part_paths(b_id)
            if part_p.exists() and meta_p.exists():
                try:
                    meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    if meta.get("batch_id") == b_id and meta.get("record_count") == rec_count:
                        return True
                except Exception:
                    pass
            return False

        # Group 4 chunks (10,000 queries) into a logical scoring batch
        group_size = max(1, 10000 // batch_size)
        total_groups = math.ceil(total_batches / group_size)

        logging.info(
            f"Controlled Concurrent Pipeline: 1 coordinator/scoring worker + {self.workers} retrieval workers. "
            f"Logical scoring batch size: {group_size * batch_size:,} ({group_size} chunks/group, {total_groups} groups)."
        )

        def stream_chunk_groups(
            batches_it: Iterator[RecordBatch], grp_sz: int
        ) -> Iterator[list[RecordBatch]]:
            curr: list[RecordBatch] = []
            for b_id, recs in batches_it:
                curr.append((b_id, recs))
                if len(curr) >= grp_sz:
                    yield curr
                    curr = []
            if curr:
                yield curr

        t0_scoring_stage = time.perf_counter()
        total_candidate_pairs = 0
        total_matched_pairs = 0
        completed_records_scored = 0
        hydration_times: list[float] = []
        scoring_times: list[float] = []

        s1_batches_for_scoring = stream_s1_batches(s1_path, batch_size=batch_size, limit=sample_limit)
        chunk_groups_iter = stream_chunk_groups(s1_batches_for_scoring, group_size)

        for group_idx, group in enumerate(chunk_groups_iter):
            chunks_to_score: list[RecordBatch] = []
            for b_id, recs in group:
                done, cand_tot, match_tot = is_score_chunk_complete(b_id, len(recs))
                if done:
                    total_candidate_pairs += cand_tot
                    total_matched_pairs += match_tot
                    completed_records_scored += len(recs)
                else:
                    chunks_to_score.append((b_id, recs))

            if not chunks_to_score:
                continue

            # Wait until all chunks in chunks_to_score have their retrieval parts ready
            while True:
                if retrieval_error:
                    raise RuntimeError(
                        f"Retrieval worker failed: {retrieval_error[0]}"
                    ) from retrieval_error[0]

                if all(is_retrieval_chunk_complete(bid, len(recs)) for bid, recs in chunks_to_score):
                    break

                if not retrieval_thread.is_alive():
                    if retrieval_error:
                        raise RuntimeError(
                            f"Retrieval thread stopped with error: {retrieval_error[0]}"
                        ) from retrieval_error[0]
                    if all(is_retrieval_chunk_complete(bid, len(recs)) for bid, recs in chunks_to_score):
                        break
                    raise RuntimeError(
                        "Retrieval thread terminated unexpectedly before producing required part."
                    )

                time.sleep(2.0)

            # Read candidate IDs from retrieved parts
            candidates_by_s1_group: dict[str, list[str]] = {}
            all_group_candidate_ids: set[str] = set()

            for b_id, recs in chunks_to_score:
                retr_part_path, _ = retriever._part_paths(b_id)
                with open(retr_part_path, "r", encoding="utf-8") as f_part:
                    for line in f_part:
                        line_str = line.strip()
                        if line_str:
                            item = json.loads(line_str)
                            s1_id = item["source1_entity_id"]
                            cands = item["candidates"]
                            candidates_by_s1_group[s1_id] = cands
                            all_group_candidate_ids.update(cands)

            # Hydrate target records from SQLite index
            t0_hydr = time.perf_counter()
            try:
                hydrated_targets = hydrate_records_from_db(self.db_path, all_group_candidate_ids)
            except Exception as e:
                if "no such column" in str(e).lower() and source == "train":
                    logging.warning(
                        "Index does not have name/addr in documents table. Using TSV hydration for train."
                    )
                    config = load_config()
                    data_root = Path(config["data"]["root"])
                    s2_path = data_root / f"dataset/{source}/{source}_source2.tsv"
                    s3_path = data_root / f"dataset/{source}/{source}_source3.tsv"
                    hydrated_targets = hydrate_target_records(all_group_candidate_ids, s2_path, s3_path)
                else:
                    raise
            t_hydr = time.perf_counter() - t0_hydr
            hydration_times.append(t_hydr)

            # Score candidate pairs using BatchMatcher
            group_s1_dicts = [
                {"id": r[0], "name": r[1], "addr": r[2], "country": r[3]}
                for _, recs in chunks_to_score
                for r in recs
            ]
            t0_score = time.perf_counter()
            scored_pairs = self.matcher.score_pairs_batch(
                s1_records=group_s1_dicts,
                candidate_records=hydrated_targets,
                candidates_by_s1=candidates_by_s1_group,
                batch_size=10000,
            )
            t_score = time.perf_counter() - t0_score
            scoring_times.append(t_score)

            group_scored_records = len(group_s1_dicts)
            group_qps = group_scored_records / t_score if t_score > 0 else 0.0

            # Atomically write each chunk's score part
            for b_id, recs in chunks_to_score:
                cand_part_path = self.score_parts_dir / f"score-part-{b_id:06d}.candidates.tsv"
                match_part_path = self.score_parts_dir / f"score-part-{b_id:06d}.matching.tsv"
                meta_part_path = self.score_parts_dir / f"score-part-{b_id:06d}.meta.json"

                cand_lines: list[str] = []
                match_lines: list[str] = []
                batch_cand_total = 0
                batch_match_total = 0

                for eid, _, _, _ in recs:
                    cands = candidates_by_s1_group.get(eid, [])
                    batch_cand_total += len(cands)
                    cand_lines.append(f"{eid}\t{','.join(cands)}\n")

                    pair_scores = scored_pairs.get(eid, [])
                    score_map = {cid: sc for cid, sc in pair_scores}
                    matched = [cid for cid in cands if score_map.get(cid, 0.0) >= self.threshold]
                    batch_match_total += len(matched)
                    match_lines.append(f"{eid}\t{','.join(matched)}\n")

                total_candidate_pairs += batch_cand_total
                total_matched_pairs += batch_match_total
                completed_records_scored += len(recs)

                # Atomic write of candidate part
                tmp_cand = cand_part_path.with_name(f".{cand_part_path.name}.{os.getpid()}.tmp")
                with open(tmp_cand, "w", encoding="utf-8", newline="\n") as f:
                    f.writelines(cand_lines)
                os.replace(tmp_cand, cand_part_path)

                # Atomic write of matching part
                tmp_match = match_part_path.with_name(f".{match_part_path.name}.{os.getpid()}.tmp")
                with open(tmp_match, "w", encoding="utf-8", newline="\n") as f:
                    f.writelines(match_lines)
                os.replace(tmp_match, match_part_path)

                chunk_frac = len(recs) / max(1, group_scored_records)
                batch_meta = {
                    "batch_id": b_id,
                    "record_count": len(recs),
                    "candidate_total": batch_cand_total,
                    "matching_total": batch_match_total,
                    "hydration_seconds": t_hydr * chunk_frac,
                    "scoring_seconds": t_score * chunk_frac,
                }
                tmp_meta = meta_part_path.with_name(f".{meta_part_path.name}.{os.getpid()}.tmp")
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump(batch_meta, f, indent=2)
                os.replace(tmp_meta, meta_part_path)

            logging.info(
                f"Scoring progress: group {group_idx + 1}/{total_groups} complete "
                f"({completed_records_scored:,}/{effective_records:,} queries, "
                f"{total_candidate_pairs:,} cand pairs, {total_matched_pairs:,} matches). "
                f"Scored {group_scored_records:,} queries in {t_score:.2f}s ({group_qps:.1f} QPS)"
            )

            del group_s1_dicts, hydrated_targets, candidates_by_s1_group, scored_pairs
            gc.collect()

        # Ensure retrieval worker thread completes cleanly
        retrieval_thread.join()
        if retrieval_error:
            raise RuntimeError(f"Retrieval worker thread failed: {retrieval_error[0]}") from retrieval_error[0]

        t_retrieval = retrieval_stats.get("end_to_end_seconds", 0.0)
        retr_qps = effective_records / t_retrieval if t_retrieval > 0 else 0.0
        logging.info(
            f"Stage 1 Retrieval complete: {effective_records:,} records in {t_retrieval:.2f}s "
            f"({retr_qps:.2f} QPS, resumed {retrieval_stats.get('resumed_batches', 0)} batches)"
        )

        t_scoring_stage = time.perf_counter() - t0_scoring_stage
        score_qps = effective_records / t_scoring_stage if t_scoring_stage > 0 else 0.0
        pairs_per_sec = total_candidate_pairs / sum(scoring_times) if sum(scoring_times) > 0 else 0.0

        logging.info(
            f"Stage 2 Scoring complete: {total_candidate_pairs:,} pairs scored in {t_scoring_stage:.2f}s "
            f"({score_qps:.2f} query QPS, {pairs_per_sec:.2f} pairs/s)"
        )

        # =====================================================================
        # STAGE 3: Synchronous Assembly of Output TSVs
        # =====================================================================
        t0_assemble = time.perf_counter()
        logging.info(f"Assembling final submission files: {candidate_tsv} and {matching_tsv}...")

        tmp_cand_tsv = candidate_tsv.with_name(f".{candidate_tsv.name}.{os.getpid()}.tmp")
        tmp_match_tsv = matching_tsv.with_name(f".{matching_tsv.name}.{os.getpid()}.tmp")

        with open(tmp_cand_tsv, "w", encoding="utf-8", newline="\n") as f_cand_out, open(
            tmp_match_tsv, "w", encoding="utf-8", newline="\n"
        ) as f_match_out:
            f_cand_out.write("source1_entity_id\tcandidate_entity_ids\n")
            f_match_out.write("source1_entity_id\tmatched_entity_ids\n")

            for batch_id in range(total_batches):
                cand_part_p = self.score_parts_dir / f"score-part-{batch_id:06d}.candidates.tsv"
                match_part_p = self.score_parts_dir / f"score-part-{batch_id:06d}.matching.tsv"

                if not cand_part_p.exists() or not match_part_p.exists():
                    raise FileNotFoundError(f"Missing score part for batch {batch_id} during assembly!")

                with open(cand_part_p, "r", encoding="utf-8") as f_cp:
                    shutil.copyfileobj(f_cp, f_cand_out)
                with open(match_part_p, "r", encoding="utf-8") as f_mp:
                    shutil.copyfileobj(f_mp, f_match_out)

        os.replace(tmp_cand_tsv, candidate_tsv)
        os.replace(tmp_match_tsv, matching_tsv)
        t_assemble = time.perf_counter() - t0_assemble
        logging.info(f"Assembled final files in {t_assemble:.2f}s")

        # =====================================================================
        # STAGE 4: Strict Output Validation
        # =====================================================================
        logging.info("Validating generated TSV files with output_validator...")
        val_report = validate_tsv_files(
            candidate_tsv_path=candidate_tsv,
            matching_tsv_path=matching_tsv,
            expected_count=effective_records,
        )

        overall_seconds = time.perf_counter() - overall_start
        overall_qps = effective_records / overall_seconds if overall_seconds > 0 else 0.0
        peak_rss = get_peak_memory_mb()

        summary_report = {
            "query_count": effective_records,
            "total_candidate_pairs": total_candidate_pairs,
            "total_matched_pairs": total_matched_pairs,
            "retrieval_seconds": round(t_retrieval, 2),
            "retrieval_qps": round(retr_qps, 2),
            "hydration_total_seconds": round(sum(hydration_times), 4),
            "hydration_mean_ms_per_batch": round(
                (sum(hydration_times) / max(1, len(hydration_times))) * 1000, 2
            ),
            "scoring_seconds": round(sum(scoring_times), 2),
            "scoring_stage_seconds": round(t_scoring_stage, 2),
            "scoring_pairs_per_sec": round(pairs_per_sec, 2),
            "overall_seconds": round(overall_seconds, 2),
            "overall_qps": round(overall_qps, 2),
            "peak_memory_mb": round(peak_rss, 2),
            "workers": self.workers,
            "budget": self.budget,
            "threshold": self.threshold,
            "validation_status": val_report["status"],
            "candidate_file": str(candidate_tsv),
            "matching_file": str(matching_tsv),
        }

        logging.info("=== END-TO-END INFERENCE SUMMARY ===")
        logging.info(f"Processed: {effective_records:,} queries")
        logging.info(f"Retrieval QPS: {retr_qps:.2f} (workers={self.workers})")
        logging.info(f"Hydration Avg: {summary_report['hydration_mean_ms_per_batch']:.2f} ms/batch")
        logging.info(f"Scoring Speed: {pairs_per_sec:.2f} pairs/sec")
        logging.info(f"End-to-End QPS: {overall_qps:.2f}")
        logging.info(f"Peak Memory: {peak_rss:.2f} MB")
        logging.info(f"Validation: {val_report['status']}")

        return summary_report

    def run_inference(
        self,
        s1_records: list[dict[str, Any]],
        s2_path: str | Path | None = None,
        s3_path: str | Path | None = None,
        cache_path: str | Path | None = None,
        batch_size: int = 10000,
    ) -> tuple[dict[str, list[str]], dict[str, Any]]:
        """In-memory inference for unit testing and backward compatibility."""
        query_count = len(s1_records)
        logging.info(f"Starting in-memory inference on {query_count} queries (B={self.budget}, tau={self.threshold})...")

        high_df = 50000
        retriever = OptimizedCandidateGenerator(
            db_path=self.db_path,
            high_df_threshold=high_df,
            posting_cache_mb=1024,
            sqlite_cache_mb=512,
            mmap_mb=8192,
        )

        try:
            t0_retr = time.perf_counter()
            candidates_by_s1: dict[str, list[str]] = {}
            all_targets: set[str] = set()

            for s1 in s1_records:
                s1_id = s1.get("id") or s1.get("entity_id")
                cands = retriever.get_candidates(
                    s1.get("name") or s1.get("business_name") or "",
                    s1.get("addr") or s1.get("business_address") or "",
                    s1.get("country") or "",
                    budget=self.budget,
                )
                candidates_by_s1[s1_id] = cands
                all_targets.update(cands)

            t_retr = time.perf_counter() - t0_retr
            retr_qps = query_count / t_retr if t_retr > 0 else 0.0

            # Hydration: try SQLite first, fallback to TSV if not in DB
            t0_hydr = time.perf_counter()
            try:
                target_records = hydrate_records_from_db(self.db_path, all_targets)
            except Exception:
                if s2_path and s3_path:
                    target_records = hydrate_target_records(
                        target_ids=all_targets,
                        s2_path=s2_path,
                        s3_path=s3_path,
                        cache_path=cache_path,
                    )
                else:
                    raise
            t_hydr = time.perf_counter() - t0_hydr

            # Scoring
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
        finally:
            retriever.close()


def run_benchmark(
    sample_size: int = 500,
    budget: int = 64,
    threshold: float = 0.60,
    workers: int = 4,
    source: str = "test",
    index_path: str | Path | None = None,
    output_dir: str | Path = "output/benchmark",
) -> dict[str, Any]:
    """Run an end-to-end benchmark on test entities, explicitly verifying France entities."""
    config = load_config()
    data_root = Path(config["data"]["root"])
    s1_path = data_root / f"dataset/{source}/{source}_source1.tsv"

    if index_path is None:
        index_path = Path(r"H:/Amazon_ML_Work/output/test_index/indices.db")
    verify_index_isolation(source, index_path)

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect representative sample of test S1 records (including France queries)
    sample_records: list[dict[str, str]] = []
    france_records: list[dict[str, str]] = []

    target_france = max(10, sample_size // 10)
    target_general = sample_size - target_france

    logging.info(
        f"Sampling {sample_size} records from {s1_path} for benchmark "
        f"({target_france} France entities + {target_general} general entities)..."
    )
    for chunk in read_tsv_chunks(s1_path, "source1", chunk_size=20000):
        for row in chunk.itertuples(index=False):
            rec = {
                "entity_id": str(row.entity_id).strip(),
                "business_name": str(row.business_name).strip() if pd.notna(row.business_name) else "",
                "business_address": str(row.business_address).strip() if pd.notna(row.business_address) else "",
                "country": str(row.country).strip() if pd.notna(row.country) else "",
            }
            norm_c = CountryNormalizer.normalize(rec["country"])
            if norm_c == "france" and len(france_records) < target_france:
                france_records.append(rec)
            elif norm_c != "france" and len(sample_records) < target_general:
                sample_records.append(rec)

            if len(sample_records) >= target_general and len(france_records) >= target_france:
                break
        if len(sample_records) >= target_general and len(france_records) >= target_france:
            break

    all_benchmark_records = sample_records + france_records
    logging.info(
        f"Benchmark sample prepared: {len(all_benchmark_records)} total entities "
        f"({len(france_records)} France entities)."
    )

    # Write temporary benchmark sample S1 file
    bench_s1_path = output_dir / f"benchmark_sample_{sample_size}.tsv"
    with open(bench_s1_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for r in all_benchmark_records:
            f.write(f"{r['entity_id']}\t{r['business_name']}\t{r['business_address']}\t{r['country']}\n")

    bench_run_dir = output_dir / f"run_w{workers}"
    if bench_run_dir.exists():
        shutil.rmtree(bench_run_dir, ignore_errors=True)

    cand_tsv = output_dir / f"benchmark_candidate_pairs_w{workers}.tsv"
    match_tsv = output_dir / f"benchmark_matching_results_w{workers}.tsv"

    pipeline = EndToEndPipeline(
        budget=budget,
        threshold=threshold,
        db_path=index_path,
        workers=workers,
        run_dir=bench_run_dir,
    )

    bench_batch_size = max(50, sample_size // (workers * 2))
    stats = pipeline.run_chunked_inference(
        s1_path=bench_s1_path,
        total_records=len(all_benchmark_records),
        batch_size=bench_batch_size,
        candidate_tsv=cand_tsv,
        matching_tsv=match_tsv,
        source=source,
    )
    stats["france_queries_count"] = len(france_records)
    return stats


def main():
    parser = argparse.ArgumentParser(description="End-to-end inference and matching pipeline")
    parser.add_argument("--source", choices=("train", "test"), default="test")
    parser.add_argument("--index-path", type=str, default=None)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2500)
    parser.add_argument("--model-path", type=str, default="output/models/lightgbm_matcher.txt")
    parser.add_argument("--candidate-tsv", type=str, default="output/candidate_pairs.tsv")
    parser.add_argument("--matching-tsv", type=str, default="output/matching_results.tsv")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--use-fallback", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    # Determine index path
    if args.index_path is None:
        if args.source == "test":
            index_path = Path(r"H:\Amazon_ML_Work\output\test_index\indices.db")
        else:
            index_path = Path(r"H:\Amazon_ML_Work\output\step5\indices.db")
    else:
        index_path = Path(args.index_path).resolve()

    verify_index_isolation(args.source, index_path)

    if args.benchmark:
        size = args.sample_size or 500
        run_benchmark(
            sample_size=size,
            budget=args.budget,
            threshold=args.threshold,
            workers=args.workers,
            source=args.source,
            index_path=index_path,
        )
        return

    config = load_config()
    data_root = Path(config["data"]["root"])
    s1_path = data_root / f"dataset/{args.source}/{args.source}_source1.tsv"

    # Total records for test S1 is exactly 1,732,544
    total_records = 1_732_544 if args.source == "test" else 2_206_821
    if args.sample_size:
        total_records = min(total_records, args.sample_size)

    pipeline = EndToEndPipeline(
        model_path=args.model_path,
        threshold=args.threshold,
        budget=args.budget,
        db_path=index_path,
        use_fallback=args.use_fallback,
        workers=args.workers,
        run_dir=args.run_dir,
    )

    pipeline.run_chunked_inference(
        s1_path=s1_path,
        total_records=total_records,
        batch_size=args.batch_size,
        candidate_tsv=args.candidate_tsv,
        matching_tsv=args.matching_tsv,
        sample_limit=args.sample_size,
        source=args.source,
    )


if __name__ == "__main__":
    main()
