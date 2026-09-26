"""Comprehensive candidate recall and blocking metrics evaluator."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

TOTAL_TARGET_UNIVERSE = 10_320_219  # |S2| (5,034,616) + |S3| (5,285,603)


class CandidateRecallEvaluator:
    """Evaluates candidate blocking quality and efficiency against ground truth."""

    def __init__(
        self,
        gt_map: dict[str, list[str]] | None = None,
        gt_path: str | Path | None = None,
    ):
        if gt_map is not None:
            self.gt_map = {k: set(v) for k, v in gt_map.items()}
        elif gt_path is not None:
            self.gt_map = self._load_gt(gt_path)
        else:
            raise ValueError("Either gt_map or gt_path must be provided to CandidateRecallEvaluator")

    @staticmethod
    def _load_gt(gt_path: str | Path) -> dict[str, set[str]]:
        gt_path = Path(gt_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        df = pd.read_csv(gt_path, sep="\t", dtype="string", keep_default_na=False)
        gt_map: dict[str, set[str]] = {}
        for row in df.itertuples(index=False):
            s1_id = str(row.source1_entity_id).strip()
            matched = str(row.matched_entity_ids).strip() if pd.notna(row.matched_entity_ids) else ""
            if matched:
                targets = {t.strip() for t in matched.split(",") if t.strip()}
                gt_map[s1_id] = targets
            else:
                gt_map[s1_id] = set()
        return gt_map

    def evaluate(
        self,
        retrieved_candidates: dict[str, list[str]],
        sample_metadata: list[dict[str, Any]] | None = None,
        runtime_seconds: float = 0.0,
        latencies: list[float] | None = None,
        peak_memory_mb: float = 0.0,
        target_universe_size: int = TOTAL_TARGET_UNIVERSE,
    ) -> dict[str, Any]:
        """Compute comprehensive blocking and retrieval efficiency metrics."""
        total_queries = len(retrieved_candidates)
        if total_queries == 0:
            return {"error": "Empty retrieved_candidates dictionary provided"}

        cand_counts: list[int] = []
        singleton_cand_counts: list[int] = []

        total_true_pairs = 0
        retrieved_true_pairs = 0

        non_singleton_entities = 0
        complete_match_entities = 0

        s2_true_pairs = 0
        s2_retrieved_pairs = 0
        s3_true_pairs = 0
        s3_retrieved_pairs = 0

        # Detailed per-entity records
        per_entity_records: dict[str, dict[str, Any]] = {}

        for s1_id, cand_list in retrieved_candidates.items():
            cand_set = set(cand_list)
            num_cands = len(cand_list)
            cand_counts.append(num_cands)

            gt_set = self.gt_map.get(s1_id, set())
            gt_len = len(gt_set)

            if gt_len == 0:
                singleton_cand_counts.append(num_cands)
                per_entity_records[s1_id] = {
                    "gt_len": 0,
                    "retrieved_count": num_cands,
                    "found_matches": 0,
                    "complete_match": True,
                }
                continue

            non_singleton_entities += 1
            total_true_pairs += gt_len

            found_set = gt_set & cand_set
            found_count = len(found_set)
            retrieved_true_pairs += found_count

            is_complete = (found_count == gt_len)
            if is_complete:
                complete_match_entities += 1

            # S2 / S3 breakdown
            s2_gt = {t for t in gt_set if t.startswith("S2-")}
            s3_gt = {t for t in gt_set if t.startswith("S3-")}

            s2_true_pairs += len(s2_gt)
            s2_retrieved_pairs += len(s2_gt & cand_set)
            s3_true_pairs += len(s3_gt)
            s3_retrieved_pairs += len(s3_gt & cand_set)

            per_entity_records[s1_id] = {
                "gt_len": gt_len,
                "retrieved_count": num_cands,
                "found_matches": found_count,
                "complete_match": is_complete,
            }

        # Aggregate metrics
        pair_level_recall = (
            retrieved_true_pairs / total_true_pairs if total_true_pairs > 0 else 1.0
        )
        complete_match_recall = (
            complete_match_entities / non_singleton_entities
            if non_singleton_entities > 0
            else 1.0
        )
        s2_recall = (
            s2_retrieved_pairs / s2_true_pairs if s2_true_pairs > 0 else 1.0
        )
        s3_recall = (
            s3_retrieved_pairs / s3_true_pairs if s3_true_pairs > 0 else 1.0
        )

        # Candidate count stats
        arr_cands = np.array(cand_counts, dtype=np.int32)
        total_retrieved = int(arr_cands.sum())
        avg_cands = float(np.mean(arr_cands))
        median_cands = float(np.median(arr_cands))
        p90_cands = float(np.percentile(arr_cands, 90))
        p95_cands = float(np.percentile(arr_cands, 95))
        p99_cands = float(np.percentile(arr_cands, 99))
        min_cands = int(arr_cands.min())
        max_cands = int(arr_cands.max())

        # Reduction ratio
        universe_total = total_queries * target_universe_size
        reduction_ratio = (
            1.0 - (total_retrieved / universe_total) if universe_total > 0 else 1.0
        )

        # Singleton stats
        singleton_count = len(singleton_cand_counts)
        if singleton_count > 0:
            arr_single = np.array(singleton_cand_counts, dtype=np.int32)
            singleton_avg = float(np.mean(arr_single))
            singleton_median = float(np.median(arr_single))
            singleton_p95 = float(np.percentile(arr_single, 95))
            singleton_zero_rate = float(np.mean(arr_single == 0))
        else:
            singleton_avg = 0.0
            singleton_median = 0.0
            singleton_p95 = 0.0
            singleton_zero_rate = 1.0

        # Efficiency & latency stats
        qps = (
            round(total_queries / runtime_seconds, 2)
            if runtime_seconds > 0
            else 0.0
        )
        latency_stats: dict[str, float] = {}
        if latencies and len(latencies) > 0:
            lat_arr = np.array(latencies, dtype=np.float64)
            latency_stats = {
                "mean_seconds": round(float(np.mean(lat_arr)), 6),
                "median_seconds": round(float(np.median(lat_arr)), 6),
                "p90_seconds": round(float(np.percentile(lat_arr, 90)), 6),
                "p95_seconds": round(float(np.percentile(lat_arr, 95)), 6),
                "p99_seconds": round(float(np.percentile(lat_arr, 99)), 6),
                "max_seconds": round(float(np.max(lat_arr)), 6),
            }

        overall_metrics = {
            "total_queries": total_queries,
            "non_singleton_queries": non_singleton_entities,
            "singleton_queries": singleton_count,
            "total_ground_truth_pairs": total_true_pairs,
            "retrieved_true_pairs": retrieved_true_pairs,
            "pair_level_recall": round(pair_level_recall, 4),
            "complete_match_set_recall": round(complete_match_recall, 4),
            "source2_recall": round(s2_recall, 4),
            "source3_recall": round(s3_recall, 4),
            "candidate_counts": {
                "total_retrieved": total_retrieved,
                "mean": round(avg_cands, 2),
                "median": round(median_cands, 2),
                "p90": int(p90_cands),
                "p95": int(p95_cands),
                "p99": int(p99_cands),
                "min": min_cands,
                "max": max_cands,
            },
            "singleton_behavior": {
                "count": singleton_count,
                "mean_candidates": round(singleton_avg, 2),
                "median_candidates": round(singleton_median, 2),
                "p95_candidates": int(singleton_p95),
                "zero_candidate_rate": round(singleton_zero_rate, 4),
            },
            "reduction_ratio": round(reduction_ratio, 8),
            "efficiency": {
                "runtime_seconds": round(runtime_seconds, 2),
                "queries_per_second": qps,
                "latency_stats": latency_stats,
                "peak_rss_mb": round(peak_memory_mb, 2) if peak_memory_mb > 0 else round(get_peak_memory_mb(), 2),
            },
        }

        # Strata diagnostics
        diagnostic_strata: dict[str, dict[str, Any]] = {}
        if sample_metadata:
            strata_to_entities: dict[str, list[str]] = defaultdict(list)
            for item in sample_metadata:
                eid = item.get("entity_id")
                strata_list = item.get("strata", [])
                if eid and eid in retrieved_candidates:
                    for s in strata_list:
                        strata_to_entities[s].append(eid)

            for s_name, eids in strata_to_entities.items():
                s_total_gt = 0
                s_retrieved_gt = 0
                s_non_single = 0
                s_complete = 0
                s_cands: list[int] = []

                for eid in eids:
                    rec = per_entity_records.get(eid, {})
                    gt_l = rec.get("gt_len", 0)
                    c_cnt = rec.get("retrieved_count", 0)
                    s_cands.append(c_cnt)
                    if gt_l > 0:
                        s_non_single += 1
                        s_total_gt += gt_l
                        s_retrieved_gt += rec.get("found_matches", 0)
                        if rec.get("complete_match", False):
                            s_complete += 1

                diag_pair_recall = (
                    round(s_retrieved_gt / s_total_gt, 4) if s_total_gt > 0 else 1.0
                )
                diag_complete_recall = (
                    round(s_complete / s_non_single, 4) if s_non_single > 0 else 1.0
                )
                diagnostic_strata[s_name] = {
                    "entity_count": len(eids),
                    "non_singleton_count": s_non_single,
                    "total_true_pairs": s_total_gt,
                    "pair_level_recall": diag_pair_recall,
                    "complete_match_set_recall": diag_complete_recall,
                    "average_candidates": round(float(np.mean(s_cands)), 2) if s_cands else 0.0,
                }

        return {
            "representative_metrics": overall_metrics,
            "diagnostic_strata": diagnostic_strata,
        }
