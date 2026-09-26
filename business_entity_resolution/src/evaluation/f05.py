"""Exact entity-level Macro-F0.5 evaluation for business entity resolution."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def compute_entity_f05(gt_targets: set[str], pred_targets: set[str]) -> float:
    """Compute exact F0.5 score for a single Source 1 entity.

    Rules:
    - True singleton (len(gt_targets) == 0):
        * empty prediction -> 1.0
        * nonempty prediction -> 0.0
    - Non-singleton (len(gt_targets) > 0):
        * TP = len(gt_targets & pred_targets)
        * FP = len(pred_targets - gt_targets)
        * FN = len(gt_targets - pred_targets)
        * F0.5 = (5 * TP) / (5 * TP + 4 * FP + FN)
        * If TP == 0 -> 0.0
    """
    if len(gt_targets) == 0:
        return 1.0 if len(pred_targets) == 0 else 0.0

    tp = len(gt_targets & pred_targets)
    if tp == 0:
        return 0.0

    fp = len(pred_targets - gt_targets)
    fn = len(gt_targets - pred_targets)

    denom = 5 * tp + 4 * fp + fn
    if denom == 0:
        return 0.0

    return (5 * tp) / denom


class MacroF05Evaluator:
    """Evaluates final predicted entity matches with exact Macro-F0.5 and diagnostics."""

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
            raise ValueError("Either gt_map or gt_path must be provided to MacroF05Evaluator")

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
        predictions: dict[str, list[str]],
        candidates: dict[str, list[str]] | None = None,
        sample_metadata: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Compute exact entity-level Macro-F0.5, pair metrics, and candidate violations."""
        total_queries = len(predictions)
        if total_queries == 0:
            return {"error": "Empty predictions dictionary provided"}

        entity_scores: dict[str, float] = {}
        per_entity_details: dict[str, dict[str, Any]] = {}

        singleton_count = 0
        singleton_correct = 0

        non_singleton_count = 0
        sum_non_singleton_score = 0.0

        total_tp = 0
        total_fp = 0
        total_fn = 0

        duplicate_predictions_count = 0
        invalid_id_format_count = 0
        candidate_subset_violations = []

        cand_map = {k: set(v) for k, v in candidates.items()} if candidates else None

        for s1_id, raw_preds in predictions.items():
            # Validate IDs & deduplicate
            cleaned_preds: list[str] = []
            seen_in_row = set()
            for p in raw_preds:
                p_str = str(p).strip()
                if not p_str:
                    continue
                if p_str in seen_in_row:
                    duplicate_predictions_count += 1
                    continue
                seen_in_row.add(p_str)

                # Validation checks: target prefix must be S2- or S3-
                if not (p_str.startswith("S2-") or p_str.startswith("S3-")):
                    invalid_id_format_count += 1

                cleaned_preds.append(p_str)

            pred_set = set(cleaned_preds)
            gt_set = self.gt_map.get(s1_id, set())

            # Candidate pool containment check
            if cand_map is not None:
                pool = cand_map.get(s1_id, set())
                outside_pool = pred_set - pool
                if outside_pool:
                    candidate_subset_violations.append({
                        "entity_id": s1_id,
                        "violating_predictions": list(outside_pool),
                    })

            score = compute_entity_f05(gt_set, pred_set)
            entity_scores[s1_id] = score

            gt_len = len(gt_set)
            if gt_len == 0:
                singleton_count += 1
                if len(pred_set) == 0:
                    singleton_correct += 1
                tp = 0
                fp = len(pred_set)
                fn = 0
            else:
                non_singleton_count += 1
                sum_non_singleton_score += score
                tp = len(gt_set & pred_set)
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            per_entity_details[s1_id] = {
                "score": score,
                "gt_len": gt_len,
                "pred_len": len(pred_set),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }

        macro_f05 = float(np.mean(list(entity_scores.values())))
        singleton_accuracy = (
            singleton_correct / singleton_count if singleton_count > 0 else 1.0
        )
        non_singleton_f05 = (
            sum_non_singleton_score / non_singleton_count
            if non_singleton_count > 0
            else 0.0
        )

        pair_precision = (
            total_tp / (total_tp + total_fp)
            if (total_tp + total_fp) > 0
            else (1.0 if total_fn == 0 else 0.0)
        )
        pair_recall = (
            total_tp / (total_tp + total_fn)
            if (total_tp + total_fn) > 0
            else 1.0
        )

        denom_pair_f05 = 5 * total_tp + 4 * total_fp + total_fn
        pair_f05 = (
            (5 * total_tp) / denom_pair_f05 if denom_pair_f05 > 0 else 0.0
        )

        overall_metrics = {
            "macro_f05": round(macro_f05, 4),
            "singleton_accuracy": round(singleton_accuracy, 4),
            "non_singleton_macro_f05": round(non_singleton_f05, 4),
            "pair_precision": round(pair_precision, 4),
            "pair_recall": round(pair_recall, 4),
            "pair_f05": round(pair_f05, 4),
            "total_queries": total_queries,
            "singleton_queries": singleton_count,
            "non_singleton_queries": non_singleton_count,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "validation_checks": {
                "duplicate_predictions_handled": duplicate_predictions_count,
                "invalid_id_format_count": invalid_id_format_count,
                "candidate_subset_violations_count": len(candidate_subset_violations),
                "candidate_subset_violation_rate": round(
                    len(candidate_subset_violations) / total_queries, 4
                ),
            },
        }

        # Diagnostic strata breakdown
        diagnostic_strata: dict[str, dict[str, Any]] = {}
        if sample_metadata:
            strata_to_entities: dict[str, list[str]] = defaultdict(list)
            for item in sample_metadata:
                eid = item.get("entity_id")
                strata_list = item.get("strata", [])
                if eid and eid in predictions:
                    for s in strata_list:
                        strata_to_entities[s].append(eid)

            for s_name, eids in strata_to_entities.items():
                s_scores = [entity_scores[eid] for eid in eids if eid in entity_scores]
                s_tp = sum(per_entity_details[eid]["tp"] for eid in eids)
                s_fp = sum(per_entity_details[eid]["fp"] for eid in eids)
                s_fn = sum(per_entity_details[eid]["fn"] for eid in eids)

                s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 1.0
                s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 1.0
                s_single = [eid for eid in eids if per_entity_details[eid]["gt_len"] == 0]
                s_single_acc = (
                    sum(1 for eid in s_single if per_entity_details[eid]["pred_len"] == 0)
                    / len(s_single)
                    if s_single
                    else 1.0
                )

                diagnostic_strata[s_name] = {
                    "entity_count": len(eids),
                    "macro_f05": round(float(np.mean(s_scores)), 4) if s_scores else 0.0,
                    "pair_precision": round(s_prec, 4),
                    "pair_recall": round(s_rec, 4),
                    "singleton_accuracy": round(s_single_acc, 4),
                }

        return {
            "representative_metrics": overall_metrics,
            "diagnostic_strata": diagnostic_strata,
            "violations_sample": candidate_subset_violations[:10],
        }
