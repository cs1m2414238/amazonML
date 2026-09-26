"""Threshold tuning utility to maximize entity-level Macro-F0.5."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np

from business_entity_resolution.src.evaluation.f05 import MacroF05Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def search_best_threshold(
    scored_pairs: list[tuple[str, str, float]],
    gt_map: dict[str, list[str]],
    thresholds: list[float] | None = None,
    sample_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Search for the score threshold that maximizes entity-level Macro-F0.5.

    Args:
        scored_pairs: List of (s1_entity_id, candidate_target_id, match_probability).
        gt_map: Ground truth dictionary mapping s1_id -> list of target IDs.
        thresholds: List of candidate thresholds to test. Defaults to 0.05..0.95 in 0.05 steps.
        sample_metadata: Optional metadata with diagnostic strata tags.

    Returns:
        Dictionary with best threshold, best macro-F0.5, metrics at best threshold,
        and full threshold evaluation curve.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.linspace(0.05, 0.95, 19)]

    evaluator = MacroF05Evaluator(gt_map=gt_map)

    # Group scored pairs by s1_id
    s1_to_candidates: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for s1_id, cand_id, score in scored_pairs:
        s1_to_candidates[s1_id].append((cand_id, float(score)))

    # Ensure all entities in gt_map are represented
    all_s1_ids = list(gt_map.keys())

    best_threshold = 0.5
    best_score = -1.0
    best_result: dict[str, Any] = {}
    curve: list[dict[str, Any]] = []

    for thresh in thresholds:
        predictions: dict[str, list[str]] = {}
        for s1_id in all_s1_ids:
            cands = s1_to_candidates.get(s1_id, [])
            accepted = [c_id for c_id, score in cands if score >= thresh]
            predictions[s1_id] = accepted

        res = evaluator.evaluate(predictions=predictions, sample_metadata=sample_metadata)
        rep = res["representative_metrics"]
        macro_f05 = rep["macro_f05"]

        curve.append({
            "threshold": thresh,
            "macro_f05": macro_f05,
            "pair_precision": rep["pair_precision"],
            "pair_recall": rep["pair_recall"],
            "singleton_accuracy": rep["singleton_accuracy"],
        })

        if macro_f05 > best_score:
            best_score = macro_f05
            best_threshold = thresh
            best_result = res

    return {
        "best_threshold": best_threshold,
        "best_macro_f05": best_score,
        "evaluation_at_best": best_result,
        "curve": curve,
    }
