"""Threshold tuning against the official entity-level macro F0.5 metric."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from business_entity_resolution.src.evaluation.f05 import MacroF05Evaluator


def make_threshold_grid(
    scores: np.ndarray | list[float],
    linear_points: int = 199,
    quantile_points: int = 201,
) -> list[float]:
    """Build a compact threshold grid covering both tails and score density."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return [0.5]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.5]
    linear = np.linspace(0.001, 0.999, linear_points)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, quantile_points))
    thresholds = np.unique(np.concatenate(([0.0, 1.0], linear, quantiles)))
    return [float(value) for value in thresholds]


def predictions_at_threshold(
    scored_pairs: list[tuple[str, str, float]],
    entity_ids: list[str],
    threshold: float,
) -> dict[str, list[str]]:
    predictions = {entity_id: [] for entity_id in entity_ids}
    for source_id, target_id, score in scored_pairs:
        if score >= threshold:
            predictions.setdefault(source_id, []).append(target_id)
    return predictions


def tune_macro_f05_threshold(
    scored_pairs: list[tuple[str, str, float]],
    gt_map: dict[str, list[str] | set[str]],
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Choose a global score threshold using exact per-entity macro F0.5."""
    if thresholds is None:
        thresholds = make_threshold_grid([row[2] for row in scored_pairs])

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for source_id, target_id, score in scored_pairs:
        grouped[source_id].append((target_id, float(score)))
    evaluator = MacroF05Evaluator(gt_map={k: list(v) for k, v in gt_map.items()})

    best_threshold = 0.5
    best_macro = -1.0
    best_result: dict[str, Any] = {}
    curve = []
    entity_ids = list(gt_map)
    for threshold in sorted(set(float(value) for value in thresholds)):
        predictions = {
            source_id: [
                target_id
                for target_id, score in grouped.get(source_id, [])
                if score >= threshold
            ]
            for source_id in entity_ids
        }
        result = evaluator.evaluate(predictions)
        metrics = result["representative_metrics"]
        macro_f05 = float(metrics["macro_f05"])
        curve.append(
            {
                "threshold": threshold,
                "macro_f05": macro_f05,
                "pair_precision": metrics["pair_precision"],
                "pair_recall": metrics["pair_recall"],
                "singleton_accuracy": metrics["singleton_accuracy"],
            }
        )
        if macro_f05 > best_macro or (
            macro_f05 == best_macro and threshold > best_threshold
        ):
            best_macro = macro_f05
            best_threshold = threshold
            best_result = result

    return {
        "best_threshold": best_threshold,
        "best_macro_f05": best_macro,
        "evaluation_at_best": best_result,
        "curve": curve,
    }
