"""High-throughput batch inference interface for business entity resolution."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from business_entity_resolution.src.features.pair_features import PairFeatureGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class BatchMatcher:
    """Batch inference coordinator for ranking and classifying candidate pairs."""

    def __init__(
        self,
        model_path: str | Path,
        feature_schema_path: str | Path | None = None,
        threshold: float = 0.5,
    ):
        self.model_path = Path(model_path).resolve()
        self.threshold = float(threshold)
        self.feat_gen = PairFeatureGenerator()

        # Load feature schema
        if feature_schema_path is None:
            schema_path = self.model_path.parent / "feature_schema.json"
        else:
            schema_path = Path(feature_schema_path).resolve()

        if schema_path.exists():
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_data = json.load(f)
            self.feature_names = schema_data["features"]
        else:
            self.feature_names = None

        # Load model (supports both Booster text format and joblib)
        logging.info(f"Loading LightGBM model from {self.model_path}...")
        if self.model_path.suffix == ".txt":
            self.booster = lgb.Booster(model_file=str(self.model_path))
            if self.feature_names is None:
                self.feature_names = self.booster.feature_name()
            self._predict_fn = self.booster.predict
        else:
            self.model = joblib.load(self.model_path)
            if hasattr(self.model, "booster_"):
                self.booster = self.model.booster_
                if self.feature_names is None:
                    self.feature_names = self.booster.feature_name()
                self._predict_fn = self.booster.predict
            else:
                self._predict_fn = lambda x: self.model.predict_proba(x)[:, 1]

        logging.info(f"BatchMatcher initialized with {len(self.feature_names)} features, threshold={self.threshold}")

    def score_pairs_batch(
        self,
        s1_records: list[dict[str, Any]],
        candidate_records: dict[str, dict[str, Any]],
        candidates_by_s1: dict[str, list[str]],
        batch_size: int = 10000,
    ) -> dict[str, list[tuple[str, float]]]:
        """Compute matching probability scores for all retrieved candidate pairs.

        Args:
            s1_records: List of S1 dicts, each with keys 'id' or 'entity_id', 'name', 'addr', 'country'.
            candidate_records: Dict mapping target entity_id -> dict with 'name', 'addr', 'country'.
            candidates_by_s1: Dict mapping S1 ID -> ordered list of retrieved target entity IDs.
            batch_size: Number of pairs to process in a single vectorized batch.

        Returns:
            Dict mapping S1 ID -> list of (candidate_id, match_probability).
        """
        s1_map = {}
        for r in s1_records:
            eid = r.get("id") or r.get("entity_id")
            s1_map[eid] = r

        results: dict[str, list[tuple[str, float]]] = {eid: [] for eid in s1_map}

        # Build pair queue
        pair_tasks: list[tuple[str, str, int]] = []
        for s1_id, cands in candidates_by_s1.items():
            if s1_id not in s1_map:
                continue
            for rank_idx, cand_id in enumerate(cands, start=1):
                pair_tasks.append((s1_id, cand_id, rank_idx))

        if not pair_tasks:
            return results

        # Process in bounded chunks
        for start_idx in range(0, len(pair_tasks), batch_size):
            chunk = pair_tasks[start_idx : start_idx + batch_size]
            features_list: list[list[float]] = []

            for s1_id, cand_id, rank in chunk:
                s1_info = s1_map[s1_id]
                cand_info = candidate_records.get(cand_id, {"id": cand_id, "name": "", "addr": "", "country": ""})
                target_src = "source3" if cand_id.startswith("S3-") else "source2"

                feat_dict = self.feat_gen.compute_features(
                    s1_dict=s1_info,
                    s2_dict=cand_info,
                    retrieval_rank=rank,
                    target_source=target_src,
                )
                features_list.append([feat_dict[f] for f in self.feature_names])

            X_chunk = np.array(features_list, dtype=np.float32)
            probs = self._predict_fn(X_chunk)

            for (s1_id, cand_id, _), prob in zip(chunk, probs):
                results[s1_id].append((cand_id, float(prob)))

        return results

    def predict_batch(
        self,
        s1_records: list[dict[str, Any]],
        candidate_records: dict[str, dict[str, Any]],
        candidates_by_s1: dict[str, list[str]],
        threshold: float | None = None,
        batch_size: int = 10000,
    ) -> dict[str, list[str]]:
        """Predict final matches for all S1 entities, preserving one-to-many matches.

        Returns:
            Dict mapping S1 ID -> list of accepted target entity IDs (score >= threshold).
        """
        thresh = self.threshold if threshold is None else float(threshold)
        scored = self.score_pairs_batch(
            s1_records=s1_records,
            candidate_records=candidate_records,
            candidates_by_s1=candidates_by_s1,
            batch_size=batch_size,
        )

        final_predictions: dict[str, list[str]] = {}
        for s1_id, cand_scores in scored.items():
            accepted = [cand_id for cand_id, score in cand_scores if score >= thresh]
            final_predictions[s1_id] = accepted

        return final_predictions

    @staticmethod
    def format_tsv(predictions: dict[str, list[str]], output_path: str | Path) -> None:
        """Export predictions to the official matching_results.tsv format."""
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id, target_list in predictions.items():
                targets_str = ",".join(target_list) if target_list else ""
                f.write(f"{s1_id}\t{targets_str}\n")

        logging.info(f"Wrote matching results to {output_path} ({len(predictions)} rows)")
