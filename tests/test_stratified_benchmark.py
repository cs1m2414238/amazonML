"""Focused tests for deterministic sampling and resumable benchmark plumbing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from business_entity_resolution.src.evaluation.f05 import MacroF05Evaluator
from business_entity_resolution.src.evaluation.stratified_benchmark import (
    RetrievalCheckpoint,
    retrieve_with_checkpoint,
    validate_completed_positions,
)
from business_entity_resolution.src.evaluation.stratified_sampling import (
    match_band,
    primary_stratum,
    scaled_quotas,
    stable_priority,
)


class FakeRetriever:
    def __init__(self):
        self.calls: list[str] = []

    def get_candidates(self, name, address, country, budget):
        self.calls.append(name)
        return [f"S2-{name}", f"S3-{country}"][:budget]


class TestStratifiedSampling(unittest.TestCase):
    def test_scaled_quotas_are_exact_and_cover_all_primary_strata(self):
        quotas = scaled_quotas(10_000)
        self.assertEqual(quotas, {
            "US|singleton": 600,
            "US|one_match": 600,
            "US|multi_2_4": 3_000,
            "US|multi_5_plus": 1_800,
            "India|singleton": 400,
            "India|one_match": 400,
            "India|multi_2_4": 2_000,
            "India|multi_5_plus": 1_200,
        })
        self.assertEqual(sum(quotas.values()), 10_000)
        self.assertTrue(all(value > 0 for value in quotas.values()))

    def test_primary_strata_and_priority_are_deterministic(self):
        self.assertEqual(match_band(0), "singleton")
        self.assertEqual(match_band(1), "one_match")
        self.assertEqual(match_band(4), "multi_2_4")
        self.assertEqual(match_band(5), "multi_5_plus")
        self.assertEqual(primary_stratum("IN", 7), "India|multi_5_plus")
        self.assertEqual(primary_stratum("US", 0), "US|singleton")
        self.assertEqual(stable_priority(42, "S1-1"), stable_priority(42, "S1-1"))
        self.assertNotEqual(stable_priority(42, "S1-1"), stable_priority(43, "S1-1"))


class TestRetrievalCheckpoint(unittest.TestCase):
    def setUp(self):
        self.records = [
            {
                "entity_id": "S1-1",
                "business_name": "alpha",
                "business_address": "one",
                "country": "US",
            },
            {
                "entity_id": "S1-2",
                "business_name": "beta",
                "business_address": "two",
                "country": "India",
            },
        ]

    def test_checkpoint_resume_does_not_repeat_completed_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.db"
            index_path = Path(tmp) / "indices.db"
            index_path.touch()
            first = FakeRetriever()
            with RetrievalCheckpoint(checkpoint_path, "hash", 2, index_path) as checkpoint:
                candidates, _, _ = retrieve_with_checkpoint(
                    self.records, first, checkpoint, max_budget=2, checkpoint_every=1
                )
            self.assertEqual(first.calls, ["alpha", "beta"])
            self.assertEqual(candidates["S1-1"], ["S2-alpha", "S3-US"])

            second = FakeRetriever()
            with RetrievalCheckpoint(checkpoint_path, "hash", 2, index_path) as checkpoint:
                candidates, _, _ = retrieve_with_checkpoint(
                    self.records, second, checkpoint, max_budget=2, checkpoint_every=1
                )
                validate_completed_positions(checkpoint.completed_positions(), self.records)
            self.assertEqual(second.calls, [])
            self.assertEqual(len(candidates), 2)

    def test_checkpoint_rejects_different_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "checkpoint.db"
            index_path = Path(tmp) / "indices.db"
            index_path.touch()
            with RetrievalCheckpoint(checkpoint_path, "hash-a", 512, index_path):
                pass
            with self.assertRaises(ValueError):
                RetrievalCheckpoint(checkpoint_path, "hash-b", 512, index_path)


class TestCompleteMacroF05Interface(unittest.TestCase):
    def test_requires_all_entities_including_singletons(self):
        evaluator = MacroF05Evaluator(
            gt_map={"S1-1": ["S2-1"], "S1-2": []}
        )
        with self.assertRaises(ValueError):
            evaluator.evaluate_complete_prediction_map({"S1-1": ["S2-1"]})

        result = evaluator.evaluate_complete_prediction_map(
            {"S1-1": ["S2-1"], "S1-2": []}
        )
        self.assertEqual(result["representative_metrics"]["macro_f05"], 1.0)

    def test_false_singleton_match_scores_zero(self):
        evaluator = MacroF05Evaluator(gt_map={"S1-1": []})
        result = evaluator.evaluate_complete_prediction_map({"S1-1": ["S3-9"]})
        self.assertEqual(result["representative_metrics"]["macro_f05"], 0.0)


if __name__ == "__main__":
    unittest.main()
