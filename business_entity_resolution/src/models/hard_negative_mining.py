"""Deterministic hard/random negative selection for candidate pairs."""

from __future__ import annotations

import hashlib
import random


def _entity_seed(entity_id: str, seed: int) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{entity_id}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def select_training_candidates(
    entity_id: str,
    candidate_ids: list[str],
    true_ids: set[str],
    hard_negatives: int = 16,
    random_negatives: int = 8,
    seed: int = 42,
) -> list[tuple[int, str, int, str]]:
    """Keep every retrieved positive plus top-ranked and random negatives.

    Returns (rank, target_id, label, sample_kind) tuples. Hard negatives
    are the earliest unmatched retrievals, while random negatives are sampled
    without replacement from the remaining candidate tail.
    """
    positives: list[tuple[int, str, int, str]] = []
    negatives: list[tuple[int, str, int, str]] = []
    for rank, target_id in enumerate(candidate_ids, start=1):
        if target_id in true_ids:
            positives.append((rank, target_id, 1, "positive"))
        else:
            negatives.append((rank, target_id, 0, "negative"))

    hard = negatives[: max(0, hard_negatives)]
    hard = [
        (rank, target_id, label, "hard_negative")
        for rank, target_id, label, _ in hard
    ]
    tail = negatives[max(0, hard_negatives) :]
    if random_negatives > 0 and tail:
        rng = random.Random(_entity_seed(entity_id, seed))
        sampled = rng.sample(tail, min(random_negatives, len(tail)))
        random_part = [
            (rank, target_id, label, "random_negative")
            for rank, target_id, label, _ in sampled
        ]
    else:
        random_part = []

    return sorted(positives + hard + random_part, key=lambda item: item[0])
