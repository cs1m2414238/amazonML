"""Deterministic, country-agnostic pair features for business matching."""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass

from business_entity_resolution.src.indexing.indexer import get_ngrams
from business_entity_resolution.src.preprocessing import (
    AddressNormalizer,
    CountryNormalizer,
    NameNormalizer,
)


_DIGITS = re.compile(r"\d+")


@dataclass(frozen=True)
class PreparedRecord:
    """Normalized views reused across every pair for the same record."""

    name_clean: str
    name_core: str
    name_tokens: frozenset[str]
    name_ngrams: frozenset[str]
    name_initials: str
    address_clean: str
    address_core: str
    address_tokens: frozenset[str]
    address_ngrams: frozenset[str]
    address_numbers: frozenset[str]
    country: str
    source: str


class PairFeatureGenerator:
    """Create numeric pair features without country-specific categories."""

    FEATURE_VERSION = "pair_features_v2"

    @staticmethod
    def _jaccard(set1, set2) -> float:
        if not set1 and not set2:
            return 0.0
        return len(set1 & set2) / max(1, len(set1 | set2))

    @staticmethod
    def _containment(set1, set2) -> float:
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / min(len(set1), len(set2))

    @staticmethod
    def _sequence_ratio(str1: str, str2: str) -> float:
        if not str1 or not str2:
            return 0.0
        return difflib.SequenceMatcher(None, str1, str2, autojunk=False).ratio()

    @staticmethod
    def _length_ratio(str1: str, str2: str) -> float:
        if not str1 or not str2:
            return 0.0
        return min(len(str1), len(str2)) / max(len(str1), len(str2))

    @staticmethod
    def prepare_record(record: dict) -> PreparedRecord:
        name = record.get("name", "")
        address = record.get("addr", "")
        name_clean = NameNormalizer.clean(name)
        name_core = NameNormalizer.core(name)
        address_clean = AddressNormalizer.clean(address)
        address_core = AddressNormalizer.core(address)
        name_tokens = frozenset(name_core.split()) if name_core else frozenset()
        address_tokens = (
            frozenset(address_core.split()) if address_core else frozenset()
        )
        source = str(record.get("source", "") or "").lower()
        return PreparedRecord(
            name_clean=name_clean,
            name_core=name_core,
            name_tokens=name_tokens,
            name_ngrams=frozenset(get_ngrams(name_clean, 3)),
            name_initials="".join(token[0] for token in sorted(name_tokens) if token),
            address_clean=address_clean,
            address_core=address_core,
            address_tokens=address_tokens,
            address_ngrams=frozenset(get_ngrams(address_clean, 3)),
            address_numbers=frozenset(_DIGITS.findall(address_core)),
            country=CountryNormalizer.normalize(record.get("country", "")),
            source=source,
        )

    def compute_prepared_features(
        self,
        s1: PreparedRecord,
        target: PreparedRecord,
        retrieval_rank: int,
        candidate_budget: int = 256,
    ) -> dict[str, float]:
        """Compute features from normalized records."""
        name_intersection = len(s1.name_tokens & target.name_tokens)
        address_intersection = len(s1.address_tokens & target.address_tokens)
        number_intersection = len(s1.address_numbers & target.address_numbers)

        name_jaccard_3gram = self._jaccard(s1.name_ngrams, target.name_ngrams)
        name_jaccard_token = self._jaccard(s1.name_tokens, target.name_tokens)
        name_sequence_ratio = self._sequence_ratio(s1.name_clean, target.name_clean)
        address_jaccard_3gram = self._jaccard(
            s1.address_ngrams, target.address_ngrams
        )
        address_jaccard_token = self._jaccard(
            s1.address_tokens, target.address_tokens
        )
        address_sequence_ratio = self._sequence_ratio(
            s1.address_clean, target.address_clean
        )

        if not s1.country or not target.country:
            country_match = 0.5
        else:
            country_match = 1.0 if s1.country == target.country else 0.0

        features = {
            "name_exact_match": float(
                bool(s1.name_clean) and s1.name_clean == target.name_clean
            ),
            "name_core_exact_match": float(
                bool(s1.name_core) and s1.name_core == target.name_core
            ),
            "name_jaccard_3gram": name_jaccard_3gram,
            "name_jaccard_token": name_jaccard_token,
            "name_token_containment": self._containment(
                s1.name_tokens, target.name_tokens
            ),
            "name_sequence_ratio": name_sequence_ratio,
            "name_length_ratio": self._length_ratio(
                s1.name_clean, target.name_clean
            ),
            "name_len_diff": float(abs(len(s1.name_clean) - len(target.name_clean))),
            "name_tok_diff": float(
                abs(len(s1.name_tokens) - len(target.name_tokens))
            ),
            "name_shared_tokens": float(name_intersection),
            "name_first_token_match": float(
                bool(s1.name_core)
                and bool(target.name_core)
                and s1.name_core.split()[0] == target.name_core.split()[0]
            ),
            "name_initials_match": float(
                bool(s1.name_initials)
                and s1.name_initials == target.name_initials
            ),
            "address_exact_match": float(
                bool(s1.address_clean)
                and s1.address_clean == target.address_clean
            ),
            "address_core_exact_match": float(
                bool(s1.address_core) and s1.address_core == target.address_core
            ),
            "address_jaccard_3gram": address_jaccard_3gram,
            "address_jaccard_token": address_jaccard_token,
            "address_token_containment": self._containment(
                s1.address_tokens, target.address_tokens
            ),
            "address_sequence_ratio": address_sequence_ratio,
            "address_length_ratio": self._length_ratio(
                s1.address_clean, target.address_clean
            ),
            "address_len_diff": float(
                abs(len(s1.address_clean) - len(target.address_clean))
            ),
            "address_tok_diff": float(
                abs(len(s1.address_tokens) - len(target.address_tokens))
            ),
            "address_shared_tokens": float(address_intersection),
            "address_number_jaccard": self._jaccard(
                s1.address_numbers, target.address_numbers
            ),
            "address_shared_numbers": float(number_intersection),
            "country_match": country_match,
            "country_both_present": float(bool(s1.country and target.country)),
            "s1_missing_name": float(not s1.name_clean),
            "s2_missing_name": float(not target.name_clean),
            "s1_missing_address": float(not s1.address_clean),
            "s2_missing_address": float(not target.address_clean),
            "both_names_present": float(bool(s1.name_clean and target.name_clean)),
            "both_addresses_present": float(
                bool(s1.address_clean and target.address_clean)
            ),
            "target_source2": float(target.source in {"s2", "source2"}),
            "target_source3": float(target.source in {"s3", "source3"}),
            "retrieval_rank": float(retrieval_rank),
            "retrieval_log_rank": math.log1p(max(0, retrieval_rank)),
            "retrieval_reciprocal_rank": 1.0 / max(1, retrieval_rank),
            "retrieval_rank_fraction": min(
                1.0, float(retrieval_rank) / max(1, candidate_budget)
            ),
        }
        features["name_similarity_max"] = max(
            name_jaccard_3gram, name_jaccard_token, name_sequence_ratio
        )
        features["address_similarity_max"] = max(
            address_jaccard_3gram,
            address_jaccard_token,
            address_sequence_ratio,
        )
        features["name_address_similarity_min"] = min(
            features["name_similarity_max"],
            features["address_similarity_max"],
        )
        return features

    def compute_features(
        self,
        s1_dict: dict,
        s2_dict: dict,
        retrieval_rank: int,
        candidate_budget: int = 256,
    ) -> dict[str, float]:
        """Backwards-compatible convenience wrapper for individual pairs."""
        return self.compute_prepared_features(
            self.prepare_record(s1_dict),
            self.prepare_record(s2_dict),
            retrieval_rank=retrieval_rank,
            candidate_budget=candidate_budget,
        )
