"""Pairwise feature engineering for business entity matching."""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from business_entity_resolution.src.indexing.indexer import get_ngrams
from business_entity_resolution.src.preprocessing import (
    AddressNormalizer,
    CountryNormalizer,
    NameNormalizer,
)

from rapidfuzz.distance import Indel

NUMERIC_RE = re.compile(r'\b\d+\b')
_DIGITS = re.compile(r'\d+')


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
    """Computes comprehensive pairwise similarity features for business entity resolution."""

    def __init__(self):
        pass

    @staticmethod
    def _jaccard(set1: set | frozenset, set2: set | frozenset) -> float:
        if not set1 and not set2:
            return 0.0
        return len(set1 & set2) / max(1, len(set1 | set2))

    @staticmethod
    def _containment(set1: set | frozenset, set2: set | frozenset) -> float:
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / min(len(set1), len(set2))

    @staticmethod
    def _sequence_ratio(str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 0.0
        if not str1 or not str2:
            return 0.0
        if str1 == str2:
            return 1.0
        return difflib.SequenceMatcher(None, str1, str2).ratio()

    @staticmethod
    def _extract_numbers(text: str) -> set[str]:
        if not text:
            return set()
        return set(NUMERIC_RE.findall(text))

    def compute_features(
        self,
        s1_dict: dict[str, Any],
        s2_dict: dict[str, Any],
        retrieval_rank: int | float = 1.0,
        target_source: str = "source2",
    ) -> dict[str, float]:
        """Compute pairwise features between S1 entity and target S2/S3 entity."""
        p1 = self.prepare_record(s1_dict)
        p2 = self.prepare_record(s2_dict)
        return self.compute_features_from_prepared(
            s1=p1,
            s2=p2,
            retrieval_rank=retrieval_rank,
            target_source=target_source,
        )

    def compute_features_from_prepared(
        self,
        s1: PreparedRecord,
        s2: PreparedRecord,
        retrieval_rank: int | float = 1.0,
        target_source: str = "source2",
    ) -> dict[str, float]:
        """Compute pairwise features directly from pre-normalized PreparedRecord objects."""
        features: dict[str, float] = {}

        # 1. Name Features
        s1_n_clean = s1.name_clean
        s2_n_clean = s2.name_clean
        s1_n_core = s1.name_core
        s2_n_core = s2.name_core

        features['name_exact_match'] = 1.0 if (s1_n_clean and s1_n_clean == s2_n_clean) else 0.0
        features['name_core_exact_match'] = 1.0 if (s1_n_core and s1_n_core == s2_n_core) else 0.0

        if s1_n_clean and s2_n_clean:
            features['name_rapidfuzz_ratio'] = fuzz.ratio(s1_n_clean, s2_n_clean) / 100.0
            features['name_rapidfuzz_token_sort'] = fuzz.token_sort_ratio(s1_n_clean, s2_n_clean) / 100.0
            features['name_rapidfuzz_token_set'] = fuzz.token_set_ratio(s1_n_clean, s2_n_clean) / 100.0
        else:
            features['name_rapidfuzz_ratio'] = 0.0
            features['name_rapidfuzz_token_sort'] = 0.0
            features['name_rapidfuzz_token_set'] = 0.0

        features['name_jaccard_3gram'] = self._jaccard(s1.name_ngrams, s2.name_ngrams)
        features['name_jaccard_token'] = self._jaccard(s1.name_tokens, s2.name_tokens)
        features['name_token_containment'] = self._containment(s1.name_tokens, s2.name_tokens)
        features['name_sequence_ratio'] = self._sequence_ratio(s1_n_clean, s2_n_clean)

        len_s1_n = len(s1_n_clean)
        len_s2_n = len(s2_n_clean)
        features['name_len_diff'] = float(abs(len_s1_n - len_s2_n))
        features['name_len_ratio'] = min(len_s1_n, len_s2_n) / max(1, max(len_s1_n, len_s2_n))
        features['name_tok_diff'] = float(abs(len(s1.name_tokens) - len(s2.name_tokens)))

        # 2. Address Features
        s1_a_clean = s1.address_clean
        s2_a_clean = s2.address_clean
        s1_a_core = s1.address_core
        s2_a_core = s2.address_core

        features['address_exact_match'] = 1.0 if (s1_a_clean and s1_a_clean == s2_a_clean) else 0.0
        features['address_core_exact_match'] = 1.0 if (s1_a_core and s1_a_core == s2_a_core) else 0.0

        if s1_a_clean and s2_a_clean:
            features['address_rapidfuzz_ratio'] = fuzz.ratio(s1_a_clean, s2_a_clean) / 100.0
            features['address_rapidfuzz_token_set'] = fuzz.token_set_ratio(s1_a_clean, s2_a_clean) / 100.0
        else:
            features['address_rapidfuzz_ratio'] = 0.0
            features['address_rapidfuzz_token_set'] = 0.0

        features['address_jaccard_3gram'] = self._jaccard(s1.address_ngrams, s2.address_ngrams)
        features['address_jaccard_token'] = self._jaccard(s1.address_tokens, s2.address_tokens)
        features['address_token_containment'] = self._containment(s1.address_tokens, s2.address_tokens)
        features['address_sequence_ratio'] = self._sequence_ratio(s1_a_clean, s2_a_clean)

        s1_nums = s1.address_numbers
        s2_nums = s2.address_numbers
        if s1_nums and s2_nums:
            features['address_numeric_match'] = self._jaccard(s1_nums, s2_nums)
            features['address_has_exact_number_match'] = 1.0 if (s1_nums & s2_nums) else 0.0
        elif not s1_nums and not s2_nums:
            features['address_numeric_match'] = 0.5
            features['address_has_exact_number_match'] = 0.5
        else:
            features['address_numeric_match'] = 0.0
            features['address_has_exact_number_match'] = 0.0

        len_s1_a = len(s1_a_clean)
        len_s2_a = len(s2_a_clean)
        features['address_len_diff'] = float(abs(len_s1_a - len_s2_a))
        features['address_len_ratio'] = min(len_s1_a, len_s2_a) / max(1, max(len_s1_a, len_s2_a))
        features['address_tok_diff'] = float(abs(len(s1.address_tokens) - len(s2.address_tokens)))

        # 3. Country Features (Open-Set Robust)
        s1_c = s1.country
        s2_c = s2.country
        if not s1_c and not s2_c:
            features['country_match'] = 0.5
        elif not s1_c or not s2_c:
            features['country_match'] = 0.5
        else:
            features['country_match'] = 1.0 if s1_c == s2_c else 0.0

        features['is_us'] = 1.0 if (s1_c in ('united states', 'us')) else 0.0
        features['is_india'] = 1.0 if (s1_c in ('india', 'in')) else 0.0

        # 4. Missing Fields
        features['s1_missing_name'] = 1.0 if not s1_n_clean else 0.0
        features['s2_missing_name'] = 1.0 if not s2_n_clean else 0.0
        features['s1_missing_address'] = 1.0 if not s1_a_clean else 0.0
        features['s2_missing_address'] = 1.0 if not s2_a_clean else 0.0
        features['both_missing_address'] = 1.0 if (not s1_a_clean and not s2_a_clean) else 0.0

        # 5. Retrieval Evidence & Target Source
        rank_val = float(retrieval_rank)
        features['retrieval_rank'] = rank_val
        features['retrieval_reciprocal_rank'] = 1.0 / max(1.0, rank_val)
        features['retrieval_log_rank'] = math.log1p(rank_val)

        is_s3 = (target_source in ("source3", "s3")) or s2.source in ("source3", "s3")
        features['is_source3'] = 1.0 if is_s3 else 0.0

        # 6. Combined Name / Address Interactions
        name_sim = features['name_rapidfuzz_ratio']
        addr_sim = features['address_rapidfuzz_ratio']
        features['name_address_product'] = name_sim * addr_sim
        features['name_address_min'] = min(name_sim, addr_sim)
        features['name_address_harmonic_mean'] = (
            (2.0 * name_sim * addr_sim) / (name_sim + addr_sim + 1e-6)
        )

        # 7. Additional compatibility keys for Agent 1 workflows
        features["name_shared_tokens"] = float(len(s1.name_tokens & s2.name_tokens))
        features["name_length_ratio"] = features["name_len_ratio"]
        s1_toks = sorted(s1.name_tokens)
        s2_toks = sorted(s2.name_tokens)
        features["name_first_token_match"] = float(
            bool(s1_n_core and s2_n_core)
            and s1_n_core.split()[0] == s2_n_core.split()[0]
        )
        features["name_initials_match"] = float(bool(s1.name_initials) and s1.name_initials == s2.name_initials)

        features["address_shared_tokens"] = float(len(s1.address_tokens & s2.address_tokens))
        features["address_length_ratio"] = features["address_len_ratio"]
        features["address_number_jaccard"] = features["address_numeric_match"]
        features["address_shared_numbers"] = float(len(s1_nums & s2_nums))

        features["country_both_present"] = float(bool(s1_c and s2_c))
        features["both_names_present"] = float(bool(s1_n_clean and s2_n_clean))
        features["both_addresses_present"] = float(bool(s1_a_clean and s2_a_clean))
        features["target_source2"] = 1.0 - features["is_source3"]
        features["target_source3"] = features["is_source3"]
        features["retrieval_rank_fraction"] = min(1.0, rank_val / max(1.0, 256.0))

        features["name_similarity_max"] = max(
            features["name_jaccard_3gram"],
            features["name_jaccard_token"],
            features["name_sequence_ratio"],
        )
        features["address_similarity_max"] = max(
            features["address_jaccard_3gram"],
            features["address_jaccard_token"],
            features["address_sequence_ratio"],
        )
        features["name_address_similarity_min"] = min(
            features["name_similarity_max"], features["address_similarity_max"]
        )

        return features

    @staticmethod
    def prepare_record(record: dict[str, Any]) -> PreparedRecord:
        name = record.get("name") or record.get("business_name") or ""
        address = record.get("addr") or record.get("business_address") or ""
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
            address_numbers=frozenset(NUMERIC_RE.findall(address_clean)),
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
        return self.compute_features_from_prepared(
            s1=s1,
            s2=target,
            retrieval_rank=retrieval_rank,
            target_source="source3" if target.source in {"s3", "source3"} else "source2",
        )

    @staticmethod
    def compute_feature_vector_from_prepared(
        s1: PreparedRecord,
        s2: PreparedRecord,
        retrieval_rank: int | float = 1.0,
        target_source: str = "source2",
    ) -> list[float]:
        """Directly compute ordered 40-float feature vector without intermediate dicts."""
        s1_n_clean = s1.name_clean
        s2_n_clean = s2.name_clean
        s1_n_core = s1.name_core
        s2_n_core = s2.name_core

        name_exact = 1.0 if (s1_n_clean and s1_n_clean == s2_n_clean) else 0.0
        name_core_exact = 1.0 if (s1_n_core and s1_n_core == s2_n_core) else 0.0

        if s1_n_clean and s2_n_clean:
            rf_ratio = fuzz.ratio(s1_n_clean, s2_n_clean) / 100.0
            rf_tok_sort = fuzz.token_sort_ratio(s1_n_clean, s2_n_clean) / 100.0
            rf_tok_set = fuzz.token_set_ratio(s1_n_clean, s2_n_clean) / 100.0
        else:
            rf_ratio = 0.0
            rf_tok_sort = 0.0
            rf_tok_set = 0.0

        s1_ng = s1.name_ngrams
        s2_ng = s2.name_ngrams
        n_jacc_3g = len(s1_ng & s2_ng) / max(1, len(s1_ng | s2_ng)) if (s1_ng or s2_ng) else 0.0

        s1_tok = s1.name_tokens
        s2_tok = s2.name_tokens
        n_jacc_tok = len(s1_tok & s2_tok) / max(1, len(s1_tok | s2_tok)) if (s1_tok or s2_tok) else 0.0
        n_contain = len(s1_tok & s2_tok) / min(len(s1_tok), len(s2_tok)) if (s1_tok and s2_tok) else 0.0

        n_seq = 1.0 if s1_n_clean == s2_n_clean else (difflib.SequenceMatcher(None, s1_n_clean, s2_n_clean).ratio() if (s1_n_clean and s2_n_clean) else 0.0)

        l1 = len(s1_n_clean)
        l2 = len(s2_n_clean)
        n_len_diff = float(abs(l1 - l2))
        n_len_ratio = min(l1, l2) / max(1, max(l1, l2))
        n_tok_diff = float(abs(len(s1_tok) - len(s2_tok)))

        s1_a_clean = s1.address_clean
        s2_a_clean = s2.address_clean
        s1_a_core = s1.address_core
        s2_a_core = s2.address_core

        addr_exact = 1.0 if (s1_a_clean and s1_a_clean == s2_a_clean) else 0.0
        addr_core_exact = 1.0 if (s1_a_core and s1_a_core == s2_a_core) else 0.0

        if s1_a_clean and s2_a_clean:
            a_rf_ratio = fuzz.ratio(s1_a_clean, s2_a_clean) / 100.0
            a_rf_tok_set = fuzz.token_set_ratio(s1_a_clean, s2_a_clean) / 100.0
        else:
            a_rf_ratio = 0.0
            a_rf_tok_set = 0.0

        s1_ang = s1.address_ngrams
        s2_ang = s2.address_ngrams
        a_jacc_3g = len(s1_ang & s2_ang) / max(1, len(s1_ang | s2_ang)) if (s1_ang or s2_ang) else 0.0

        s1_atok = s1.address_tokens
        s2_atok = s2.address_tokens
        a_jacc_tok = len(s1_atok & s2_atok) / max(1, len(s1_atok | s2_atok)) if (s1_atok or s2_atok) else 0.0
        a_contain = len(s1_atok & s2_atok) / min(len(s1_atok), len(s2_atok)) if (s1_atok and s2_atok) else 0.0

        a_seq = 1.0 if s1_a_clean == s2_a_clean else (difflib.SequenceMatcher(None, s1_a_clean, s2_a_clean).ratio() if (s1_a_clean and s2_a_clean) else 0.0)

        s1_nums = s1.address_numbers
        s2_nums = s2.address_numbers
        if s1_nums and s2_nums:
            a_num_match = len(s1_nums & s2_nums) / len(s1_nums | s2_nums)
            a_has_exact_num = 1.0 if (s1_nums & s2_nums) else 0.0
        elif not s1_nums and not s2_nums:
            a_num_match = 0.5
            a_has_exact_num = 0.5
        else:
            a_num_match = 0.0
            a_has_exact_num = 0.0

        al1 = len(s1_a_clean)
        al2 = len(s2_a_clean)
        a_len_diff = float(abs(al1 - al2))
        a_len_ratio = min(al1, al2) / max(1, max(al1, al2))
        a_tok_diff = float(abs(len(s1_atok) - len(s2_atok)))

        s1_c = s1.country
        s2_c = s2.country
        if not s1_c or not s2_c:
            c_match = 0.5
        else:
            c_match = 1.0 if s1_c == s2_c else 0.0

        is_us = 1.0 if (s1_c in ('united states', 'us')) else 0.0
        is_india = 1.0 if (s1_c in ('india', 'in')) else 0.0

        s1_miss_n = 1.0 if not s1_n_clean else 0.0
        s2_miss_n = 1.0 if not s2_n_clean else 0.0
        s1_miss_a = 1.0 if not s1_a_clean else 0.0
        s2_miss_a = 1.0 if not s2_a_clean else 0.0
        both_miss_a = 1.0 if (not s1_a_clean and not s2_a_clean) else 0.0

        rank_f = float(retrieval_rank)
        recip_rank = 1.0 / max(1.0, rank_f)
        log_rank = math.log1p(rank_f)

        is_s3 = 1.0 if ((target_source in ('source3', 's3')) or s2.source in ('source3', 's3')) else 0.0

        na_prod = rf_ratio * a_rf_ratio
        na_min = min(rf_ratio, a_rf_ratio)
        na_harm = (2.0 * rf_ratio * a_rf_ratio) / (rf_ratio + a_rf_ratio + 1e-6)

        return [
            name_exact, name_core_exact, rf_ratio, rf_tok_sort, rf_tok_set,
            n_jacc_3g, n_jacc_tok, n_contain, n_seq, n_len_diff, n_len_ratio, n_tok_diff,
            addr_exact, addr_core_exact, a_rf_ratio, a_rf_tok_set, a_jacc_3g, a_jacc_tok, a_contain, a_seq,
            a_num_match, a_has_exact_num, a_len_diff, a_len_ratio, a_tok_diff,
            c_match, is_us, is_india, s1_miss_n, s2_miss_n, s1_miss_a, s2_miss_a, both_miss_a,
            rank_f, recip_rank, log_rank, is_s3, na_prod, na_min, na_harm
        ]
