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
    def _jaccard(set1: set, set2: set) -> float:
        if not set1 and not set2:
            return 0.0
        return len(set1 & set2) / max(1, len(set1 | set2))

    @staticmethod
    def _containment(set1: set, set2: set) -> float:
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / min(len(set1), len(set2))

    @staticmethod
    def _sequence_ratio(str1: str, str2: str) -> float:
        if not str1 and not str2:
            return 0.0
        if not str1 or not str2:
            return 0.0
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
        """Compute pairwise features between S1 entity and target S2/S3 entity.

        s1_dict, s2_dict should contain keys: 'name', 'addr', 'country'
        """
        raw_s1_name = s1_dict.get('name', '') or ''
        raw_s2_name = s2_dict.get('name', '') or ''
        raw_s1_addr = s1_dict.get('addr', '') or ''
        raw_s2_addr = s2_dict.get('addr', '') or ''
        raw_s1_country = s1_dict.get('country', '') or ''
        raw_s2_country = s2_dict.get('country', '') or ''

        # Normalization
        s1_name_clean = NameNormalizer.clean(raw_s1_name)
        s1_name_core = NameNormalizer.core(raw_s1_name)
        s2_name_clean = NameNormalizer.clean(raw_s2_name)
        s2_name_core = NameNormalizer.core(raw_s2_name)

        s1_addr_clean = AddressNormalizer.clean(raw_s1_addr)
        s1_addr_core = AddressNormalizer.core(raw_s1_addr)
        s2_addr_clean = AddressNormalizer.clean(raw_s2_addr)
        s2_addr_core = AddressNormalizer.core(raw_s2_addr)

        s1_country = CountryNormalizer.normalize(raw_s1_country)
        s2_country = CountryNormalizer.normalize(raw_s2_country)

        # Name Token Sets
        s1_name_tok = set(s1_name_core.split()) if s1_name_core else set()
        s2_name_tok = set(s2_name_core.split()) if s2_name_core else set()
        s1_name_3g = get_ngrams(s1_name_clean, 3)
        s2_name_3g = get_ngrams(s2_name_clean, 3)

        # Address Token Sets
        s1_addr_tok = set(s1_addr_core.split()) if s1_addr_core else set()
        s2_addr_tok = set(s2_addr_core.split()) if s2_addr_core else set()
        s1_addr_3g = get_ngrams(s1_addr_clean, 3)
        s2_addr_3g = get_ngrams(s2_addr_clean, 3)

        features: dict[str, float] = {}

        # 1. Name Features
        features['name_exact_match'] = 1.0 if (s1_name_clean and s1_name_clean == s2_name_clean) else 0.0
        features['name_core_exact_match'] = 1.0 if (s1_name_core and s1_name_core == s2_name_core) else 0.0

        if s1_name_clean and s2_name_clean:
            features['name_rapidfuzz_ratio'] = fuzz.ratio(s1_name_clean, s2_name_clean) / 100.0
            features['name_rapidfuzz_token_sort'] = fuzz.token_sort_ratio(s1_name_clean, s2_name_clean) / 100.0
            features['name_rapidfuzz_token_set'] = fuzz.token_set_ratio(s1_name_clean, s2_name_clean) / 100.0
        else:
            features['name_rapidfuzz_ratio'] = 0.0
            features['name_rapidfuzz_token_sort'] = 0.0
            features['name_rapidfuzz_token_set'] = 0.0

        features['name_jaccard_3gram'] = self._jaccard(s1_name_3g, s2_name_3g)
        features['name_jaccard_token'] = self._jaccard(s1_name_tok, s2_name_tok)
        features['name_token_containment'] = self._containment(s1_name_tok, s2_name_tok)
        features['name_sequence_ratio'] = self._sequence_ratio(s1_name_clean, s2_name_clean)

        len_s1_n = len(s1_name_clean)
        len_s2_n = len(s2_name_clean)
        features['name_len_diff'] = float(abs(len_s1_n - len_s2_n))
        features['name_len_ratio'] = min(len_s1_n, len_s2_n) / max(1, max(len_s1_n, len_s2_n))
        features['name_tok_diff'] = float(abs(len(s1_name_tok) - len(s2_name_tok)))

        # 2. Address Features
        features['address_exact_match'] = 1.0 if (s1_addr_clean and s1_addr_clean == s2_addr_clean) else 0.0
        features['address_core_exact_match'] = 1.0 if (s1_addr_core and s1_addr_core == s2_addr_core) else 0.0

        if s1_addr_clean and s2_addr_clean:
            features['address_rapidfuzz_ratio'] = fuzz.ratio(s1_addr_clean, s2_addr_clean) / 100.0
            features['address_rapidfuzz_token_set'] = fuzz.token_set_ratio(s1_addr_clean, s2_addr_clean) / 100.0
        else:
            features['address_rapidfuzz_ratio'] = 0.0
            features['address_rapidfuzz_token_set'] = 0.0

        features['address_jaccard_3gram'] = self._jaccard(s1_addr_3g, s2_addr_3g)
        features['address_jaccard_token'] = self._jaccard(s1_addr_tok, s2_addr_tok)
        features['address_token_containment'] = self._containment(s1_addr_tok, s2_addr_tok)
        features['address_sequence_ratio'] = self._sequence_ratio(s1_addr_clean, s2_addr_clean)

        s1_nums = self._extract_numbers(s1_addr_clean)
        s2_nums = self._extract_numbers(s2_addr_clean)
        if s1_nums and s2_nums:
            features['address_numeric_match'] = self._jaccard(s1_nums, s2_nums)
            features['address_has_exact_number_match'] = 1.0 if (s1_nums & s2_nums) else 0.0
        elif not s1_nums and not s2_nums:
            features['address_numeric_match'] = 0.5
            features['address_has_exact_number_match'] = 0.5
        else:
            features['address_numeric_match'] = 0.0
            features['address_has_exact_number_match'] = 0.0

        len_s1_a = len(s1_addr_clean)
        len_s2_a = len(s2_addr_clean)
        features['address_len_diff'] = float(abs(len_s1_a - len_s2_a))
        features['address_len_ratio'] = min(len_s1_a, len_s2_a) / max(1, max(len_s1_a, len_s2_a))
        features['address_tok_diff'] = float(abs(len(s1_addr_tok) - len(s2_addr_tok)))

        # 3. Country Features (Open-Set Robust)
        if not s1_country and not s2_country:
            features['country_match'] = 0.5
        elif not s1_country or not s2_country:
            features['country_match'] = 0.5
        else:
            features['country_match'] = 1.0 if s1_country == s2_country else 0.0

        features['is_us'] = 1.0 if (s1_country in ('united states', 'us')) else 0.0
        features['is_india'] = 1.0 if (s1_country in ('india', 'in')) else 0.0

        # 4. Missing Fields
        features['s1_missing_name'] = 1.0 if not s1_name_clean else 0.0
        features['s2_missing_name'] = 1.0 if not s2_name_clean else 0.0
        features['s1_missing_address'] = 1.0 if not s1_addr_clean else 0.0
        features['s2_missing_address'] = 1.0 if not s2_addr_clean else 0.0
        features['both_missing_address'] = 1.0 if (not s1_addr_clean and not s2_addr_clean) else 0.0

        # 5. Retrieval Evidence & Target Source
        rank_val = float(retrieval_rank)
        features['retrieval_rank'] = rank_val
        features['retrieval_reciprocal_rank'] = 1.0 / max(1.0, rank_val)
        features['retrieval_log_rank'] = math.log1p(rank_val)

        target_id = str(s2_dict.get('id', ''))
        is_s3 = (target_source == 'source3') or target_id.startswith('S3-')
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
        features["name_shared_tokens"] = float(len(s1_name_tok & s2_name_tok))
        features["name_length_ratio"] = features["name_len_ratio"]
        features["name_first_token_match"] = float(
            bool(s1_name_core and s2_name_core)
            and s1_name_core.split()[0] == s2_name_core.split()[0]
        )
        s1_initials = "".join(token[0] for token in sorted(s1_name_tok) if token)
        s2_initials = "".join(token[0] for token in sorted(s2_name_tok) if token)
        features["name_initials_match"] = float(bool(s1_initials) and s1_initials == s2_initials)

        features["address_shared_tokens"] = float(len(s1_addr_tok & s2_addr_tok))
        features["address_length_ratio"] = features["address_len_ratio"]
        features["address_number_jaccard"] = features["address_numeric_match"]
        features["address_shared_numbers"] = float(len(s1_nums & s2_nums))

        features["country_both_present"] = float(bool(s1_country and s2_country))
        features["both_names_present"] = float(bool(s1_name_clean and s2_name_clean))
        features["both_addresses_present"] = float(bool(s1_addr_clean and s2_addr_clean))
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
        s1_dict = {
            "name": s1.name_clean,
            "addr": s1.address_clean,
            "country": s1.country,
        }
        target_dict = {
            "name": target.name_clean,
            "addr": target.address_clean,
            "country": target.country,
            "source": target.source,
        }
        return self.compute_features(
            s1_dict=s1_dict,
            s2_dict=target_dict,
            retrieval_rank=retrieval_rank,
            target_source="source3" if target.source in {"s3", "source3"} else "source2",
        )
