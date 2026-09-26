"""Pairwise feature engineering for business entity matching."""

from __future__ import annotations

import difflib
import math
import re
from typing import Any

from rapidfuzz import fuzz

from business_entity_resolution.src.indexing.indexer import get_ngrams
from business_entity_resolution.src.preprocessing import (
    AddressNormalizer,
    CountryNormalizer,
    NameNormalizer,
)

NUMERIC_RE = re.compile(r'\b\d+\b')


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

        return features
