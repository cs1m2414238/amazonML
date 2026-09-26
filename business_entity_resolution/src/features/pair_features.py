import difflib
import math

from business_entity_resolution.src.preprocessing import NameNormalizer, AddressNormalizer, CountryNormalizer
from business_entity_resolution.src.indexing.indexer import get_ngrams

class PairFeatureGenerator:
    def __init__(self):
        pass

    def _jaccard(self, set1, set2):
        if not set1 and not set2:
            return 0.0
        return len(set1 & set2) / max(1, len(set1 | set2))

    def _sequence_ratio(self, str1, str2):
        if not str1 and not str2:
            return 0.0
        if not str1 or not str2:
            return 0.0
        return difflib.SequenceMatcher(None, str1, str2).ratio()

    def compute_features(self, s1_dict, s2_dict, retrieval_rank):
        """
        s1_dict, s2_dict should contain: 'name', 'addr', 'country'
        Returns a flat dictionary of numeric features.
        """
        # Normalize
        s1_name_clean = NameNormalizer.clean(s1_dict.get('name', ''))
        s1_name_core = NameNormalizer.core(s1_dict.get('name', ''))
        s2_name_clean = NameNormalizer.clean(s2_dict.get('name', ''))
        s2_name_core = NameNormalizer.core(s2_dict.get('name', ''))

        s1_addr_clean = AddressNormalizer.clean(s1_dict.get('addr', ''))
        s1_addr_core = AddressNormalizer.core(s1_dict.get('addr', ''))
        s2_addr_clean = AddressNormalizer.clean(s2_dict.get('addr', ''))
        s2_addr_core = AddressNormalizer.core(s2_dict.get('addr', ''))

        s1_country = CountryNormalizer.normalize(s1_dict.get('country', ''))
        s2_country = CountryNormalizer.normalize(s2_dict.get('country', ''))

        # Name Sets
        s1_name_tok = set(s1_name_core.split()) if s1_name_core else set()
        s2_name_tok = set(s2_name_core.split()) if s2_name_core else set()
        s1_name_3g = get_ngrams(s1_name_clean, 3)
        s2_name_3g = get_ngrams(s2_name_clean, 3)

        # Addr Sets
        s1_addr_tok = set(s1_addr_core.split()) if s1_addr_core else set()
        s2_addr_tok = set(s2_addr_core.split()) if s2_addr_core else set()
        s1_addr_3g = get_ngrams(s1_addr_clean, 3)
        s2_addr_3g = get_ngrams(s2_addr_clean, 3)

        features = {}

        # 1. Name Features
        features['name_exact_match'] = 1.0 if (s1_name_clean and s1_name_clean == s2_name_clean) else 0.0
        features['name_jaccard_3gram'] = self._jaccard(s1_name_3g, s2_name_3g)
        features['name_jaccard_token'] = self._jaccard(s1_name_tok, s2_name_tok)
        features['name_sequence_ratio'] = self._sequence_ratio(s1_name_clean, s2_name_clean)
        features['name_len_diff'] = abs(len(s1_name_clean) - len(s2_name_clean))
        features['name_tok_diff'] = abs(len(s1_name_tok) - len(s2_name_tok))
        
        # 2. Address Features
        features['address_exact_match'] = 1.0 if (s1_addr_clean and s1_addr_clean == s2_addr_clean) else 0.0
        features['address_jaccard_3gram'] = self._jaccard(s1_addr_3g, s2_addr_3g)
        features['address_jaccard_token'] = self._jaccard(s1_addr_tok, s2_addr_tok)
        features['address_sequence_ratio'] = self._sequence_ratio(s1_addr_clean, s2_addr_clean)
        features['address_len_diff'] = abs(len(s1_addr_clean) - len(s2_addr_clean))
        features['address_tok_diff'] = abs(len(s1_addr_tok) - len(s2_addr_tok))
        
        # 3. Country Feature
        if not s1_country and not s2_country:
            features['country_match'] = 0.5
        elif not s1_country or not s2_country:
            features['country_match'] = 0.5
        else:
            features['country_match'] = 1.0 if s1_country == s2_country else 0.0
            
        # 4. Missing Fields
        features['s1_missing_name'] = 1.0 if not s1_name_clean else 0.0
        features['s2_missing_name'] = 1.0 if not s2_name_clean else 0.0
        features['s1_missing_address'] = 1.0 if not s1_addr_clean else 0.0
        features['s2_missing_address'] = 1.0 if not s2_addr_clean else 0.0
        
        # 5. Retrieval Evidence (Rank proxy for IDF)
        features['retrieval_rank'] = float(retrieval_rank)
        features['retrieval_reciprocal_rank'] = 1.0 / max(1, retrieval_rank)
        
        return features
