import unittest
from business_entity_resolution.src.features.pair_features import PairFeatureGenerator

class TestPairFeatures(unittest.TestCase):
    def setUp(self):
        self.feat_gen = PairFeatureGenerator()
        
    def test_exact_match(self):
        s1 = {'name': 'Apple Inc.', 'addr': '1 Infinite Loop, Cupertino CA', 'country': 'US'}
        s2 = {'name': 'apple inc', 'addr': '1 infinite loop, cupertino ca', 'country': 'us'}
        
        feat = self.feat_gen.compute_features(s1, s2, retrieval_rank=1)
        self.assertEqual(feat['name_exact_match'], 1.0)
        self.assertEqual(feat['address_exact_match'], 1.0)
        self.assertEqual(feat['country_match'], 1.0)
        
    def test_missing_fields(self):
        s1 = {'name': 'Apple Inc.', 'addr': '', 'country': 'US'}
        s2 = {'name': '', 'addr': '1 Infinite Loop, Cupertino CA', 'country': ''}
        
        feat = self.feat_gen.compute_features(s1, s2, retrieval_rank=5)
        self.assertEqual(feat['s1_missing_address'], 1.0)
        self.assertEqual(feat['s2_missing_name'], 1.0)
        # s2 missing country, s1 has country -> mismatch/partial match
        self.assertEqual(feat['country_match'], 0.5)
        
    def test_jaccard_and_sequence(self):
        s1 = {'name': 'McDonalds', 'addr': '', 'country': ''}
        s2 = {'name': 'McDonald', 'addr': '', 'country': ''}
        
        feat = self.feat_gen.compute_features(s1, s2, retrieval_rank=2)
        # sequence ratio > 0.9
        self.assertTrue(feat['name_sequence_ratio'] > 0.9)
        # not exact
        self.assertEqual(feat['name_exact_match'], 0.0)

    def test_retrieval_evidence(self):
        s1 = {'name': 'A', 'addr': 'B', 'country': 'US'}
        feat = self.feat_gen.compute_features(s1, s1, retrieval_rank=10)
        self.assertEqual(feat['retrieval_rank'], 10.0)
        self.assertEqual(feat['retrieval_reciprocal_rank'], 0.1)

if __name__ == "__main__":
    unittest.main()
