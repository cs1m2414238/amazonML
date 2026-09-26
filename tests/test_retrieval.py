import unittest
import sqlite3
import os
import shutil
from contextlib import closing
from pathlib import Path
from array import array

from business_entity_resolution.src.retrieval.country_filter import CountryFilter
from business_entity_resolution.src.retrieval.candidate_generator import CandidateGenerator

class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("tests/temp_retrieval_test")
        cls.test_dir.mkdir(exist_ok=True)
        cls.db_path = cls.test_dir / "test_indices.db"
        
        # Setup mock db
        with closing(sqlite3.connect(cls.db_path)) as conn:
            conn.execute("CREATE TABLE documents (doc_id INTEGER PRIMARY KEY, entity_id TEXT, source TEXT, country TEXT)")
            tables = ["name_tokens", "name_ngrams", "address_tokens", "address_ngrams"]
            for t in tables:
                conn.execute(f"CREATE TABLE {t} (token TEXT PRIMARY KEY, postings BLOB)")
                
            docs = [
                (1, "S2-1", "source2", "us"),
                (2, "S3-2", "source3", "france"),
                (3, "S2-3", "source2", "missing"),
            ]
            conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", docs)
            
            # Insert fake postings
            # doc 1 & 2 have "apple"
            # doc 3 has "banana"
            apple_blob = array('I', [1, 2]).tobytes()
            banana_blob = array('I', [3]).tobytes()
            conn.execute("INSERT INTO name_tokens VALUES (?, ?)", ("apple", apple_blob))
            conn.execute("INSERT INTO name_tokens VALUES (?, ?)", ("banana", banana_blob))
            conn.commit()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_country_filter(self):
        cf = CountryFilter(str(self.db_path))
        # "us" matches doc 1 and missing doc 3
        self.assertTrue(cf.is_match(1, "us"))
        self.assertFalse(cf.is_match(2, "us"))
        self.assertTrue(cf.is_match(3, "us")) # doc 3 is missing, matches anything
        
        # "france" matches doc 2 and doc 3
        self.assertTrue(cf.is_match(2, "france"))
        self.assertTrue(cf.is_match(3, "france"))
        
        # S1 country is missing -> matches all
        self.assertTrue(cf.is_match(1, "missing"))
        self.assertTrue(cf.is_match(2, "missing"))

    def test_candidate_generator(self):
        gen = CandidateGenerator(str(self.db_path))
        self.addCleanup(gen.close)
        
        # Query 'apple' with country 'us' -> Should return S2-1 (doc 1) because doc 2 is france
        res = gen.get_candidates("apple", "", "us", budget=10)
        self.assertEqual(res, ["S2-1"])
        
        # Query 'apple' with missing country -> Should return both S2-1 and S3-2
        res2 = gen.get_candidates("apple", "", "missing", budget=10)
        self.assertEqual(set(res2), {"S2-1", "S3-2"})
        
    def test_budget_truncation(self):
        gen = CandidateGenerator(str(self.db_path))
        self.addCleanup(gen.close)
        # Query 'apple' missing country gives 2 docs. If B=1, we should only get 1 doc back.
        res = gen.get_candidates("apple", "", "missing", budget=1)
        self.assertEqual(len(res), 1)

if __name__ == "__main__":
    unittest.main()
