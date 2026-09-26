import unittest
import sqlite3
import os
import shutil
from array import array
from pathlib import Path
import pandas as pd

from business_entity_resolution.src.indexing.indexer import SPIMIIndexer

class TestIndexConstruction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("tests/temp_index_test")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create a synthetic dataset
        cls.synth_tsv = cls.test_dir / "test_synth_source2.tsv"
        data = [
            ("entity_id", "business_name", "business_address", "country"),
            ("S2-0001", "Apple Inc.", "1 Infinite Loop, Cupertino CA", "US"),
            ("S2-0002", "Banana Services Services", "2 Yellow Rd", "US"), # test duplicates in same doc
            ("S3-0003", "nan", "  ", " France "), # test missing values and open-set country
        ]
        with open(cls.synth_tsv, "w") as f:
            for row in data:
                f.write("\t".join(row) + "\n")
                
        cls.db_path = cls.test_dir / "test_indices.db"
        
        # Build index
        cls.indexer = SPIMIIndexer(cls.db_path, num_shards=2, chunk_size=10)
        cls.indexer.map_phase(cls.synth_tsv, "source2")
        cls.indexer.reduce_phase()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_close_reopen_and_retrieve(self):
        # Open connection
        conn = sqlite3.connect(self.db_path)
        
        # Query known name token 'apple'
        cursor = conn.execute("SELECT postings FROM name_tokens WHERE token = 'apple'")
        blob = cursor.fetchone()[0]
        
        # Decode blob
        doc_ids = array('I')
        doc_ids.frombytes(blob)
        self.assertEqual(len(doc_ids), 1)
        
        # Map doc_id to entity_id and source
        cursor = conn.execute("SELECT entity_id, source FROM documents WHERE doc_id = ?", (doc_ids[0],))
        entity_id, source = cursor.fetchone()
        
        self.assertEqual(entity_id, "S2-0001")
        self.assertEqual(source, "source2")
        conn.close()

    def test_ngrams(self):
        conn = sqlite3.connect(self.db_path)
        # 'app' is an ngram of 'apple'
        cursor = conn.execute("SELECT postings FROM name_ngrams WHERE token = 'app'")
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        conn.close()
        
    def test_duplicate_posting_prevention(self):
        conn = sqlite3.connect(self.db_path)
        # 'services' appears twice in S2-0002
        cursor = conn.execute("SELECT postings FROM name_tokens WHERE token = 'services'")
        blob = cursor.fetchone()[0]
        doc_ids = array('I')
        doc_ids.frombytes(blob)
        # Should only be recorded once for that document
        self.assertEqual(len(doc_ids), 1)
        conn.close()
        
    def test_missing_values_no_empty_keys(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM name_tokens WHERE token = ''")
        self.assertEqual(cursor.fetchone()[0], 0)
        conn.close()
        
    def test_open_set_country(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT country FROM documents WHERE entity_id = 'S3-0003'")
        self.assertEqual(cursor.fetchone()[0], "france")
        conn.close()

    def test_deterministic_rebuild(self):
        # Rebuild again
        db_path_2 = self.test_dir / "test_indices_2.db"
        indexer2 = SPIMIIndexer(db_path_2, num_shards=2, chunk_size=10)
        indexer2.map_phase(self.synth_tsv, "source2")
        indexer2.reduce_phase()
        
        # Compare file sizes (rough proxy for determinism since timestamps inside sqlite might vary slightly, 
        # but rows/blobs should be identical). Better yet, compare row counts in tables.
        conn1 = sqlite3.connect(self.db_path)
        conn2 = sqlite3.connect(db_path_2)
        
        count1 = conn1.execute("SELECT COUNT(*) FROM name_tokens").fetchone()[0]
        count2 = conn2.execute("SELECT COUNT(*) FROM name_tokens").fetchone()[0]
        self.assertEqual(count1, count2)
        conn1.close()
        conn2.close()

if __name__ == "__main__":
    unittest.main()
