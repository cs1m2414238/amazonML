import os
import sqlite3
import pickle
from array import array
from collections import defaultdict
import time
import shutil
import logging
from pathlib import Path
from contextlib import closing

from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.preprocessing.name_normalizer import NameNormalizer
from business_entity_resolution.src.preprocessing.address_normalizer import AddressNormalizer
from business_entity_resolution.src.preprocessing.country_normalizer import CountryNormalizer
from business_entity_resolution.src.preprocessing.missing_handler import is_missing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_ngrams(text, n=3):
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text)-n+1)}

class SPIMIIndexer:
    def __init__(self, db_path, num_shards=64, chunk_size=100_000):
        self.db_path = Path(db_path)
        self.num_shards = num_shards
        self.chunk_size = chunk_size
        self.shard_dir = self.db_path.with_name(self.db_path.name + "_shards")
        
        self.doc_counter = 0
        self.missing_stats = {
            "business_name": {"raw_missing": 0, "normalized_empty": 0},
            "business_address": {"raw_missing": 0, "normalized_empty": 0},
            "country": {"raw_missing": 0, "normalized_empty": 0},
        }
        
        self._init_db()
        os.makedirs(self.shard_dir, exist_ok=True)

    def _init_db(self):
        # Create empty DB and tables
        if self.db_path.exists():
            self.db_path.unlink()
            
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA journal_mode = MEMORY")
            
            conn.execute("""
                CREATE TABLE documents (
                    doc_id INTEGER PRIMARY KEY,
                    entity_id TEXT,
                    source TEXT,
                    country TEXT,
                    name TEXT,
                    addr TEXT
                )
            """)
            
            tables = ["name_tokens", "name_ngrams", "address_tokens", "address_ngrams"]
            for t in tables:
                conn.execute(f"""
                    CREATE TABLE {t} (
                        token TEXT PRIMARY KEY,
                        postings BLOB
                    )
                """)
            conn.commit()
                
    def _get_shard_id(self, token):
        return hash(token) % self.num_shards

    def map_phase(self, file_path, source_name):
        logging.info(f"Starting MAP phase for {file_path}")
        source_start = time.time()
        source_docs = 0
        
        # We write documents to sqlite immediately to save memory
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        
        for chunk in read_tsv_chunks(file_path, source_name, self.chunk_size):
            # Local RAM aggregation per chunk to massively reduce I/O
            local_idx = [defaultdict(set) for _ in range(4)]
            docs_to_insert = []
            
            for row in chunk.itertuples(index=False):
                self.doc_counter += 1
                source_docs += 1
                doc_id = self.doc_counter
                
                # Tracking missing values
                if is_missing(row.business_name): self.missing_stats["business_name"]["raw_missing"] += 1
                if is_missing(row.business_address): self.missing_stats["business_address"]["raw_missing"] += 1
                if is_missing(row.country): self.missing_stats["country"]["raw_missing"] += 1
                
                # Normalization
                norm_country = CountryNormalizer.normalize(row.country)
                norm_name_core = NameNormalizer.core(row.business_name)
                norm_name_clean = NameNormalizer.clean(row.business_name)
                norm_address_core = AddressNormalizer.core(row.business_address)
                norm_address_clean = AddressNormalizer.clean(row.business_address)
                
                if not norm_name_core: self.missing_stats["business_name"]["normalized_empty"] += 1
                if not norm_address_core: self.missing_stats["business_address"]["normalized_empty"] += 1
                if not norm_country: self.missing_stats["country"]["normalized_empty"] += 1

                # Routing to DB
                raw_name = "" if is_missing(row.business_name) else str(row.business_name).strip()
                raw_addr = "" if is_missing(row.business_address) else str(row.business_address).strip()
                docs_to_insert.append((doc_id, str(row.entity_id).strip(), source_name, norm_country, raw_name, raw_addr))
                
                # Tokenization & N-grams (Using sets per document to prevent duplicate postings)
                tokens_name = set(norm_name_core.split()) if norm_name_core else set()
                ngrams_name = get_ngrams(norm_name_clean, 3)
                tokens_addr = set(norm_address_core.split()) if norm_address_core else set()
                ngrams_addr = get_ngrams(norm_address_clean, 3)
                
                # Map to local index
                for t in tokens_name: local_idx[0][t].add(doc_id)
                for t in ngrams_name: local_idx[1][t].add(doc_id)
                for t in tokens_addr: local_idx[2][t].add(doc_id)
                for t in ngrams_addr: local_idx[3][t].add(doc_id)
                
            # Flush documents to DB
            conn.executemany("INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)", docs_to_insert)
            conn.commit()
            
            # Distribute local_idx to shard files
            # shard_data structure: shard_id -> type_id -> token -> list(doc_ids)
            shard_data = [[defaultdict(list) for _ in range(4)] for _ in range(self.num_shards)]
            
            for type_id in range(4):
                for token, docs in local_idx[type_id].items():
                    s = self._get_shard_id(token)
                    shard_data[s][type_id][token].extend(docs)
                    
            for s in range(self.num_shards):
                if any(shard_data[s]): # only write if not empty
                    shard_path = self.shard_dir / f"shard_{s}.pkl"
                    with open(shard_path, "ab") as f:
                        pickle.dump(shard_data[s], f)

            if source_docs % 1_000_000 < len(chunk):
                elapsed = time.time() - source_start
                rate = source_docs / elapsed if elapsed else 0.0
                logging.info(
                    f"MAP {source_name}: {source_docs:,} rows "
                    f"({rate:,.0f} rows/s)"
                )

        elapsed = time.time() - source_start
        rate = source_docs / elapsed if elapsed else 0.0
        logging.info(
            f"Completed MAP {source_name}: {source_docs:,} rows in "
            f"{elapsed:.1f}s ({rate:,.0f} rows/s)"
        )
                        
        conn.close()

    def reduce_phase(self):
        logging.info("Starting REDUCE phase")
        tables = ["name_tokens", "name_ngrams", "address_tokens", "address_ngrams"]
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA journal_mode = MEMORY")
        
        stats = {
            "unique_keys": [0, 0, 0, 0],
            "largest_posting_list": [(0, ""), (0, ""), (0, ""), (0, "")]
        }
        
        for s in range(self.num_shards):
            shard_path = self.shard_dir / f"shard_{s}.pkl"
            if not shard_path.exists():
                continue
            logging.info(f"REDUCE shard {s + 1}/{self.num_shards}")
                
            # Load and merge all chunks for this shard
            global_shard = [defaultdict(set) for _ in range(4)]
            with open(shard_path, "rb") as f:
                while True:
                    try:
                        data = pickle.load(f)
                        for type_id in range(4):
                            for token, docs in data[type_id].items():
                                global_shard[type_id][token].update(docs)
                    except EOFError:
                        break
                        
            # Write global_shard to DB
            for type_id in range(4):
                insert_data = []
                for token, docs in global_shard[type_id].items():
                    if not token: # Ensure no empty string keys
                        continue
                        
                    sorted_docs = sorted(list(docs))
                    blob = array('I', sorted_docs).tobytes()
                    insert_data.append((token, blob))
                    
                    # Tracking stats
                    stats["unique_keys"][type_id] += 1
                    if len(sorted_docs) > stats["largest_posting_list"][type_id][0]:
                        stats["largest_posting_list"][type_id] = (len(sorted_docs), token)
                        
                if insert_data:
                    conn.executemany(f"""
                        INSERT INTO {tables[type_id]} (token, postings) VALUES (?, ?)
                        ON CONFLICT(token) DO UPDATE SET postings = postings || excluded.postings
                    """, insert_data)
                    
            # Delete shard file to save disk space
            shard_path.unlink()
            
        logging.info("Creating index on documents(entity_id)...")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_entity_id ON documents(entity_id)")
        conn.commit()
        conn.close()
        
        # Cleanup shard directory
        shutil.rmtree(self.shard_dir, ignore_errors=True)
        return stats
