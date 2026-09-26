import sqlite3
import math
from array import array
from collections import defaultdict
import logging

from business_entity_resolution.src.preprocessing import NameNormalizer, AddressNormalizer, CountryNormalizer
from business_entity_resolution.src.indexing.indexer import get_ngrams
from business_entity_resolution.src.retrieval.country_filter import CountryFilter

class CandidateGenerator:
    def __init__(self, db_path: str, high_df_threshold: int = 50000):
        self.db_path = db_path
        self.high_df_threshold = high_df_threshold
        self.conn = sqlite3.connect(self.db_path)
        self.country_filter = CountryFilter(self.db_path)
        
        # We need the total number of docs for IDF calculation
        cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
        self.total_docs = cursor.fetchone()[0] or 1

    def close(self):
        """Close the SQLite connection held by this generator."""
        conn = getattr(self, "conn", None)
        if conn is not None:
            conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Destructors must not mask interpreter-shutdown errors.
            pass
        
    def _fetch_posting_list(self, table: str, token: str):
        """Fetches the posting list for a given token from a specific table."""
        cursor = self.conn.execute(f"SELECT postings FROM {table} WHERE token = ?", (token,))
        row = cursor.fetchone()
        if not row:
            return array('I')
        doc_ids = array('I')
        doc_ids.frombytes(row[0])
        return doc_ids

    def _get_idf(self, df: int) -> float:
        if df == 0:
            return 0.0
        # Smooth IDF
        return math.log(self.total_docs / df)

    def _score_block(self, table: str, tokens: set, current_scores: dict):
        """
        Retrieves and scores tokens from a specific index block.
        Implements Intersection-first for High-DF tokens to prevent explosion.
        """
        token_postings = {}
        for token in tokens:
            if not token:
                continue
            postings = self._fetch_posting_list(table, token)
            if len(postings) > 0:
                token_postings[token] = postings

        if not token_postings:
            return

        # Separate into low_df (anchors) and high_df (boosters)
        low_df_tokens = {t: p for t, p in token_postings.items() if len(p) <= self.high_df_threshold}
        high_df_tokens = {t: p for t, p in token_postings.items() if len(p) > self.high_df_threshold}
        
        # 1. Score Low-DF Tokens (Unions)
        for token, postings in low_df_tokens.items():
            idf = self._get_idf(len(postings))
            for doc in postings:
                current_scores[doc] += idf

        # 2. Score High-DF Tokens
        if low_df_tokens or current_scores:
            # If we have an anchor base, we only score the High-DF tokens for candidates already in our pool
            # This is a massive optimization (Intersection)
            for token, postings in high_df_tokens.items():
                idf = self._get_idf(len(postings))
                # Convert postings to set for fast membership testing
                posting_set = set(postings)
                for doc in list(current_scores.keys()):
                    if doc in posting_set:
                        current_scores[doc] += idf
        else:
            # Desperation: The query ONLY consists of High-DF tokens.
            # We must intersect them with each other to reduce the candidate pool.
            if high_df_tokens:
                # Sort by length (smallest first)
                sorted_high = sorted(high_df_tokens.items(), key=lambda x: len(x[1]))
                # Start with the smallest High-DF token as the base
                base_token, base_postings = sorted_high[0]
                base_idf = self._get_idf(len(base_postings))
                
                intersected_pool = set(base_postings)
                for token, postings in sorted_high[1:]:
                    intersected_pool &= set(postings)
                    if not intersected_pool:
                        break
                        
                # Now add the intersected pool to current_scores
                for doc in intersected_pool:
                    # Give them the score of all matching tokens
                    score = 0
                    for token, postings in high_df_tokens.items():
                        if doc in set(postings):
                            score += self._get_idf(len(postings))
                    current_scores[doc] += score

    def get_candidates(self, s1_name: str, s1_address: str, s1_country: str, budget: int):
        """
        Retrieves Top-B candidates using an adaptive, tiered approach.
        """
        norm_country = CountryNormalizer.normalize(s1_country)
        norm_country = "missing" if not norm_country else norm_country

        norm_name_core = NameNormalizer.core(s1_name)
        norm_name_clean = NameNormalizer.clean(s1_name)
        norm_address_core = AddressNormalizer.core(s1_address)
        norm_address_clean = AddressNormalizer.clean(s1_address)

        name_tokens = set(norm_name_core.split()) if norm_name_core else set()
        addr_tokens = set(norm_address_core.split()) if norm_address_core else set()
        
        scores = defaultdict(float)

        # Stage 1: Strict Anchoring
        self._score_block("name_tokens", name_tokens, scores)
        self._score_block("address_tokens", addr_tokens, scores)

        # Apply Country Filter
        for doc in list(scores.keys()):
            if not self.country_filter.is_match(doc, norm_country):
                del scores[doc]

        # Stage 2: Adaptive Expansion
        # Triggers: Very few candidates OR low max score OR missing country
        max_score = max(scores.values()) if scores else 0.0
        
        if len(scores) < budget or max_score < 10.0 or norm_country == "missing" or norm_country == "france":
            # Expand to N-grams
            name_ngrams = get_ngrams(norm_name_clean, 3)
            addr_ngrams = get_ngrams(norm_address_clean, 3)
            
            # To avoid noise explosion, we ONLY expand with relatively specific n-grams
            # (Handled natively by High-DF intersection logic in _score_block)
            self._score_block("name_ngrams", name_ngrams, scores)
            self._score_block("address_ngrams", addr_ngrams, scores)
            
            # Re-apply country filter on newly added docs
            for doc in list(scores.keys()):
                if not self.country_filter.is_match(doc, norm_country):
                    del scores[doc]

        # Stage 3: Top-B Truncation & Resolution
        if not scores:
            return []

        # Sort by score DESC, then doc_id ASC for determinism
        sorted_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        
        # Take Top-B
        top_b_docs = [doc_id for doc_id, score in sorted_docs[:budget]]
        
        # Resolve to entity_id and source
        placeholders = ",".join(["?"] * len(top_b_docs))
        query = f"SELECT doc_id, entity_id, source FROM documents WHERE doc_id IN ({placeholders})"
        
        results = []
        cursor = self.conn.execute(query, top_b_docs)
        doc_to_entity = {row[0]: (row[1], row[2]) for row in cursor}
        
        final_candidates = []
        for doc_id in top_b_docs:
            if doc_id in doc_to_entity:
                final_candidates.append(doc_to_entity[doc_id][0])
                
        return final_candidates
