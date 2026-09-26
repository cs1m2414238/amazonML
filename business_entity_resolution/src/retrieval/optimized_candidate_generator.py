"""Faster candidate retrieval that preserves the baseline scoring semantics."""

from __future__ import annotations

import math
import sqlite3
import time
from heapq import nsmallest
from array import array
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

from business_entity_resolution.src.indexing.indexer import get_ngrams
from business_entity_resolution.src.preprocessing import (
    AddressNormalizer,
    CountryNormalizer,
    NameNormalizer,
)
from business_entity_resolution.src.retrieval.country_filter import CountryFilter


class PostingCache:
    """Byte-bounded LRU cache for decoded posting arrays."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self._items = OrderedDict()

    def get(self, key):
        value = self._items.get(key)
        if value is None:
            return None
        self._items.move_to_end(key)
        return value

    def put(self, key, postings):
        if self.max_bytes <= 0:
            return
        size = len(postings) * postings.itemsize
        if size > self.max_bytes:
            return
        old = self._items.pop(key, None)
        if old is not None:
            self.current_bytes -= len(old) * old.itemsize
        self._items[key] = postings
        self.current_bytes += size
        while self.current_bytes > self.max_bytes and self._items:
            _, evicted = self._items.popitem(last=False)
            self.current_bytes -= len(evicted) * evicted.itemsize

    @property
    def entries(self):
        return len(self._items)


class OptimizedCandidateGenerator:
    """Drop-in retrieval equivalent with lower SQLite and posting-list overhead."""

    TABLES = {"name_tokens", "name_ngrams", "address_tokens", "address_ngrams"}

    def __init__(
        self,
        db_path: str | Path,
        high_df_threshold: int = 50_000,
        posting_cache_mb: int = 1024,
        sqlite_cache_mb: int = 512,
        mmap_mb: int = 8192,
    ):
        self.db_path = Path(db_path).resolve()
        self.high_df_threshold = high_df_threshold
        uri = f"file:{self.db_path.as_posix()}?mode=ro&immutable=1"
        self.conn = sqlite3.connect(uri, uri=True)
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.execute(f"PRAGMA cache_size = -{int(sqlite_cache_mb) * 1024}")
        self.conn.execute(f"PRAGMA mmap_size = {int(mmap_mb) * 1024 * 1024}")
        self.country_filter = CountryFilter(str(self.db_path))
        self.posting_cache = PostingCache(posting_cache_mb * 1024 * 1024)

        cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
        self.total_docs = cursor.fetchone()[0] or 1
        self.profile = defaultdict(float)

    def close(self):
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
            pass

    def reset_profile(self):
        self.profile.clear()

    def profile_snapshot(self):
        result = dict(self.profile)
        result["posting_cache_entries"] = self.posting_cache.entries
        result["posting_cache_mb"] = round(
            self.posting_cache.current_bytes / (1024 * 1024), 2
        )
        return result

    def _get_idf(self, df: int) -> float:
        if df == 0:
            return 0.0
        return math.log(self.total_docs / df)

    def _fetch_posting_lists(self, table: str, tokens) -> dict[str, array]:
        if table not in self.TABLES:
            raise ValueError(f"Unsupported postings table: {table}")

        token_list = [token for token in tokens if token]
        result = {}
        misses = []
        for token in token_list:
            key = (table, token)
            cached = self.posting_cache.get(key)
            if cached is None:
                misses.append(token)
                self.profile["cache_misses"] += 1
            else:
                result[token] = cached
                self.profile["cache_hits"] += 1

        for start in range(0, len(misses), 900):
            batch = misses[start : start + 900]
            placeholders = ",".join("?" for _ in batch)
            query = f"SELECT token, postings FROM {table} WHERE token IN ({placeholders})"

            t0 = time.perf_counter()
            rows = self.conn.execute(query, batch).fetchall()
            self.profile["sqlite_fetch_seconds"] += time.perf_counter() - t0
            self.profile["sqlite_batch_queries"] += 1

            returned = set()
            for token, blob in rows:
                returned.add(token)
                t0 = time.perf_counter()
                postings = array("I")
                postings.frombytes(blob)
                self.profile["posting_decode_seconds"] += time.perf_counter() - t0
                self.profile["posting_lists_decoded"] += 1
                self.profile["posting_ids_decoded"] += len(postings)
                result[token] = postings
                self.posting_cache.put((table, token), postings)

            for token in batch:
                if token not in returned:
                    postings = array("I")
                    result[token] = postings
                    self.posting_cache.put((table, token), postings)

        return {token: result[token] for token in token_list}

    @staticmethod
    def _as_numpy(postings: array) -> np.ndarray:
        """Expose an array('I') as a zero-copy sorted uint32 NumPy view."""
        return np.frombuffer(postings, dtype=np.uint32)

    def _score_block(self, table: str, tokens: set, current_scores: dict):
        t_fetch = time.perf_counter()
        token_postings = {
            token: postings
            for token, postings in self._fetch_posting_lists(table, tokens).items()
            if postings
        }
        self.profile["fetch_total_seconds"] += time.perf_counter() - t_fetch
        if not token_postings:
            return

        low_df_tokens = {
            token: postings
            for token, postings in token_postings.items()
            if len(postings) <= self.high_df_threshold
        }
        high_df_tokens = {
            token: postings
            for token, postings in token_postings.items()
            if len(postings) > self.high_df_threshold
        }

        t0 = time.perf_counter()
        for token, postings in low_df_tokens.items():
            idf = self._get_idf(len(postings))
            for doc_id in postings:
                current_scores[doc_id] += idf
        self.profile["low_df_scoring_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        if low_df_tokens or current_scores:
            candidate_ids = np.fromiter(
                current_scores.keys(), dtype=np.uint32, count=len(current_scores)
            )
            candidate_scores = np.fromiter(
                current_scores.values(), dtype=np.float64, count=len(current_scores)
            )
            changed = np.zeros(len(candidate_ids), dtype=np.bool_)
            for token, postings in high_df_tokens.items():
                idf = self._get_idf(len(postings))
                posting_ids = self._as_numpy(postings)
                positions = np.searchsorted(posting_ids, candidate_ids)
                valid = positions < len(posting_ids)
                matched = np.zeros(len(candidate_ids), dtype=np.bool_)
                matched[valid] = posting_ids[positions[valid]] == candidate_ids[valid]
                candidate_scores[matched] += idf
                changed |= matched
            for doc_id, score in zip(candidate_ids[changed], candidate_scores[changed]):
                current_scores[int(doc_id)] = float(score)
        elif high_df_tokens:
            sorted_high = sorted(high_df_tokens.items(), key=lambda item: len(item[1]))
            intersection = self._as_numpy(sorted_high[0][1])
            for _, postings in sorted_high[1:]:
                intersection = np.intersect1d(
                    intersection, self._as_numpy(postings), assume_unique=True
                )
                if not len(intersection):
                    break
            total_score = sum(
                self._get_idf(len(postings)) for postings in high_df_tokens.values()
            )
            for doc_id in intersection:
                current_scores[int(doc_id)] += total_score
        self.profile["high_df_scoring_seconds"] += time.perf_counter() - t0

    def _apply_country_filter(self, scores: dict, target_country: str):
        if not scores or not target_country or target_country == "missing":
            return

        candidate_ids = np.fromiter(
            scores.keys(), dtype=np.uint32, count=len(scores)
        )
        country_ids = np.frombuffer(self.country_filter.doc_countries, dtype=np.uint8)
        target_id = self.country_filter.country_to_id.get(target_country, -1)
        valid = candidate_ids < len(country_ids)
        allowed = np.zeros(len(candidate_ids), dtype=np.bool_)
        allowed[valid] = (
            (country_ids[candidate_ids[valid]] == 0)
            | (country_ids[candidate_ids[valid]] == target_id)
        )
        for doc_id in candidate_ids[~allowed]:
            del scores[int(doc_id)]

    def get_candidates(
        self,
        s1_name: str,
        s1_address: str,
        s1_country: str,
        budget: int,
    ):
        query_start = time.perf_counter()
        t0 = query_start

        norm_country = CountryNormalizer.normalize(s1_country) or "missing"
        norm_name_core = NameNormalizer.core(s1_name)
        norm_name_clean = NameNormalizer.clean(s1_name)
        norm_address_core = AddressNormalizer.core(s1_address)
        norm_address_clean = AddressNormalizer.clean(s1_address)
        name_tokens = set(norm_name_core.split()) if norm_name_core else set()
        addr_tokens = set(norm_address_core.split()) if norm_address_core else set()
        self.profile["normalization_seconds"] += time.perf_counter() - t0

        scores = defaultdict(float)
        self._score_block("name_tokens", name_tokens, scores)
        self._score_block("address_tokens", addr_tokens, scores)

        t0 = time.perf_counter()
        self._apply_country_filter(scores, norm_country)
        self.profile["country_filter_seconds"] += time.perf_counter() - t0

        max_score = max(scores.values()) if scores else 0.0
        expanded = (
            len(scores) < budget
            or max_score < 10.0
            or norm_country == "missing"
            or norm_country == "france"
        )
        if expanded:
            self.profile["expanded_queries"] += 1
            self._score_block("name_ngrams", get_ngrams(norm_name_clean, 3), scores)
            self._score_block(
                "address_ngrams", get_ngrams(norm_address_clean, 3), scores
            )
            t0 = time.perf_counter()
            self._apply_country_filter(scores, norm_country)
            self.profile["country_filter_seconds"] += time.perf_counter() - t0

        if not scores:
            self.profile["queries"] += 1
            self.profile["query_total_seconds"] += time.perf_counter() - query_start
            return []

        t0 = time.perf_counter()
        ranking_key = lambda item: (-item[1], item[0])
        if len(scores) > budget:
            top_items = nsmallest(budget, scores.items(), key=ranking_key)
        else:
            top_items = sorted(scores.items(), key=ranking_key)
        top_b_docs = [doc_id for doc_id, _ in top_items]
        self.profile["ranking_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        placeholders = ",".join("?" for _ in top_b_docs)
        query = f"SELECT doc_id, entity_id FROM documents WHERE doc_id IN ({placeholders})"
        rows = self.conn.execute(query, top_b_docs).fetchall()
        doc_to_entity = {doc_id: entity_id for doc_id, entity_id in rows}
        candidates = [doc_to_entity[doc_id] for doc_id in top_b_docs if doc_id in doc_to_entity]
        self.profile["document_resolution_seconds"] += time.perf_counter() - t0
        self.profile["queries"] += 1
        self.profile["query_total_seconds"] += time.perf_counter() - query_start
        self.profile["candidate_pool_total"] += len(scores)
        self.profile["candidate_pool_max"] = max(
            self.profile["candidate_pool_max"], len(scores)
        )
        return candidates
