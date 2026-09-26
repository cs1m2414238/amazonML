import sqlite3
import array
import logging
from contextlib import closing

class CountryFilter:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.country_to_id = {"missing": 0}
        self.id_to_country = {0: "missing"}
        
        # We don't know the exact max doc_id until we query it.
        # But roughly 10.3M records means we can safely allocate a bytearray of ~12M.
        self.doc_countries = None
        self._load()

    def _load(self):
        logging.info("Loading Country Filter into RAM...")
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute("SELECT MAX(doc_id) FROM documents")
            max_id = cursor.fetchone()[0]
            if max_id is None:
                max_id = 100_000
                
            self.doc_countries = bytearray(max_id + 1)
            
            cursor = conn.execute("SELECT doc_id, country FROM documents")
            for doc_id, country in cursor:
                # Treat empty string as missing
                if not country:
                    country = "missing"
                    
                if country not in self.country_to_id:
                    cid = len(self.country_to_id)
                    self.country_to_id[country] = cid
                    self.id_to_country[cid] = country
                    
                self.doc_countries[doc_id] = self.country_to_id[country]
                
        logging.info(f"Loaded Country Filter: {len(self.country_to_id)} distinct countries.")

    def is_match(self, doc_id: int, target_country: str) -> bool:
        """
        Returns True if the document's country matches the target_country.
        If target_country is empty/'missing', it matches everything (global fallback).
        If the document's country is missing, it is eligible to match anything.
        """
        if not target_country or target_country == "missing":
            return True
            
        try:
            doc_cid = self.doc_countries[doc_id]
        except IndexError:
            return False
            
        if doc_cid == 0:  # Document is missing country
            return True
            
        target_cid = self.country_to_id.get(target_country, -1)
        return doc_cid == target_cid
