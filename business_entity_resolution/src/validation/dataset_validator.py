import json
import time
from pathlib import Path
from collections import defaultdict
import logging

from business_entity_resolution.src.ingestion.reader import read_tsv_chunks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class DatasetValidator:
    def __init__(self, data_root, chunk_size=100_000):
        self.data_root = Path(data_root)
        self.chunk_size = chunk_size
        self.report = {
            "train": {},
            "test": {},
            "cross_file_consistency": {}
        }
        
        # We store just integer IDs to save massive amounts of RAM.
        # e.g., 'S1-12345' -> 12345
        self.unseen_gt_s1 = set()
        self.unseen_gt_s2 = set()
        self.unseen_gt_s3 = set()
        
        # Set to safely keep track of what prefix maps to what set
        self._unseen_target_sets = {
            "S2": self.unseen_gt_s2,
            "S3": self.unseen_gt_s3
        }

    def _extract_id_num(self, entity_id, prefix):
        # Optimized ID extraction and validation
        if not entity_id.startswith(f"{prefix}-"):
            return None
        try:
            return int(entity_id[len(prefix)+1:])
        except ValueError:
            return None

    def validate_ground_truth(self):
        file_path = self.data_root / "dataset/train/train_ground_truth.tsv"
        logging.info(f"Validating Ground Truth: {file_path.name}")
        
        stats = {
            "total_rows": 0,
            "zero_matches": 0,
            "one_match": 0,
            "more_than_one_match": 0,
            "total_matched_pairs": 0,
            "max_matches": 0,
            "invalid_s1_prefix": 0,
            "invalid_matched_prefix": 0,
            "s1_in_matched": 0,
            "intra_row_duplicates": 0,
        }
        
        for chunk in read_tsv_chunks(file_path, "ground_truth", self.chunk_size):
            for row in chunk.itertuples(index=False):
                stats["total_rows"] += 1
                
                s1_id = row.source1_entity_id
                s1_num = self._extract_id_num(s1_id, "S1")
                if s1_num is None:
                    stats["invalid_s1_prefix"] += 1
                else:
                    self.unseen_gt_s1.add(s1_num)
                
                matched_str = row.matched_entity_ids
                if not matched_str:
                    stats["zero_matches"] += 1
                    continue
                
                matched_ids = matched_str.split(',')
                num_matches = len(matched_ids)
                
                if num_matches == 1:
                    stats["one_match"] += 1
                else:
                    stats["more_than_one_match"] += 1
                    
                stats["total_matched_pairs"] += num_matches
                stats["max_matches"] = max(stats["max_matches"], num_matches)
                
                # Intra-row deduplication check
                if len(set(matched_ids)) != num_matches:
                    stats["intra_row_duplicates"] += 1
                
                for mid in matched_ids:
                    if mid.startswith("S1-"):
                        stats["s1_in_matched"] += 1
                    elif mid.startswith("S2-"):
                        mnum = self._extract_id_num(mid, "S2")
                        if mnum is None:
                            stats["invalid_matched_prefix"] += 1
                        else:
                            self.unseen_gt_s2.add(mnum)
                    elif mid.startswith("S3-"):
                        mnum = self._extract_id_num(mid, "S3")
                        if mnum is None:
                            stats["invalid_matched_prefix"] += 1
                        else:
                            self.unseen_gt_s3.add(mnum)
                    else:
                        stats["invalid_matched_prefix"] += 1

        self.report["train"]["ground_truth"] = stats
        logging.info(f"GT Stats: {stats}")

    def validate_source(self, split, source_name):
        file_path = self.data_root / f"dataset/{split}/{split}_{source_name}.tsv"
        logging.info(f"Validating Source: {file_path.name}")
        
        prefix = "S" + source_name[-1] # "S1", "S2", or "S3"
        
        stats = {
            "total_rows": 0,
            "missing_business_name": 0,
            "missing_business_address": 0,
            "missing_country": 0,
            "invalid_prefix_count": 0,
            "duplicate_rows": 0,
            "country_distribution": defaultdict(int),
        }
        
        # Local memory-efficient set of integers for duplicate check
        seen_ids = set()
        
        # Target set for cross-file check (only if train split)
        target_gt_set = None
        if split == "train":
            if prefix == "S1":
                target_gt_set = self.unseen_gt_s1
            elif prefix == "S2":
                target_gt_set = self.unseen_gt_s2
            elif prefix == "S3":
                target_gt_set = self.unseen_gt_s3

        for chunk in read_tsv_chunks(file_path, source_name, self.chunk_size):
            # Missing checks
            stats["missing_business_name"] += int((chunk["business_name"] == "").sum())
            stats["missing_business_address"] += int((chunk["business_address"] == "").sum())
            stats["missing_country"] += int((chunk["country"] == "").sum())
            
            # Group by country
            vc = chunk["country"].value_counts().to_dict()
            for c, count in vc.items():
                stats["country_distribution"][c] += count
                
            for entity_id in chunk["entity_id"]:
                stats["total_rows"] += 1
                
                id_num = self._extract_id_num(entity_id, prefix)
                if id_num is None:
                    stats["invalid_prefix_count"] += 1
                else:
                    if id_num in seen_ids:
                        stats["duplicate_rows"] += 1
                    else:
                        seen_ids.add(id_num)
                        
                    # Remove from unseen GT set if tracking
                    if target_gt_set is not None:
                        target_gt_set.discard(id_num)
                        
        # Ensure country distribution is purely a dictionary for JSON output
        stats["country_distribution"] = dict(stats["country_distribution"])
        self.report[split][source_name] = stats
        logging.info(f"Source Stats for {file_path.name}: {stats}")
        seen_ids.clear() # Free up local duplicate set immediately

    def run_all(self):
        t0 = time.time()
        
        # 1. Validate Ground Truth first to populate sets
        self.validate_ground_truth()
        
        # 2. Validate Train Sources (and dynamically discard found GT IDs)
        for src in ["source1", "source2", "source3"]:
            self.validate_source("train", src)
            
        # 3. Check Cross-File Consistency
        missing_s1 = len(self.unseen_gt_s1)
        missing_s2 = len(self.unseen_gt_s2)
        missing_s3 = len(self.unseen_gt_s3)
        
        self.report["cross_file_consistency"] = {
            "missing_s1_in_train_source1": missing_s1,
            "missing_s2_in_train_source2": missing_s2,
            "missing_s3_in_train_source3": missing_s3,
        }
        logging.info(f"Cross File Consistency: {self.report['cross_file_consistency']}")
        
        # We can free the sets now to save memory before test
        self.unseen_gt_s1.clear()
        self.unseen_gt_s2.clear()
        self.unseen_gt_s3.clear()
        
        # 4. Validate Test Sources
        for src in ["source1", "source2", "source3"]:
            self.validate_source("test", src)
            
        t1 = time.time()
        self.report["metadata"] = {
            "validation_runtime_seconds": round(t1 - t0, 2)
        }
        
        return self.report
