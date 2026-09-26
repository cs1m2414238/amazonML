import pandas as pd
from collections import defaultdict
import logging

class RecallEvaluator:
    def __init__(self, ground_truth_path: str):
        self.ground_truth_path = ground_truth_path
        self.gt_map = defaultdict(set)
        self._load()

    def _load(self):
        logging.info("Loading Ground Truth for Evaluation...")
        df = pd.read_csv(self.ground_truth_path, sep="\t", dtype="string")
        # Ensure we capture source2 and source3 properly
        for row in df.itertuples(index=False):
            if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
                targets = str(row.matched_entity_ids).split(",")
                for t in targets:
                    self.gt_map[row.source1_entity_id].add(t.strip())
        logging.info(f"Loaded GT for {len(self.gt_map)} Source 1 entities.")

    def evaluate(self, retrieved_results: dict):
        """
        retrieved_results: dict mapping S1_entity_id -> list of retrieved S2/S3 entity_ids
        Returns a dictionary of metrics.
        """
        total_true_pairs = 0
        found_true_pairs = 0
        
        s1_total = 0
        s1_full_match = 0
        
        stats = {
            "zero_match_s1_processed": 0,
            "one_match_s1": {"total": 0, "full_match": 0},
            "multi_match_s1": {"total": 0, "full_match": 0},
            "macro_recall_sum": 0.0
        }
        
        for s1_id, candidates in retrieved_results.items():
            gt_set = self.gt_map.get(s1_id, set())
            cand_set = set(candidates)
            
            gt_len = len(gt_set)
            
            if gt_len == 0:
                stats["zero_match_s1_processed"] += 1
                continue
                
            s1_total += 1
            total_true_pairs += gt_len
            
            intersection = gt_set & cand_set
            found = len(intersection)
            found_true_pairs += found
            
            recall_fraction = found / gt_len
            stats["macro_recall_sum"] += recall_fraction
            
            is_full = (found == gt_len)
            if is_full:
                s1_full_match += 1
                
            if gt_len == 1:
                stats["one_match_s1"]["total"] += 1
                if is_full:
                    stats["one_match_s1"]["full_match"] += 1
            else:
                stats["multi_match_s1"]["total"] += 1
                if is_full:
                    stats["multi_match_s1"]["full_match"] += 1
                    
        pair_recall = found_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
        s1_full_match_recall = s1_full_match / s1_total if s1_total > 0 else 0.0
        macro_recall = stats["macro_recall_sum"] / s1_total if s1_total > 0 else 0.0
        
        one_match_recall = stats["one_match_s1"]["full_match"] / stats["one_match_s1"]["total"] if stats["one_match_s1"]["total"] > 0 else 0.0
        multi_match_recall = stats["multi_match_s1"]["full_match"] / stats["multi_match_s1"]["total"] if stats["multi_match_s1"]["total"] > 0 else 0.0

        return {
            "pair_level_recall": round(pair_recall, 4),
            "s1_level_full_match_recall": round(s1_full_match_recall, 4),
            "macro_recall": round(macro_recall, 4),
            "zero_match_s1_processed": stats["zero_match_s1_processed"],
            "one_match_s1_recall": round(one_match_recall, 4),
            "multi_match_s1_recall": round(multi_match_recall, 4),
        }
