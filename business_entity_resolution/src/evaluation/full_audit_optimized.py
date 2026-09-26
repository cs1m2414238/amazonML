import pandas as pd
import sqlite3
import logging
from collections import defaultdict
import random
import time
import numpy as np
import json
import difflib
from pathlib import Path

from business_entity_resolution.src.retrieval.candidate_generator import CandidateGenerator
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.preprocessing import NameNormalizer
from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_ratio(s1, s2):
    if not s1 and not s2: return 0.0
    if not s1 or not s2: return 0.0
    return difflib.SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()

def is_abbreviation(s1, s2):
    if not s1 or not s2: return False
    w1 = str(s1).lower().split()
    w2 = str(s2).lower().split()
    if len(w1) == 1 and len(w2) > 1:
        acronym = "".join([w[0] for w in w2 if w])
        if str(s1).lower() == acronym: return True
    if len(w2) == 1 and len(w1) > 1:
        acronym = "".join([w[0] for w in w1 if w])
        if str(s2).lower() == acronym: return True
    return False

def main():
    random.seed(42)
    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    s1_path = data_root / "dataset/train/train_source1.tsv"
    s2_path = data_root / "dataset/train/train_source2.tsv"
    s3_path = data_root / "dataset/train/train_source3.tsv"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    db_path = output_root / "step5/indices.db"
    
    t_start = time.time()
    
    logging.info("1. Parsing Ground Truth...")
    gt_map = defaultdict(set)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype="string")
    for row in df_gt.itertuples(index=False):
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            targets = str(row.matched_entity_ids).split(",")
            for t in targets:
                gt_map[row.source1_entity_id].add(t.strip())
                
    max_match_len = max([len(v) for v in gt_map.values()] or [0])
                
    logging.info("2. Streaming S2/S3 to identify targets with Missing Addresses...")
    missing_address_targets = set()
    # To keep memory bounded, we only store IDs
    for path, source in [(s2_path, "source2"), (s3_path, "source3")]:
        for chunk in read_tsv_chunks(path, source):
            for row in chunk.itertuples(index=False):
                if pd.isna(row.business_address) or not str(row.business_address).strip():
                    missing_address_targets.add(row.entity_id)
                    
    logging.info("3. Building Representative Groups from FULL S1...")
    groups = {
        "A_Random": [], "B_Zero": [], "C_One": [], "D_Multi": [],
        "E_5Plus": [], "F_Max": [], "G_Similar": [], "H_Short": [],
        "I_MissingAddr": [], "J_Lexical": []
    }
    
    target_sizes = {
        "A_Random": 500, "B_Zero": 150, "C_One": 150, "D_Multi": 150,
        "E_5Plus": 150, "F_Max": 150, "G_Similar": 150, "H_Short": 150,
        "I_MissingAddr": 150, "J_Lexical": 150
    }
    
    # We will buffer up to 50,000 S1 records to ensure we can hydrate targets for Group J
    # but we stream the entire file to find Group I (missing address targets) and Group F (Max) if needed.
    s1_buffer = []
    idx = 0
    for chunk in read_tsv_chunks(s1_path, "source1"):
        for row in chunk.itertuples(index=False):
            idx += 1
            s1_id = row.entity_id
            name = str(row.business_name).strip() if pd.notna(row.business_name) else ""
            addr = str(row.business_address).strip() if pd.notna(row.business_address) else ""
            country = str(row.country).strip() if pd.notna(row.country) else ""
            gt_set = gt_map.get(s1_id, set())
            gt_len = len(gt_set)
            rec = (s1_id, name, addr, country)
            
            # Fill groups
            if idx % 1000 == 0 and len(groups["A_Random"]) < target_sizes["A_Random"]:
                groups["A_Random"].append(rec)
            if gt_len == 0 and len(groups["B_Zero"]) < target_sizes["B_Zero"]:
                groups["B_Zero"].append(rec)
            if gt_len == 1 and len(groups["C_One"]) < target_sizes["C_One"]:
                groups["C_One"].append(rec)
            if 2 <= gt_len <= 4 and len(groups["D_Multi"]) < target_sizes["D_Multi"]:
                groups["D_Multi"].append(rec)
            if gt_len >= 5 and len(groups["E_5Plus"]) < target_sizes["E_5Plus"]:
                groups["E_5Plus"].append(rec)
            if gt_len == max_match_len and len(groups["F_Max"]) < target_sizes["F_Max"]:
                groups["F_Max"].append(rec)
            if gt_len > 1 and len(name) <= 12 and len(groups["G_Similar"]) < target_sizes["G_Similar"]:
                groups["G_Similar"].append(rec)
            if len(name.split()) == 1 and gt_len > 0 and len(groups["H_Short"]) < target_sizes["H_Short"]:
                groups["H_Short"].append(rec)
            
            # Missing Candidate Address (Group I)
            if gt_len > 0 and len(groups["I_MissingAddr"]) < target_sizes["I_MissingAddr"]:
                if any(t in missing_address_targets for t in gt_set):
                    groups["I_MissingAddr"].append(rec)
                    
            if len(s1_buffer) < 50000:
                s1_buffer.append((rec, gt_set))
                
        if all(len(lst) >= target_sizes[k] for k, lst in groups.items() if k != "J_Lexical" and k != "F_Max"):
            break
            
    logging.info("Hydrating targets for Group J (Lexical Cases)...")
    target_ids_needed = set()
    for rec, gt_set in s1_buffer:
        target_ids_needed.update(gt_set)
        
    target_dict = {}
    for path, source in [(s2_path, "source2"), (s3_path, "source3")]:
        for chunk in read_tsv_chunks(path, source):
            for row in chunk.itertuples(index=False):
                if row.entity_id in target_ids_needed:
                    target_dict[row.entity_id] = {
                        "name": str(row.business_name).strip() if pd.notna(row.business_name) else "",
                        "addr": str(row.business_address).strip() if pd.notna(row.business_address) else ""
                    }
                    
    for rec, gt_set in s1_buffer:
        if len(groups["J_Lexical"]) >= target_sizes["J_Lexical"]: break
        if len(gt_set) > 0:
            for t_id in gt_set:
                t_data = target_dict.get(t_id)
                if t_data and get_ratio(rec[1], t_data["name"]) < 0.4:
                    groups["J_Lexical"].append(rec)
                    break
                    
    # Combine uniquely
    eval_set_dict = {}
    for gname, lst in groups.items():
        for r in lst:
            eval_set_dict[r[0]] = r
    eval_set = list(eval_set_dict.values())
    
    logging.info(f"Final Eval Set Size: {len(eval_set)}")
    
    # Audit Missing Address
    grp_i_cnt = len(groups["I_MissingAddr"])
    grp_i_gt_counts = [len(gt_map.get(r[0], set())) for r in groups["I_MissingAddr"]]
    
    # 4. Equivalence Test
    generator = CandidateGenerator(db_path, high_df_threshold=50000)
    equiv_passed = True
    for r in eval_set[:5]:
        c1024 = generator.get_candidates(r[1], r[2], r[3], budget=1024)
        for b in [256, 512]:
            cb = generator.get_candidates(r[1], r[2], r[3], budget=b)
            if cb != c1024[:b]:
                equiv_passed = False
                break
                
    # 5. Optimized Retrieval (B=1024 for all, B=10000 only for misses)
    logging.info("Running retrieval at B=1024...")
    master_results = {}
    
    t_retrieval_start = time.time()
    for idx, r in enumerate(eval_set):
        if idx > 0 and idx % 200 == 0:
            logging.info(f"Retrieved {idx}/{len(eval_set)}...")
        cands = generator.get_candidates(r[1], r[2], r[3], budget=1024)
        master_results[r[0]] = cands
    t_retrieval_end = time.time()
    
    # 6. Evaluate Fixed & Adaptive
    adaptive_cands = {}
    expansion_stats = {"at_256": 0, "at_512": 0, "at_1024": 0}
    useful_expansions = 0
    unproductive_expansions = 0
    
    for r in eval_set:
        s1_id = r[0]
        cands = master_results[s1_id]
        c256 = cands[:256]
        
        is_missing_addr = (not r[2])
        is_short_name = (len(r[1].split()) <= 2)
        
        if len(c256) == 256 and (is_missing_addr or is_short_name):
            c512 = cands[:512]
            if len(c512) == 512 and (is_missing_addr and is_short_name):
                final_c = cands[:1024]
                expansion_stats["at_1024"] += 1
                base_c = c512
            else:
                final_c = c512
                expansion_stats["at_512"] += 1
                base_c = c256
                
            gt = gt_map.get(s1_id, set())
            if len(gt) > 0:
                if len(gt & set(final_c)) > len(gt & set(base_c)):
                    useful_expansions += 1
                else:
                    unproductive_expansions += 1
        else:
            final_c = c256
            expansion_stats["at_256"] += 1
            
        adaptive_cands[s1_id] = final_c
        
    def eval_cands(cands_dict, budget_cap=None):
        metrics = {"total_found": 0, "s1_full_matches": 0, "cand_counts": []}
        group_metrics = {g: {"found": 0, "total": 0, "cands": []} for g in groups.keys()}
        zero_match_correct = 0
        total_zero = 0
        
        for r in eval_set:
            s1_id = r[0]
            gt = gt_map.get(s1_id, set())
            cands = cands_dict[s1_id]
            if budget_cap: cands = cands[:budget_cap]
            cands = set(cands)
            
            metrics["cand_counts"].append(len(cands))
            
            if len(gt) == 0:
                total_zero += 1
                zero_match_correct += 1
                continue
                
            found = len(gt & cands)
            metrics["total_found"] += found
            if found == len(gt):
                metrics["s1_full_matches"] += 1
                
            for gname, lst in groups.items():
                if any(x[0] == s1_id for x in lst):
                    group_metrics[gname]["total"] += len(gt)
                    group_metrics[gname]["found"] += found
                    group_metrics[gname]["cands"].append(len(cands))
                    
        return metrics, group_metrics, (zero_match_correct / max(1, total_zero))

    f256, g256, z256 = eval_cands(master_results, 256)
    f512, g512, z512 = eval_cands(master_results, 512)
    f1024, g1024, z1024 = eval_cands(master_results, 1024)
    fa, ga, za = eval_cands(adaptive_cands, None)
    
    # 7. Deep Query Misses for Failure & Rank Analysis
    logging.info("Running deep queries for missed records...")
    ranks = {"257-512": 0, "513-1024": 0, "1025-2048": 0, "2049-10000": 0}
    block_failures = []
    
    for r in eval_set:
        s1_id = r[0]
        gt = gt_map.get(s1_id, set())
        ac = set(adaptive_cands[s1_id])
        
        misses = gt - ac
        if not misses: continue
        
        # Deep query specifically for this miss
        cands_10k = generator.get_candidates(r[1], r[2], r[3], budget=10000)
        
        for m in misses:
            try:
                rank = cands_10k.index(m) + 1
                if 257 <= rank <= 512: ranks["257-512"] += 1
                elif 513 <= rank <= 1024: ranks["513-1024"] += 1
                elif 1025 <= rank <= 2048: ranks["1025-2048"] += 1
                elif rank > 2048: ranks["2049-10000"] += 1
            except ValueError:
                block_failures.append({"s1_id": s1_id, "s1_name": r[1], "s1_addr": r[2], "t_id": m})
                
    # Block failure categorization
    bf_categories = defaultdict(int)
    for bf in block_failures:
        # Check missing target address
        if bf["t_id"] in missing_address_targets:
            bf_categories["missing_address"] += 1
        else:
            t_data = target_dict.get(bf["t_id"])
            if t_data:
                if is_abbreviation(bf["s1_name"], t_data["name"]):
                    bf_categories["abbreviation"] += 1
                elif get_ratio(bf["s1_name"], t_data["name"]) < 0.3:
                    bf_categories["severe_name_mutation"] += 1
                else:
                    bf_categories["other"] += 1
            else:
                bf_categories["other_not_hydrated"] += 1
                
    # Build Report
    def compile_stats(cand_list):
        if not cand_list: return {}
        return {
            "mean": float(np.mean(cand_list)),
            "median": float(np.median(cand_list)),
            "p95": float(np.percentile(cand_list, 95)),
            "p99": float(np.percentile(cand_list, 99)),
            "max": float(np.max(cand_list)),
            "total": sum(cand_list)
        }
        
    report = {
        "set_size": len(eval_set),
        "total_true_pairs": sum(len(gt_map.get(r[0], set())) for r in eval_set),
        "group_i_audit": {
            "s1_records": grp_i_cnt,
            "zero_match": sum(1 for c in grp_i_gt_counts if c == 0),
            "one_match": sum(1 for c in grp_i_gt_counts if c == 1),
            "multi_match": sum(1 for c in grp_i_gt_counts if c > 1)
        },
        "fixed_budgets": {
            "B_256": {"metrics": f256, "group_recalls": g256, "stats": compile_stats(f256["cand_counts"])},
            "B_512": {"metrics": f512, "group_recalls": g512, "stats": compile_stats(f512["cand_counts"])},
            "B_1024": {"metrics": f1024, "group_recalls": g1024, "stats": compile_stats(f1024["cand_counts"])},
        },
        "adaptive": {
            "expansion_counts": expansion_stats,
            "useful_expansions": useful_expansions,
            "unproductive_expansions": unproductive_expansions,
            "expansion_yield": useful_expansions / max(1, useful_expansions + unproductive_expansions),
            "unproductive_expansion_rate": unproductive_expansions / max(1, useful_expansions + unproductive_expansions),
            "metrics": fa,
            "group_recalls": ga,
            "stats": compile_stats(fa["cand_counts"]),
            "group_stats": {g: compile_stats(ga[g]["cands"]) for g in groups.keys()}
        },
        "budget_casualties_ranks": ranks,
        "block_failures_categories": dict(bf_categories),
        "equivalence_passed": equiv_passed,
        "performance": {
            "runtime_retrieval": t_retrieval_end - t_retrieval_start,
            "total_runtime": time.time() - t_start,
            "peak_ram_mb": get_peak_memory_mb(),
            "total_records_processed": idx
        }
    }
    
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "step6_full_audit.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    logging.info("Full audit complete.")

if __name__ == "__main__":
    main()
