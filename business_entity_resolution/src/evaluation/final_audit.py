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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def get_ratio(s1, s2):
    if not s1 and not s2: return 0.0
    if not s1 or not s2: return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def is_abbreviation(s1, s2):
    if not s1 or not s2: return False
    w1 = s1.split()
    w2 = s2.split()
    if len(w1) == 1 and len(w2) > 1:
        acronym = "".join([w[0] for w in w2 if w]).lower()
        if s1.lower() == acronym: return True
    if len(w2) == 1 and len(w1) > 1:
        acronym = "".join([w[0] for w in w1 if w]).lower()
        if s2.lower() == acronym: return True
    return False

def main():
    random.seed(42)
    config = load_config()
    data_root = Path(config["data"]["root"])
    output_root = Path(config["output"]["directory"])
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    s1_path = data_root / "dataset/train/train_source1.tsv"
    s2_path = data_root / "dataset/train/train_source2.tsv"
    s3_path = data_root / "dataset/train/train_source3.tsv"
    db_path = output_root / "step5/indices.db"
    
    logging.info("1. Parsing Ground Truth...")
    gt_map = defaultdict(set)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype="string")
    for row in df_gt.itertuples(index=False):
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            targets = str(row.matched_entity_ids).split(",")
            for t in targets:
                gt_map[row.source1_entity_id].add(t.strip())
                
    logging.info("2. Building Target Dictionary for Group J identification...")
    # First, let's stream S1 and collect a buffer of candidate S1s and their targets
    s1_buffer = []
    target_ids_needed = set()
    idx = 0
    for chunk in read_tsv_chunks(s1_path, "source1"):
        for row in chunk.itertuples(index=False):
            idx += 1
            s1_id = row.entity_id
            name = str(row.business_name).strip() if pd.notna(row.business_name) else ""
            addr = str(row.business_address).strip() if pd.notna(row.business_address) else ""
            country = str(row.country).strip() if pd.notna(row.country) else ""
            gt_set = gt_map.get(s1_id, set())
            
            s1_buffer.append((s1_id, name, addr, country, gt_set))
            target_ids_needed.update(gt_set)
            
            if len(s1_buffer) >= 20000:
                break
        if len(s1_buffer) >= 20000:
            break
            
    logging.info(f"Collected {len(s1_buffer)} S1 records for pool. Need {len(target_ids_needed)} targets.")
    
    target_dict = {}
    for path, source in [(s2_path, "source2"), (s3_path, "source3")]:
        for chunk in read_tsv_chunks(path, source):
            for row in chunk.itertuples(index=False):
                if row.entity_id in target_ids_needed:
                    target_dict[row.entity_id] = {
                        "name": str(row.business_name).strip() if pd.notna(row.business_name) else "",
                        "addr": str(row.business_address).strip() if pd.notna(row.business_address) else ""
                    }
                    
    logging.info("3. Distributing into Groups A-J...")
    groups = {
        "A_Random": [], "B_Zero": [], "C_One": [], "D_Multi": [],
        "E_5Plus": [], "F_10_11": [], "G_Similar": [], "H_Short": [],
        "I_MissingAddr": [], "J_Lexical": []
    }
    
    target_sizes = {
        "A_Random": 500, "B_Zero": 150, "C_One": 150, "D_Multi": 150,
        "E_5Plus": 150, "F_10_11": 150, "G_Similar": 150, "H_Short": 150,
        "I_MissingAddr": 150, "J_Lexical": 150
    }
    
    # We will also pull from the full S1 file for Group A and I if needed, 
    # but the buffer of 20000 might have enough of most groups.
    for rec in s1_buffer:
        s1_id, name, addr, country, gt_set = rec
        gt_len = len(gt_set)
        r_tup = (s1_id, name, addr, country)
        
        # A
        if len(groups["A_Random"]) < target_sizes["A_Random"]:
            groups["A_Random"].append(r_tup)
            
        # B
        if gt_len == 0 and len(groups["B_Zero"]) < target_sizes["B_Zero"]:
            groups["B_Zero"].append(r_tup)
        # C
        if gt_len == 1 and len(groups["C_One"]) < target_sizes["C_One"]:
            groups["C_One"].append(r_tup)
        # D
        if 2 <= gt_len <= 4 and len(groups["D_Multi"]) < target_sizes["D_Multi"]:
            groups["D_Multi"].append(r_tup)
        # E
        if gt_len >= 5 and len(groups["E_5Plus"]) < target_sizes["E_5Plus"]:
            groups["E_5Plus"].append(r_tup)
        # F
        if gt_len >= 10 and len(groups["F_10_11"]) < target_sizes["F_10_11"]:
            groups["F_10_11"].append(r_tup)
            
        # H Short
        if len(name.split()) == 1 and gt_len > 0 and len(groups["H_Short"]) < target_sizes["H_Short"]:
            groups["H_Short"].append(r_tup)
            
        # I Missing Address
        # Need to be explicitly sure addr is truly empty or NaN.
        # It was stripped and nan replaced with "".
        if not addr and gt_len > 0 and len(groups["I_MissingAddr"]) < target_sizes["I_MissingAddr"]:
            groups["I_MissingAddr"].append(r_tup)
            
        # J Difficult Lexical
        if gt_len > 0 and len(groups["J_Lexical"]) < target_sizes["J_Lexical"]:
            # Check if any target has a bad ratio
            for t_id in gt_set:
                t_dict = target_dict.get(t_id)
                if t_dict:
                    if get_ratio(name, t_dict["name"]) < 0.4:
                        groups["J_Lexical"].append(r_tup)
                        break
                        
        # G Collision Heavy / Highly Similar
        if gt_len > 1 and len(name) <= 15 and len(groups["G_Similar"]) < target_sizes["G_Similar"]:
            # Proxy for collision: short name, multiple matches
            groups["G_Similar"].append(r_tup)
            
    # Combine uniquely
    eval_set_dict = {}
    for gname, lst in groups.items():
        for r in lst:
            eval_set_dict[r[0]] = r
            
    eval_set = list(eval_set_dict.values())
    logging.info(f"Final Eval Set Size: {len(eval_set)}")
    
    # Check Group I explicitly
    grp_i_cnt = len(groups["I_MissingAddr"])
    grp_i_gt_counts = [len(gt_map.get(r[0], set())) for r in groups["I_MissingAddr"]]
    i_zeros = sum(1 for c in grp_i_gt_counts if c == 0)
    i_ones = sum(1 for c in grp_i_gt_counts if c == 1)
    logging.info(f"GROUP I AUDIT: Selected {grp_i_cnt}, Zeros={i_zeros}, Ones={i_ones}, Avg={np.mean(grp_i_gt_counts) if grp_i_gt_counts else 0}")
    
    # 4. Deep Retrieval
    generator = CandidateGenerator(db_path, high_df_threshold=50000)
    master_results = {}
    deep_results = {}
    
    t0 = time.time()
    for idx, r in enumerate(eval_set):
        if idx > 0 and idx % 200 == 0:
            logging.info(f"Retrieved {idx}/{len(eval_set)}...")
        cands = generator.get_candidates(r[1], r[2], r[3], budget=10000)
        deep_results[r[0]] = cands
        master_results[r[0]] = cands[:1024]
        
    t1 = time.time()
    runtime = t1 - t0
    
    # 5. Evaluate Fixed Budgets & Candidate Volumes
    def eval_budget(budget, cands_dict):
        metrics = {"total_found": 0, "s1_full_matches": 0}
        group_recalls = {g: {"found": 0, "total": 0} for g in groups.keys()}
        cand_counts = {g: [] for g in groups.keys()}
        cand_counts["overall"] = []
        
        for r in eval_set:
            s1_id = r[0]
            gt = gt_map.get(s1_id, set())
            cands = set(cands_dict[s1_id][:budget])
            
            cand_counts["overall"].append(len(cands))
            
            if len(gt) > 0:
                found = len(gt & cands)
                metrics["total_found"] += found
                if found == len(gt):
                    metrics["s1_full_matches"] += 1
                    
            for gname, lst in groups.items():
                if any(x[0] == s1_id for x in lst):
                    cand_counts[gname].append(len(cands))
                    if len(gt) > 0:
                        group_recalls[gname]["total"] += len(gt)
                        group_recalls[gname]["found"] += len(gt & cands)
                        
        # Calc stats
        cand_stats = {}
        for k, v in cand_counts.items():
            if v:
                cand_stats[k] = {
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "p95": float(np.percentile(v, 95)),
                    "p99": float(np.percentile(v, 99)),
                    "max": float(np.max(v)),
                    "total": sum(v)
                }
        return metrics, group_recalls, cand_stats
        
    f_metrics = {}
    for b in [256, 512, 1024]:
        m, gm, cs = eval_budget(b, master_results)
        f_metrics[f"B_{b}"] = {"metrics": m, "group_recalls": gm, "cand_stats": cs}
        
    # 6. Adaptive Retrieval
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
        
    a_m, a_gm, a_cs = eval_budget(1024, adaptive_cands)
    
    # 7. Rank Analysis of Budget Casualties
    ranks = {"257-512": 0, "513-1024": 0, "1025-2048": 0, "2049-10000": 0}
    block_failures = []
    
    for r in eval_set:
        s1_id = r[0]
        gt = gt_map.get(s1_id, set())
        ac = set(adaptive_cands[s1_id])
        
        misses = gt - ac
        if not misses: continue
        
        deep = deep_results[s1_id]
        for m in misses:
            try:
                rank = deep.index(m) + 1
                if 257 <= rank <= 512: ranks["257-512"] += 1
                elif 513 <= rank <= 1024: ranks["513-1024"] += 1
                elif 1025 <= rank <= 2048: ranks["1025-2048"] += 1
                elif rank > 2048: ranks["2049-10000"] += 1
            except ValueError:
                block_failures.append({"s1_id": s1_id, "s1_name": r[1], "s1_addr": r[2], "t_id": m})
                
    # 8. Block Failure Categorization
    bf_categories = defaultdict(int)
    for bf in block_failures:
        t_dict = target_dict.get(bf["t_id"])
        if not t_dict:
            bf_categories["other_not_hydrated"] += 1
            continue
            
        s1n = bf["s1_name"]
        s1a = bf["s1_addr"]
        t2n = t_dict["name"]
        t2a = t_dict["addr"]
        
        if is_abbreviation(s1n, t2n):
            bf_categories["abbreviation"] += 1
        elif get_ratio(s1n, t2n) < 0.3:
            bf_categories["severe_name_mutation"] += 1
        elif not s1a or not t2a:
            bf_categories["missing_address"] += 1
        else:
            bf_categories["other"] += 1
            
    # Compile
    report = {
        "set_size": len(eval_set),
        "total_true_pairs": sum(len(gt_map.get(r[0], set())) for r in eval_set),
        "group_i_audit": {"selected": grp_i_cnt, "zeros": i_zeros, "ones": i_ones, "avg": np.mean(grp_i_gt_counts) if grp_i_gt_counts else 0},
        "fixed_budgets": f_metrics,
        "adaptive": {
            "expansion_counts": expansion_stats,
            "expansion_yield": useful_expansions / max(1, useful_expansions + unproductive_expansions),
            "unproductive_expansion_rate": unproductive_expansions / max(1, useful_expansions + unproductive_expansions),
            "metrics": a_m,
            "group_recalls": a_gm,
            "cand_stats": a_cs
        },
        "budget_casualties_ranks": ranks,
        "block_failures_categories": dict(bf_categories)
    }
    
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "step6_audit_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    logging.info("Audit complete.")

if __name__ == "__main__":
    main()
