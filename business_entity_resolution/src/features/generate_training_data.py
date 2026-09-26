import pandas as pd
import sqlite3
import logging
from collections import defaultdict
from pathlib import Path

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.retrieval.candidate_generator import CandidateGenerator
from business_entity_resolution.src.ingestion.reader import read_tsv_chunks
from business_entity_resolution.src.features.pair_features import PairFeatureGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    config = load_config()
        
    data_root = Path(config["data"]["root"])
    s1_path = data_root / "dataset/train/train_source1.tsv"
    s2_path = data_root / "dataset/train/train_source2.tsv"
    s3_path = data_root / "dataset/train/train_source3.tsv"
    gt_path = data_root / "dataset/train/train_ground_truth.tsv"
    
    db_path = Path(config["output"]["directory"]) / "step5/indices.db"
    output_dir = Path(config["output"]["directory"]) / "step7"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    budget = 64  # Generating training data from Top-64 to balance classes and keep sizes manageable
    # But wait, the prompt asks to "Keep the B=256 single-pass optimization". 
    # I'll retrieve at 256 and generate features for all 256, so the ML model can learn to rank up to 256.
    budget = 256
    
    generator = CandidateGenerator(str(db_path), high_df_threshold=50000)
    feat_gen = PairFeatureGenerator()
    
    logging.info("Loading S1 records...")
    s1_records = []
    # Let's take the first 2000 records for the training set creation demonstration
    for chunk in read_tsv_chunks(s1_path, "source1", 2000):
        for row in chunk.itertuples(index=False):
            s1_records.append({
                'id': row.entity_id,
                'name': row.business_name,
                'addr': row.business_address,
                'country': row.country
            })
            if len(s1_records) >= 2000:
                break
        if len(s1_records) >= 2000:
            break
            
    logging.info("Loading Ground Truth...")
    gt_map = defaultdict(set)
    df_gt = pd.read_csv(gt_path, sep="\t", dtype="string")
    for row in df_gt.itertuples(index=False):
        if pd.notna(row.matched_entity_ids) and str(row.matched_entity_ids).strip():
            targets = str(row.matched_entity_ids).split(",")
            for t in targets:
                gt_map[row.source1_entity_id].add(t.strip())
                
    logging.info("Retrieving candidates for S1...")
    retrieved_results = {}
    target_id_set = set()
    
    for idx, s1 in enumerate(s1_records):
        if idx > 0 and idx % 500 == 0:
            logging.info(f"Retrieved {idx}/2000...")
        cands = generator.get_candidates(s1['name'], s1['addr'], s1['country'], budget=budget)
        retrieved_results[s1['id']] = cands
        target_id_set.update(cands)
        
    logging.info(f"Total unique target entities to hydrate: {len(target_id_set)}")
    
    # Hydrate targets by streaming S2 and S3
    target_dict = {}
    logging.info("Hydrating S2 targets...")
    for chunk in read_tsv_chunks(s2_path, "source2"):
        for row in chunk.itertuples(index=False):
            if row.entity_id in target_id_set:
                target_dict[row.entity_id] = {
                    'name': row.business_name,
                    'addr': row.business_address,
                    'country': row.country
                }
                
    logging.info("Hydrating S3 targets...")
    for chunk in read_tsv_chunks(s3_path, "source3"):
        for row in chunk.itertuples(index=False):
            if row.entity_id in target_id_set:
                target_dict[row.entity_id] = {
                    'name': row.business_name,
                    'addr': row.business_address,
                    'country': row.country
                }

    logging.info("Generating Features...")
    feature_rows = []
    
    pos_count = 0
    neg_count = 0
    
    for s1 in s1_records:
        cands = retrieved_results[s1['id']]
        gt_set = gt_map.get(s1['id'], set())
        
        s1_dict = {'name': s1['name'], 'addr': s1['addr'], 'country': s1['country']}
        
        for rank, cand_id in enumerate(cands, start=1):
            s2_dict = target_dict.get(cand_id, {})
            # label
            label = 1 if cand_id in gt_set else 0
            
            if label == 1:
                pos_count += 1
            else:
                neg_count += 1
                
            features = feat_gen.compute_features(s1_dict, s2_dict, retrieval_rank=rank)
            
            row = {
                's1_id': s1['id'],
                'target_id': cand_id,
                'label': label
            }
            row.update(features)
            feature_rows.append(row)
            
    df_features = pd.DataFrame(feature_rows)
    out_path = output_dir / "training_features.csv"
    df_features.to_csv(out_path, index=False)
    
    logging.info("=== Feature Generation Report ===")
    logging.info(f"Total Pairs: {len(df_features)}")
    logging.info(f"Positives: {pos_count} ({(pos_count/max(1, len(df_features)))*100:.2f}%)")
    logging.info(f"Negatives: {neg_count} ({(neg_count/max(1, len(df_features)))*100:.2f}%)")
    logging.info(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
