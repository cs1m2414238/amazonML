import time
import logging
import os
import platform
from pathlib import Path
import json

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.indexing.indexer import SPIMIIndexer
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    config = load_config()
        
    data_root = Path(config["data"]["root"])
    chunk_size = config["runtime"]["chunk_size"]
    
    # We will index train Source 2 and Source 3 as per Step 5 specs.
    files_to_index = [
        (data_root / "dataset/train/train_source2.tsv", "source2"),
        (data_root / "dataset/train/train_source3.tsv", "source3"),
    ]
    
    output_dir = Path(config["output"]["directory"]) / "step5"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    db_path = output_dir / "indices.db"
    
    indexer = SPIMIIndexer(db_path=db_path, num_shards=64, chunk_size=chunk_size)
    
    t0 = time.time()
    
    for fp, src in files_to_index:
        indexer.map_phase(fp, src)
        
    reduce_stats = indexer.reduce_phase()
    
    t1 = time.time()
    build_time = t1 - t0
    records_per_sec = indexer.doc_counter / build_time if build_time > 0 else 0
    
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    peak_mem_mb = get_peak_memory_mb()
    
    report = {
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
            "chunk_size": chunk_size,
            "num_shards": indexer.num_shards,
        },
        "build_time_seconds": round(build_time, 2),
        "total_records_indexed": indexer.doc_counter,
        "records_per_second": round(records_per_sec, 2),
        "peak_rss_mb": round(peak_mem_mb, 2),
        "index_disk_size_mb": round(db_size_mb, 2),
        "unique_key_counts": {
            "name_tokens": reduce_stats["unique_keys"][0],
            "name_ngrams": reduce_stats["unique_keys"][1],
            "address_tokens": reduce_stats["unique_keys"][2],
            "address_ngrams": reduce_stats["unique_keys"][3],
        },
        "largest_posting_lists": {
            "name_tokens": reduce_stats["largest_posting_list"][0],
            "name_ngrams": reduce_stats["largest_posting_list"][1],
            "address_tokens": reduce_stats["largest_posting_list"][2],
            "address_ngrams": reduce_stats["largest_posting_list"][3],
        },
        "missing_value_statistics": indexer.missing_stats
    }
    
    report_path = output_dir / "step5_build_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info(f"Build complete. Report saved to {report_path}")

if __name__ == "__main__":
    main()
