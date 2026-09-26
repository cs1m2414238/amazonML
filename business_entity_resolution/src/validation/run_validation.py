import json
import logging

from business_entity_resolution.src.config import load_config
from business_entity_resolution.src.runtime_metrics import get_peak_memory_mb
from business_entity_resolution.src.validation.dataset_validator import DatasetValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    config = load_config()
        
    data_root = config["data"]["root"]
    chunk_size = config["runtime"]["chunk_size"]
    
    logging.info(f"Starting Step 3B Dataset Validation.")
    logging.info(f"Data Root: {data_root}")
    logging.info(f"Chunk Size: {chunk_size}")
    
    validator = DatasetValidator(data_root=data_root, chunk_size=chunk_size)
    report = validator.run_all()
    
    peak_mem = get_peak_memory_mb()
    report["metadata"]["peak_memory_mb"] = round(peak_mem, 2)
    logging.info(f"Peak Memory: {peak_mem:.2f} MB")
    
    # Write to output folder
    from pathlib import Path

    output_dir = Path(config["output"]["directory"])
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / "dataset_validation_report.json"
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logging.info(f"Validation report saved to {report_path}")

if __name__ == "__main__":
    main()
