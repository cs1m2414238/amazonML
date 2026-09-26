# Business Entity Resolution

## Windows development setup

Use 64-bit Python 3.13 for the current blocking baseline:

```powershell
py -3.13 -m venv H:\Amazon_ML_Work\.venv
H:\Amazon_ML_Work\.venv\Scripts\python.exe -m pip install -r business_entity_resolution\requirements.txt
```

The default dataset path is the sibling `student_resource` directory. Large outputs
can be redirected to an SSD without changing committed configuration:

```powershell
$env:AMAZON_ML_OUTPUT_ROOT = "H:\Amazon_ML_Work\output"
```

From the project root, validate the dataset, build the training S2/S3 index, and run
the unchanged 2,000-record retrieval baseline:

```powershell
H:\Amazon_ML_Work\.venv\Scripts\python.exe -m business_entity_resolution.src.validation.run_validation
H:\Amazon_ML_Work\.venv\Scripts\python.exe -m business_entity_resolution.src.indexing.run_indexer
H:\Amazon_ML_Work\.venv\Scripts\python.exe -m business_entity_resolution.src.retrieval.run_budget_experiment
```

Run the optimized cold-cache benchmark on the same 2,000 records, including an
exact ordered-candidate comparison against the reference generator:

```powershell
H:\Amazon_ML_Work\.venv\Scripts\python.exe -m business_entity_resolution.src.retrieval.run_optimized_budget_experiment --sample-size 2000 --equivalence-sample 100 --posting-cache-mb 1024
```

The benchmark writes `step6\optimized_budget_experiment_results.json` below the
configured output root. The reference index is opened read-only and is not rebuilt.

Override the extracted dataset location with `AMAZON_ML_DATA_ROOT` when needed.
