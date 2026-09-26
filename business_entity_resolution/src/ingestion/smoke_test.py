from pathlib import Path

from business_entity_resolution.src.config import load_config
from .reader import read_tsv_chunks


DATA_ROOT = Path(load_config()["data"]["root"])


FILES = {
    "source1": DATA_ROOT / "dataset/train/train_source1.tsv",
    "source2": DATA_ROOT / "dataset/train/train_source2.tsv",
    "source3": DATA_ROOT / "dataset/train/train_source3.tsv",
}


for source_name, file_path in FILES.items():

    print(f"\nChecking {source_name}")
    print(f"File: {file_path}")

    total_rows = 0

    for chunk in read_tsv_chunks(
        file_path,
        source_name,
        chunk_size=100_000,
    ):
        total_rows += len(chunk)

        print(
            f"First chunk shape: {chunk.shape}"
        )

        print(
            f"Columns: {list(chunk.columns)}"
        )

        print(
            f"First row:\n{chunk.iloc[0].to_dict()}"
        )

        break

    print(
        f"Smoke test passed for {source_name}"
    )
