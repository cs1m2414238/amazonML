from pathlib import Path

import pandas as pd

from .schema import validate_columns, validate_source


def read_tsv_chunks(
    file_path,
    source_name,
    chunk_size=100_000,
):
    validate_source(source_name)

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    reader = pd.read_csv(
        file_path,
        sep="\t",
        dtype="string",
        chunksize=chunk_size,
        keep_default_na=False,
    )

    first_chunk = True

    for chunk in reader:

        if first_chunk:
            validate_columns(chunk.columns, source_name)
            first_chunk = False

        yield chunk
        
