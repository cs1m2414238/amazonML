EXPECTED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

EXPECTED_GROUND_TRUTH_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids"
]

def validate_columns(columns, source_name=None):
    columns = list(columns)
    
    expected = EXPECTED_GROUND_TRUTH_COLUMNS if source_name == "ground_truth" else EXPECTED_COLUMNS

    if columns != expected:
        raise ValueError(
            f"Invalid columns for {source_name}.\n"
            f"Expected: {expected}\n"
            f"Found:    {columns}"
        )


def validate_source(source_name):
    valid_sources = {"source1", "source2", "source3", "ground_truth"}

    if source_name not in valid_sources:
        raise ValueError(
            f"Invalid source: {source_name}. "
            f"Expected one of {valid_sources}"
        )