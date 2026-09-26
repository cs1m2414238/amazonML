import re

MISSING_PATTERNS = {"", "nan", "none", "null", "na", "-", "."}

def is_missing(val: str) -> bool:
    """
    Case-insensitive check for missing values after stripping whitespace.
    """
    if not isinstance(val, str):
        return True
    val = val.strip().lower()
    return not val or val in MISSING_PATTERNS

def handle_missing(val: str) -> str:
    """
    Returns an empty string if the value is considered missing, 
    otherwise returns the stripped original string.
    Do NOT modify the original raw field in place; this just returns the derived representation.
    """
    return "" if is_missing(val) else val.strip()
