import re
import unicodedata
from unidecode import unidecode
from .missing_handler import handle_missing

class NameNormalizer:
    # Match isolated generic legal/business tokens
    LEGAL_TOKENS = re.compile(
        r'\b(pvt|private|ltd|limited|inc|incorporated|llc|corp|corporation|co|company)\b', 
        re.IGNORECASE
    )
    # Match standard ASCII punctuation characters
    PUNCTUATION = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]')

    @staticmethod
    def clean(name: str) -> str:
        """
        Normalized Unicode representation:
        - NFKC normalization
        - Lowercase
        - Replace '&' with ' and '
        - Replace punctuation with spaces
        - Strip and collapse whitespace
        (Preserves original scripts like Devanagari or French accents).
        """
        name_clean = handle_missing(name)
        if not name_clean:
            return ""
            
        # NFKC standardizes ligatures and full-width characters
        name_clean = unicodedata.normalize('NFKC', name_clean)
        name_clean = name_clean.lower()
        
        # Specific abbreviation handling before punctuation stripping
        name_clean = name_clean.replace('&', ' and ')
        
        # Replace all punctuation with spaces
        name_clean = NameNormalizer.PUNCTUATION.sub(' ', name_clean)
        
        # Collapse multiple spaces
        return re.sub(r'\s+', ' ', name_clean).strip()

    @staticmethod
    def to_ascii(name: str) -> str:
        """
        Transliterated representation:
        Applies unidecode to the clean representation to map to base Latin/ASCII.
        """
        cleaned = NameNormalizer.clean(name)
        if not cleaned:
            return ""
        return unidecode(cleaned)

    @staticmethod
    def core(name: str) -> str:
        """
        High-Recall Tokens:
        Takes the ASCII transliterated representation and removes generic legal suffixes.
        """
        ascii_name = NameNormalizer.to_ascii(name)
        if not ascii_name:
            return ""
            
        core_name = NameNormalizer.LEGAL_TOKENS.sub(' ', ascii_name)
        return re.sub(r'\s+', ' ', core_name).strip()
