import re
import unicodedata
from unidecode import unidecode
from .missing_handler import handle_missing

class AddressNormalizer:
    # Match standard ASCII punctuation characters
    PUNCTUATION = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]')
    
    # Common street abbreviations to expand for high-recall token matching
    ABBREVIATIONS = {
        r'\b(rd)\b': 'road',
        r'\b(st)\b': 'street',
        r'\b(apt)\b': 'apartment',
        r'\b(bldg)\b': 'building',
        r'\b(nr)\b': 'near',
        r'\b(opp)\b': 'opposite',
        r'\b(ste)\b': 'suite',
        r'\b(ave)\b': 'avenue',
        r'\b(blvd)\b': 'boulevard',
        r'\b(dr)\b': 'drive',
        r'\b(hwy)\b': 'highway',
        r'\b(fl)\b': 'floor',
    }

    @staticmethod
    def clean(address: str) -> str:
        """
        Normalized Unicode representation:
        - NFKC normalization
        - Lowercase
        - Replace punctuation with spaces
        - Strip and collapse whitespace
        """
        addr_clean = handle_missing(address)
        if not addr_clean:
            return ""
            
        addr_clean = unicodedata.normalize('NFKC', addr_clean).lower()
        addr_clean = AddressNormalizer.PUNCTUATION.sub(' ', addr_clean)
        return re.sub(r'\s+', ' ', addr_clean).strip()

    @staticmethod
    def to_ascii(address: str) -> str:
        """
        Transliterated representation:
        Applies unidecode to map everything to base Latin/ASCII.
        """
        cleaned = AddressNormalizer.clean(address)
        if not cleaned:
            return ""
        return unidecode(cleaned)

    @staticmethod
    def core(address: str) -> str:
        """
        High-Recall Tokens:
        Takes the ASCII transliterated representation and expands common abbreviations.
        """
        core_addr = AddressNormalizer.to_ascii(address)
        if not core_addr:
            return ""
            
        for pattern, replacement in AddressNormalizer.ABBREVIATIONS.items():
            core_addr = re.sub(pattern, replacement, core_addr)
            
        return re.sub(r'\s+', ' ', core_addr).strip()
