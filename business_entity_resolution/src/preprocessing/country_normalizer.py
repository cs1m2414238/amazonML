from .missing_handler import handle_missing

class CountryNormalizer:
    @staticmethod
    def normalize(country: str) -> str:
        """
        Open-set normalization: lowercase and strip whitespace.
        Does not assume specific countries (like US or India) to support 
        unseen values like France.
        """
        country_clean = handle_missing(country)
        if not country_clean:
            return ""
        return country_clean.lower()
