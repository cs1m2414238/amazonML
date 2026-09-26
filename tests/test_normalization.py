import unittest
from business_entity_resolution.src.preprocessing import (
    is_missing, handle_missing, CountryNormalizer, NameNormalizer, AddressNormalizer
)

class TestNormalization(unittest.TestCase):

    def test_missing_handler(self):
        # Should be identified as missing
        self.assertTrue(is_missing("nan"))
        self.assertTrue(is_missing("None"))
        self.assertTrue(is_missing("  null  "))
        self.assertTrue(is_missing(""))
        self.assertTrue(is_missing("   "))
        self.assertTrue(is_missing("na"))
        self.assertTrue(is_missing("-"))
        self.assertTrue(is_missing("."))
        
        # Should not be missing
        self.assertFalse(is_missing("valid name"))
        self.assertFalse(is_missing("0"))
        
        # Handler check
        self.assertEqual(handle_missing("nan"), "")
        self.assertEqual(handle_missing(" valid  "), "valid")

    def test_country_normalizer(self):
        self.assertEqual(CountryNormalizer.normalize("  United States  "), "united states")
        self.assertEqual(CountryNormalizer.normalize("France"), "france")
        self.assertEqual(CountryNormalizer.normalize("nan"), "")

    def test_name_normalizer(self):
        raw1 = "राम मार्केटिंग प्राइवेट लिमिटेड"
        self.assertEqual(NameNormalizer.clean(raw1), raw1)
        # Note: unidecode on Devanagari can produce imperfect transliteration depending on library version, 
        # but the test checks it doesn't crash and returns ASCII.
        ascii1 = NameNormalizer.to_ascii(raw1)
        self.assertTrue(all(ord(c) < 128 for c in ascii1))
        
        # Suffix removal test
        raw2 = "Café & Bistro, LLC"
        self.assertEqual(NameNormalizer.clean(raw2), "café and bistro llc")
        self.assertEqual(NameNormalizer.to_ascii(raw2), "cafe and bistro llc")
        self.assertEqual(NameNormalizer.core(raw2), "cafe and bistro")

    def test_address_normalizer(self):
        raw1 = "Opp. SBI, M.G. Rd, New York"
        self.assertEqual(AddressNormalizer.clean(raw1), "opp sbi m g rd new york")
        self.assertEqual(AddressNormalizer.to_ascii(raw1), "opp sbi m g rd new york")
        self.assertEqual(AddressNormalizer.core(raw1), "opposite sbi m g road new york")
        
if __name__ == "__main__":
    unittest.main()
