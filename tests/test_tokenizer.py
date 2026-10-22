import unittest

from retrieval.tokenizer import Tokenizer, normalize


class NormalizeTests(unittest.TestCase):
    def test_legacy_and_atomic_chillu_are_equal(self):
        legacy = "സര്‍ക്കാര്‍"  # consonant + virama + ZWJ (as in the corpus)
        atomic = "സർക്കാർ"               # atomic chillu ർ (modern keyboards)
        self.assertEqual(normalize(legacy), normalize(atomic))

    def test_zwnj_removed(self):
        self.assertEqual(normalize("വകുപ്പ്‌"), "വകുപ്പ്")

    def test_legacy_font_sequences(self):
        self.assertEqual(normalize("ഉളള"), normalize("ഉള്ള"))
        self.assertEqual(normalize("ററ"), normalize("റ്റ"))

    def test_au_sign_variants(self):
        self.assertEqual(normalize("കൌ"), normalize("കൗ"))

    def test_malayalam_digits(self):
        self.assertEqual(normalize("൯൨"), "92")

    def test_casefold(self):
        self.assertEqual(normalize("Section"), "section")


class TokenizeTests(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer(subword="prefix", prefix_len=5)

    def test_malayalam_word_not_split_on_vowel_signs(self):
        tokens = self.tok.tokenize("വകുപ്പിന്റെ")
        self.assertIn("വകുപ്പിന്റെ", tokens)

    def test_inflections_share_prefix_feature(self):
        a = set(self.tok.tokenize("നികുതി"))
        b = set(self.tok.tokenize("നികുതിയുടെയോ"))
        self.assertTrue(a & b, "inflected forms should share a subword feature")

    def test_identifier_kept_whole_and_split(self):
        tokens = self.tok.tokenize("നമ്പര്‍ 18/ലെഗ്‌.സി3/2024/നിയമം.")
        self.assertIn("18/ലെഗ്.സി3/2024/നിയമം", tokens)
        for part in ("18", "സി3", "2024", "നിയമം"):
            self.assertIn(part, tokens)

    def test_english_identifier_with_and_without_space_share_parts(self):
        a = set(self.tok.tokenize("No. 79/Leg. C3/2025/Law."))
        b = set(self.tok.tokenize("Notification No. 79/Leg.C3/2025/Law"))
        self.assertTrue({"79", "leg", "c3", "2025", "law"} <= a & b)

    def test_dates_and_leading_zeros(self):
        tokens = self.tok.tokenize("dated 05-09-2012")
        self.assertIn("05-09-2012", tokens)
        self.assertIn("5", tokens)
        self.assertIn("2012", tokens)

    def test_indian_digit_grouping(self):
        self.assertIn("10000000", self.tok.tokenize("Rs.1,00,00,000/-"))

    def test_no_english_stemming_or_stopword_removal(self):
        tokens = self.tok.tokenize("the appeals shall be disposed")
        self.assertEqual(tokens, ["the", "appeals", "shall", "be", "disposed"])

    def test_section_numbers_kept(self):
        self.assertEqual(self.tok.tokenize("section 74"), ["section", "74"])

    def test_single_letter_noise_dropped(self):
        self.assertEqual(self.tok.tokenize("മ a ക"), [])

    def test_empty(self):
        self.assertEqual(self.tok.tokenize(""), [])
        self.assertEqual(self.tok.tokenize("   "), [])

    def test_version_reflects_settings(self):
        self.assertNotEqual(Tokenizer(subword="prefix", prefix_len=5).version,
                            Tokenizer(subword="ngram", ngram_n=4).version)


if __name__ == "__main__":
    unittest.main()
