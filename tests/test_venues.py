import unittest

from reference_finder.venues import VENUES, is_excluded, source_matches


class VenueTests(unittest.TestCase):
    def test_official_names_proceedings_and_aliases(self):
        examples = {
            "ccs": ["Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security", "CCS '24", "ACM CCS 2023", "Conference on Computer and Communications Security"],
            "usenix-security": ["33rd USENIX Security Symposium (USENIX Security 24)", "USENIX Security '25", "USENIXSecurity24"],
            "sp": ["2024 IEEE Symposium on Security and Privacy (SP)", "IEEE S&P 2023", "S&P", "Symposium on Security and Privacy"],
            "eurosp": ["2024 IEEE European Symposium on Security and Privacy (EuroS&P)", "EuroS&P 2023", "IEEE EuroSP"],
            "asiaccs": ["Proceedings of the 19th ACM Asia Conference on Computer and Communications Security", "ACM Symposium on Information, Computer, and Communications Security", "ASIA CCS '24"],
            "crypto": ["Advances in Cryptology – CRYPTO 2024, Part I", "44th Annual International Cryptology Conference", "CRYPTO 2024"],
            "eurocrypt": ["Advances in Cryptology – EUROCRYPT 2024", "Annual International Conference on the Theory and Applications of Cryptographic Techniques", "EUROCRYPT 2024"],
            "asiacrypt": ["Advances in Cryptology - ASIACRYPT 2024", "International Conference on the Theory and Application of Cryptology and Information Security", "ASIACRYPT 2024"],
        }
        for key, labels in examples.items():
            for label in labels:
                with self.subTest(key=key, label=label):
                    self.assertTrue(VENUES[key].matches(label))
                    self.assertEqual([v.key for v in VENUES.values() if v.matches(label)], [key])

    def test_adjacent_events_are_excluded(self):
        labels = [
            "2024 IEEE Symposium on Security and Privacy Workshops (SPW)",
            "2024 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW)",
            "CCS 2024 Companion", "ACM AsiaCCS 2024 Posters",
            "Workshop on Advances in Cryptology - CRYPTO 2024",
            "USENIX Security Symposium Poster Session",
        ]
        for label in labels:
            with self.subTest(label=label):
                self.assertFalse(any(venue.matches(label) for venue in VENUES.values()))

    def test_unrelated_or_generic_sources_are_excluded(self):
        labels = [
            "Lecture Notes in Computer Science", "LNCS", "Advances in Cryptology",
            "IEEE Security & Privacy", "Security and Privacy", "USENIX Annual Technical Conference",
            "Cryptography and Communications", "International Conference on Applied Cryptography and Network Security",
            "International Conference on Security and Privacy in Communication Networks",
            "International Symposium on Security and Privacy",
            "Crypto Finance Conference", "Eurocryptography Journal", "", None,
        ]
        for label in labels:
            with self.subTest(label=label):
                self.assertFalse(any(venue.matches(label) for venue in VENUES.values()))

    def test_catalog_queries_and_semantic_names(self):
        self.assertEqual(len(VENUES), 8)
        for key, venue in VENUES.items():
            self.assertEqual(key, venue.key)
            self.assertTrue(venue.openalex_queries)
            for name in venue.semantic_scholar_names:
                with self.subTest(key=key, name=name):
                    self.assertTrue(venue.matches(name))

    def test_openalex_source_alternate_title(self):
        self.assertTrue(source_matches(VENUES["crypto"], {"display_name": "Unrecognized", "alternate_titles": ["CRYPTO"]}))
        self.assertFalse(source_matches(VENUES["crypto"], {"display_name": "Lecture Notes in Computer Science"}))
        self.assertFalse(source_matches(VENUES["sp"], {"display_name": "IEEE Symposium on Security and Privacy Workshops", "alternate_titles": ["SP"]}))

    def test_exclusion_helper(self):
        for label in ("SPW", "EuroSPW", "EuroS&PW", "IEEE S&PW 2024", "CCS Companion"):
            with self.subTest(label=label):
                self.assertTrue(is_excluded(label))
        self.assertFalse(is_excluded("CRYPTO"))
        self.assertFalse(is_excluded(None))


if __name__ == "__main__":
    unittest.main()
