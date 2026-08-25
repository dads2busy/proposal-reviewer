import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proposal_reviewer import slot_labels


class TestSlotLabels(unittest.TestCase):
    def test_distinct_models_keep_their_names(self):
        self.assertEqual(slot_labels(["gemma3:27b", "qwen2.5:32b"]),
                         ["gemma3:27b", "qwen2.5:32b"])

    def test_identical_models_get_sample_suffixes(self):
        self.assertEqual(slot_labels(["uva:Kimi K2.5", "uva:Kimi K2.5"]),
                         ["uva:Kimi K2.5 (sample 1)", "uva:Kimi K2.5 (sample 2)"])

    def test_labels_are_always_distinct(self):
        for pair in (["a", "a"], ["a", "b"]):
            labels = slot_labels(pair)
            self.assertEqual(len(set(labels)), 2)


from proposal_reviewer import is_single_family

CFG = {"providers": {"uva": {"kind": "openai",
                             "base_url": "https://example.invalid/api/",
                             "api_key_env": "TEST_KEY"}}}


class TestSingleFamily(unittest.TestCase):
    def test_same_provider_and_model_is_single_family(self):
        self.assertTrue(is_single_family(CFG, ["uva:Kimi K2.5", "uva:Kimi K2.5"]))

    def test_different_models_is_not(self):
        self.assertFalse(is_single_family(CFG, ["gemma3:27b", "qwen2.5:32b"]))

    def test_same_model_different_providers_is_not(self):
        cfg = {"providers": {"uva": CFG["providers"]["uva"],
                             "other": {"kind": "openai",
                                       "base_url": "https://other.invalid/api/",
                                       "api_key_env": "TEST_KEY"}}}
        self.assertFalse(is_single_family(cfg, ["uva:qwen", "other:qwen"]))

    def test_prefixed_and_bare_same_ollama_model_is_single_family(self):
        self.assertTrue(is_single_family({}, ["ollama:gemma3:27b", "gemma3:27b"]))


if __name__ == "__main__":
    unittest.main()
