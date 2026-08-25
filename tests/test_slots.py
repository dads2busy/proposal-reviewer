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


if __name__ == "__main__":
    unittest.main()
