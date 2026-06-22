import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proposal_reviewer import is_required_criterion, is_missing_result


class TestDetectionHelpers(unittest.TestCase):
    def test_required_detects_data_management_plan(self):
        self.assertTrue(is_required_criterion(
            {"name": "Eval: data management plan", "look_for": "DMP covering access"}))

    def test_required_detects_letter_of_support(self):
        self.assertTrue(is_required_criterion(
            {"name": "Capacity: collaborations", "look_for": "at least one letter of support"}))

    def test_required_detects_org_chart(self):
        self.assertTrue(is_required_criterion(
            {"name": "Project management", "look_for": "organizational chart with roles"}))

    def test_required_false_for_plain_criterion(self):
        self.assertFalse(is_required_criterion(
            {"name": "Background", "look_for": "influenza knowledge and clear problems"}))

    def test_missing_true_on_section_missing_justification(self):
        self.assertTrue(is_missing_result(
            {"justification": "Required section missing.", "gaps": []}))

    def test_missing_true_on_placeholder_gap(self):
        self.assertTrue(is_missing_result(
            {"justification": "", "gaps": ["The section is entirely placeholder text"]}))

    def test_missing_false_on_substantive_result(self):
        self.assertFalse(is_missing_result(
            {"justification": "Solid section", "gaps": ["one minor detail could be clearer"]}))


if __name__ == "__main__":
    unittest.main()
