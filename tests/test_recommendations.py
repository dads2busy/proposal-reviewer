import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proposal_reviewer import is_required_criterion, is_missing_result, build_recommendation_digest


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


def _sample_cfg_and_results():
    cfg = {"criteria": [
        {"section": "1.1", "name": "Background", "max": 5, "look_for": "influenza"},
        {"section": "2.5", "name": "Eval: data management plan", "max": 3, "look_for": "dmp"},
        {"section": "3.1", "name": "Strong capacity", "max": 10, "look_for": "expertise"},
    ]}
    A, B = "m1", "m2"
    results = {
        A: [
            {"awarded": 0, "gaps": ["entirely placeholder text"], "justification": ""},
            {"awarded": 0, "gaps": [], "justification": "Required section missing."},
            {"awarded": 10, "gaps": [], "justification": "great"},
        ],
        B: [
            {"awarded": 0, "gaps": ["no content provided"], "justification": ""},
            {"awarded": 0, "gaps": [], "justification": "missing"},
            {"awarded": 9, "gaps": ["one minor gap"], "justification": "great"},
        ],
    }
    return cfg, [A, B], results


class TestDigest(unittest.TestCase):
    def test_excludes_item_when_one_model_at_max(self):
        cfg, models, results = _sample_cfg_and_results()
        digest = build_recommendation_digest(cfg, models, results)
        self.assertNotIn("Strong capacity", [d["name"] for d in digest])

    def test_points_at_stake_sums_both_shortfalls(self):
        cfg, models, results = _sample_cfg_and_results()
        digest = build_recommendation_digest(cfg, models, results)
        bg = next(d for d in digest if d["name"] == "Background")
        self.assertEqual(bg["points_at_stake"], 10)  # (5-0)+(5-0)

    def test_required_and_missing_flags_set(self):
        cfg, models, results = _sample_cfg_and_results()
        digest = build_recommendation_digest(cfg, models, results)
        dmp = next(d for d in digest if "data management" in d["name"].lower())
        self.assertTrue(dmp["required"])
        self.assertTrue(dmp["missing"])

    def test_gaps_merged_from_both_models(self):
        cfg, models, results = _sample_cfg_and_results()
        digest = build_recommendation_digest(cfg, models, results)
        bg = next(d for d in digest if d["name"] == "Background")
        self.assertEqual(set(bg["gaps"]), {"entirely placeholder text", "no content provided"})


if __name__ == "__main__":
    unittest.main()
