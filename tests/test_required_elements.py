"""Required attachments are per-opportunity, so they must come from the config.

The engine used to hardcode the CDC NOFO's list ("data management plan", "letters of
support", "organizational chart") and name it verbatim in the synthesis system prompt.
Against any other opportunity that invents a requirement: the ARPA-H IGoR run produced
a top-ranked "provide a Data Management Plan, a mandatory deliverable" recommendation
when the phrase "data management" appears nowhere in that solicitation.
"""
import unittest

from proposal_reviewer import (build_recommendation_digest, is_required_criterion,
                               synthesize_recommendations, synthesis_system_prompt)

CRIT_DMP = {"name": "Eval: data management plan", "max": 3,
            "look_for": "Data management plan covering data generated."}
CRIT_TA1 = {"name": "TA1 — Digital twins", "max": 8, "look_for": "Mechanistic models."}


class TestIsRequiredCriterion(unittest.TestCase):
    def test_no_config_list_means_nothing_is_required(self):
        self.assertFalse(is_required_criterion(CRIT_DMP, []))
        self.assertFalse(is_required_criterion(CRIT_TA1, []))

    def test_config_list_marks_matching_criterion(self):
        self.assertTrue(is_required_criterion(CRIT_DMP, ["data management plan"]))
        self.assertFalse(is_required_criterion(CRIT_TA1, ["data management plan"]))

    def test_matching_is_case_insensitive(self):
        self.assertTrue(is_required_criterion(CRIT_DMP, ["DATA Management Plan"]))


class TestSynthesisPrompt(unittest.TestCase):
    def test_prompt_omits_required_clause_when_none_declared(self):
        p = synthesis_system_prompt("a reviewer", 7, [])
        self.assertNotIn("REQUIRED", p)
        self.assertNotIn("data management plan", p.lower())

    def test_prompt_names_only_the_declared_elements(self):
        p = synthesis_system_prompt("a reviewer", 7, ["letters of support"])
        self.assertIn("REQUIRED", p)
        self.assertIn("letters of support", p)
        self.assertNotIn("data management plan", p.lower())

    def test_igor_style_config_never_mentions_a_dmp(self):
        cfg = {"criteria": [CRIT_TA1], "reviewer_role": "an ARPA-H reviewer"}
        seen = {}

        def fake(model, system, user, **kw):
            seen["system"] = system
            return '[{"title": "t", "points_at_stake": 2, "criteria": [], "do": "d", "why": "w"}]'

        results = {"a": [{"awarded": 6, "max": 8, "gaps": ["g"], "justification": ""}],
                   "b": [{"awarded": 7, "max": 8, "gaps": ["g"], "justification": ""}]}
        synthesize_recommendations(cfg, ["a", "b"], results, chat_fn=fake)
        self.assertNotIn("data management", seen["system"].lower())
        self.assertNotIn("organizational chart", seen["system"].lower())


class TestDigestFlagging(unittest.TestCase):
    def _digest(self, cfg):
        results = {"a": [{"awarded": 1, "max": 3, "gaps": [], "justification": "thin"}],
                   "b": [{"awarded": 1, "max": 3, "gaps": [], "justification": "thin"}]}
        return build_recommendation_digest(cfg, ["a", "b"], results)

    def test_not_required_without_config_declaration(self):
        d = self._digest({"criteria": [CRIT_DMP]})
        self.assertFalse(d[0]["required"])

    def test_required_when_declared(self):
        d = self._digest({"criteria": [CRIT_DMP], "required_elements": ["data management plan"]})
        self.assertTrue(d[0]["required"])

    def test_missing_section_still_flags_required_without_a_list(self):
        # An absent section is a hard miss regardless of the opportunity's attachments.
        results = {"a": [{"awarded": 0, "max": 3, "gaps": ["Section 2.5 not found in proposal."],
                          "justification": "Required section missing."}],
                   "b": [{"awarded": 0, "max": 3, "gaps": [], "justification": ""}]}
        d = build_recommendation_digest({"criteria": [CRIT_DMP]}, ["a", "b"], results)
        self.assertTrue(d[0]["required"])


if __name__ == "__main__":
    unittest.main()
