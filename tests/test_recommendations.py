import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proposal_reviewer import is_required_criterion, is_missing_result, build_recommendation_digest


# What a NOFO requires is declared by its config; this is the CDC NOFO's list.
CDC_REQUIRED = ["data management plan", "letter of support", "letters of support",
                "organizational chart", "org chart"]


class TestDetectionHelpers(unittest.TestCase):
    def test_required_detects_data_management_plan(self):
        self.assertTrue(is_required_criterion(
            {"name": "Eval: data management plan", "look_for": "DMP covering access"},
            CDC_REQUIRED))

    def test_required_detects_letter_of_support(self):
        self.assertTrue(is_required_criterion(
            {"name": "Capacity: collaborations", "look_for": "at least one letter of support"},
            CDC_REQUIRED))

    def test_required_detects_org_chart(self):
        self.assertTrue(is_required_criterion(
            {"name": "Project management", "look_for": "organizational chart with roles"},
            CDC_REQUIRED))

    def test_required_false_for_plain_criterion(self):
        self.assertFalse(is_required_criterion(
            {"name": "Background", "look_for": "influenza knowledge and clear problems"},
            CDC_REQUIRED))

    def test_missing_true_on_section_missing_justification(self):
        self.assertTrue(is_missing_result(
            {"justification": "Required section missing.", "gaps": []}))

    def test_missing_true_on_placeholder_gap(self):
        self.assertTrue(is_missing_result(
            {"justification": "", "gaps": ["The section is entirely placeholder text"]}))

    def test_missing_false_on_substantive_result(self):
        self.assertFalse(is_missing_result(
            {"justification": "Solid section", "gaps": ["one minor detail could be clearer"]}))

    def test_missing_false_on_benign_missing_phrase(self):
        self.assertFalse(is_missing_result(
            {"justification": "", "gaps": ["the discussion is missing a citation"]}))


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


from proposal_reviewer import synthesize_recommendations, deterministic_recommendations


class TestSynthesis(unittest.TestCase):
    def test_uses_model_output_when_valid(self):
        cfg, models, results = _sample_cfg_and_results()
        fake = lambda m, s, u: (
            '[{"title":"Write Background","points_at_stake":10,'
            '"criteria":["§1.1"],"do":"Add the section","why":"empty"}]')
        recs = synthesize_recommendations(cfg, models, results, chat_fn=fake)
        self.assertEqual(recs[0]["title"], "Write Background")

    def test_strips_code_fences(self):
        cfg, models, results = _sample_cfg_and_results()
        fake = lambda m, s, u: '```json\n[{"title":"X","points_at_stake":1,"criteria":[],"do":"d","why":"w"}]\n```'
        recs = synthesize_recommendations(cfg, models, results, chat_fn=fake)
        self.assertEqual(recs[0]["title"], "X")

    def test_respects_top_n(self):
        cfg, models, results = _sample_cfg_and_results()
        cfg["top_recommendations"] = 1
        fake = lambda m, s, u: (
            '[{"title":"a","points_at_stake":10,"criteria":[],"do":"x","why":"y"},'
            '{"title":"b","points_at_stake":5,"criteria":[],"do":"x","why":"y"}]')
        recs = synthesize_recommendations(cfg, models, results, chat_fn=fake)
        self.assertEqual(len(recs), 1)

    def test_falls_back_on_exception(self):
        cfg, models, results = _sample_cfg_and_results()
        def boom(m, s, u):
            raise TimeoutError("timed out")
        recs = synthesize_recommendations(cfg, models, results, chat_fn=boom)
        self.assertTrue(recs)  # deterministic list, non-empty
        # required/missing items rank first; Background & DMP both qualify
        self.assertIn(recs[0]["title"], ("Background", "Eval: data management plan"))

    def test_falls_back_on_garbage(self):
        cfg, models, results = _sample_cfg_and_results()
        recs = synthesize_recommendations(cfg, models, results, chat_fn=lambda m, s, u: "not json at all")
        self.assertTrue(recs)
        self.assertIn("points_at_stake", recs[0])

    def test_empty_when_no_gaps(self):
        cfg = {"criteria": [{"section": "1", "name": "x", "max": 5, "look_for": ""}]}
        results = {"m1": [{"awarded": 5, "gaps": []}], "m2": [{"awarded": 5, "gaps": []}]}
        recs = synthesize_recommendations(cfg, ["m1", "m2"], results, chat_fn=lambda *a: "[]")
        self.assertEqual(recs, [])

    def test_deterministic_orders_required_first(self):
        cfg, models, results = _sample_cfg_and_results()
        from proposal_reviewer import build_recommendation_digest
        digest = build_recommendation_digest(cfg, models, results)
        recs = deterministic_recommendations(digest, 7)
        # Background (10 pts, required) before DMP (6 pts, required)
        self.assertEqual(recs[0]["title"], "Background")


class TestSynthesisRetry(unittest.TestCase):
    def test_retries_then_succeeds_instead_of_fallback(self):
        cfg, models, results = _sample_cfg_and_results()
        seq = ["garbage",
               '[{"title":"Fix","points_at_stake":10,"criteria":[],"do":"d","why":"w"}]']
        calls = []
        def fake(m, s, u):
            calls.append(u)
            return seq[len(calls) - 1]
        recs = synthesize_recommendations(cfg, models, results, chat_fn=fake)
        self.assertEqual(recs[0]["title"], "Fix")   # model output, not deterministic fallback
        self.assertEqual(len(calls), 2)


from proposal_reviewer import render_recommendations


class TestRender(unittest.TestCase):
    def test_header_and_item_format(self):
        md = render_recommendations([
            {"title": "Write Background", "points_at_stake": 10,
             "criteria": ["§1.1"], "do": "Add the section", "why": "currently empty"}])
        self.assertIn("## 2. Top recommendations (synthesized, ranked by importance)", md)
        self.assertIn("### 1. Write Background (§1.1) — 10 pts at stake", md)
        self.assertIn("**Do:** Add the section", md)
        self.assertIn("**Why:** currently empty", md)

    def test_numbers_multiple_items(self):
        md = render_recommendations([
            {"title": "A", "points_at_stake": 5, "criteria": [], "do": "x", "why": "y"},
            {"title": "B", "points_at_stake": 3, "criteria": [], "do": "x", "why": "y"}])
        self.assertIn("### 1. A", md)
        self.assertIn("### 2. B", md)

    def test_empty_recs_message(self):
        md = render_recommendations([])
        self.assertIn("## 2. Top recommendations", md)
        self.assertIn("No material gaps", md)

    def test_render_omits_none_points(self):
        md = render_recommendations([{
            "title": "X",
            "points_at_stake": None,
            "criteria": [],
            "do": "d",
            "why": "w"
        }])
        self.assertIn("### 1. X", md)
        self.assertNotIn("None pts at stake", md)


from proposal_reviewer import _chat_json, JSONRetryError, MAX_JSON_ATTEMPTS


class TestChatJSONRetry(unittest.TestCase):
    def test_returns_validated_on_first_success(self):
        calls = []
        def fake(m, s, u):
            calls.append(u)
            return '{"awarded": 5}'
        out = _chat_json("m", "sys", "user", validate=lambda d: d, chat_fn=fake)
        self.assertEqual(out, {"awarded": 5})
        self.assertEqual(len(calls), 1)

    def test_retries_then_succeeds_with_nudge(self):
        calls = []
        seq = ["not json", '{"ok": true}']
        def fake(m, s, u):
            calls.append(u)
            return seq[len(calls) - 1]
        out = _chat_json("m", "sys", "USERPROMPT", validate=lambda d: d, chat_fn=fake)
        self.assertEqual(out, {"ok": True})
        self.assertEqual(len(calls), 2)
        self.assertIn("USERPROMPT", calls[1])          # original prompt retained
        self.assertIn("not valid JSON", calls[1])      # corrective nudge appended on retry

    def test_raises_after_exhausting_attempts(self):
        calls = []
        def fake(m, s, u):
            calls.append(u)
            return "still not json"
        with self.assertRaises(JSONRetryError) as ctx:
            _chat_json("m", "sys", "user", validate=lambda d: d, chat_fn=fake)
        self.assertEqual(len(calls), MAX_JSON_ATTEMPTS)
        self.assertEqual(ctx.exception.raw, "still not json")

    def test_validate_rejection_triggers_retry(self):
        calls = []
        seq = ['{"x": 1}', '[1, 2]']
        def fake(m, s, u):
            calls.append(u)
            return seq[len(calls) - 1]
        def validate(parsed):
            if not isinstance(parsed, list):
                raise ValueError("need a list")
            return parsed
        out = _chat_json("m", "sys", "user", validate=validate, chat_fn=fake)
        self.assertEqual(out, [1, 2])
        self.assertEqual(len(calls), 2)


from proposal_reviewer import build_combined


class TestBuildCombinedIntegration(unittest.TestCase):
    def _cfg_results_recs(self):
        cfg = {"name": "T", "total_points": 5,
               "criteria": [{"section": "1.1", "name": "Background", "max": 5,
                             "look_for": "x", "group": "G"}],
               "section_groups": {"1": "G"}, "section_totals": {"G": 5}}
        A, B = "m1", "m2"
        results = {
            A: [{"awarded": 0, "max": 5, "strengths": [], "gaps": ["placeholder"]}],
            B: [{"awarded": 0, "max": 5, "strengths": [], "gaps": ["empty"]}],
        }
        recs = [{"title": "Fix Background", "points_at_stake": 10,
                 "criteria": ["§1.1"], "do": "Write it", "why": "empty"}]
        return cfg, [A, B], results, recs

    def test_recommendations_section_is_second(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertLess(md.index("## 2. Top recommendations"),
                        md.index("## 3. Per-criterion"))

    def test_renders_recommendation_and_renumbered_final(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertIn("### 1. Fix Background", md)
        self.assertIn("## 7. Final", md)

    def test_default_wording_claims_two_models(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertIn("both models dock points", md)
        self.assertIn("models diverge", md)
        self.assertNotIn("No independent second opinion", md)

    def test_single_family_relabels_and_warns(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs,
                            single_family=True)
        self.assertIn("consistent across both samples", md)
        self.assertIn("unstable score", md)
        self.assertIn("consistently scored", md)
        self.assertIn("No independent second opinion", md)
        self.assertNotIn("both models dock points", md)

    def test_single_family_keeps_section_numbering(self):
        cfg, models, results, recs = self._cfg_results_recs()
        md = build_combined(cfg, models, results, 100.0, [], recs,
                            single_family=True)
        for heading in ("## 1. Overall scores", "## 3. Per-criterion",
                        "## 4. Priority gaps", "## 5. Strengths",
                        "## 6. Missing RFA phrases", "## 7. Final"):
            self.assertIn(heading, md)

    def test_local_only_run_is_described_as_offline(self):
        cfg, models, results, recs = self._cfg_results_recs()
        cfg["models"] = ["gemma3:27b", "qwen2.5:32b"]
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertIn("offline (local models)", md)

    def test_remote_run_is_not_described_as_local(self):
        # This line ends up in a document a PI may forward; it must not claim
        # a local run when both models were provider-hosted.
        cfg, models, results, recs = self._cfg_results_recs()
        cfg["providers"] = {"uva": {"kind": "openai",
                                    "base_url": "https://example.invalid/api/",
                                    "api_key_env": "TEST_KEY"}}
        cfg["models"] = ["uva:Kimi K2.5", "uva:Llama 4"]
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertNotIn("offline", md)
        self.assertNotIn("local models", md)

    def test_hybrid_run_claims_neither_purely_local_nor_purely_remote(self):
        cfg, models, results, recs = self._cfg_results_recs()
        cfg["providers"] = {"uva": {"kind": "openai",
                                    "base_url": "https://example.invalid/api/",
                                    "api_key_env": "TEST_KEY"}}
        cfg["models"] = ["uva:Kimi K2.5", "qwen2.5:32b"]
        md = build_combined(cfg, models, results, 100.0, [], recs)
        self.assertNotIn("offline (local models)", md)
        self.assertIn("mixed", md)


from proposal_reviewer import ask_model


class TestAskModelRetry(unittest.TestCase):
    CRIT = {"name": "Test", "look_for": "stuff", "max": 10}

    def test_parses_and_clamps_on_success(self):
        fake = lambda m, s, u: '{"awarded": 15, "strengths": [], "gaps": []}'
        res = ask_model("m", "sys", self.CRIT, "Title", "body text", chat_fn=fake)
        self.assertEqual(res["awarded"], 10)   # clamped to max
        self.assertEqual(res["max"], 10)

    def test_retries_bad_then_good(self):
        calls = []
        seq = ["garbage", '{"awarded": 7}']
        def fake(m, s, u):
            calls.append(u)
            return seq[len(calls) - 1]
        res = ask_model("m", "sys", self.CRIT, "Title", "body", chat_fn=fake)
        self.assertEqual(res["awarded"], 7)
        self.assertEqual(len(calls), 2)

    def test_unparsed_fallback_after_exhaustion(self):
        fake = lambda m, s, u: "never json"
        res = ask_model("m", "sys", self.CRIT, "Title", "body", chat_fn=fake)
        self.assertEqual(res["awarded"], 0)
        self.assertEqual(res["max"], 10)
        self.assertIn("UNPARSED MODEL OUTPUT", res["justification"])
        self.assertIn("never json", res["justification"])


from pathlib import Path as _Path
from proposal_reviewer import stamped


class TestStamped(unittest.TestCase):
    def test_composes_timestamped_name(self):
        self.assertEqual(
            stamped("compliance_report_combined", "20260622-211500", "md"),
            "compliance_report_combined_20260622-211500.md")

    def test_with_suffix_swaps_only_extension(self):
        # render() derives html/pdf from the combined .md path via with_suffix;
        # the timestamp must survive because it lives in the stem, not the extension.
        p = _Path(stamped("compliance_report_combined", "20260622-211500", "md"))
        self.assertEqual(p.with_suffix(".pdf").name,
                         "compliance_report_combined_20260622-211500.pdf")


if __name__ == "__main__":
    unittest.main()
