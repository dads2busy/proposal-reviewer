"""Section parsing: config-supplied heading patterns and the body-size cap."""
import json
import unittest
from unittest import mock

from proposal_reviewer import (DEFAULT_MAX_SECTION_CHARS, DEFAULT_NUM_CTX,
                               _ollama_chat, ask_model, parse_sections,
                               review_with_model)

# A proposal whose headings carry no section numbers the numeric pass can see.
UNNUMBERED = """\
1. PROPOSAL SUMMARY **3 PAGES
Summary body.

4. Technical Plan **25 pages
System architecture and interfaces spanning all TAs **2.5 pages
Architecture body.

TA1. Modular, Mechanistic Models **5 pages
TA1 body.
TA1.A: A subtask that must NOT start a new section.
More TA1 body.

TA2. New Science Engine **5 pages
TA2 body.
"""

PATTERNS = [
    {"pattern": r"^1\.\s+PROPOSAL SUMMARY", "id": "1"},
    {"pattern": r"^System architecture and interfaces", "id": "4.0"},
    {"pattern": r"^TA1\.\s", "id": "4.1"},
    {"pattern": r"^TA2\.\s", "id": "4.2"},
]


class TestHeadingPatterns(unittest.TestCase):
    def test_without_patterns_nothing_is_found(self):
        self.assertEqual(parse_sections(UNNUMBERED), {})

    def test_patterns_assign_section_ids(self):
        sec = parse_sections(UNNUMBERED, PATTERNS)
        self.assertEqual(sorted(sec), ["1", "4.0", "4.1", "4.2"])

    def test_body_runs_to_the_next_matched_heading(self):
        sec = parse_sections(UNNUMBERED, PATTERNS)
        self.assertIn("TA1 body.", sec["4.1"]["body"])
        self.assertIn("More TA1 body.", sec["4.1"]["body"])
        self.assertNotIn("TA2 body.", sec["4.1"]["body"])

    def test_subtask_label_does_not_split_the_section(self):
        # "TA1.A:" has no space after the dot, so `^TA1\.\s` must not match it.
        sec = parse_sections(UNNUMBERED, PATTERNS)
        self.assertIn("TA1.A: A subtask", sec["4.1"]["body"])

    def test_padded_table_cell_does_not_anchor_a_section(self):
        # A table cell like "TA1.      TAZ effects ..." looks like the heading unless
        # the pattern demands a single space then capital-then-lowercase.
        md = "TA1.        TAZ effects on the microcirculation\nCell text.\n" + UNNUMBERED
        strict = [{"pattern": r"^TA1\.\s[A-Z][a-z]", "id": "4.1"}]
        sec = parse_sections(md, strict)
        self.assertNotIn("Cell text.", sec["4.1"]["body"])
        self.assertIn("TA1 body.", sec["4.1"]["body"])

    def test_toc_lines_are_ignored(self):
        toc = "TA1. Modular, Mechanistic Models **5 pages .................. 13\n" + UNNUMBERED
        sec = parse_sections(toc, PATTERNS)
        self.assertIn("TA1 body.", sec["4.1"]["body"])

    def test_numeric_headings_still_parse_alongside_patterns(self):
        md = UNNUMBERED + "\n7.1 Team expertise\nTeam body.\n"
        sec = parse_sections(md, PATTERNS)
        self.assertIn("7.1", sec)
        self.assertIn("Team body.", sec["7.1"]["body"])


class TestBodyCap(unittest.TestCase):
    CRIT = {"name": "C", "max": 10, "look_for": "things"}

    def _capture(self):
        seen = {}

        def fake(model, system, user, timeout=900, **kw):
            seen["user"] = user
            return '{"awarded": 5, "max": 10, "strengths": [], "gaps": [], "justification": "j"}'

        return seen, fake

    def test_default_cap_applies(self):
        seen, fake = self._capture()
        body = "z" * (DEFAULT_MAX_SECTION_CHARS + 5000)
        ask_model("m", "sys", self.CRIT, "T", body, chat_fn=fake)
        self.assertEqual(seen["user"].count("z"), DEFAULT_MAX_SECTION_CHARS)

    def test_cap_can_be_raised(self):
        seen, fake = self._capture()
        body = "z" * 30000
        ask_model("m", "sys", self.CRIT, "T", body, max_chars=32000, chat_fn=fake)
        self.assertEqual(seen["user"].count("z"), 30000)

    def test_review_with_model_honors_config_cap(self):
        seen, fake = self._capture()
        cfg = {"criteria": [dict(self.CRIT, section="*")], "max_section_chars": 25000}
        review_with_model("m", cfg, {}, "y" * 30000, chat_fn=fake)
        self.assertEqual(seen["user"].count("y"), 25000)


if __name__ == "__main__":
    unittest.main()


class TestNumCtx(unittest.TestCase):
    """num_ctx must be configurable: a raised max_section_chars is useless if the
    section still overflows the model's context window."""

    def _payload(self, **kw):
        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["payload"] = json.loads(req.data.decode())
            raise RuntimeError("stop after capturing payload")

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaises(RuntimeError):
                _ollama_chat("m", "sys", "user", **kw)
        return seen["payload"]

    def test_default_num_ctx(self):
        self.assertEqual(self._payload()["options"]["num_ctx"], DEFAULT_NUM_CTX)

    def test_num_ctx_can_be_raised(self):
        self.assertEqual(self._payload(num_ctx=32768)["options"]["num_ctx"], 32768)

    def test_review_with_model_passes_config_num_ctx(self):
        seen = {}

        def fake(model, system, user, timeout=900, num_ctx=None):
            seen["num_ctx"] = num_ctx
            return '{"awarded": 1, "max": 10, "strengths": [], "gaps": [], "justification": "j"}'

        cfg = {"criteria": [{"name": "C", "max": 10, "look_for": "x", "section": "*"}],
               "num_ctx": 32768}
        review_with_model("m", cfg, {}, "body", chat_fn=fake)
        self.assertEqual(seen["num_ctx"], 32768)


class TestReportHeader(unittest.TestCase):
    """A per-model report has to name the slot it came from: 'local' is false for
    a remote provider, and two samples of one model need distinguishing."""

    CFG = {"criteria": [{"name": "C", "max": 10, "look_for": "x", "section": "*"}]}

    def _fake(self, model, system, user, timeout=900, num_ctx=None):
        return '{"awarded": 1, "max": 10, "strengths": [], "gaps": [], "justification": "j"}'

    def test_header_uses_the_slot_label_when_given(self):
        _, report = review_with_model("Kimi K2.5", self.CFG, {}, "body",
                                      chat_fn=self._fake,
                                      label="uva:Kimi K2.5 (sample 2)")
        self.assertIn("# Compliance Assessment \u2014 uva:Kimi K2.5 (sample 2)", report)

    def test_header_does_not_claim_local_for_a_remote_model(self):
        _, report = review_with_model("Kimi K2.5", self.CFG, {}, "body",
                                      chat_fn=self._fake,
                                      label="uva:Kimi K2.5")
        self.assertNotIn("(local)", report)

    def test_header_falls_back_to_the_model_name(self):
        _, report = review_with_model("gemma3:27b", self.CFG, {}, "body",
                                      chat_fn=self._fake)
        self.assertIn("# Compliance Assessment \u2014 gemma3:27b", report)
