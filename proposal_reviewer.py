#!/usr/bin/env python3
"""
proposal_reviewer.py — proposal-agnostic offline compliance reviewer.

Runs a two-model ensemble plus a deterministic keyword pass against a grant
proposal, scores it against a rubric defined entirely in a YAML config, and merges
the results into a combined report (.md / .html / .pdf). Models run on a local
Ollama or on UVA's self-hosted RC GenAI — never on a commercial provider.

The engine is generic. Everything proposal-specific — the criteria, point caps,
section map, keywords, models — lives in the config file, so one engine serves
every proposal you review.

Usage:
    python proposal_reviewer.py PROPOSAL --config configs/cdc-ip-26-0102.yaml
    python proposal_reviewer.py --config CONFIG --check-provider  # test backends, then exit
    python proposal_reviewer.py PROPOSAL --config CONFIG --dry-run   # validate mapping, no model calls
    python proposal_reviewer.py PROPOSAL --config CONFIG --models llama3.3:70b qwen3.6:27b

Requires: pyyaml (pip install pyyaml); pandoc + pdftotext for extraction/rendering;
and, for local models, an Ollama (>=0.19) with those models pulled. Remote providers
are declared per-config under `providers:` and keyed from .env — see README.md.
"""

import argparse, datetime, json, os, re, subprocess, sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency: pyyaml.  Install with:  pip install pyyaml")

import providers
from providers import ProviderError

# Section text handed to a model per criterion. Raise via `max_section_chars` in
# the config when a proposal's sections are long — and raise `num_ctx` with it, since
# a section that fits the cap but overflows the context window is silently discarded
# by the model rather than by us. Roughly 4 chars per token.
DEFAULT_MAX_SECTION_CHARS = 14000
DEFAULT_NUM_CTX = 16384

DEFAULT_ROLE = "an experienced grant reviewer"
SYSTEM_TEMPLATE = (
    "You are {role}. Given ONE scoring criterion (with its point maximum) and the "
    "relevant proposal section, award an integer 0..max for the EXTENT the applicant "
    "satisfies it. Be specific and critical, grounded only in the provided text; if a "
    "required element is absent, say so and withhold points. "
    'Respond ONLY with JSON: {{"awarded": <int>, "max": <int>, "strengths": [<str>], '
    '"gaps": [<str>], "justification": <str>}}'
)


# ── extraction ──────────────────────────────────────────────────────
# pandoc's markdown writer defaults to SETEXT headings ("===" underlines) up to 2.10
# and to ATX ("#") from 3.0. parse_sections keys on "#", so a setext conversion loses
# every "#"-derived section — and, worse than losing them, the sections that survive
# via heading_patterns absorb the dropped text and get scored against it. The flag
# forcing ATX was renamed at 2.11 and each version rejects the other's spelling.
ATX_FLAGS = ("--markdown-headings=atx", "--atx-headers")


def _run_or_exit(cmd, what):
    """Run an extractor and refuse to hand back nothing.

    The old code took .stdout unconditionally, so a missing binary or an unreadable
    file produced empty text, which scores every criterion 0 with no error anywhere."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{what} failed ({cmd[0]} exit {r.returncode}): {r.stderr.strip()}")
    if not r.stdout.strip():
        sys.exit(f"{what} produced no text. Is the file empty or corrupt?")
    return r.stdout


def _pandoc_markdown(path: Path) -> str:
    """docx -> markdown with ATX headings, whatever pandoc version is installed."""
    for i, flag in enumerate(ATX_FLAGS):
        cmd = ["pandoc", str(path), "-t", "markdown", flag]
        if i == len(ATX_FLAGS) - 1:
            return _run_or_exit(cmd, f"Converting {path.name}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            if not r.stdout.strip():
                sys.exit(f"Converting {path.name} produced no text. "
                         f"Is the file empty or corrupt?")
            return r.stdout


def extract_text(path: Path, stop_marker=None) -> str:
    s = path.suffix.lower()
    if s in {".md", ".txt", ".tex"}:
        text = path.read_text(errors="ignore")
    elif s == ".docx":
        text = _pandoc_markdown(path)
    elif s == ".pdf":
        text = _run_or_exit(["pdftotext", "-layout", str(path), "-"],
                            f"Extracting {path.name}")
    else:
        sys.exit(f"Unsupported file type: {s}")
    if stop_marker and stop_marker in text:
        text = text.split(stop_marker)[0]
    return text


def clean_heading(raw: str) -> str:
    raw = re.sub(r"\[?\\?\[[^\]]*\\?\]\]?\{\.mark\}", "", raw)
    raw = re.sub(r"\\?\[[^\]]*\\?\]", "", raw)
    return raw.replace("\\", "").strip()


def parse_sections(md: str, heading_patterns=None) -> dict:
    """Map numbered heading IDs (1.1, 2.3, ...) to body text. Unnumbered
    subheadings fold into the preceding numbered section.

    Handles three heading styles:
      - Config-supplied `heading_patterns`: [{pattern: <regex>, id: "4.1"}, ...],
        which give a section ID to headings that carry no usable number of their
        own (e.g. "TA1. Digital Twins"). These win over the two passes below.
      - Markdown (pandoc/docx): ## 1.1 Background
      - Plain text (pdftotext): 1.1 Background  (column 0, no TOC dots)
    Only the first occurrence of each section ID is kept so that reference
    numbers later in the document (e.g. "3.4 (2017): e83.") don't overwrite
    the real section content.
    """
    lines = md.splitlines()
    custom = [(re.compile(h["pattern"]), str(h["id"])) for h in (heading_patterns or [])]
    numbered = []
    for i, ln in enumerate(lines):
        # Config-supplied headings, checked first so a proposal with unnumbered
        # headings can still be addressed section by section.
        if custom and "...." not in ln:
            stripped = ln.strip()
            hit = next((sid for rx, sid in custom if rx.search(stripped)), None)
            if hit:
                numbered.append((i, hit, stripped))
                continue
        # Markdown headings produced by pandoc (docx → md conversion)
        m = re.match(r"^#+\s+(.*)$", ln)
        if m:
            title = clean_heading(m.group(1))
            num = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+(.*)$", title)
            if num:
                numbered.append((i, num.group(1), title))
            continue
        # Plain-text numbered headings from pdftotext (e.g. "1.1 Background").
        # Must start at column 0, have at least one dot, begin with a letter
        # after the number, and not be a TOC entry (which contains "....").
        m2 = re.match(r"^(\d+\.\d+(?:\.\d+)?)[.):]?\s+([A-Za-z].{1,})$", ln)
        if m2 and "...." not in ln:
            numbered.append((i, m2.group(1), ln.strip()))

    # Keep only the first occurrence of each section ID.
    seen: set = set()
    deduped = []
    for item in numbered:
        if item[1] not in seen:
            seen.add(item[1])
            deduped.append(item)

    sections = {}
    for idx, (li, sid, title) in enumerate(deduped):
        end = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(lines)
        sections[sid] = {"title": title, "body": "\n".join(lines[li + 1:end]).strip()}
    return sections


def get_body(sections, full_text, sid):
    if sid in (None, "", "*", "all"):
        return full_text, "(full document)"
    sec = sections.get(str(sid))
    if sec and sec["body"]:
        return sec["body"], sec["title"]
    return "", None


# ── grouping for section subtotals ──────────────────────────────────
def resolve_group(crit, section_groups):
    if crit.get("group"):
        return crit["group"]
    sid = str(crit.get("section", ""))
    best = None
    for prefix, grp in (section_groups or {}).items():
        if sid == prefix or sid.startswith(prefix + "."):
            if best is None or len(prefix) > len(best[0]):
                best = (prefix, grp)
    return best[1] if best else "Ungrouped"


# ── model call ──────────────────────────────────────────────────────
def ask_model(model, system_prompt, crit, sect_title, body, rfa_context="",
              max_chars=DEFAULT_MAX_SECTION_CHARS, num_ctx=None, chat_fn=None):
    if chat_fn is None:
        chat_fn = _ollama_chat
    user = (f"CRITERION: {crit['name']}  (max {crit['max']} points)\n"
            f"WHAT REVIEWERS LOOK FOR: {crit['look_for']}\n")
    if rfa_context:
        user += f"\nRELEVANT RFA CONTEXT:\n{rfa_context}\n"
    user += f"\nPROPOSAL SECTION — {sect_title}:\n{body[:max_chars]}"

    def coerce(d):
        d["awarded"] = max(0, min(int(d.get("awarded", 0) or 0), crit["max"]))
        d["max"] = crit["max"]
        return d

    try:
        return _chat_json(model, system_prompt, user, coerce,
                          num_ctx=num_ctx, chat_fn=chat_fn)
    except JSONRetryError as e:
        return {"awarded": 0, "max": crit["max"], "strengths": [], "gaps": [],
                "justification": "UNPARSED MODEL OUTPUT:\n" + e.raw}


def review_with_model(model, cfg, sections, full_text, chat_fn=None, label=None):
    role = cfg.get("reviewer_role", DEFAULT_ROLE)
    system_prompt = cfg.get("system_prompt") or SYSTEM_TEMPLATE.format(role=role)
    rfa_context = cfg.get("rfa_context", "")
    max_chars = int(cfg.get("max_section_chars", DEFAULT_MAX_SECTION_CHARS))
    num_ctx = int(cfg.get("num_ctx", DEFAULT_NUM_CTX))
    results = []
    # Display name only: the slot label distinguishes two samples of one model,
    # and not every provider is local any more.
    lines = [f"# Compliance Assessment — {label or model}\n",
             f"_Generated {datetime.date.today()}_\n"]
    for crit in cfg["criteria"]:
        body, title = get_body(sections, full_text, crit.get("section"))
        if not body:
            res = {"awarded": 0, "max": crit["max"], "strengths": [],
                   "gaps": [f"Section {crit.get('section')} not found in proposal."],
                   "justification": "Required section missing."}
            lines.append(f"\n## {crit['name']} — 0/{crit['max']}  ⚠ SECTION MISSING\n")
        else:
            res = ask_model(model, system_prompt, crit, title, body, rfa_context,
                            max_chars=max_chars, num_ctx=num_ctx, chat_fn=chat_fn)
            lines.append(f"\n## {crit['name']} — {res['awarded']}/{crit['max']}\n"
                         f"_{res.get('justification','').strip()}_\n")
            lines += [f"- ✓ {s}" for s in res.get("strengths", [])]
            lines += [f"- ✗ {g}" for g in res.get("gaps", [])]
        results.append(res)
    return results, "\n".join(lines)


# ── keyword pass ────────────────────────────────────────────────────
def keyword_scan(text, keywords):
    low = text.lower()
    rows = []
    for kw in keywords:
        pat = kw["pattern"]
        label = kw.get("label") or re.sub(r"[\\^$.?*+()\[\]|]", "", pat)[:42]
        rows.append((label, kw.get("category", ""), bool(re.search(pat, low))))
    found = sum(1 for *_, f in rows if f)
    pct = found / len(rows) * 100 if rows else 100.0
    out = [f"# RFA Keyword Coverage\n", f"**{found}/{len(rows)} phrases found ({pct:.0f}%)**\n",
           "| Category | Phrase | Status |", "|---|---|---|"]
    for label, cat, f in rows:
        out.append(f"| {cat} | {label} | {'FOUND' if f else 'MISSING'} |")
    missing = [(l, c) for l, c, f in rows if not f]
    return "\n".join(out), pct, missing


# ── recommendation synthesis ────────────────────────────────────────
MISSING_MARKERS = ("not found", "section missing", "missing.", "placeholder", "still need")


def is_required_criterion(crit, required_elements):
    """True if the criterion hinges on a required attachment the narrative can only
    reference. `required_elements` comes from the config: which attachments an
    opportunity mandates is a property of that opportunity, not of the engine.
    An empty list means the config declares none, and nothing is flagged."""
    text = (crit.get("name", "") + " " + crit.get("look_for", "")).lower()
    return any(p.lower() in text for p in (required_elements or []))


def is_missing_result(res):
    """True if a model's result indicates the scored section was absent or placeholder."""
    blob = (res.get("justification", "") + " " + " ".join(res.get("gaps", []) or [])).lower()
    return any(m in blob for m in MISSING_MARKERS)


def build_recommendation_digest(cfg, models, all_results):
    """Per-criterion candidates for recommendations: items where BOTH models docked
    points (same selection rule as §3/§4). Order follows cfg['criteria']."""
    A, B = models
    ra, rb = all_results[A], all_results[B]
    digest = []
    for crit, a, b in zip(cfg["criteria"], ra, rb):
        mx = crit["max"]
        if a["awarded"] < mx and b["awarded"] < mx:
            missing = is_missing_result(a) or is_missing_result(b)
            required = is_required_criterion(crit, cfg.get("required_elements")) or missing
            digest.append({
                "section": str(crit.get("section", "")),
                "name": crit["name"],
                "max": mx,
                "awarded": [a["awarded"], b["awarded"]],
                "points_at_stake": (mx - a["awarded"]) + (mx - b["awarded"]),
                "gaps": (a.get("gaps") or []) + (b.get("gaps") or []),
                "required": required,
                "missing": missing,
            })
    return digest


SYNTH_SYSTEM = (
    "You are {role}. You are given the gaps two independent reviewers found in a grant "
    "proposal, each with the points at stake. Produce the TOP {n} most important "
    "recommendations to improve the proposal. MERGE related or duplicate gaps from the two "
    "reviewers into a single recommendation. Rank by importance: points at stake is the "
    "primary signal, but {required_clause}a hard compliance miss may outrank a higher-point "
    "soft gap. Ground every recommendation in the reviewer gaps supplied below; do NOT "
    "introduce requirements that do not appear in them. "
    "Return FEWER than {n} items if there are fewer real gaps. "
    'Respond ONLY with a JSON array; each item: {{"title": <str>, "points_at_stake": <int>, '
    '"criteria": [<str>], "do": <str>, "why": <str>}}'
)


def synthesis_system_prompt(role, n, required_elements):
    """Build the synthesis system prompt. The REQUIRED-element clause only appears when
    the config declares such elements — naming examples unconditionally invites the model
    to recommend them whether or not the solicitation asks for them."""
    elements = [e for e in (required_elements or []) if e]
    clause = f"a missing REQUIRED element ({', '.join(elements)}) or " if elements else ""
    return SYNTH_SYSTEM.format(role=role, n=n, required_clause=clause)


def _ollama_chat(model, system_prompt, user_prompt, timeout=900, num_ctx=DEFAULT_NUM_CTX):
    """Single local Ollama chat call; returns the message content string.
    Retained as the default chat_fn for callers that don't resolve a provider."""
    return providers.make_chat_fn({}, "ollama")(
        model, system_prompt, user_prompt, timeout=timeout, num_ctx=num_ctx)


class JSONRetryError(Exception):
    """Raised when a model never returns parseable, valid JSON within the attempt budget.
    `.raw` holds the last unparseable content for the caller's fallback message."""
    def __init__(self, raw):
        self.raw = raw
        super().__init__("model did not return valid JSON")


MAX_JSON_ATTEMPTS = 3
_JSON_NUDGE = ("\n\nIMPORTANT: your previous reply was not valid JSON. "
               "Respond with ONLY the JSON, no prose and no code fences.")


def _chat_json(model, system, user, validate, attempts=MAX_JSON_ATTEMPTS, num_ctx=None,
               chat_fn=_ollama_chat):
    """Call the model and parse its reply as JSON, retrying with a corrective nudge on
    parse/validate failure. `validate(parsed)` returns the cleaned result or raises to
    trigger a retry. Network errors from chat_fn propagate (NOT retried here).
    Raises JSONRetryError(last_raw) if no attempt yields valid JSON.

    num_ctx is forwarded only when set, so a chat_fn with the older three-argument
    signature keeps working."""
    extra = {} if num_ctx is None else {"num_ctx": num_ctx}
    raw = ""
    for i in range(attempts):
        prompt = user if i == 0 else user + _JSON_NUDGE
        raw = chat_fn(model, system, prompt, **extra)
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        try:
            return validate(json.loads(cleaned))
        except Exception:
            print(f"  {model}: unparseable JSON (attempt {i + 1}/{attempts})", file=sys.stderr)
            continue
    raise JSONRetryError(raw)


def _digest_to_prompt(digest):
    lines = []
    for d in digest:
        flag = "  [REQUIRED/MISSING]" if d["required"] else ""
        lines.append(f"- {d['name']} (§{d['section']}) — {d['points_at_stake']} pts at stake{flag}")
        for g in d["gaps"]:
            lines.append(f"    • {g}")
    return "\n".join(lines)


def deterministic_recommendations(digest, n):
    """Fallback ranking when the synthesis model is unavailable or returns garbage.
    Sort by (required, points_at_stake) descending."""
    ordered = sorted(digest, key=lambda d: (d["required"], d["points_at_stake"]), reverse=True)
    recs = []
    for d in ordered[:n]:
        why = ("Section is missing or placeholder; both reviewers scored it near zero."
               if d["missing"] else "Both reviewers docked points here.")
        recs.append({
            "title": d["name"],
            "points_at_stake": d["points_at_stake"],
            "criteria": [f"§{d['section']}"] if d["section"] else [],
            "do": d["gaps"][0] if d["gaps"] else "Address the reviewers' gaps for this criterion.",
            "why": why,
        })
    return recs


def synthesize_recommendations(cfg, models, all_results, chat_fn=_ollama_chat,
                               synth_model=None, num_ctx=None):
    """Ranked, deduplicated recommendations merging both models' gaps. Falls back to a
    deterministic ranking if the synthesis call fails or returns unusable output."""
    digest = build_recommendation_digest(cfg, models, all_results)
    if not digest:
        return []
    n = int(cfg.get("top_recommendations", 7))
    model = synth_model or cfg.get("synthesis_model") or models[0]
    role = cfg.get("reviewer_role", DEFAULT_ROLE)
    system = synthesis_system_prompt(role, n, cfg.get("required_elements"))
    user = ("CRITERIA WITH GAPS (ranked candidates):\n" + _digest_to_prompt(digest)
            + f"\n\nReturn the top {n} merged recommendations as a JSON array.")
    def _valid_recs(parsed):
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("empty or non-list synthesis output")
        return parsed

    try:
        recs = _chat_json(model, system, user, _valid_recs, num_ctx=num_ctx, chat_fn=chat_fn)
        return recs[:n]
    except Exception as e:
        print(f"  synthesis failed ({e}); using deterministic ranking", file=sys.stderr)
        return deterministic_recommendations(digest, n)


def render_recommendations(recs):
    """Render the synthesized recommendations as the combined-report §2 body."""
    out = ["\n## 2. Top recommendations (synthesized, ranked by importance)\n"]
    if not recs:
        out.append("No material gaps — both reviewers scored every criterion at or near max.")
        return "\n".join(out)
    for i, r in enumerate(recs, 1):
        crits = ", ".join(r.get("criteria") or [])
        suffix = f" ({crits})" if crits else ""
        pts = r.get("points_at_stake", "")
        pts_str = f" — {pts} pts at stake" if pts not in ("", None) else ""
        out.append(f"\n### {i}. {r.get('title', '(untitled)')}{suffix}{pts_str}")
        out.append(f"**Do:** {str(r.get('do', '')).strip()}")
        out.append(f"**Why:** {str(r.get('why', '')).strip()}")
    return "\n".join(out)


# ── combine ─────────────────────────────────────────────────────────
def section_subtotals(cfg, results):
    groups = {}
    for crit, res in zip(cfg["criteria"], results):
        g = resolve_group(crit, cfg.get("section_groups", {}))
        groups.setdefault(g, [0, 0])
        groups[g][0] += res["awarded"]
        groups[g][1] += res["max"]
    return groups


def combined_provenance(cfg, single_family):
    """How the combined report describes where its scores came from.

    Never claim a local run for a provider-hosted one: this line goes into a
    document a PI may forward, and it is the report's only statement about where
    the draft was sent."""
    if single_family:
        return "single model, two samples"
    refs = cfg.get("models") or []
    if not refs:
        return "two-model ensemble"
    known = providers.known_provider_names(cfg)
    kinds = {providers.split_model_ref(r, known)[0] for r in refs}
    if kinds == {"ollama"}:
        return "offline (local models)"
    if "ollama" in kinds:
        return "mixed local and provider-hosted models"
    return "provider-hosted models"


def build_combined(cfg, models, all_results, kw_pct, kw_missing, recs,
                   single_family=False):
    A, B = models
    ra, rb = all_results[A], all_results[B]
    ga, gb = section_subtotals(cfg, ra), section_subtotals(cfg, rb)
    ta, tb = sum(r["awarded"] for r in ra), sum(r["awarded"] for r in rb)
    total = cfg.get("total_points", sum(c["max"] for c in cfg["criteria"]))
    section_totals = cfg.get("section_totals") or {g: ga[g][1] for g in ga}

    # Two samples of one model agree for reasons that have nothing to do with the
    # proposal being sound, so every heading that implies corroboration is reworded.
    provenance = combined_provenance(cfg, single_family)
    if single_family:
        per_crit_hd = "Per-criterion scores (★ = unstable score, model is guessing)"
        gaps_hd = "Priority gaps — consistent across both samples (ranked by points at stake)"
        strengths_hd = "Strengths — consistently scored ≥80% of max"
        banner = ("\n> **No independent second opinion.** Both columns are the same "
                  "model sampled twice, so agreement between them reflects "
                  "consistency, not corroboration. Shared blind spots stay "
                  "invisible here.\n")
    else:
        per_crit_hd = "Per-criterion scores (★ = models diverge ≥2 pts)"
        gaps_hd = "Priority gaps — both models dock points (ranked by points at stake)"
        strengths_hd = "Strengths — both models score ≥80% of max"
        banner = ""

    L = [f"# Combined Compliance Assessment: {A} vs {B}",
         f"\n_{cfg.get('name','')} — generated {datetime.date.today()}, {provenance}._\n",
         banner,
         "---\n## 1. Overall scores\n", f"| Section | {A} | {B} |", "|---|---|---|"]
    for sec, mx in section_totals.items():
        L.append(f"| {sec} | **{ga.get(sec,[0,0])[0]}/{mx}** | **{gb.get(sec,[0,0])[0]}/{mx}** |")
    L.append(f"| RFA Keyword Coverage | {kw_pct:.0f}% | {kw_pct:.0f}% |")
    L.append(f"| **TOTAL** | **{ta}/{total}** | **{tb}/{total}** |\n")

    L.append(render_recommendations(recs))

    L.append(f"\n## 3. {per_crit_hd}\n")
    L += ["| Criterion | Max | " + A + " | " + B + " |", "|---|---|---|---|"]
    for crit, x, y in zip(cfg["criteria"], ra, rb):
        flag = " ★" if abs(x["awarded"] - y["awarded"]) >= 2 else ""
        L.append(f"| {crit['name']}{flag} | {crit['max']} | {x['awarded']} | {y['awarded']} |")

    gaps = []
    for crit, x, y in zip(cfg["criteria"], ra, rb):
        if x["awarded"] < crit["max"] and y["awarded"] < crit["max"]:
            lost = (crit["max"] - x["awarded"]) + (crit["max"] - y["awarded"])
            gaps.append((lost, crit, x, y))
    gaps.sort(key=lambda t: t[0], reverse=True)
    L.append(f"\n## 4. {gaps_hd}\n")
    for lost, crit, x, y in gaps:
        L.append(f"\n### {crit['name']} — {A} {x['awarded']}/{crit['max']}, {B} {y['awarded']}/{crit['max']}  (−{lost})")
        for who, r in ((A, x), (B, y)):
            for g in r.get("gaps", []):
                L.append(f"- _{who}:_ {g}")

    L.append(f"\n## 5. {strengths_hd}\n")
    for crit, x, y in zip(cfg["criteria"], ra, rb):
        if x["awarded"] >= 0.8 * crit["max"] and y["awarded"] >= 0.8 * crit["max"]:
            strs = (x.get("strengths") or []) + (y.get("strengths") or [])
            L.append(f"- **{crit['name']}** ({x['awarded']}/{crit['max']}, {y['awarded']}/{crit['max']})"
                     + (f" — {strs[0]}" if strs else ""))

    L.append("\n## 6. Missing RFA phrases (deterministic scan)\n")
    if kw_missing:
        L += ["| Phrase | Category |", "|---|---|"] + [f"| {l} | {c} |" for l, c in kw_missing]
    else:
        L.append("None — all tracked phrases present.")

    L.append(f"\n## 7. Final\n**{A}: {ta}/{total} · {B}: {tb}/{total} · Keywords: {kw_pct:.0f}%**\n")
    if gaps:
        L.append("Remaining work, by points at stake: "
                 + ", ".join(f"{c['name']} (−{lost})" for lost, c, *_ in gaps[:5]) + ".")
    return "\n".join(L)


UNICODE_FONTS = ("Arial Unicode MS",   # macOS
                 "DejaVu Sans",        # Linux, and shipped with TeX Live
                 "Noto Sans",
                 "Liberation Sans",
                 "FreeSans")


def render(md_path: Path):
    html = md_path.with_suffix(".html")
    subprocess.run(["pandoc", str(md_path), "-s", "-o", str(html)], check=False)
    pdf = md_path.with_suffix(".pdf")
    # The report uses Unicode (★ ≥ ✓ ✗ • —), so pdflatex (Latin Modern) mangles it.
    # wkhtmltopdf/weasyprint render from HTML/CSS and handle Unicode natively; the
    # fontspec TeX engines (xelatex/lualatex) need a Unicode main font. Each attempt
    # that errors (engine absent, or font absent) returns non-zero and we fall through.
    # One font name is not portable: "Arial Unicode MS" exists on macOS and nowhere
    # else, so a Linux box with a complete TeX Live still produced no PDF at all.
    # Each name that is absent fails fast, so trying several costs little.
    attempts = [("wkhtmltopdf", []), ("weasyprint", [])]
    attempts += [(eng, ["-V", f"mainfont={font}"])
                 for eng in ("xelatex", "lualatex") for font in UNICODE_FONTS]
    attempts += [("tectonic", []), ("pdflatex", [])]
    for eng, extra in attempts:
        try:
            r = subprocess.run(["pandoc", str(md_path), f"--pdf-engine={eng}", *extra, "-o", str(pdf)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return f"{html.name}, {pdf.name}"
        except FileNotFoundError:
            continue
    return f"{html.name} (no PDF engine found — install weasyprint or a Unicode TeX (xelatex))"


def dry_run(cfg, sections, full_text):
    print(f"Config: {cfg.get('name','(unnamed)')}")
    print(f"Models: {cfg.get('models')}")
    print(f"Criteria: {len(cfg['criteria'])}  |  Total points: "
          f"{cfg.get('total_points', sum(c['max'] for c in cfg['criteria']))}")
    print(f"Parsed numbered sections: {sorted(sections)}")
    max_chars = int(cfg.get("max_section_chars", DEFAULT_MAX_SECTION_CHARS))
    num_ctx = int(cfg.get("num_ctx", DEFAULT_NUM_CTX))
    # ~4 chars/token, less a reserve for the system prompt, criterion and reply.
    ctx_chars = max(0, (num_ctx - 2500)) * 4
    note = "" if max_chars <= ctx_chars else f"  ⚠ EXCEEDS num_ctx (fits ~{ctx_chars} chars)"
    print(f"Section text cap: {max_chars} chars  |  num_ctx: {num_ctx}{note}\n")
    print(f"{'section':>8}  {'found':>5}  {'words':>6}  {'chars':>7}  criterion")
    for crit in cfg["criteria"]:
        body, title = get_body(sections, full_text, crit.get("section"))
        flag = "  ⚠ TRUNCATED" if len(body) > max_chars else ""
        print(f"{str(crit.get('section')):>8}  {'yes' if body else 'NO':>5}  "
              f"{len(body.split()):>6}  {len(body):>7}  {crit['name']} (/{crit['max']}){flag}")
    _, pct, missing = keyword_scan(full_text, cfg.get("keywords", []))
    print(f"\nKeyword coverage: {pct:.0f}%  |  missing: {[m[0] for m in missing]}")


def resolve_outdir(proposal: Path, outdir_arg):
    """Directory the run's reports go in. Defaults beside the proposal.

    Resolved before any model call on purpose: with --outdir it is easy to name a
    path that does not exist or cannot be written, and discovering that after 30-odd
    model calls wastes the whole run."""
    if not outdir_arg:
        return proposal.parent
    out = Path(outdir_arg).expanduser()
    if out.exists() and not out.is_dir():
        sys.exit(f"--outdir {out}: exists but is not a directory.")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        sys.exit(f"--outdir {out}: cannot use this directory ({e.strerror}).")
    if not os.access(out, os.W_OK):
        sys.exit(f"--outdir {out}: directory is not writable.")
    return out


def stamped(base, ts, ext):
    """Compose a timestamped output filename, e.g.
    stamped('compliance_report_combined', '20260622-211500', 'md')
    -> 'compliance_report_combined_20260622-211500.md'."""
    return f"{base}_{ts}.{ext}"


def slot_labels(models):
    """Distinct display keys for the two ensemble slots.

    Results are keyed by label, so two samples of the same model must not
    collide — otherwise the second overwrites the first and the combined
    report compares a result against itself."""
    a, b = models
    if a == b:
        return [f"{a} (sample 1)", f"{b} (sample 2)"]
    return [a, b]


def check_providers(cfg, models, environ=None, transport=None):
    """One cheap call per slot. Returns (label, ok, detail) rows; never raises.

    This is how you confirm a machine can reach its backends — in particular
    whether it is inside the HPC security zone the UVA API requires. Each slot is
    resolved separately so one broken backend still reports on the other."""
    rows = []
    known = providers.known_provider_names(cfg)
    labels = slot_labels(models)
    num_ctx = int(cfg.get("num_ctx", DEFAULT_NUM_CTX))
    for label, ref in zip(labels, models):
        pname, bare = providers.split_model_ref(ref, known)
        try:
            fn = providers.make_chat_fn(cfg, pname, environ=environ,
                                        transport=transport)
            # A local ping is the call that loads the model, which off GPFS can
            # outlast a minute. Loading it at the run's num_ctx also spares the
            # first real call a reload.
            local = providers.provider_spec(cfg, pname).get("kind", "ollama") == "ollama"
            reply = fn(bare, "Reply with the single word: ok.", "ping",
                       timeout=600 if local else 60,
                       num_ctx=num_ctx if local else None)
            rows.append((label, True, f"{pname} -> {bare}: "
                                      f"{reply.strip()[:40] or '(empty reply)'}"))
        except ProviderError as e:
            rows.append((label, False, str(e)))
        except Exception as e:                # report, don't crash: this is a diagnostic
            rows.append((label, False, f"{type(e).__name__}: {e}"))
    return rows


def format_check_rows(rows):
    return "\n".join(f"{'OK  ' if ok else 'FAIL'}  {label}: {detail}"
                     for label, ok, detail in rows)


def preflight(cfg, models, environ=None, transport=None):
    """Ping both slots before a full run; exit if either fails.

    Slot A reviews every criterion before slot B makes its first call, so an
    unreachable B would otherwise surface only after A's entire pass."""
    print("Checking model backends ...", file=sys.stderr)
    rows = check_providers(cfg, models, environ=environ, transport=transport)
    print(format_check_rows(rows), file=sys.stderr)
    if not all(ok for _, ok, _ in rows):
        sys.exit("A model backend failed its check, so nothing was reviewed. "
                 "Fix it, or pass --no-check to skip this check.")
    return rows


def is_single_family(cfg, models):
    """True when both ensemble slots resolve to the same provider AND model.

    Two samples of one model are a reliability check, not two independent
    opinions — the combined report must not claim corroboration."""
    known = providers.known_provider_names(cfg)
    a = providers.split_model_ref(models[0], known)
    b = providers.split_model_ref(models[1], known)
    return a == b


def resolve_chat_fns(cfg, models, environ=None, temperatures=(0.2, 0.2),
                     transport=None):
    """One (label, bare_model, chat_fn) per ensemble slot, in order.

    The label is the display/results key; the bare model is what gets sent to
    the backend with any provider prefix stripped."""
    known = providers.known_provider_names(cfg)
    labels = slot_labels(models)
    slots = []
    for i, ref in enumerate(models):
        pname, bare = providers.split_model_ref(ref, known)
        fn = providers.make_chat_fn(cfg, pname, temperature=temperatures[i],
                                    environ=environ, transport=transport)
        slots.append((labels[i], bare, fn))
    return slots


def resolve_synthesis_slot(cfg, slots, environ=None, transport=None):
    """(bare_model, chat_fn) for the synthesis call.

    `synthesis_model` may name any provider, so it gets its own chat_fn rather
    than borrowing slot 0's — otherwise a local model named there would be
    posted to slot 0's remote endpoint. Unset, slot 0 writes the synthesis."""
    ref = cfg.get("synthesis_model")
    if not ref:
        return slots[0][1], slots[0][2]
    pname, bare = providers.split_model_ref(ref, providers.known_provider_names(cfg))
    return bare, providers.make_chat_fn(cfg, pname, environ=environ,
                                        transport=transport)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proposal", nargs="?")
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs=2, default=None, metavar=("MODEL_A", "MODEL_B"))
    ap.add_argument("--dry-run", action="store_true", help="Validate config/section mapping without calling models")
    ap.add_argument("--check-provider", action="store_true",
                    help="Test connectivity and auth for each configured model, then exit")
    ap.add_argument("--outdir", default=None, metavar="DIR",
                    help="Write reports here instead of beside the proposal "
                         "(created if missing)")
    ap.add_argument("--no-check", action="store_true",
                    help="Skip pinging both backends before a full run")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    if args.models:
        cfg["models"] = args.models
    models = cfg.get("models")
    if not models or len(models) != 2:
        sys.exit("Config must specify exactly two models (or pass --models A B).")

    if args.check_provider:
        providers.load_dotenv()
        rows = check_providers(cfg, models)
        print(format_check_rows(rows))
        sys.exit(0 if all(ok for _, ok, _ in rows) else 1)

    if not args.proposal:
        sys.exit("PROPOSAL is required (omit it only with --check-provider).")

    path = Path(args.proposal).expanduser()
    full_text = extract_text(path, cfg.get("stop_marker"))
    sections = parse_sections(full_text, cfg.get("heading_patterns"))

    if args.dry_run:
        dry_run(cfg, sections, full_text)
        return

    outdir = resolve_outdir(path, args.outdir)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    num_ctx = int(cfg.get("num_ctx", DEFAULT_NUM_CTX))

    providers.load_dotenv()
    if not args.no_check:
        preflight(cfg, models)
    # At 0.2 two samples of one model risk being near-identical, which makes the
    # second call pure cost. Sampling wider is what makes "is this score stable?"
    # a real question.
    single = is_single_family(cfg, models)
    temps = (0.2, 0.6) if single else (0.2, 0.2)
    try:
        slots = resolve_chat_fns(cfg, models, temperatures=temps)
        synth_model, synth_fn = resolve_synthesis_slot(cfg, slots)
    except ProviderError as e:
        sys.exit(str(e))

    all_results = {}
    for label, bare, chat_fn in slots:
        print(f"Reviewing with {label} ...", file=sys.stderr)
        try:
            res, report = review_with_model(bare, cfg, sections, full_text,
                                            chat_fn=chat_fn, label=label)
        except ProviderError as e:
            sys.exit(str(e))
        all_results[label] = res
        safe = label.replace(":", "_").replace("/", "_").replace(" ", "_")
        (outdir / stamped(f"compliance_report_{safe}", ts, "md")).write_text(report)

    labels = [s[0] for s in slots]

    kw_text, kw_pct, kw_missing = keyword_scan(full_text, cfg.get("keywords", []))
    (outdir / stamped("compliance_report_keywords", ts, "md")).write_text(kw_text)

    print("Synthesizing recommendations ...", file=sys.stderr)
    recs = synthesize_recommendations(cfg, labels, all_results,
                                      chat_fn=synth_fn, synth_model=synth_model,
                                      num_ctx=num_ctx)
    combined = build_combined(cfg, labels, all_results, kw_pct, kw_missing, recs,
                              single_family=single)
    cpath = outdir / stamped("compliance_report_combined", ts, "md")
    cpath.write_text(combined)
    rendered = render(cpath)
    print(f"Wrote per-model reports, keywords, and {cpath.name} ({rendered})", file=sys.stderr)


if __name__ == "__main__":
    main()
