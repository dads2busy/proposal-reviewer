#!/usr/bin/env python3
"""
proposal_reviewer.py — proposal-agnostic offline compliance reviewer.

Runs a two-model LOCAL ensemble (via Ollama) plus a deterministic keyword pass
against a grant proposal, scores it against a rubric defined entirely in a YAML
config, and merges the results into a combined report (.md / .html / .pdf).

The engine is generic. Everything proposal-specific — the criteria, point caps,
section map, keywords, models — lives in the config file, so one engine serves
every proposal you review.

Usage:
    python proposal_reviewer.py PROPOSAL --config configs/cdc-ip-26-0102.yaml
    python proposal_reviewer.py PROPOSAL --config CONFIG --dry-run   # validate mapping, no model calls
    python proposal_reviewer.py PROPOSAL --config CONFIG --models llama3.3:70b qwen3.6:27b

Requires: pyyaml (pip install pyyaml); pandoc + pdftotext for extraction/rendering;
a local Ollama (>=0.19) with the configured models pulled.
"""

import argparse, datetime, json, re, subprocess, sys, urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("Missing dependency: pyyaml.  Install with:  pip install pyyaml")

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

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
def extract_text(path: Path, stop_marker=None) -> str:
    s = path.suffix.lower()
    if s in {".md", ".txt", ".tex"}:
        text = path.read_text(errors="ignore")
    elif s == ".docx":
        text = subprocess.run(["pandoc", str(path), "-t", "markdown"], capture_output=True, text=True).stdout
    elif s == ".pdf":
        text = subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True).stdout
    else:
        sys.exit(f"Unsupported file type: {s}")
    if stop_marker and stop_marker in text:
        text = text.split(stop_marker)[0]
    return text


def clean_heading(raw: str) -> str:
    raw = re.sub(r"\[?\\?\[[^\]]*\\?\]\]?\{\.mark\}", "", raw)
    raw = re.sub(r"\\?\[[^\]]*\\?\]", "", raw)
    return raw.replace("\\", "").strip()


def parse_sections(md: str) -> dict:
    """Map numbered heading IDs (1.1, 2.3, ...) to body text. Unnumbered
    subheadings fold into the preceding numbered section."""
    lines = md.splitlines()
    numbered = []
    for i, ln in enumerate(lines):
        m = re.match(r"^#+\s+(.*)$", ln)
        if not m:
            continue
        title = clean_heading(m.group(1))
        num = re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+(.*)$", title)
        if num:
            numbered.append((i, num.group(1), title))
    sections = {}
    for idx, (li, sid, title) in enumerate(numbered):
        end = numbered[idx + 1][0] if idx + 1 < len(numbered) else len(lines)
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
def ask_model(model, system_prompt, crit, sect_title, body, rfa_context=""):
    user = (f"CRITERION: {crit['name']}  (max {crit['max']} points)\n"
            f"WHAT REVIEWERS LOOK FOR: {crit['look_for']}\n")
    if rfa_context:
        user += f"\nRELEVANT RFA CONTEXT:\n{rfa_context}\n"
    user += f"\nPROPOSAL SECTION — {sect_title}:\n{body[:14000]}"
    payload = {"model": model, "temperature": 0.2,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user}],
               "options": {"num_ctx": 16384}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        content = json.loads(r.read())["choices"][0]["message"]["content"]
    cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        d = json.loads(cleaned)
        d["awarded"] = max(0, min(int(d.get("awarded", 0) or 0), crit["max"]))
        d["max"] = crit["max"]
        return d
    except Exception:
        return {"awarded": 0, "max": crit["max"], "strengths": [], "gaps": [],
                "justification": "UNPARSED MODEL OUTPUT:\n" + content}


def review_with_model(model, cfg, sections, full_text):
    role = cfg.get("reviewer_role", DEFAULT_ROLE)
    system_prompt = cfg.get("system_prompt") or SYSTEM_TEMPLATE.format(role=role)
    rfa_context = cfg.get("rfa_context", "")
    results = []
    lines = [f"# Compliance Assessment — {model} (local)\n", f"_Generated {datetime.date.today()}_\n"]
    for crit in cfg["criteria"]:
        body, title = get_body(sections, full_text, crit.get("section"))
        if not body:
            res = {"awarded": 0, "max": crit["max"], "strengths": [],
                   "gaps": [f"Section {crit.get('section')} not found in proposal."],
                   "justification": "Required section missing."}
            lines.append(f"\n## {crit['name']} — 0/{crit['max']}  ⚠ SECTION MISSING\n")
        else:
            res = ask_model(model, system_prompt, crit, title, body, rfa_context)
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
REQUIRED_PATTERNS = (
    "data management plan",
    "letter of support", "letters of support",
    "organizational chart", "org chart",
)
MISSING_MARKERS = ("not found", "section missing", "missing.", "placeholder", "still need")


def is_required_criterion(crit):
    """True if the criterion hinges on a required attachment the narrative can only reference."""
    text = (crit.get("name", "") + " " + crit.get("look_for", "")).lower()
    return any(p in text for p in REQUIRED_PATTERNS)


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
            digest.append({
                "section": str(crit.get("section", "")),
                "name": crit["name"],
                "max": mx,
                "awarded": [a["awarded"], b["awarded"]],
                "points_at_stake": (mx - a["awarded"]) + (mx - b["awarded"]),
                "gaps": (a.get("gaps") or []) + (b.get("gaps") or []),
                "required": is_required_criterion(crit) or missing,
                "missing": missing,
            })
    return digest


SYNTH_SYSTEM = (
    "You are {role}. You are given the gaps two independent reviewers found in a grant "
    "proposal, each with the points at stake. Produce the TOP {n} most important "
    "recommendations to improve the proposal. MERGE related or duplicate gaps from the two "
    "reviewers into a single recommendation. Rank by importance: points at stake is the "
    "primary signal, but a missing REQUIRED element (data management plan, letters of "
    "support, organizational chart) or a hard compliance miss may outrank a higher-point "
    "soft gap. Return FEWER than {n} items if there are fewer real gaps. "
    'Respond ONLY with a JSON array; each item: {{"title": <str>, "points_at_stake": <int>, '
    '"criteria": [<str>], "do": <str>, "why": <str>}}'
)


def _ollama_chat(model, system_prompt, user_prompt, timeout=900):
    """Single local Ollama chat call; returns the message content string."""
    payload = {"model": model, "temperature": 0.2,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}],
               "options": {"num_ctx": 16384}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


class JSONRetryError(Exception):
    """Raised when a model never returns parseable, valid JSON within the attempt budget.
    `.raw` holds the last unparseable content for the caller's fallback message."""
    def __init__(self, raw):
        self.raw = raw
        super().__init__("model did not return valid JSON")


MAX_JSON_ATTEMPTS = 3
_JSON_NUDGE = ("\n\nIMPORTANT: your previous reply was not valid JSON. "
               "Respond with ONLY the JSON, no prose and no code fences.")


def _chat_json(model, system, user, validate, attempts=MAX_JSON_ATTEMPTS, chat_fn=_ollama_chat):
    """Call the model and parse its reply as JSON, retrying with a corrective nudge on
    parse/validate failure. `validate(parsed)` returns the cleaned result or raises to
    trigger a retry. Network errors from chat_fn propagate (NOT retried here).
    Raises JSONRetryError(last_raw) if no attempt yields valid JSON."""
    raw = ""
    for i in range(attempts):
        prompt = user if i == 0 else user + _JSON_NUDGE
        raw = chat_fn(model, system, prompt)
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


def synthesize_recommendations(cfg, models, all_results, chat_fn=_ollama_chat):
    """Ranked, deduplicated recommendations merging both models' gaps. Falls back to a
    deterministic ranking if the synthesis call fails or returns unusable output."""
    digest = build_recommendation_digest(cfg, models, all_results)
    if not digest:
        return []
    n = int(cfg.get("top_recommendations", 7))
    model = cfg.get("synthesis_model") or models[0]
    role = cfg.get("reviewer_role", DEFAULT_ROLE)
    system = SYNTH_SYSTEM.format(role=role, n=n)
    user = ("CRITERIA WITH GAPS (ranked candidates):\n" + _digest_to_prompt(digest)
            + f"\n\nReturn the top {n} merged recommendations as a JSON array.")
    try:
        content = chat_fn(model, system, user)
        cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
        recs = json.loads(cleaned)
        if not isinstance(recs, list) or not recs:
            raise ValueError("empty or non-list synthesis output")
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


def build_combined(cfg, models, all_results, kw_pct, kw_missing, recs):
    A, B = models
    ra, rb = all_results[A], all_results[B]
    ga, gb = section_subtotals(cfg, ra), section_subtotals(cfg, rb)
    ta, tb = sum(r["awarded"] for r in ra), sum(r["awarded"] for r in rb)
    total = cfg.get("total_points", sum(c["max"] for c in cfg["criteria"]))
    section_totals = cfg.get("section_totals") or {g: ga[g][1] for g in ga}

    L = [f"# Combined Compliance Assessment: {A} vs {B}",
         f"\n_{cfg.get('name','')} — generated {datetime.date.today()}, offline (local models)._\n",
         "---\n## 1. Overall scores\n", f"| Section | {A} | {B} |", "|---|---|---|"]
    for sec, mx in section_totals.items():
        L.append(f"| {sec} | **{ga.get(sec,[0,0])[0]}/{mx}** | **{gb.get(sec,[0,0])[0]}/{mx}** |")
    L.append(f"| RFA Keyword Coverage | {kw_pct:.0f}% | {kw_pct:.0f}% |")
    L.append(f"| **TOTAL** | **{ta}/{total}** | **{tb}/{total}** |\n")

    L.append(render_recommendations(recs))

    L.append("\n## 3. Per-criterion scores (★ = models diverge ≥2 pts)\n")
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
    L.append("\n## 4. Priority gaps — both models dock points (ranked by points at stake)\n")
    for lost, crit, x, y in gaps:
        L.append(f"\n### {crit['name']} — {A} {x['awarded']}/{crit['max']}, {B} {y['awarded']}/{crit['max']}  (−{lost})")
        for who, r in ((A, x), (B, y)):
            for g in r.get("gaps", []):
                L.append(f"- _{who}:_ {g}")

    L.append("\n## 5. Strengths — both models score ≥80% of max\n")
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


def render(md_path: Path):
    html = md_path.with_suffix(".html")
    subprocess.run(["pandoc", str(md_path), "-s", "-o", str(html)], check=False)
    pdf = md_path.with_suffix(".pdf")
    # The report uses Unicode (★ ≥ ✓ ✗ • —), so pdflatex (Latin Modern) mangles it.
    # wkhtmltopdf/weasyprint render from HTML/CSS and handle Unicode natively; the
    # fontspec TeX engines (xelatex/lualatex) need a Unicode main font. Each attempt
    # that errors (engine absent, or font absent) returns non-zero and we fall through.
    unicode_font = "Arial Unicode MS"  # broad coverage on macOS
    attempts = [("wkhtmltopdf", []), ("weasyprint", []),
                ("xelatex", ["-V", f"mainfont={unicode_font}"]),
                ("lualatex", ["-V", f"mainfont={unicode_font}"]),
                ("tectonic", []), ("pdflatex", [])]
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
    print(f"Parsed numbered sections: {sorted(sections)}\n")
    print(f"{'section':>8}  {'found':>5}  {'words':>6}  criterion")
    for crit in cfg["criteria"]:
        body, title = get_body(sections, full_text, crit.get("section"))
        print(f"{str(crit.get('section')):>8}  {'yes' if body else 'NO':>5}  "
              f"{len(body.split()):>6}  {crit['name']} (/{crit['max']})")
    _, pct, missing = keyword_scan(full_text, cfg.get("keywords", []))
    print(f"\nKeyword coverage: {pct:.0f}%  |  missing: {[m[0] for m in missing]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proposal")
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs=2, default=None, metavar=("MODEL_A", "MODEL_B"))
    ap.add_argument("--dry-run", action="store_true", help="Validate config/section mapping without calling models")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).expanduser().read_text())
    if args.models:
        cfg["models"] = args.models
    models = cfg.get("models")
    if not models or len(models) != 2:
        sys.exit("Config must specify exactly two models (or pass --models A B).")

    path = Path(args.proposal).expanduser()
    full_text = extract_text(path, cfg.get("stop_marker"))
    sections = parse_sections(full_text)

    if args.dry_run:
        dry_run(cfg, sections, full_text)
        return

    outdir = path.parent
    all_results = {}
    for model in models:
        print(f"Reviewing with {model} ...", file=sys.stderr)
        res, report = review_with_model(model, cfg, sections, full_text)
        all_results[model] = res
        safe = model.replace(":", "_").replace("/", "_")
        (outdir / f"compliance_report_{safe}.md").write_text(report)

    kw_text, kw_pct, kw_missing = keyword_scan(full_text, cfg.get("keywords", []))
    (outdir / "compliance_report_keywords.md").write_text(kw_text)

    print("Synthesizing recommendations ...", file=sys.stderr)
    recs = synthesize_recommendations(cfg, models, all_results)
    combined = build_combined(cfg, models, all_results, kw_pct, kw_missing, recs)
    cpath = outdir / "compliance_report_combined.md"
    cpath.write_text(combined)
    rendered = render(cpath)
    print(f"Wrote per-model reports, keywords, and {cpath.name} ({rendered})", file=sys.stderr)


if __name__ == "__main__":
    main()
