#!/usr/bin/env python3
"""
page_reducer.py — suggest page-count reductions for a grant proposal.

Copies the original .docx, sends all body paragraphs to a local Ollama model
in one call (large context window), and marks suggested cuts (~~strikethrough~~)
and rewrites ([RED]red bold text[/RED]) inline.  Heading paragraphs are left
untouched.

Output: a copy of the proposal with strikethrough + red markup applied to runs,
        suitable for opening in Word and accepting/rejecting changes manually.

Usage:
    python page_reducer.py PROPOSAL.docx
    python page_reducer.py PROPOSAL.docx --model qwen3.6:27b --target-reduction 0.20
"""

import argparse, datetime, difflib, json, re, shutil, sys, urllib.request
from pathlib import Path

try:
    from docx import Document
    from docx.shared import RGBColor
    from docx.oxml.ns import qn
except ImportError:
    sys.exit("Missing dependency: python-docx.  Install with:  pip install python-docx")

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.6:27b"

SYSTEM_PROMPT = (
    "You are an expert scientific grant writer helping to reduce the length of a grant "
    "proposal — without changing the scientific content, without changing margins or fonts.\n\n"
    "You will receive ALL body paragraphs of the proposal, separated by '|||PARA|||'. "
    "You will also be given a current word count and a target word count. Rewrite each "
    "paragraph to be more concise, cutting enough text across the document to reach the target.\n\n"
    "Hard rules:\n"
    "  - NEVER touch references of the form [Author+Year] or [XX+##] or any [bracket] citation\n"
    "  - Focus on: redundant phrases, wordy qualifiers, repeated ideas across sections, "
    "verbose constructions, unnecessary throat-clearing\n"
    "  - Do NOT change section headings or activity labels (e.g. 'S1.RA1:')\n"
    "  - Return EXACTLY the same number of paragraphs in the same order\n"
    "  - Separate them with the EXACT string '|||PARA|||' and nothing else between them\n"
    "  - If a paragraph needs no changes, return it verbatim\n"
    "  - Do NOT add commentary, preamble, or explanation — only the rewritten paragraphs"
)


# ── Ollama call ─────────────────────────────────────────────────────────────

def ollama_chat(model: str, system: str, user: str, num_ctx: int, think: bool = False, timeout: int = 1800) -> str:
    # Native /api/chat with streaming so we can show live token progress.
    # think=True enables extended reasoning on qwen3 models (slower, better rewrites).
    payload = {
        "model": model,
        "think": think,
        "stream": True,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {"num_ctx": num_ctx, "temperature": 0.3},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.strip():
                continue
            obj = json.loads(line)
            piece = obj.get("message", {}).get("content", "")
            if piece:
                chunks.append(piece)
                tokens += 1
                if tokens % 200 == 0:
                    print(f"    {tokens} tokens ...", file=sys.stderr, flush=True)
            if obj.get("done"):
                eval_count = obj.get("eval_count", tokens)
                print(f"    done ({eval_count} tokens)", file=sys.stderr, flush=True)
                break
    return "".join(chunks)


# ── diff-based markup ────────────────────────────────────────────────────────

def diff_segments(original: str, revised: str):
    """Yield (text, fmt) segments by word-level diff of original vs revised.
    fmt is 'normal' (kept), 'strike' (deleted), or 'red' (inserted)."""
    orig_words = original.split()
    rev_words = revised.split()
    matcher = difflib.SequenceMatcher(None, orig_words, rev_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            yield " ".join(orig_words[i1:i2]), "normal"
        elif tag == "delete":
            yield " ".join(orig_words[i1:i2]), "strike"
        elif tag == "insert":
            yield " ".join(rev_words[j1:j2]), "red"
        elif tag == "replace":
            yield " ".join(orig_words[i1:i2]), "strike"
            yield " ".join(rev_words[j1:j2]), "red"


# ── docx manipulation ────────────────────────────────────────────────────────

def is_heading(para) -> bool:
    return "heading" in para.style.name.lower()


def clear_runs(para):
    """Remove all content from a paragraph, keeping its paragraph properties."""
    p = para._p
    for child in list(p):
        if child.tag != qn("w:pPr"):
            p.remove(child)


def apply_diff(para, original: str, revised: str):
    """Rebuild paragraph runs showing a word-level diff of original vs revised."""
    clear_runs(para)
    prev_fmt = None
    buf = []

    def flush():
        nonlocal buf, prev_fmt
        if buf:
            text = " ".join(buf)
            run = para.add_run(text)
            if prev_fmt == "strike":
                run.font.strike = True
            elif prev_fmt == "red":
                run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
                run.bold = True
        buf = []

    for text, fmt in diff_segments(original, revised):
        if not text:
            continue
        if fmt != prev_fmt:
            flush()
            # Add a space between adjacent segments of different types
            if prev_fmt is not None:
                run = para.add_run(" ")
        buf.append(text)
        prev_fmt = fmt
    flush()


# ── analysis helpers ─────────────────────────────────────────────────────────

def count_words(paras) -> int:
    return sum(len(p.text.split()) for p in paras)


def para_net_reduction(orig: str, revised: str) -> tuple[int, int]:
    """Return (words_deleted, words_inserted) for one paragraph pair."""
    deleted = inserted = 0
    for text, fmt in diff_segments(orig, revised):
        n = len(text.split())
        if fmt == "strike":
            deleted += n
        elif fmt == "red":
            inserted += n
    return deleted, inserted


def select_to_target(orig_texts: list[str], revised_texts: list[str],
                     current_words: int, target_words: int) -> list[bool]:
    """Return a boolean mask: True = accept this paragraph's changes.
    Greedily accepts paragraphs sorted by net savings (largest first) until
    the target word count is reached."""
    need = current_words - target_words
    if need <= 0:
        return [False] * len(orig_texts)

    candidates = []
    for i, (orig, revised) in enumerate(zip(orig_texts, revised_texts)):
        if not revised or orig == revised:
            continue
        deleted, inserted = para_net_reduction(orig, revised)
        net = deleted - inserted
        if net > 0:
            candidates.append((i, net))

    candidates.sort(key=lambda x: x[1], reverse=True)

    accepted = [False] * len(orig_texts)
    saved = 0
    for i, net in candidates:
        if saved >= need:
            break
        accepted[i] = True
        saved += net

    return accepted


def report_reduction(orig_texts: list[str], revised_texts: list[str],
                     accepted: list[bool], current_words: int, target_words: int):
    deleted = inserted = unchanged = 0
    for orig, revised, acc in zip(orig_texts, revised_texts, accepted):
        if not acc or orig == revised:
            unchanged += len(orig.split())
            continue
        d, ins = para_net_reduction(orig, revised)
        deleted += d
        inserted += ins
    net = current_words - deleted + inserted
    skipped = sum(1 for a in accepted if not a)
    print(f"\nMarkup summary:", file=sys.stderr)
    print(f"  Paragraphs with accepted changes: {sum(accepted)}  "
          f"({skipped} model suggestions skipped to stay on target)", file=sys.stderr)
    print(f"  Words deleted:  {deleted:,}", file=sys.stderr)
    print(f"  Words inserted: {inserted:,}", file=sys.stderr)
    print(f"  Net word count: {net:,}  (target {target_words:,}, was {current_words:,})",
          file=sys.stderr)
    pct = (current_words - net) / current_words * 100
    print(f"  Net reduction:  {pct:.1f}%", file=sys.stderr)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Suggest page-count reductions for a .docx proposal"
    )
    ap.add_argument("proposal", help="Path to the .docx file")
    ap.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model to use (default: {DEFAULT_MODEL})"
    )
    ap.add_argument(
        "--target-reduction", type=float, default=0.20,
        help="Fraction of words to cut (default: 0.20 = 20%%)"
    )
    ap.add_argument(
        "--think", action="store_true", default=False,
        help="Enable extended reasoning for qwen3 models (slower, better rewrites)"
    )
    ap.add_argument(
        "--from-heading", default=None, metavar="TEXT",
        help="Only process paragraphs from the first heading containing TEXT onwards"
    )
    args = ap.parse_args()

    path = Path(args.proposal).expanduser()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = path.parent / f"page_reduction_{ts}.docx"
    shutil.copy(str(path), str(out_path))
    print(f"Copied → {out_path.name}", file=sys.stderr)

    doc = Document(str(out_path))

    # Split paragraphs into skip (before target heading) and process (from it on).
    all_paras = [p for p in doc.paragraphs if p.text.strip()]
    if args.from_heading:
        after = False
        body_paras = []
        for p in all_paras:
            if is_heading(p) and args.from_heading in p.text:
                after = True
            if after and not is_heading(p):
                body_paras.append(p)
        if not body_paras:
            sys.exit(f"No paragraphs found after heading containing {args.from_heading!r}")
        print(f"Starting from heading containing {args.from_heading!r}", file=sys.stderr)
    else:
        body_paras = [p for p in all_paras if not is_heading(p)]

    heading_count = len([p for p in all_paras if is_heading(p)])

    current_words = count_words(body_paras)
    target_words = int(current_words * (1 - args.target_reduction))
    words_to_cut = current_words - target_words

    # Context window: need to hold input (~chars/4 tokens) + output (similar).
    # Add generous headroom; qwen3.6:27b supports 262K.
    input_tokens = sum(len(p.text) for p in body_paras) // 4
    num_ctx = max(16384, (input_tokens * 3) // 1024 * 1024 + 16384)  # round up with buffer

    print(
        f"Body paragraphs: {len(body_paras)}  |  Headings skipped: {heading_count}\n"
        f"Current words:   {current_words:,}\n"
        f"Target words:    {target_words:,}  (cut {words_to_cut:,} words, "
        f"{args.target_reduction:.0%} reduction)\n"
        f"Context window:  {num_ctx:,} tokens\n"
        f"Model:           {args.model}  (thinking: {'on' if args.think else 'off'})",
        file=sys.stderr,
    )

    texts = [p.text for p in body_paras]
    user = (
        f"Current word count: {current_words:,}. Target: {target_words:,} words "
        f"(cut {words_to_cut:,} words — {args.target_reduction:.0%} reduction).\n\n"
        f"Here are all {len(texts)} body paragraphs. Mark enough text for deletion "
        f"or condensing to reach the target word count. Be aggressive where the text "
        f"is repetitive or verbose — this is a hard page limit.\n\n"
        + "|||PARA|||".join(texts)
    )

    print("Sending to model (single call, full document) ...", file=sys.stderr, flush=True)
    raw = ollama_chat(args.model, SYSTEM_PROMPT, user, num_ctx=num_ctx, think=args.think)

    # Strip accidental code-fence wrapping
    raw = re.sub(r"^```\w*\n?", "", raw.strip())
    raw = re.sub(r"\n?```$", "", raw)

    parts = [p.strip() for p in raw.split("|||PARA|||")]
    print(f"Received {len(parts)} paragraphs back (sent {len(texts)})", file=sys.stderr)

    # Select only enough paragraph changes to reach the target word count,
    # ranked by net savings (largest first). This prevents the model overshooting.
    accepted = select_to_target(texts, parts, current_words, target_words)

    for para, orig, revised, acc in zip(body_paras, texts, parts, accepted):
        if acc and revised and revised != orig:
            apply_diff(para, orig, revised)

    report_reduction(texts, parts, accepted, current_words, target_words)
    doc.save(str(out_path))
    print(f"Saved → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
