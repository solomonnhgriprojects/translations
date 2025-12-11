#!/usr/bin/env python3
"""
phrase_retention_analysis.py

Parse highlighted phrases from a Word document and compute per-row retention metrics
against an Excel sheet, plus token-level Jaccard similarity between two text columns.

- GREEN highlights in the Word doc are treated as "demographic" phrases.
- YELLOW highlights are treated as "phenotypic" phrases.
- Unhighlighted short paragraphs (<= 80 chars) start new "sections" in order.
- Column A is assumed to contain a row "ID" that maps to section order (1..N).

Outputs (written into specified columns, row 0 contains headers):
  H: jaccard_similarity
  I: demographic_phrases_exact_retained_ratio
  J: demographic_phrases_fuzzy_retained_ratio
  K: phenotypic_phrases_exact_retained_ratio
  L: phenotypic_phrases_fuzzy_retained_ratio

Example:
  python phrase_retention_analysis.py \
      --word-doc path/to/HighlightedDescriptions.docx \
      --input-dir path/to/xlsx_dir \
      --sheet "Main Data" \
      --col-a A --col-e E --col-g G \
      --col-h H --col-i I --col-j J --col-k K --col-l L \
      --out-suffix "_retained"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

# Hard dependency: python-docx
try:
    from docx import Document
    from docx.enum.text import WD_COLOR_INDEX
except Exception as e:
    raise SystemExit(
        "[error] python-docx is required. Install with: pip install python-docx"
    ) from e

# Optional: Unidecode fallback for diacritics
try:
    from unidecode import unidecode
    _HAVE_UNIDECODE = True
except Exception:
    _HAVE_UNIDECODE = False


# -------------------- CLI --------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute phrase retention metrics from a Word doc vs. Excel files."
    )

    # Inputs (mutually exclusive: --files or --input-dir)
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--files", nargs="+", help="One or more .xlsx files to process.")
    src.add_argument("--input-dir", help="Directory containing .xlsx files to process.")
    p.add_argument("--word-doc", required=True, help="Path to the Word document (.docx) with highlights.")
    p.add_argument("--sheet", default="Main Data", help="Worksheet name (default: 'Main Data').")

    # Column mapping (letters or 0-based indices)
    p.add_argument("--col-a", default="A", help="Column with row IDs mapping to section order (default: A).")
    p.add_argument("--col-e", default="E", help="Text column E (default: E).")
    p.add_argument("--col-g", default="G", help="Text column G (default: G).")
    p.add_argument("--col-h", default="H", help="Output column for Jaccard similarity (default: H).")
    p.add_argument("--col-i", default="I", help="Output column for demographic exact retention (default: I).")
    p.add_argument("--col-j", default="J", help="Output column for demographic fuzzy retention (default: J).")
    p.add_argument("--col-k", default="K", help="Output column for phenotypic exact retention (default: K).")
    p.add_argument("--col-l", default="L", help="Output column for phenotypic fuzzy retention (default: L).")

    # Output
    p.add_argument("--out-suffix", default="_retained", help="Suffix for output files (default: _retained).")

    # Normalization toggles (for fuzzy checks & Jaccard)
    p.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing.")
    p.add_argument("--no-remove-punct", action="store_true", help="Keep punctuation.")
    p.add_argument("--no-remove-diacritics", action="store_true", help="Keep diacritics.")
    p.add_argument("--no-collapse-ws", action="store_true", help="Do not collapse whitespace.")
    p.add_argument("--min-token-len", type=int, default=1, help="Minimum token length to keep (default: 1).")
    p.add_argument(
        "--stopwords",
        nargs="*",
        default=[
            "a","an","and","are","as","at","be","but","by","for","from","has","have","he","in","into",
            "is","it","its","of","on","or","that","the","their","them","they","this","to","was","were",
            "will","with","her","his","she","him","we","you","your","yours","i","our","ours","not",
            "due","who","had","also","been","than"
        ],
        help="Stopword list used in fuzzy checks (space-separated)."
    )

    return p.parse_args(argv)


# -------------------- Column helpers --------------------

_COL_RE = re.compile(r"^[A-Za-z]+$")

def col_to_index(c: str | int) -> int:
    """Convert Excel column letter(s) (e.g., 'A', 'H', 'AA') or 0-based int to 0-based index."""
    if isinstance(c, int):
        if c < 0:
            raise ValueError("Column index must be >= 0")
        return c
    s = str(c).strip()
    if s.isdigit():
        i = int(s)
        if i < 0:
            raise ValueError("Column index must be >= 0")
        return i
    if not _COL_RE.match(s):
        raise ValueError(f"Invalid column reference: {c!r}")
    s = s.upper()
    idx = 0
    for ch in s:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1


# -------------------- Text normalization --------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

def strip_diacritics(text: str) -> str:
    try:
        norm = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in norm if not unicodedata.combining(ch))
    except Exception:
        if _HAVE_UNIDECODE:
            return unidecode(text)
        return text

def normalize_string(
    s: Optional[str],
    lowercase: bool,
    remove_punct: bool,
    remove_diacritics: bool,
    collapse_ws: bool,
) -> str:
    if s is None:
        return ""
    t = str(s)
    if remove_diacritics:
        t = strip_diacritics(t)
    if lowercase:
        t = t.lower()
    if remove_punct:
        t = _PUNCT_RE.sub(" ", t)
    if collapse_ws:
        t = _WS_RE.sub(" ", t).strip()
    return t

def tokenize(
    s: str,
    *,
    stopwords: Set[str],
    min_len: int,
    drop_stopwords: bool = True,
) -> List[str]:
    if not s:
        return []
    toks = s.split()
    if drop_stopwords:
        toks = [t for t in toks if t not in stopwords]
    return [t for t in toks if len(t) >= min_len]


# -------------------- Metrics --------------------

def jaccard_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    A, B = set(tokens_a), set(tokens_b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def exact_retained_ratio(phrases: List[str], g_raw_text: str, norm_flags: Dict[str, bool]) -> float:
    """Fraction of phrases that appear as exact substrings in G (case-insensitive if lowercase=True)."""
    if not phrases:
        return float("nan")
    g_cmp = normalize_string(g_raw_text, **norm_flags, remove_punct=False)  # keep punct for exact matching
    kept = 0
    for p in phrases:
        p_cmp = normalize_string(p, **norm_flags, remove_punct=False)
        if p_cmp and p_cmp in g_cmp:
            kept += 1
    return kept / len(phrases)

def fuzzy_retained_ratio(
    phrases: List[str],
    g_raw_text: str,
    norm_flags: Dict[str, bool],
    stopwords: Set[str],
    min_token_len: int,
) -> float:
    """
    Fraction of phrases where ALL content words (stopwords removed) of the phrase
    appear somewhere in G after normalization; order-agnostic.
    """
    if not phrases:
        return float("nan")
    g_norm = normalize_string(g_raw_text, **norm_flags)
    g_tokens = set(tokenize(g_norm, stopwords=stopwords, min_len=min_token_len, drop_stopwords=True))
    kept = 0
    for p in phrases:
        p_norm = normalize_string(p, **norm_flags)
        p_tokens = set(tokenize(p_norm, stopwords=stopwords, min_len=min_token_len, drop_stopwords=True))
        if p_tokens and p_tokens.issubset(g_tokens):
            kept += 1
    return kept / len(phrases)


# -------------------- Word parsing --------------------

def parse_highlighted_phrases_by_order(doc_path: str) -> List[Dict[str, List[str]]]:
    """
    Treat any unhighlighted, short paragraph as a section title.
    Collect GREEN (demographic) and YELLOW (phenotypic) highlighted spans
    in subsequent paragraphs until the next title.
    """
    doc = Document(doc_path)
    ordered: List[Dict[str, List[str]]] = []
    current: Optional[Dict[str, List[str] | str]] = None

    def flush_current():
        nonlocal current
        if current:
            for key in ("demographic", "phenotypic"):
                seen, uniq = set(), []
                for p in current[key]:  # type: ignore[index]
                    if p and p not in seen:
                        uniq.append(p)
                        seen.add(p)
                current[key] = uniq  # type: ignore[index]
            ordered.append({"title": current["title"], "demographic": current["demographic"], "phenotypic": current["phenotypic"]})  # type: ignore[index]
        current = None

    GREEN_SET = {WD_COLOR_INDEX.BRIGHT_GREEN, WD_COLOR_INDEX.GREEN}
    YELLOW = WD_COLOR_INDEX.YELLOW

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        has_high = any(run.font.highlight_color is not None for run in para.runs)

        # Title heuristic: unhighlighted & short-ish
        if not has_high and len(text) <= 80:
            flush_current()
            current = {"title": text, "demographic": [], "phenotypic": []}  # type: ignore[assignment]
            continue

        if current is None:
            # Skip preamble (if any) before first title
            continue

        # Group contiguous highlighted runs
        buf = ""
        buf_color = None

        def flush_buf():
            nonlocal buf, buf_color
            if buf and buf_color:
                phrase = buf.strip()
                if phrase:
                    if buf_color in GREEN_SET:
                        current["demographic"].append(phrase)  # type: ignore[index]
                    elif buf_color == YELLOW:
                        current["phenotypic"].append(phrase)  # type: ignore[index]
            buf, buf_color = "", None

        for run in para.runs:
            color, rtext = run.font.highlight_color, run.text
            if color == buf_color and (color in GREEN_SET or color == YELLOW):
                buf += rtext
            else:
                flush_buf()
                if color in GREEN_SET or color == YELLOW:
                    buf, buf_color = rtext, color
                else:
                    buf, buf_color = "", None
        flush_buf()

    flush_current()
    return ordered


# -------------------- Main processing --------------------

def find_excel_files(files: Optional[List[str]], input_dir: Optional[str]) -> List[str]:
    if files:
        return [f for f in files if f.lower().endswith(".xlsx")]
    if input_dir:
        return sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
    return sorted(glob.glob("*.xlsx"))

def ensure_cols(df: pd.DataFrame, max_idx: int) -> None:
    if df.shape[1] <= max_idx:
        for _ in range(max_idx + 1 - df.shape[1]):
            df[df.shape[1]] = None

def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Column indexes
    try:
        A = col_to_index(args.col_a)
        E = col_to_index(args.col_e)
        G = col_to_index(args.col_g)
        H = col_to_index(args.col_h)
        I = col_to_index(args.col_i)
        J = col_to_index(args.col_j)
        K = col_to_index(args.col_k)
        L = col_to_index(args.col_l)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    files = find_excel_files(args.files, args.input_dir)
    if not files:
        print("[error] No .xlsx files found. Provide --files or --input-dir (or run in a folder with .xlsx).", file=sys.stderr)
        return 2

    # Parse Word document -> ordered sections (1..N)
    sections = parse_highlighted_phrases_by_order(args.word_doc)
    if not sections:
        print("[error] No sections parsed from the Word document. Check highlighting and title formatting.", file=sys.stderr)
        return 2

    id_to_phrases = {
        i + 1: {"demographic": s["demographic"], "phenotypic": s["phenotypic"]}
        for i, s in enumerate(sections)
    }

    # Normalization flags
    norm_flags = dict(
        lowercase=not args.no_lowercase,
        remove_punct=not args.no_remove_punct,
        remove_diacritics=not args.no_remove_diacritics,
        collapse_ws=not args.no_collapse_ws,
    )
    stopwords = set(args.stopwords)
    min_token_len = args.min_token_len

    print("Configuration:")
    print(f"  word_doc       : {args.word_doc}")
    print(f"  sheet          : {args.sheet}")
    print(f"  cols A,E,G     : {args.col_a}({A}), {args.col_e}({E}), {args.col_g}({G})")
    print(f"  cols H–L       : {args.col_h}({H}), {args.col_i}({I}), {args.col_j}({J}), {args.col_k}({K}), {args.col_l}({L})")
    print(f"  out_suffix     : {args.out_suffix}")
    print(f"  norm flags     : {norm_flags}")
    print(f"  min_token_len  : {min_token_len}")
    print(f"  stopwords      : {len(stopwords)} items")
    print(f"  files          : {len(files)} file(s)")

    failures = 0
    for path in files:
        print(f"\nProcessing {os.path.basename(path)} ...")
        try:
            df = pd.read_excel(path, sheet_name=args.sheet, engine="openpyxl", header=None, dtype=object)
        except ValueError as e:
            print(f"  [warn] Sheet '{args.sheet}' not found in {path} ({e}). Skipping.")
            continue

        # Make sure output columns exist
        ensure_cols(df, max(H, I, J, K, L))

        # Headers at row 0
        df.iat[0, H] = "jaccard_similarity"
        df.iat[0, I] = "demographic_phrases_exact_retained_ratio"
        df.iat[0, J] = "demographic_phrases_fuzzy_retained_ratio"
        df.iat[0, K] = "phenotypic_phrases_exact_retained_ratio"
        df.iat[0, L] = "phenotypic_phrases_fuzzy_retained_ratio"

        # Iterate rows (skip header row 0)
        for i in range(1, len(df)):
            a_val = df.iat[i, A]
            e_text = df.iat[i, E]
            g_text = df.iat[i, G]
            g_safe = "" if g_text is None else str(g_text)

            # H: Jaccard on normalized tokens (without stopword removal)
            e_norm = normalize_string(e_text, **norm_flags)
            g_norm = normalize_string(g_text, **norm_flags)
            e_tok = tokenize(e_norm, stopwords=stopwords, min_len=min_token_len, drop_stopwords=False)
            g_tok = tokenize(g_norm, stopwords=stopwords, min_len=min_token_len, drop_stopwords=False)
            df.iat[i, H] = jaccard_similarity(e_tok, g_tok)

            # Resolve row ID from Column A (int from "12" or "12.0" or embedded digits)
            row_id: Optional[int] = None
            if a_val is not None:
                try:
                    row_id = int(float(str(a_val)))
                except Exception:
                    m = re.search(r"\d+", str(a_val))
                    row_id = int(m.group(0)) if m else None

            if not row_id or row_id not in id_to_phrases:
                # No matching section; leave I–L as NaN
                continue

            demo = id_to_phrases[row_id]["demographic"]
            pheno = id_to_phrases[row_id]["phenotypic"]

            # I–L: retention ratios vs Column G
            df.iat[i, I] = exact_retained_ratio(demo, g_safe, norm_flags)
            df.iat[i, J] = fuzzy_retained_ratio(demo, g_safe, norm_flags, stopwords, min_token_len)
            df.iat[i, K] = exact_retained_ratio(pheno, g_safe, norm_flags)
            df.iat[i, L] = fuzzy_retained_ratio(pheno, g_safe, norm_flags, stopwords, min_token_len)

        # Save beside original
        base, ext = os.path.splitext(path)
        out_path = base + f"{args.out_suffix}.xlsx"
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=args.sheet, header=False, index=False)
        print(f"  ✓ Wrote -> {out_path}")

    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
        return 1

    print("\nAll workbooks processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

