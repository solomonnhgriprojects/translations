#!/usr/bin/env python3
"""
jaccard_text_similarity.py

Compute token-level Jaccard similarity between two Excel columns, with configurable
text normalization (lowercasing, punctuation removal, diacritics stripping, whitespace
collapsing) and optional phonetic (Double Metaphone) and spell-correction passes.

Usage examples:
  # Process all .xlsx files in a folder, reading sheet "Main Data",
  # comparing columns E and G, and writing similarity to column H:
  python jaccard_text_similarity.py \
      --input-dir path/to/folder \
      --sheet "Main Data" \
      --col-a E --col-b G --col-out H

  # Process specific files and put output suffix "_jaccard":
  python jaccard_text_similarity.py \
      --files file1.xlsx file2.xlsx \
      --sheet "Sheet1" \
      --col-a B --col-b C --col-out D \
      --out-suffix "_jaccard"

Dependencies:
  - pandas
  - openpyxl (for .xlsx I/O)
  - (optional) Unidecode
  - (optional) Metaphone
  - (optional) pyspellchecker
"""

from __future__ import annotations

import argparse
import os
import sys
import glob
import re
import unicodedata
from typing import Iterable, List, Set, Tuple

import pandas as pd

# Optional imports (loaded lazily if enabled)
_have_unidecode = False
_have_metaphone = False
_have_spellchecker = False

try:
    from unidecode import unidecode
    _have_unidecode = True
except Exception:
    pass

try:
    from metaphone import doublemetaphone
    _have_metaphone = True
except Exception:
    pass

try:
    from spellchecker import SpellChecker
    _have_spellchecker = True
except Exception:
    pass


# ---------- Utilities ----------

_COL_RE = re.compile(r"^[A-Za-z]+$")  # support A, H, AA, etc.
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def col_to_index(col: str | int) -> int:
    """Convert Excel column letter(s) (e.g., 'A', 'H', 'AA') or 0-based int to 0-based index."""
    if isinstance(col, int):
        if col < 0:
            raise ValueError("Column index must be >= 0")
        return col
    c = col.strip()
    if not _COL_RE.match(c):
        raise ValueError(f"Invalid column reference: {col!r}")
    c = c.upper()
    idx = 0
    for ch in c:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1  # 1-based -> 0-based


def strip_diacritics(text: str) -> str:
    # Prefer standard Unicode NFKD; fall back to Unidecode if available, else unchanged.
    try:
        norm = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in norm if not unicodedata.combining(ch))
    except Exception:
        if _have_unidecode:
            return unidecode(text)
        return text


def normalize_tokens(
    value,
    *,
    lowercase: bool,
    remove_punct: bool,
    remove_diacritics: bool,
    collapse_ws: bool,
    use_metaphone: bool,
    use_spell_correct: bool,
    min_token_len: int,
) -> List[str]:
    """Normalize and tokenize a cell value into a list of tokens."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    s = str(value)

    if remove_diacritics:
        s = strip_diacritics(s)
    if lowercase:
        s = s.lower()
    if remove_punct:
        s = _PUNCT_RE.sub(" ", s)
    if collapse_ws:
        s = _WS_RE.sub(" ", s).strip()

    tokens = s.split() if s else []

    # Spell correction
    if use_spell_correct and tokens:
        if not _have_spellchecker:
            raise RuntimeError("Spell correction requested, but 'pyspellchecker' is not installed.")
        sp = SpellChecker()
        alpha = [t for t in tokens if t.isalpha()]
        mapping = {t: (sp.correction(t) or t) for t in set(alpha)}
        tokens = [mapping.get(t, t) if t.isalpha() else t for t in tokens]

    # Phonetic mapping
    if use_metaphone and tokens:
        if not _have_metaphone:
            raise RuntimeError("Metaphone requested, but 'Metaphone' is not installed.")
        def to_meta(t: str) -> str:
            if t.isalpha():
                m1, m2 = doublemetaphone(t)
                return m1 or m2 or t
            return t
        tokens = [to_meta(t) for t in tokens]

    # Length filter
    tokens = [t for t in tokens if len(t) >= min_token_len]
    return tokens


def jaccard_similarity(tokens_a: Iterable[str], tokens_b: Iterable[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    set_a: Set[str] = set(tokens_a)
    set_b: Set[str] = set(tokens_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def find_excel_files(files: List[str], input_dir: str | None) -> List[str]:
    if files:
        return [f for f in files if f.lower().endswith(".xlsx")]
    if input_dir:
        return sorted(glob.glob(os.path.join(input_dir, "*.xlsx")))
    return sorted(glob.glob("*.xlsx"))


# ---------- Main processing ----------

def process_workbook(
    path: str,
    sheet: str,
    col_a_idx: int,
    col_b_idx: int,
    col_out_idx: int,
    *,
    lowercase: bool,
    remove_punct: bool,
    remove_diacritics: bool,
    collapse_ws: bool,
    use_metaphone: bool,
    use_spell_correct: bool,
    min_token_len: int,
    out_suffix: str,
) -> Tuple[str, int]:
    """Process a single workbook; returns (output_path, n_rows_written)."""
    try:
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", header=None, dtype=object)
    except ValueError as e:
        raise RuntimeError(f"Sheet {sheet!r} not found in {os.path.basename(path)}") from e

    # Ensure output column exists
    if df.shape[1] <= col_out_idx:
        for _ in range(col_out_idx + 1 - df.shape[1]):
            df[df.shape[1]] = None

    # Bounds check for input columns
    if df.shape[1] <= max(col_a_idx, col_b_idx):
        raise RuntimeError(
            f"Not enough columns for A/B indices in {os.path.basename(path)} "
            f"(need ≥ {max(col_a_idx, col_b_idx)+1}, have {df.shape[1]})."
        )

    # Compute similarities (skip header row by default if your sheet has one; make it configurable if needed)
    start_row = 1  # change to 0 if your sheet has no header row
    rows = 0
    for i in range(start_row, len(df)):
        a = df.iat[i, col_a_idx]
        b = df.iat[i, col_b_idx]
        tok_a = normalize_tokens(
            a,
            lowercase=lowercase,
            remove_punct=remove_punct,
            remove_diacritics=remove_diacritics,
            collapse_ws=collapse_ws,
            use_metaphone=use_metaphone,
            use_spell_correct=use_spell_correct,
            min_token_len=min_token_len,
        )
        tok_b = normalize_tokens(
            b,
            lowercase=lowercase,
            remove_punct=remove_punct,
            remove_diacritics=remove_diacritics,
            collapse_ws=collapse_ws,
            use_metaphone=use_metaphone,
            use_spell_correct=use_spell_correct,
            min_token_len=min_token_len,
        )
        df.iat[i, col_out_idx] = jaccard_similarity(tok_a, tok_b)
        rows += 1

    base, ext = os.path.splitext(path)
    out_path = f"{base}{out_suffix}.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, header=False, index=False)

    return out_path, rows


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Jaccard similarity between two Excel columns.")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--files", nargs="+", help="One or more .xlsx files to process.")
    src.add_argument("--input-dir", help="Directory containing .xlsx files to process.")

    p.add_argument("--sheet", default="Main Data", help="Worksheet name (default: 'Main Data').")
    p.add_argument("--col-a", default="E", help="First input column (letter(s) or 0-based index). Default: E")
    p.add_argument("--col-b", default="G", help="Second input column (letter(s) or 0-based index). Default: G")
    p.add_argument("--col-out", default="H", help="Output column for similarity (letter(s) or 0-based index). Default: H")
    p.add_argument("--out-suffix", default="_jaccard", help="Suffix for output files (default: _jaccard).")

    # Normalization toggles
    p.add_argument("--no-lowercase", action="store_true", help="Disable lowercasing.")
    p.add_argument("--no-remove-punct", action="store_true", help="Keep punctuation.")
    p.add_argument("--no-remove-diacritics", action="store_true", help="Keep diacritics.")
    p.add_argument("--no-collapse-ws", action="store_true", help="Do not collapse whitespace.")
    p.add_argument("--use-metaphone", action="store_true", help="Enable Double Metaphone phonetic mapping.")
    p.add_argument("--use-spell-correct", action="store_true", help="Enable basic spell correction.")
    p.add_argument("--min-token-len", type=int, default=1, help="Minimum token length to keep (default: 1).")

    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        col_a_idx = col_to_index(args.col_a) if isinstance(args.col_a, str) else int(args.col_a)
        col_b_idx = col_to_index(args.col_b) if isinstance(args.col_b, str) else int(args.col_b)
        col_out_idx = col_to_index(args.col_out) if isinstance(args.col_out, str) else int(args.col_out)
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    files = find_excel_files(args.files or [], args.input_dir)
    if not files:
        print("[error] No .xlsx files found. Provide --files or --input-dir.", file=sys.stderr)
        return 2

    # Echo configuration
    print("Configuration:")
    print(f"  sheet           : {args.sheet}")
    print(f"  col A, B, out   : {args.col_a} ({col_a_idx}), {args.col_b} ({col_b_idx}), {args.col_out} ({col_out_idx})")
    print(f"  lowercase       : {not args.no_lowercase}")
    print(f"  remove_punct    : {not args.no_remove_punct}")
    print(f"  remove_diacritics: {not args.no_remove_diacritics}")
    print(f"  collapse_ws     : {not args.no_collapse_ws}")
    print(f"  use_metaphone   : {args.use_metaphone}")
    print(f"  use_spell_correct: {args.use_spell_correct}")
    print(f"  min_token_len   : {args.min_token_len}")
    print(f"  out_suffix      : {args.out_suffix}")
    print(f"  files           : {len(files)} file(s)")

    # Validate optional deps if toggled
    if args.use_metaphone and not _have_metaphone:
        print("[error] --use-metaphone was set, but 'Metaphone' package is not installed.", file=sys.stderr)
        return 2
    if args.use_spell_correct and not _have_spellchecker:
        print("[error] --use-spell-correct was set, but 'pyspellchecker' package is not installed.", file=sys.stderr)
        return 2

    failures = 0
    for path in files:
        print(f"\nProcessing {os.path.basename(path)} ...")
        try:
            out_path, nrows = process_workbook(
                path,
                args.sheet,
                col_a_idx,
                col_b_idx,
                col_out_idx,
                lowercase=not args.no_lowercase,
                remove_punct=not args.no_remove_punct,
                remove_diacritics=not args.no_remove_diacritics,
                collapse_ws=not args.no_collapse_ws,
                use_metaphone=args.use_metaphone,
                use_spell_correct=args.use_spell_correct,
                min_token_len=args.min_token_len,
                out_suffix=args.out_suffix,
            )
            print(f"  ✓ Wrote {nrows} row(s) -> {out_path}")
        except Exception as e:
            failures += 1
            print(f"  ✗ Failed: {e}", file=sys.stderr)

    if failures:
        print(f"\nCompleted with {failures} failure(s).", file=sys.stderr)
        return 1

    print("\nAll workbooks processed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

