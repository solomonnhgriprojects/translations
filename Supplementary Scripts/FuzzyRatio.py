#!/usr/bin/env python3
"""
Compute the fuzzy retained ratio between an original phrase and a
back-translated phrase, following the method described in:

    Duong et al., "Quantifying the effects of translation methods on
    large language model recognition of genetic conditions"

The fuzzy ratio is case-insensitive, ignores punctuation, diacritics,
spacing, and stopwords, and is order-agnostic.  It measures what fraction
of the content words in the original phrase appear somewhere in the
back translation.

Usage:
    python FuzzyRatio.py "original phrase" "back-translated phrase"
    python FuzzyRatio.py --file phrases.xlsx
"""

import argparse
import os
import re
import sys
import unicodedata

# ---------------------------------------------------------------------------
# English stopwords (no external dependency required)
# ---------------------------------------------------------------------------
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "am", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "that", "this", "these", "those", "it", "its", "he", "she",
    "his", "her", "they", "them", "their", "we", "us", "our", "you",
    "your", "who", "whom", "which", "what", "where", "when", "how", "not",
    "no", "nor", "so", "as", "up", "out", "about", "into", "over",
    "after", "before", "between", "under", "again", "then", "once", "than",
})


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def strip_diacritics(text: str) -> str:
    """Remove diacritical marks (accents, cedillas, etc.)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")


def normalise(text: str) -> str:
    """Lowercase, strip diacritics, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = strip_diacritics(text)
    text = re.sub(r"[^\w\s]", " ", text)   # punctuation -> space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_words(text: str) -> list[str]:
    """Return the list of content words after normalisation and stopword removal."""
    tokens = normalise(text).split()
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
# Fuzzy ratio
# ---------------------------------------------------------------------------

def fuzzy_ratio(original: str, back_translation: str) -> float:
    """Compute the fuzzy retained ratio.

    Returns the fraction of content words in *original* that appear
    anywhere in *back_translation*.  Single-token phrases (original has
    only one content word) return NaN, following the paper's convention
    of excluding them.

    Parameters
    ----------
    original : str
        The original English phrase (demographic or phenotypic).
    back_translation : str
        The back-translated English text to search within.

    Returns
    -------
    float
        Value in [0, 1], or NaN if the original is a single-token phrase.
    """
    orig_words = content_words(original)

    # Exclude single-token phrases per the paper
    if len(orig_words) <= 1:
        return float("nan")

    bt_words = set(content_words(back_translation))

    matches = sum(1 for w in orig_words if w in bt_words)
    return matches / len(orig_words)


# ---------------------------------------------------------------------------
# Batch processing from Excel
# ---------------------------------------------------------------------------

def process_file(path: str, original_col: str, backtrans_col: str, output_path: str | None):
    """Read an Excel file and compute fuzzy ratios for each row."""
    import pandas as pd

    df = pd.read_excel(path)

    if original_col not in df.columns:
        print(f"Error: column '{original_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)
    if backtrans_col not in df.columns:
        print(f"Error: column '{backtrans_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    ratios = []
    for _, row in df.iterrows():
        orig = str(row[original_col]).strip()
        bt = str(row[backtrans_col]).strip()
        if not orig or orig == "nan" or not bt or bt == "nan":
            ratios.append(float("nan"))
        else:
            ratios.append(round(fuzzy_ratio(orig, bt), 4))

    df["fuzzy_ratio"] = ratios

    if output_path is None:
        base, ext = os.path.splitext(path)
        output_path = f"{base}_fuzzy_ratio{ext}"

    df.to_excel(output_path, index=False)
    print(f"Results written to: {output_path}")

    valid = [r for r in ratios if r == r]  # exclude NaN
    if valid:
        import numpy as np
        print(f"  Rows: {len(ratios)}  (valid: {len(valid)}, excluded single-token: {len(ratios) - len(valid)})")
        print(f"  Mean fuzzy ratio: {np.mean(valid):.4f}")
        print(f"  Median:           {np.median(valid):.4f}")
        print(f"  SD:               {np.std(valid, ddof=1):.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute the fuzzy retained ratio between phrases.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- pair mode ---
    pair_parser = subparsers.add_parser("pair", help="Compare a single pair of phrases")
    pair_parser.add_argument("original", help="Original phrase")
    pair_parser.add_argument("back_translation", help="Back-translated text")

    # --- file mode ---
    file_parser = subparsers.add_parser("file", help="Batch-process an Excel file")
    file_parser.add_argument("path", help="Path to .xlsx file")
    file_parser.add_argument("--original-col", default="original_phrase",
                             help="Column name for original phrases (default: original_phrase)")
    file_parser.add_argument("--backtrans-col", default="back_translation",
                             help="Column name for back translations (default: back_translation)")
    file_parser.add_argument("--output", default=None, help="Output .xlsx path (default: <input>_fuzzy_ratio.xlsx)")

    args = parser.parse_args()

    if args.command == "pair":
        ratio = fuzzy_ratio(args.original, args.back_translation)
        orig_words = content_words(args.original)
        bt_words = set(content_words(args.back_translation))
        matched = [w for w in orig_words if w in bt_words]
        missed = [w for w in orig_words if w not in bt_words]

        print(f"Original:         {args.original!r}")
        print(f"Back translation: {args.back_translation!r}")
        print(f"Content words:    {orig_words}")
        print(f"Matched:          {matched}")
        print(f"Missed:           {missed}")

        if ratio != ratio:  # NaN check
            print(f"Fuzzy ratio:      N/A (single-token phrase, excluded per paper)")
        else:
            print(f"Fuzzy ratio:      {ratio:.4f}")

    elif args.command == "file":
        process_file(args.path, args.original_col, args.backtrans_col, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
