#!/usr/bin/env python3
"""
Compute BioLORD-2023 cosine similarity for Table 1 comparisons.

Parses back-translated sentences to extract demographic and phenotypic portions,
then computes cosine similarity between:
  1. Full original sentences vs. full back-translated sentences
  2. Original demographic phrases vs. extracted demographic portions of back-translations
  3. Original phenotypic phrases vs. extracted phenotypic portions of back-translations

Outputs results in the same structure as Table 1 from the manuscript:
  NMT_139, NMT_70, LLM_70 with Mann-Whitney U test for NMT_70 vs LLM_70.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

PHRASES_PATH = os.path.join(ROOT_DIR, "Original_Phenotype_Genotype_Phrases.xlsx")
TRANSLATIONS_PATH = os.path.join(
    ROOT_DIR, "Semantic Value Comparisons (All Languages)", "All Translations.xlsx"
)
OUTPUT_PATH = os.path.join(ROOT_DIR, "Table1_BioLORD_Cosine_Similarity.xlsx")

# ---------------------------------------------------------------------------
# Back-translation parser
# ---------------------------------------------------------------------------

# Demographic pattern: matches age/sex descriptors at the start of a sentence.
# Captures the subject descriptor and stops before the first clinical connector.
# Built from the 40 known demographic phrases + common back-translation synonyms.
_DEMO_PATTERN = re.compile(
    r"^"
    r"(?:(?:A|An|The|This)\s+)?"  # optional article
    r"("
    # Compound descriptors (order: longest first)
    r"wheelchair[- ]bound\s+young\s+(?:man|woman|boy|girl|person|adult)"
    r"|macrosomic\s+(?:newborn|neonate|infant|baby)"
    r"|middle[- ]aged\s+(?:adult|man|woman|person|individual)"
    r"|(?:young|old|elderly|newborn|older|short|tall)\s+"
      r"(?:adult|man|woman|boy|girl|child|person|people|infant|baby)"
    # Two-word age+sex
    r"|(?:male|female)\s+(?:neonate|newborn|infant|baby|child|adolescent|adult|teenager|toddler)"
    r"|(?:adolescent|teenage|teenaged)\s+(?:female|male|girl|boy|woman|man|child)"
    # Single-word descriptors
    r"|(?:neonate|neonates|newborn|newborns|infant|infants|baby|babies"
      r"|child|children|toddler|toddlers"
      r"|adolescent|adolescents|teenager|teenagers"
      r"|adult|adults|man|woman|boy|girl|person|individual|patient)"
    r")"
    r"(?=\s|,|;|\.|\b)",  # must be followed by space, punctuation, or word boundary
    re.IGNORECASE,
)

# Separator patterns between demographic and phenotypic content.
# We find the EARLIEST match across all of these.
_CONNECTOR_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwith\b",
        r"\bwho\b",
        r"\bpresenting\b",
        r"\bpresents\b",
        r"\bpresented\b",
        r"\bdiagnosed\b",
        r"\bsuffering\b",
        r"\bnoted\b",
        r"\bobserved\b",
        r"\bfound\b",
        r"\bdescribed\b",
        r"\bcharacterized\b",
        r"\bexhibiting\b",
        r"\bexhibited\b",
        r"\bdevelops\b",
        r"\bdeveloped\b",
        r"\bdeveloping\b",
        r"\bstarted\b",
        r"\bbegan\b",
        r"\bhas\b",
        r"\bhad\b",
        r"\bhaving\b",
        r"\bin whom\b",
        r"\breadmitted\b",
        r"\bis lethargic\b",
        r"\bis noted\b",
    ]
]

# Leading articles to strip
_LEADING_ARTICLES = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)


def parse_back_translation(sentence):
    """
    Split a back-translated sentence into (demographic_part, phenotypic_part).

    Strategy:
      1. Try to match a demographic phrase pattern at the start of the sentence.
      2. If matched, demographic = the matched text. Phenotypic = everything after.
      3. If no demographic pattern matches, find the earliest connector word
         and split there. Everything before = demographic, after = phenotypic.
      4. Fallback: first 2 words = demographic, rest = phenotypic.

    Returns (demographic_str, phenotypic_str).
    """
    s = sentence.strip()
    if not s:
        return ("", "")

    # Strategy 1: Match a known demographic pattern at the start
    m = _DEMO_PATTERN.match(s)
    if m:
        demo = m.group(1).strip()
        pheno = s[m.end():].strip()
        # Strip leading connector from pheno if present
        # e.g., "with short stature..." -> keep "with" as it provides context
        return (demo, pheno)

    # Strategy 2: Find the earliest connector and split there
    earliest_pos = len(s)
    for pat in _CONNECTOR_PATTERNS:
        match = pat.search(s)
        if match and match.start() < earliest_pos and match.start() > 1:
            earliest_pos = match.start()

    if earliest_pos < len(s):
        demo = s[:earliest_pos].strip().rstrip(",;:")
        demo = _LEADING_ARTICLES.sub("", demo).strip()
        pheno = s[earliest_pos:].strip()
        if len(demo) >= 2:
            return (demo, pheno)

    # Strategy 3: Fallback - first 2 words
    words = s.split()
    if len(words) <= 2:
        return (s, "")
    demo = " ".join(words[:2])
    demo = _LEADING_ARTICLES.sub("", demo).strip()
    pheno = " ".join(words[2:])
    return (demo, pheno)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_biolord():
    from sentence_transformers import SentenceTransformer
    print("Loading BioLORD-2023 model...")
    model = SentenceTransformer("FremyCompany/BioLORD-2023")
    print("Model loaded.")
    return model


def embed_texts(model, texts):
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def pairwise_cosine(emb1, emb2):
    return np.array(
        [cosine_similarity(emb1[i : i + 1], emb2[i : i + 1])[0, 0]
         for i in range(len(emb1))]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Load data ----
    print("Loading phrases...")
    phrases_df = pd.read_excel(PHRASES_PATH, sheet_name="All Data")

    print("Loading translations...")
    originals_sheet = pd.read_excel(TRANSLATIONS_PATH, sheet_name="Originals")
    nmt_139 = pd.read_excel(TRANSLATIONS_PATH, sheet_name="NMT 139 back and forward")
    nmt_70 = pd.read_excel(TRANSLATIONS_PATH, sheet_name="NMT 70 back and forward")
    llm_70 = pd.read_excel(TRANSLATIONS_PATH, sheet_name="LLM 70 back and forward")

    for df in [originals_sheet, nmt_139, nmt_70, llm_70]:
        df.columns = [c.strip() for c in df.columns]

    orig_map = originals_sheet.set_index("condition_index")["original_sentence"]
    for df in [nmt_139, nmt_70, llm_70]:
        df["original_sentence"] = df["condition_index"].map(orig_map)

    print(f"NMT_139: {nmt_139.shape}, NMT_70: {nmt_70.shape}, LLM_70: {llm_70.shape}")

    # ---- Phrase lookups ----
    demo_phrases = {}
    pheno_phrases = {}
    for i, row in phrases_df.iterrows():
        idx = i + 1
        demo_phrases[idx] = str(row["Demographic Phrases"]).strip()
        pheno_phrases[idx] = str(row["Phenotypic Phrases"]).strip()

    # ---- Parser sanity check on originals ----
    print("\n=== Parser sanity check (originals) ===")
    for ci in range(1, 41):
        sent = orig_map[ci]
        d, p = parse_back_translation(sent)
        expected = demo_phrases[ci]
        match = "OK" if d.lower() == expected.lower() else "MISMATCH"
        if match == "MISMATCH":
            print(f"  [{ci:2d}] {match}: got '{d}' | expected '{expected}'")
    print("  (only mismatches shown above)")

    # Also show a sample of back-translation parses
    print("\n=== Parser sample (NMT back-translations, French) ===")
    french = nmt_139[nmt_139["target_language_name"] == "French"]
    for _, row in french.head(10).iterrows():
        ci = row["condition_index"]
        d, p = parse_back_translation(row["back_translation"])
        print(f"  [{ci:2d}] DEMO='{d}' | PHENO='{p[:70]}...'")
        print(f"        Expected DEMO='{demo_phrases[ci]}'")
    print()

    # ---- Load model ----
    model = load_biolord()

    # ---- Process each method ----
    methods = {"NMT_139": nmt_139, "NMT_70": nmt_70, "LLM_70": llm_70}
    all_results = []

    for method_name, trans_df in methods.items():
        n = len(trans_df)
        n_langs = trans_df["target_language_name"].nunique()
        print(f"\nProcessing {method_name} ({n} rows, {n_langs} languages)...")

        orig_sents = trans_df["original_sentence"].tolist()
        back_sents = trans_df["back_translation"].tolist()
        cond_idx = trans_df["condition_index"].tolist()

        # Original phrase lists
        orig_demo_list = [demo_phrases.get(ci, "") for ci in cond_idx]
        orig_pheno_list = [pheno_phrases.get(ci, "") for ci in cond_idx]

        # Parse back-translations
        print("  Parsing back-translations...")
        parsed = [parse_back_translation(s) for s in back_sents]
        back_demo_list = [p[0] for p in parsed]
        back_pheno_list = [p[1] for p in parsed]

        # Log parse stats
        n_empty_pheno = sum(1 for p in back_pheno_list if len(p.strip()) < 3)
        n_empty_demo = sum(1 for d in back_demo_list if len(d.strip()) < 2)
        print(f"  Parse stats: {n_empty_demo} empty demos, {n_empty_pheno} empty phenos out of {n}")

        # ---- Embed everything ----
        print("  Embedding full sentences...")
        emb_orig_full = embed_texts(model, orig_sents)
        emb_back_full = embed_texts(model, back_sents)

        print("  Embedding demographic phrases (original vs. parsed)...")
        emb_orig_demo = embed_texts(model, orig_demo_list)
        emb_back_demo = embed_texts(model, back_demo_list)

        print("  Embedding phenotypic phrases (original vs. parsed)...")
        emb_orig_pheno = embed_texts(model, orig_pheno_list)
        emb_back_pheno = embed_texts(model, back_pheno_list)

        # ---- Cosine similarities ----
        print("  Computing cosine similarities...")
        full_sim = pairwise_cosine(emb_orig_full, emb_back_full)
        demo_sim = pairwise_cosine(emb_orig_demo, emb_back_demo)
        pheno_sim = pairwise_cosine(emb_orig_pheno, emb_back_pheno)

        for j in range(n):
            all_results.append({
                "method": method_name,
                "condition_index": cond_idx[j],
                "target_language": trans_df.iloc[j].get(
                    "target_language_name",
                    trans_df.iloc[j].get("target_language_code", ""),
                ),
                "original_sentence": orig_sents[j],
                "back_translation": back_sents[j],
                "orig_demo": orig_demo_list[j],
                "parsed_back_demo": back_demo_list[j],
                "orig_pheno": orig_pheno_list[j],
                "parsed_back_pheno": back_pheno_list[j],
                "full_sentence_cosine": full_sim[j],
                "demographic_cosine": demo_sim[j],
                "phenotypic_cosine": pheno_sim[j],
            })

    results_df = pd.DataFrame(all_results)

    # ---- Summary for Table 1 ----
    print("\n\n=== SUMMARY FOR TABLE 1 ===\n")

    metrics = [
        ("full_sentence_cosine", "BioLORD cosine similarity (full sentence)"),
        ("demographic_cosine", "BioLORD cosine similarity (demographic phrases)"),
        ("phenotypic_cosine", "BioLORD cosine similarity (phenotypic phrases)"),
    ]

    summary_rows = []
    for col, label in metrics:
        nmt139 = results_df.loc[results_df["method"] == "NMT_139", col].dropna()
        nmt70 = results_df.loc[results_df["method"] == "NMT_70", col].dropna()
        llm70 = results_df.loc[results_df["method"] == "LLM_70", col].dropna()

        u_stat, p_value = stats.mannwhitneyu(nmt70, llm70, alternative="two-sided")

        row = {
            "Metric": label,
            "NMT 139 Mean (SD, Median)": f"{nmt139.mean():.3f} ({nmt139.std():.3f}, {nmt139.median():.3f})",
            "NMT 70 Mean (SD, Median)": f"{nmt70.mean():.3f} ({nmt70.std():.3f}, {nmt70.median():.3f})",
            "LLM 70 Mean (SD, Median)": f"{llm70.mean():.3f} ({llm70.std():.3f}, {llm70.median():.3f})",
            "p-value (NMT 70 vs LLM 70)": f"{p_value:.2e}",
            "NMT 139 Mean": nmt139.mean(),
            "NMT 70 Mean": nmt70.mean(),
            "LLM 70 Mean": llm70.mean(),
            "NMT 139 SD": nmt139.std(),
            "NMT 70 SD": nmt70.std(),
            "LLM 70 SD": llm70.std(),
            "NMT 139 Median": nmt139.median(),
            "NMT 70 Median": nmt70.median(),
            "LLM 70 Median": llm70.median(),
            "U statistic": u_stat,
            "p-value": p_value,
        }
        summary_rows.append(row)

        print(f"{label}:")
        print(f"  NMT 139: {nmt139.mean():.3f} ({nmt139.std():.3f}, {nmt139.median():.3f})")
        print(f"  NMT  70: {nmt70.mean():.3f} ({nmt70.std():.3f}, {nmt70.median():.3f})")
        print(f"  LLM  70: {llm70.mean():.3f} ({llm70.std():.3f}, {llm70.median():.3f})")
        print(f"  p-value (NMT 70 vs LLM 70): {p_value:.2e}")
        print()

    summary_df = pd.DataFrame(summary_rows)

    # ---- Save ----
    print(f"Saving results to {OUTPUT_PATH}...")
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        summary_df[
            ["Metric", "NMT 139 Mean (SD, Median)", "NMT 70 Mean (SD, Median)",
             "LLM 70 Mean (SD, Median)", "p-value (NMT 70 vs LLM 70)"]
        ].to_excel(writer, sheet_name="Table 1 Summary", index=False)

        summary_df.to_excel(writer, sheet_name="Detailed Summary", index=False)

        results_df.to_excel(writer, sheet_name="All Results", index=False)

        # Parser audit sheet
        audit_cols = [
            "method", "condition_index", "target_language",
            "orig_demo", "parsed_back_demo", "demographic_cosine",
            "orig_pheno", "parsed_back_pheno", "phenotypic_cosine",
        ]
        results_df[audit_cols].to_excel(writer, sheet_name="Parser Audit", index=False)

        # Per-condition aggregation
        cond_agg = (
            results_df.groupby(["method", "condition_index"])
            .agg({
                "full_sentence_cosine": ["mean", "std", "median"],
                "demographic_cosine": ["mean", "std", "median"],
                "phenotypic_cosine": ["mean", "std", "median"],
            })
            .reset_index()
        )
        cond_agg.columns = ["_".join(c).strip("_") for c in cond_agg.columns]
        cond_agg.to_excel(writer, sheet_name="By Condition", index=False)

        # Per-language aggregation
        lang_agg = (
            results_df.groupby(["method", "target_language"])
            .agg({
                "full_sentence_cosine": ["mean", "std", "median"],
                "demographic_cosine": ["mean", "std", "median"],
                "phenotypic_cosine": ["mean", "std", "median"],
            })
            .reset_index()
        )
        lang_agg.columns = ["_".join(c).strip("_") for c in lang_agg.columns]
        lang_agg.to_excel(writer, sheet_name="By Language", index=False)

    print(f"Done! Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
