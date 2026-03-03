#!/usr/bin/env python3
"""
Unified semantic similarity comparison tool.

Subcommands:
  greek       Compare Greek back-translations against originals
  all         Compare all-language back-translations against originals
  test-pairs  Compare semantic similarity test pairs

Uses three embedding models:
  1. nomic-embed-text    (local Ollama API)
  2. sentence-t5-base    (sentence-transformers, general-purpose)
  3. BioLORD-2023        (sentence-transformers, medical/clinical SOTA)
"""

import argparse
import gc
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # parent of the Supplementary Scripts/ directory

# ---------------------------------------------------------------------------
# Ollama config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_RETRIES = 3
OLLAMA_TIMEOUT = 60

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODELS = {
    "nomic_embed_text": {
        "display_name": "nomic-embed-text",
        "type": "ollama",
        "model_id": "nomic-embed-text",
    },
    "sentence_t5_base": {
        "display_name": "sentence-t5-base",
        "type": "sentence_transformer",
        "model_id": "sentence-transformers/sentence-t5-base",
    },
    "biolord_2023": {
        "display_name": "BioLORD-2023",
        "type": "sentence_transformer",
        "model_id": "FremyCompany/BioLORD-2023",
    },
}

# ---------------------------------------------------------------------------
# Embedding providers
# ---------------------------------------------------------------------------

class OllamaEmbedder:
    """Embeddings via local Ollama API."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def embed(self, text: str) -> np.ndarray:
        for attempt in range(1, OLLAMA_RETRIES + 1):
            try:
                resp = requests.post(
                    OLLAMA_URL,
                    json={"model": self.model_id, "input": text},
                    timeout=OLLAMA_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()
                return np.array(data["embeddings"][0], dtype=np.float64)
            except Exception as e:
                if attempt == OLLAMA_RETRIES:
                    raise
                wait = 2 ** attempt
                print(f"    [WARN] Ollama attempt {attempt} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        results = []
        for i, t in enumerate(texts):
            results.append(self.embed(t))
            if (i + 1) % 500 == 0:
                print(f"      {i+1}/{len(texts)} embedded...")
        return results

    def close(self):
        pass


class SentenceTransformerEmbedder:
    """Embeddings via HuggingFace sentence-transformers."""

    def __init__(self, model_id: str):
        from sentence_transformers import SentenceTransformer
        print(f"    Loading model {model_id} ...")
        self.model = SentenceTransformer(model_id)
        print(f"    Model loaded.")

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True)
        return vec.astype(np.float64)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=128)
        return [v.astype(np.float64) for v in vecs]

    def close(self):
        del self.model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def make_embedder(model_cfg: dict):
    if model_cfg["type"] == "ollama":
        return OllamaEmbedder(model_cfg["model_id"])
    elif model_cfg["type"] == "sentence_transformer":
        return SentenceTransformerEmbedder(model_cfg["model_id"])
    else:
        raise ValueError(f"Unknown embedder type: {model_cfg['type']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def compute_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols)["cosine_similarity"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = group_cols + ["mean", "median", "std", "min", "max", "count"]
    for col in ["mean", "median", "std", "min", "max"]:
        summary[col] = summary[col].round(4)
    return summary


# ---------------------------------------------------------------------------
# Greek helpers
# ---------------------------------------------------------------------------

def derive_method_name(filename: str) -> str:
    """Derive translation_method from back-translation filename.

    E.g. 'Gemini_2.0_Flash_Eirini_back_translation_greek.xlsx'
      -> 'Gemini_2.0_Flash_Eirini'
    """
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_back_translation_greek$", "", name)
    return name


def discover_backtranslation_files(backtrans_dir: str) -> list[dict]:
    """Find all back-translation xlsx files and return metadata."""
    pattern = os.path.join(backtrans_dir, "*.xlsx")
    files = sorted(glob.glob(pattern))
    result = []
    for fp in files:
        basename = os.path.basename(fp)
        if basename.startswith("~$"):
            continue
        method = derive_method_name(basename)
        result.append({"path": fp, "method": method, "filename": basename})
    return result


# ===========================================================================
# Subcommand: greek
# ===========================================================================

def run_greek():
    """Compare Greek back-translated sentences against originals."""
    greek_dir = os.path.join(PROJECT_DIR, "Greek Data")
    originals_path = os.path.join(greek_dir, "Original_Sentences.xlsx")
    backtrans_dir = os.path.join(greek_dir, "Greek Translations", "Greek Back Translations")
    out_base = os.path.join(greek_dir, "Semantic Value Comparisons")

    # 1. Load originals
    print(f"Loading originals from: {originals_path}")
    originals = pd.read_excel(originals_path)
    print(f"  {len(originals)} original sentences loaded.")
    orig_lookup = dict(zip(originals["condition_index"], originals["original_sentence"]))

    # 2. Discover back-translation files
    bt_files = discover_backtranslation_files(backtrans_dir)
    if not bt_files:
        print(f"\nNo back-translation files found in:\n  {backtrans_dir}")
        sys.exit(1)
    print(f"\nDiscovered {len(bt_files)} back-translation files:")
    for bt in bt_files:
        print(f"  {bt['method']}  ({bt['filename']})")

    # Load all back-translation data upfront
    all_bt_rows = []
    for bt in bt_files:
        df = pd.read_excel(bt["path"])
        for _, row in df.iterrows():
            idx = row["condition_index"]
            bt_text = str(row.get("back_translation", "")).strip()
            if bt_text and bt_text != "nan" and idx in orig_lookup:
                all_bt_rows.append({
                    "condition_index": idx,
                    "translation_method": bt["method"],
                    "original_sentence": orig_lookup[idx],
                    "back_translation": bt_text,
                })
    print(f"  {len(all_bt_rows)} total sentence pairs to compare.")

    # Collect unique texts for batch embedding
    original_texts = list(dict.fromkeys(
        r["original_sentence"] for r in all_bt_rows
    ))
    bt_texts = [r["back_translation"] for r in all_bt_rows]

    os.makedirs(out_base, exist_ok=True)

    # 3. Run each model
    model_detailed = {}  # model_key -> DataFrame of detailed results

    for model_key, model_cfg in MODELS.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_cfg['display_name']}  (key: {model_key})")
        print(f"{'='*70}")

        embedder = make_embedder(model_cfg)

        # Embed originals
        print(f"  Embedding {len(original_texts)} unique original sentences...")
        start = time.time()
        orig_vecs = embedder.embed_batch(original_texts)
        orig_emb_map = dict(zip(original_texts, orig_vecs))
        print(f"    Done in {time.time()-start:.1f}s")

        # Embed back-translations
        print(f"  Embedding {len(bt_texts)} back-translations...")
        start = time.time()
        bt_vecs = embedder.embed_batch(bt_texts)
        print(f"    Done in {time.time()-start:.1f}s")

        # Compute similarities
        print(f"  Computing cosine similarities...")
        results = []
        for i, row_data in enumerate(all_bt_rows):
            orig_emb = orig_emb_map[row_data["original_sentence"]]
            bt_emb = bt_vecs[i]
            sim = cosine_similarity(orig_emb, bt_emb)
            results.append({
                "condition_index": row_data["condition_index"],
                "translation_method": row_data["translation_method"],
                "original_sentence": row_data["original_sentence"],
                "back_translation": row_data["back_translation"],
                "cosine_similarity": round(sim, 6),
            })

        results_df = pd.DataFrame(results)
        model_detailed[model_key] = results_df

        # Write per-model outputs
        model_dir = os.path.join(out_base, model_key)
        os.makedirs(model_dir, exist_ok=True)

        detailed_path = os.path.join(model_dir, "detailed_similarity.xlsx")
        results_df.to_excel(detailed_path, index=False)
        print(f"  Detailed: {detailed_path}")

        summary_df = compute_summary(results_df, ["translation_method"])
        summary_path = os.path.join(model_dir, "summary_statistics.xlsx")
        summary_df.to_excel(summary_path, index=False)
        print(f"  Summary:  {summary_path}")

        # Console summary
        print(f"\n  {model_cfg['display_name']} — Summary by translation method:")
        print(summary_df.to_string(index=False))

        # Release model memory before loading next
        embedder.close()
        del embedder, orig_vecs, bt_vecs, orig_emb_map
        gc.collect()

    # 4. Cross-model comparison
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}")

    # Build merged detailed sheet
    base_cols = ["condition_index", "translation_method", "original_sentence", "back_translation"]
    cross_df = model_detailed[list(MODELS.keys())[0]][base_cols].copy()

    for model_key in MODELS:
        col_name = f"cosine_{model_key}"
        cross_df[col_name] = model_detailed[model_key]["cosine_similarity"].values

    cross_path = os.path.join(out_base, "cross_model_comparison.xlsx")

    # Write two sheets: detailed + summary
    with pd.ExcelWriter(cross_path, engine="openpyxl") as writer:
        cross_df.to_excel(writer, sheet_name="Detailed", index=False)

        # Summary: mean/median per model per translation_method
        summary_rows = []
        for model_key in MODELS:
            col_name = f"cosine_{model_key}"
            grouped = cross_df.groupby("translation_method")[col_name].agg(
                ["mean", "median", "std", "min", "max", "count"]
            ).reset_index()
            grouped.insert(0, "model", MODELS[model_key]["display_name"])
            grouped.columns = [
                "model", "translation_method",
                "mean", "median", "std", "min", "max", "count",
            ]
            for c in ["mean", "median", "std", "min", "max"]:
                grouped[c] = grouped[c].round(4)
            summary_rows.append(grouped)

        summary_cross = pd.concat(summary_rows, ignore_index=True)
        summary_cross.to_excel(writer, sheet_name="Summary", index=False)

    print(f"  Cross-model comparison: {cross_path}")

    # Console cross-model summary
    print("\n  Summary (mean cosine similarity per model × translation method):")
    pivot = summary_cross.pivot_table(
        index="translation_method", columns="model", values="mean"
    )
    print(pivot.round(4).to_string())

    # Overall means per model
    print("\n  Overall mean per model:")
    for model_key in MODELS:
        col = f"cosine_{model_key}"
        mean_val = cross_df[col].mean()
        print(f"    {MODELS[model_key]['display_name']:25s} {mean_val:.4f}")

    print(f"\n{'='*70}")
    print("DONE. All outputs saved to:")
    print(f"  {out_base}")
    print(f"{'='*70}")


# ===========================================================================
# Subcommand: all
# ===========================================================================

TRANSLATION_SHEETS = [
    {"sheet": "NMT 139 back and forward", "method": "NMT_139"},
    {"sheet": "NMT 70 back and forward",  "method": "NMT_70"},
    {"sheet": "LLM 70 back and forward",  "method": "LLM_70"},
]


def run_all():
    """Compare all-language back-translated sentences against originals."""
    input_path = os.path.join(PROJECT_DIR, "Semantic Value Comparisons (All Languages)", "All Translations.xlsx")
    out_base = os.path.join(PROJECT_DIR, "Semantic Value Comparisons (All Languages)")

    # 1. Load originals
    print(f"Loading originals from: {input_path}")
    originals = pd.read_excel(input_path, sheet_name="Originals")
    print(f"  {len(originals)} original sentences loaded.")
    orig_lookup = dict(zip(originals["condition_index"], originals["original_sentence"]))

    # 2. Load all back-translation data
    all_bt_rows = []
    for info in TRANSLATION_SHEETS:
        df = pd.read_excel(input_path, sheet_name=info["sheet"])
        method = info["method"]
        print(f"  Sheet '{info['sheet']}': {len(df)} rows")
        for _, row in df.iterrows():
            idx = row["condition_index"]
            bt_text = str(row.get("back_translation", "")).strip()
            lang_code = row["target_language_code"]
            lang_name = row["target_language_name"]
            if bt_text and bt_text != "nan" and idx in orig_lookup:
                all_bt_rows.append({
                    "condition_index": idx,
                    "method": method,
                    "target_language_code": lang_code,
                    "target_language_name": lang_name,
                    "original_sentence": orig_lookup[idx],
                    "back_translation": bt_text,
                })

    print(f"\n  {len(all_bt_rows)} total sentence pairs to compare.")

    # Collect unique texts for batch embedding
    original_texts = list(dict.fromkeys(r["original_sentence"] for r in all_bt_rows))
    bt_texts = [r["back_translation"] for r in all_bt_rows]

    # Deduplicate back-translations for embedding efficiency
    unique_bt_texts = list(dict.fromkeys(bt_texts))
    print(f"  {len(unique_bt_texts)} unique back-translation texts (of {len(bt_texts)} total)")

    # 3. Run each model
    model_similarities = {}  # model_key -> list of cosine_similarity values (aligned with all_bt_rows)

    for model_key, model_cfg in MODELS.items():
        print(f"\n{'='*70}")
        print(f"MODEL: {model_cfg['display_name']}  (key: {model_key})")
        print(f"{'='*70}")

        embedder = make_embedder(model_cfg)

        # Embed originals
        print(f"  Embedding {len(original_texts)} unique original sentences...")
        start = time.time()
        orig_vecs = embedder.embed_batch(original_texts)
        orig_emb_map = dict(zip(original_texts, orig_vecs))
        print(f"    Done in {time.time()-start:.1f}s")

        # Embed unique back-translations
        print(f"  Embedding {len(unique_bt_texts)} unique back-translations...")
        start = time.time()
        bt_unique_vecs = embedder.embed_batch(unique_bt_texts)
        bt_emb_map = dict(zip(unique_bt_texts, bt_unique_vecs))
        elapsed = time.time() - start
        print(f"    Done in {elapsed:.1f}s ({elapsed/max(1,len(unique_bt_texts)):.3f}s/text)")

        # Compute similarities
        print(f"  Computing cosine similarities for {len(all_bt_rows)} pairs...")
        sims = []
        for row_data in all_bt_rows:
            orig_emb = orig_emb_map[row_data["original_sentence"]]
            bt_emb = bt_emb_map[row_data["back_translation"]]
            sim = cosine_similarity(orig_emb, bt_emb)
            sims.append(round(sim, 6))

        model_similarities[model_key] = sims

        # Build per-model detailed DataFrame
        results_df = pd.DataFrame(all_bt_rows)
        results_df["cosine_similarity"] = sims

        # Write per-model outputs
        model_dir = os.path.join(out_base, model_key)
        os.makedirs(model_dir, exist_ok=True)

        detailed_path = os.path.join(model_dir, "detailed_similarity.xlsx")
        results_df.to_excel(detailed_path, index=False)
        print(f"  Detailed: {detailed_path}")

        # Summary by method + language
        summary_ml = compute_summary(results_df, ["method", "target_language_code", "target_language_name"])
        summary_ml = summary_ml.sort_values(["method", "mean"], ascending=[True, False])
        ml_path = os.path.join(model_dir, "summary_by_method_language.xlsx")
        summary_ml.to_excel(ml_path, index=False)
        print(f"  By method+language: {ml_path}")

        # Summary by method
        summary_m = compute_summary(results_df, ["method"])
        m_path = os.path.join(model_dir, "summary_by_method.xlsx")
        summary_m.to_excel(m_path, index=False)
        print(f"  By method: {m_path}")

        # Summary by language
        summary_l = compute_summary(results_df, ["target_language_code", "target_language_name"])
        summary_l = summary_l.sort_values("mean", ascending=False)
        l_path = os.path.join(model_dir, "summary_by_language.xlsx")
        summary_l.to_excel(l_path, index=False)
        print(f"  By language: {l_path}")

        # Console summary
        print(f"\n  {model_cfg['display_name']} - Summary by method:")
        print(summary_m.to_string(index=False))

        # Release model memory
        embedder.close()
        del embedder, orig_vecs, bt_unique_vecs, orig_emb_map, bt_emb_map
        gc.collect()

    # 4. Cross-model comparison
    print(f"\n{'='*70}")
    print("CROSS-MODEL COMPARISON")
    print(f"{'='*70}")

    # Build merged detailed sheet
    base_cols = ["condition_index", "method", "target_language_code", "target_language_name",
                 "original_sentence", "back_translation"]
    cross_df = pd.DataFrame(all_bt_rows)[base_cols].copy()

    for model_key in MODELS:
        col_name = f"cosine_{model_key}"
        cross_df[col_name] = model_similarities[model_key]

    cross_path = os.path.join(out_base, "cross_model_comparison.xlsx")

    with pd.ExcelWriter(cross_path, engine="openpyxl") as writer:
        cross_df.to_excel(writer, sheet_name="Detailed", index=False)

        # Summary by method
        summary_rows = []
        for model_key in MODELS:
            col_name = f"cosine_{model_key}"
            grouped = cross_df.groupby("method")[col_name].agg(
                ["mean", "median", "std", "min", "max", "count"]
            ).reset_index()
            grouped.insert(0, "model", MODELS[model_key]["display_name"])
            grouped.columns = ["model", "method", "mean", "median", "std", "min", "max", "count"]
            for c in ["mean", "median", "std", "min", "max"]:
                grouped[c] = grouped[c].round(4)
            summary_rows.append(grouped)
        summary_method = pd.concat(summary_rows, ignore_index=True)
        summary_method.to_excel(writer, sheet_name="Summary by Method", index=False)

        # Summary by language (across all methods)
        lang_rows = []
        for model_key in MODELS:
            col_name = f"cosine_{model_key}"
            grouped = cross_df.groupby(["target_language_code", "target_language_name"])[col_name].agg(
                ["mean", "median"]
            ).reset_index()
            grouped.columns = ["target_language_code", "target_language_name",
                               f"mean_{model_key}", f"median_{model_key}"]
            lang_rows.append(grouped.set_index(["target_language_code", "target_language_name"]))
        lang_cross = pd.concat(lang_rows, axis=1).reset_index()
        lang_cross = lang_cross.sort_values(f"mean_{list(MODELS.keys())[0]}", ascending=False)
        lang_cross.to_excel(writer, sheet_name="Summary by Language", index=False)

    print(f"  Cross-model comparison: {cross_path}")

    # Console cross-model summary
    print("\n  Summary (mean cosine similarity per model x method):")
    pivot = summary_method.pivot_table(index="method", columns="model", values="mean")
    print(pivot.round(4).to_string())

    # Overall means per model
    print("\n  Overall mean per model:")
    for model_key in MODELS:
        col = f"cosine_{model_key}"
        mean_val = cross_df[col].mean()
        print(f"    {MODELS[model_key]['display_name']:25s} {mean_val:.4f}")

    # Top/bottom languages
    first_model_key = list(MODELS.keys())[0]
    first_mean_col = f"mean_{first_model_key}"
    print(f"\n  Top 5 languages (by {MODELS[first_model_key]['display_name']} mean):")
    print(lang_cross.head(5)[["target_language_code", "target_language_name"] +
          [f"mean_{k}" for k in MODELS]].to_string(index=False))
    print(f"\n  Bottom 5 languages:")
    print(lang_cross.tail(5)[["target_language_code", "target_language_name"] +
          [f"mean_{k}" for k in MODELS]].to_string(index=False))

    print(f"\n{'='*70}")
    print("DONE. All outputs saved to:")
    print(f"  {out_base}")
    print(f"{'='*70}")


# ===========================================================================
# Subcommand: test-pairs
# ===========================================================================

def run_test_pairs():
    """Run semantic similarity on the test pairs."""
    input_path = os.path.join(PROJECT_DIR, "Semantic Similarity Test Pairs", "semantic_similarity_test_pairs.xlsx")
    out_path = os.path.join(PROJECT_DIR, "Semantic Similarity Test Pairs", "test_pairs_results.xlsx")

    print(f"Loading test pairs from: {input_path}")
    df = pd.read_excel(input_path)
    print(f"  {len(df)} test pairs loaded.")

    # Collect all unique sentences for batch embedding
    all_a = df["sentence_a"].tolist()
    all_b = df["sentence_b"].tolist()
    unique_texts = list(dict.fromkeys(all_a + all_b))
    print(f"  {len(unique_texts)} unique sentences to embed.")

    # Start with the original columns
    results = df.copy()

    for model_key, model_cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_cfg['display_name']}  (key: {model_key})")
        print(f"{'='*60}")

        embedder = make_embedder(model_cfg)

        print(f"  Embedding {len(unique_texts)} unique sentences...")
        start = time.time()
        vecs = embedder.embed_batch(unique_texts)
        emb_map = dict(zip(unique_texts, vecs))
        print(f"    Done in {time.time()-start:.1f}s")

        print(f"  Computing cosine similarities...")
        sims = []
        for _, row in df.iterrows():
            vec_a = emb_map[row["sentence_a"]]
            vec_b = emb_map[row["sentence_b"]]
            sims.append(round(cosine_similarity(vec_a, vec_b), 6))

        col_name = f"cosine_{model_key}"
        results[col_name] = sims

        # Console summary by category
        results_temp = results.copy()
        results_temp["_sim"] = sims
        summary = results_temp.groupby("category")["_sim"].agg(["mean", "median", "min", "max"]).round(4)
        print(f"\n  {model_cfg['display_name']} — Summary by category:")
        print(summary.to_string())

        embedder.close()
        del embedder, vecs, emb_map
        gc.collect()

    # Write output
    print(f"\n{'='*60}")
    print("Writing results...")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Detailed results
        results.to_excel(writer, sheet_name="Detailed", index=False)

        # Summary by category
        cosine_cols = [f"cosine_{k}" for k in MODELS]
        summary_rows = []
        for col in cosine_cols:
            model_key = col.replace("cosine_", "")
            display = MODELS[model_key]["display_name"]
            grouped = results.groupby(["category", "expected_similarity"])[col].agg(
                ["mean", "median", "std", "min", "max", "count"]
            ).reset_index()
            grouped.insert(0, "model", display)
            grouped.columns = ["model", "category", "expected_similarity",
                               "mean", "median", "std", "min", "max", "count"]
            for c in ["mean", "median", "std", "min", "max"]:
                grouped[c] = grouped[c].round(4)
            summary_rows.append(grouped)

        summary_df = pd.concat(summary_rows, ignore_index=True)
        summary_df.to_excel(writer, sheet_name="Summary by Category", index=False)

    print(f"  Results: {out_path}")
    print(f"{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified semantic similarity comparison tool.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("greek", help="Compare Greek back-translations against originals")
    subparsers.add_parser("all", help="Compare all-language back-translations against originals")
    subparsers.add_parser("test-pairs", help="Compare semantic similarity test pairs")

    args = parser.parse_args()

    if args.command == "greek":
        run_greek()
    elif args.command == "all":
        run_all()
    elif args.command == "test-pairs":
        run_test_pairs()


if __name__ == "__main__":
    main()
