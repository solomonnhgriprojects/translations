"""
Compute prediction accuracy against Prediction_Correct_Answers.xlsx and
correlate results with PubMed citation counts.

Outputs (per variant):
  1. Accuracy_Summary.xlsx          – top-1/3/5 accuracy for every prediction file (with SE / 95% CI)
  2. Per_Condition_Accuracy.xlsx     – per-disease accuracy across all methods
  3. PubMed_Correlation.xlsx         – correlation between PubMed citations and accuracy
  4. Statistical_Comparisons.xlsx    – pairwise paired t-tests between
                                       prediction models, categories, and translation sources

Usage:
  py -3 AccuracyResults.py
"""

import os
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)  # parent of the Supplementary Scripts/ directory

VARIANTS = [
    ("Greek Data (Original)", "Greek Data (Original)"),
    ("Greek Data (English Inclusions Removed)", "Greek Data (English Inclusions Removed)"),
]

CATEGORIES = [
    "Back Translation Predictions",
    "Forward Translation Predictions",
    "Original Description Predictions",
]

MODEL_FOLDERS = [
    "3.5-turbo",
    "4o-mini",
    "gemini-2.0-flash",
    "gemini-3-flash-preview",
    "krikri-8b",
    "llama3.1-8b",
]

PRED_COLS = ["Pred1", "Pred2", "Pred3", "Pred4", "Pred5"]


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_correct_answers(text):
    """Split semicolon-separated synonyms and normalise for matching."""
    return [s.strip().lower() for s in str(text).split(";") if s.strip()]


def _normalize(text):
    """Lowercase, replace hyphens with spaces, and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().replace("-", " ")).strip()


def is_match(prediction, correct_set, correct_set_normalized):
    """Case-insensitive check of a single prediction against the set of acceptable answers.
    Falls back to hyphen/space-normalised comparison if exact match fails."""
    pred = str(prediction).strip().lower()
    if not pred or pred == "nan" or pred == "unknown":
        return False
    if pred in correct_set:
        return True
    return _normalize(pred) in correct_set_normalized


def compute_topk_hits(row, correct_set, correct_set_normalized, k):
    """Return 1 if any of Pred1..Predk matches the correct set, else 0."""
    for col in PRED_COLS[:k]:
        if is_match(row[col], correct_set, correct_set_normalized):
            return 1
    return 0


# ── main ─────────────────────────────────────────────────────────────────────

def process_variant(variant_label, variant_folder):
    pred_base = os.path.join(BASE, "Greek Data", variant_folder, "Greek Predictions")
    answers_path = os.path.join(pred_base, "Prediction_Correct_Answers.xlsx")

    if not os.path.exists(answers_path):
        print(f"  [SKIP] {answers_path} not found")
        return

    answers_df = pd.read_excel(answers_path)

    # Build lookup: condition_index -> (set of lowered answers, set of normalised answers)
    correct_lookup = {}
    pubmed_lookup = {}
    for _, r in answers_df.iterrows():
        idx = int(r["condition_index"])
        lowered = set(parse_correct_answers(r["Correct answers"]))
        normalized = set(_normalize(s) for s in lowered)
        correct_lookup[idx] = (lowered, normalized)
        pubmed_lookup[idx] = int(r["Numer of PubMed entries"])

    condition_indices = sorted(correct_lookup.keys())

    # Collect rows for the summary table
    summary_rows = []
    # Collect per-condition hits for the detail table (list of dicts)
    per_cond_records = []  # one record per (file, condition)

    for category in CATEGORIES:
        cat_dir = os.path.join(pred_base, category)
        if not os.path.isdir(cat_dir):
            continue
        for model_folder in MODEL_FOLDERS:
            model_dir = os.path.join(cat_dir, model_folder)
            if not os.path.isdir(model_dir):
                continue
            for fname in sorted(os.listdir(model_dir)):
                if not fname.endswith(".xlsx"):
                    continue
                fpath = os.path.join(model_dir, fname)
                try:
                    df = pd.read_excel(fpath)
                except Exception as e:
                    print(f"  [ERR] {fpath}: {e}")
                    continue

                if "Pred1" not in df.columns or "condition_index" not in df.columns:
                    continue

                # Derive the translation source from the filename
                source = fname.replace(".xlsx", "")
                # Strip prediction-model suffix to get the translation source
                for suf in ["_back_tran_" + model_folder.replace(".", "_").replace("-", "_"),
                            "_back_tran_" + model_folder.replace(".", "_"),
                            "_back_tran_" + model_folder,
                            "_forw_tran_" + model_folder.replace(".", "_").replace("-", "_"),
                            "_forw_tran_" + model_folder.replace(".", "_"),
                            "_forw_tran_" + model_folder,
                            "_" + model_folder.replace(".", "_").replace("-", "-"),
                            "_" + model_folder.replace(".", "_"),
                            "_" + model_folder]:
                    if source.endswith(suf):
                        source = source[: -len(suf)]
                        break

                # Determine short category label
                if "Back" in category:
                    cat_short = "Back Translation"
                elif "Forward" in category:
                    cat_short = "Forward Translation"
                else:
                    cat_short = "Original Description"

                top1_hits = 0
                top3_hits = 0
                top5_hits = 0
                n = 0

                for _, row in df.iterrows():
                    ci = int(row["condition_index"])
                    if ci not in correct_lookup:
                        continue
                    cs, cs_norm = correct_lookup[ci]
                    n += 1
                    h1 = compute_topk_hits(row, cs, cs_norm, 1)
                    h3 = compute_topk_hits(row, cs, cs_norm, 3)
                    h5 = compute_topk_hits(row, cs, cs_norm, 5)
                    top1_hits += h1
                    top3_hits += h3
                    top5_hits += h5

                    per_cond_records.append({
                        "condition_index": ci,
                        "category": cat_short,
                        "translation_source": source,
                        "prediction_model": model_folder,
                        "file": fname,
                        "top1_hit": h1,
                        "top3_hit": h3,
                        "top5_hit": h5,
                        "pubmed_entries": pubmed_lookup[ci],
                    })

                if n == 0:
                    continue

                summary_rows.append({
                    "category": cat_short,
                    "translation_source": source,
                    "prediction_model": model_folder,
                    "file": fname,
                    "n_conditions": n,
                    "top1_correct": top1_hits,
                    "top1_accuracy": round(top1_hits / n, 4),
                    "top3_correct": top3_hits,
                    "top3_accuracy": round(top3_hits / n, 4),
                    "top5_correct": top5_hits,
                    "top5_accuracy": round(top5_hits / n, 4),
                })

    # ── 1. Accuracy Summary (with SE and 95% CI) ────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    # Each file has n=40 binary outcomes per condition; SE = sqrt(p*(1-p)/n)
    for metric in ["top1", "top3", "top5"]:
        acc = summary_df[f"{metric}_accuracy"]
        n = summary_df["n_conditions"]
        se = np.sqrt(acc * (1 - acc) / n)
        summary_df[f"{metric}_se"] = se.round(4)
        summary_df[f"{metric}_ci_lo"] = (acc - 1.96 * se).clip(lower=0).round(4)
        summary_df[f"{metric}_ci_hi"] = (acc + 1.96 * se).clip(upper=1).round(4)
    summary_df.sort_values(["category", "translation_source", "prediction_model"], inplace=True)

    # ── 2. Per-Condition Accuracy ────────────────────────────────────────────
    pcond_df = pd.DataFrame(per_cond_records)

    # Aggregate: per condition across ALL files
    cond_agg = (
        pcond_df.groupby("condition_index")
        .agg(
            n_predictions=("top1_hit", "count"),
            top1_correct=("top1_hit", "sum"),
            top3_correct=("top3_hit", "sum"),
            top5_correct=("top5_hit", "sum"),
            pubmed_entries=("pubmed_entries", "first"),
        )
        .reset_index()
    )
    cond_agg["top1_accuracy"] = (cond_agg["top1_correct"] / cond_agg["n_predictions"]).round(4)
    cond_agg["top3_accuracy"] = (cond_agg["top3_correct"] / cond_agg["n_predictions"]).round(4)
    cond_agg["top5_accuracy"] = (cond_agg["top5_correct"] / cond_agg["n_predictions"]).round(4)

    # Add condition name for readability
    cond_names = {int(r["condition_index"]): r["Correct answers"].split(";")[0].strip()
                  for _, r in answers_df.iterrows()}
    cond_agg["condition_name"] = cond_agg["condition_index"].map(cond_names)
    cond_agg = cond_agg[["condition_index", "condition_name", "pubmed_entries",
                          "n_predictions", "top1_correct", "top1_accuracy",
                          "top3_correct", "top3_accuracy", "top5_correct", "top5_accuracy"]]
    cond_agg.sort_values("condition_index", inplace=True)

    # Also aggregate per condition PER category
    cond_by_cat = (
        pcond_df.groupby(["condition_index", "category"])
        .agg(
            n_predictions=("top1_hit", "count"),
            top1_correct=("top1_hit", "sum"),
            top3_correct=("top3_hit", "sum"),
            top5_correct=("top5_hit", "sum"),
            pubmed_entries=("pubmed_entries", "first"),
        )
        .reset_index()
    )
    cond_by_cat["top1_accuracy"] = (cond_by_cat["top1_correct"] / cond_by_cat["n_predictions"]).round(4)
    cond_by_cat["top3_accuracy"] = (cond_by_cat["top3_correct"] / cond_by_cat["n_predictions"]).round(4)
    cond_by_cat["top5_accuracy"] = (cond_by_cat["top5_correct"] / cond_by_cat["n_predictions"]).round(4)
    cond_by_cat["condition_name"] = cond_by_cat["condition_index"].map(cond_names)
    cond_by_cat.sort_values(["condition_index", "category"], inplace=True)

    # ── 3. PubMed Correlation ────────────────────────────────────────────────
    corr_rows = []

    # Overall correlation (across all files, per condition)
    for metric in ["top1_accuracy", "top3_accuracy", "top5_accuracy"]:
        x = cond_agg["pubmed_entries"].values
        y = cond_agg[metric].values
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        corr_rows.append({
            "scope": "Overall",
            "metric": metric,
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 6),
            "spearman_rho": round(sr, 4),
            "spearman_p": round(sp, 6),
            "n_conditions": len(x),
        })

    # Correlation per category
    for cat in cond_by_cat["category"].unique():
        sub = cond_by_cat[cond_by_cat["category"] == cat]
        for metric in ["top1_accuracy", "top3_accuracy", "top5_accuracy"]:
            x = sub["pubmed_entries"].values
            y = sub[metric].values
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            corr_rows.append({
                "scope": cat,
                "metric": metric,
                "pearson_r": round(pr, 4),
                "pearson_p": round(pp, 6),
                "spearman_rho": round(sr, 4),
                "spearman_p": round(sp, 6),
                "n_conditions": len(x),
            })

    # Correlation per prediction model (across all categories)
    for model in pcond_df["prediction_model"].unique():
        sub_model = pcond_df[pcond_df["prediction_model"] == model]
        model_cond = (
            sub_model.groupby("condition_index")
            .agg(
                top1_accuracy=("top1_hit", "mean"),
                top3_accuracy=("top3_hit", "mean"),
                top5_accuracy=("top5_hit", "mean"),
                pubmed_entries=("pubmed_entries", "first"),
            )
            .reset_index()
        )
        for metric in ["top1_accuracy", "top3_accuracy", "top5_accuracy"]:
            x = model_cond["pubmed_entries"].values
            y = model_cond[metric].values
            pr, pp = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
            corr_rows.append({
                "scope": f"Model: {model}",
                "metric": metric,
                "pearson_r": round(pr, 4),
                "pearson_p": round(pp, 6),
                "spearman_rho": round(sr, 4),
                "spearman_p": round(sp, 6),
                "n_conditions": len(x),
            })

    corr_df = pd.DataFrame(corr_rows)

    # ── 4. Pairwise Statistical Comparisons ──────────────────────────────────
    # Strategy: for each grouping factor (prediction model, category,
    # translation source), compute the per-condition mean accuracy across
    # all files in that group.  This gives 40 paired observations per group,
    # enabling paired t-tests for every pair.

    def pairwise_ttest(pcond, group_col, metrics=("top1_hit", "top3_hit", "top5_hit")):
        """Return a DataFrame of pairwise paired t-tests."""
        # Per-condition mean accuracy for each level of group_col
        agg = (
            pcond.groupby([group_col, "condition_index"])
            [list(metrics)]
            .mean()
            .reset_index()
        )
        levels = sorted(agg[group_col].unique())
        rows = []
        for a, b in combinations(levels, 2):
            da = agg[agg[group_col] == a].set_index("condition_index").sort_index()
            db = agg[agg[group_col] == b].set_index("condition_index").sort_index()
            # Align on shared conditions
            shared = da.index.intersection(db.index)
            da, db = da.loc[shared], db.loc[shared]
            for m in metrics:
                diff = da[m].values - db[m].values
                mean_a = da[m].mean()
                mean_b = db[m].mean()
                # t-test needs non-zero variance in differences
                if np.all(diff == 0):
                    stat, p = np.nan, 1.0
                else:
                    stat, p = stats.ttest_rel(da[m].values, db[m].values)
                metric_label = m.replace("_hit", "_accuracy")
                rows.append({
                    "group_a": a,
                    "group_b": b,
                    "metric": metric_label,
                    "mean_a": round(mean_a, 4),
                    "mean_b": round(mean_b, 4),
                    "mean_diff": round(mean_a - mean_b, 4),
                    "t_stat": round(stat, 2) if not np.isnan(stat) else np.nan,
                    "p_value": round(p, 6),
                    "n_conditions": len(shared),
                })
        return pd.DataFrame(rows)

    pw_model = pairwise_ttest(pcond_df, "prediction_model")
    pw_category = pairwise_ttest(pcond_df, "category")
    pw_source = pairwise_ttest(
        pcond_df[pcond_df["category"] != "Original Description"],  # Original has only one source
        "translation_source",
    )

    # Also compare Original vs Back-Translation vs Forward-Translation
    # with variability: per-condition mean ± SD across files within each group
    group_stats_rows = []
    for group_col, label in [("prediction_model", "Prediction Model"),
                              ("category", "Category"),
                              ("translation_source", "Translation Source")]:
        sub = pcond_df if group_col != "translation_source" else pcond_df[pcond_df["category"] != "Original Description"]
        agg = (
            sub.groupby([group_col, "condition_index"])
            [["top1_hit", "top3_hit", "top5_hit"]]
            .mean()
            .reset_index()
        )
        for level in sorted(agg[group_col].unique()):
            ldata = agg[agg[group_col] == level]
            for m, mlabel in [("top1_hit", "top1_accuracy"), ("top3_hit", "top3_accuracy"), ("top5_hit", "top5_accuracy")]:
                vals = ldata[m].values
                group_stats_rows.append({
                    "factor": label,
                    "level": level,
                    "metric": mlabel,
                    "mean": round(np.mean(vals), 4),
                    "std": round(np.std(vals, ddof=1), 4),
                    "se": round(np.std(vals, ddof=1) / np.sqrt(len(vals)), 4),
                    "ci_lo": round(np.mean(vals) - 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)), 4),
                    "ci_hi": round(np.mean(vals) + 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)), 4),
                    "n_conditions": len(vals),
                })
    group_stats_df = pd.DataFrame(group_stats_rows)

    # ── Write outputs ────────────────────────────────────────────────────────
    out_dir = os.path.join(pred_base, "Accuracy Results")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "Accuracy_Summary.xlsx")
    percond_path = os.path.join(out_dir, "Per_Condition_Accuracy.xlsx")
    corr_path = os.path.join(out_dir, "PubMed_Correlation.xlsx")

    with pd.ExcelWriter(summary_path, engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="By File", index=False)

        # Also add aggregated views as extra sheets
        by_model = (
            summary_df.groupby("prediction_model")
            .agg(
                n_files=("file", "count"),
                mean_top1=("top1_accuracy", "mean"),
                mean_top3=("top3_accuracy", "mean"),
                mean_top5=("top5_accuracy", "mean"),
            )
            .reset_index()
        )
        by_model[["mean_top1", "mean_top3", "mean_top5"]] = by_model[["mean_top1", "mean_top3", "mean_top5"]].round(4)
        by_model.to_excel(w, sheet_name="By Prediction Model", index=False)

        by_cat = (
            summary_df.groupby("category")
            .agg(
                n_files=("file", "count"),
                mean_top1=("top1_accuracy", "mean"),
                mean_top3=("top3_accuracy", "mean"),
                mean_top5=("top5_accuracy", "mean"),
            )
            .reset_index()
        )
        by_cat[["mean_top1", "mean_top3", "mean_top5"]] = by_cat[["mean_top1", "mean_top3", "mean_top5"]].round(4)
        by_cat.to_excel(w, sheet_name="By Category", index=False)

        by_source = (
            summary_df.groupby("translation_source")
            .agg(
                n_files=("file", "count"),
                mean_top1=("top1_accuracy", "mean"),
                mean_top3=("top3_accuracy", "mean"),
                mean_top5=("top5_accuracy", "mean"),
            )
            .reset_index()
        )
        by_source[["mean_top1", "mean_top3", "mean_top5"]] = by_source[["mean_top1", "mean_top3", "mean_top5"]].round(4)
        by_source.to_excel(w, sheet_name="By Translation Source", index=False)

        by_cat_model = (
            summary_df.groupby(["category", "prediction_model"])
            .agg(
                n_files=("file", "count"),
                mean_top1=("top1_accuracy", "mean"),
                mean_top3=("top3_accuracy", "mean"),
                mean_top5=("top5_accuracy", "mean"),
            )
            .reset_index()
        )
        by_cat_model[["mean_top1", "mean_top3", "mean_top5"]] = by_cat_model[["mean_top1", "mean_top3", "mean_top5"]].round(4)
        by_cat_model.to_excel(w, sheet_name="By Category x Model", index=False)

    with pd.ExcelWriter(percond_path, engine="openpyxl") as w:
        cond_agg.to_excel(w, sheet_name="Overall", index=False)
        cond_by_cat.to_excel(w, sheet_name="By Category", index=False)

    corr_df.to_excel(corr_path, index=False)

    stat_path = os.path.join(out_dir, "Statistical_Comparisons.xlsx")
    with pd.ExcelWriter(stat_path, engine="openpyxl") as w:
        group_stats_df.to_excel(w, sheet_name="Group Means (with SE)", index=False)
        pw_model.to_excel(w, sheet_name="Pairwise Models", index=False)
        pw_category.to_excel(w, sheet_name="Pairwise Categories", index=False)
        pw_source.to_excel(w, sheet_name="Pairwise Trans Sources", index=False)

    print(f"  -> {summary_path}")
    print(f"  -> {percond_path}")
    print(f"  -> {corr_path}")
    print(f"  -> {stat_path}")

    # Print quick summary to console
    print()
    total = len(summary_df)
    print(f"  Files analysed: {total}")
    print(f"  Overall mean top-1 accuracy: {summary_df['top1_accuracy'].mean():.4f}")
    print(f"  Overall mean top-3 accuracy: {summary_df['top3_accuracy'].mean():.4f}")
    print(f"  Overall mean top-5 accuracy: {summary_df['top5_accuracy'].mean():.4f}")
    print()
    print("  By prediction model:")
    for _, r in by_model.iterrows():
        print(f"    {r['prediction_model']:<25}  top1={r['mean_top1']:.4f}  top3={r['mean_top3']:.4f}  top5={r['mean_top5']:.4f}")
    print()
    print("  By category:")
    for _, r in by_cat.iterrows():
        print(f"    {r['category']:<25}  top1={r['mean_top1']:.4f}  top3={r['mean_top3']:.4f}  top5={r['mean_top5']:.4f}")
    print()
    print("  PubMed correlation (overall):")
    overall_corr = corr_df[corr_df["scope"] == "Overall"]
    for _, r in overall_corr.iterrows():
        print(f"    {r['metric']:<18}  Pearson r={r['pearson_r']:+.4f} (p={r['pearson_p']:.6f})  "
              f"Spearman rho={r['spearman_rho']:+.4f} (p={r['spearman_p']:.6f})")

    # Print pairwise model comparison highlights (top-1 only for brevity)
    print()
    print("  Pairwise model comparisons (top1, paired t-test):")
    t1_models = pw_model[pw_model["metric"] == "top1_accuracy"].copy()
    for _, r in t1_models.iterrows():
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"   {sig} {r['group_a']:<25} vs {r['group_b']:<25}  "
              f"diff={r['mean_diff']:+.4f}  p={r['p_value']:.6f}")

    print()
    print("  Pairwise category comparisons (top1, paired t-test):")
    t1_cats = pw_category[pw_category["metric"] == "top1_accuracy"].copy()
    for _, r in t1_cats.iterrows():
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"   {sig} {r['group_a']:<25} vs {r['group_b']:<25}  "
              f"diff={r['mean_diff']:+.4f}  p={r['p_value']:.6f}")

    return pcond_df, summary_df


# ── Cross-variant comparison ────────────────────────────────────────────────

def compare_variants(pcond_orig, pcond_removed, summary_orig, summary_removed):
    """Compare Original vs English-Inclusions-Removed using paired t-tests
    on per-condition accuracy.  Each prediction file appears in both datasets
    with the same 40 conditions, so they pair naturally."""

    out_path = os.path.join(BASE, "Greek Data",
                            "Dataset Comparison (Predictions).xlsx")

    hit_cols = ["top1_hit", "top3_hit", "top5_hit"]
    metric_labels = ["top1_accuracy", "top3_accuracy", "top5_accuracy"]

    # Tag each dataset
    pcond_orig = pcond_orig.copy()
    pcond_removed = pcond_removed.copy()
    pcond_orig["variant"] = "Original"
    pcond_removed["variant"] = "English Inclusions Removed"

    # ── 1. Overall comparison ────────────────────────────────────────────────
    # Per condition, average across all files in each variant → paired values
    def agg_by_condition(df):
        return (
            df.groupby("condition_index")[hit_cols]
            .mean()
            .reset_index()
            .sort_values("condition_index")
        )

    def align_and_compare(oc, rc):
        """Align two condition-aggregated DataFrames on shared conditions."""
        shared = sorted(set(oc["condition_index"]) & set(rc["condition_index"]))
        oc = oc[oc["condition_index"].isin(shared)].sort_values("condition_index").reset_index(drop=True)
        rc = rc[rc["condition_index"].isin(shared)].sort_values("condition_index").reset_index(drop=True)
        return oc, rc

    orig_cond, rem_cond = align_and_compare(
        agg_by_condition(pcond_orig), agg_by_condition(pcond_removed))

    overall_rows = []
    for hc, ml in zip(hit_cols, metric_labels):
        o = orig_cond[hc].values
        r = rem_cond[hc].values
        diff = o - r
        if np.all(diff == 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = stats.wilcoxon(o, r)
        overall_rows.append({
            "scope": "Overall",
            "metric": ml,
            "mean_original": round(np.mean(o), 4),
            "mean_removed": round(np.mean(r), 4),
            "mean_diff": round(np.mean(diff), 4),
            "sd_diff": round(np.std(diff, ddof=1), 4),
            "t_stat": round(stat, 2) if not np.isnan(stat) else np.nan,
            "p_value": round(p, 6),
            "n_conditions": len(o),
        })

    # ── 2. Per prediction model ──────────────────────────────────────────────
    model_rows = []
    for model in sorted(pcond_orig["prediction_model"].unique()):
        o_sub = pcond_orig[pcond_orig["prediction_model"] == model]
        r_sub = pcond_removed[pcond_removed["prediction_model"] == model]
        oc, rc = align_and_compare(agg_by_condition(o_sub), agg_by_condition(r_sub))
        for hc, ml in zip(hit_cols, metric_labels):
            o = oc[hc].values
            r = rc[hc].values
            diff = o - r
            if np.all(diff == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = stats.ttest_rel(o, r)
            model_rows.append({
                "prediction_model": model,
                "metric": ml,
                "mean_original": round(np.mean(o), 4),
                "mean_removed": round(np.mean(r), 4),
                "mean_diff": round(np.mean(diff), 4),
                "sd_diff": round(np.std(diff, ddof=1), 4),
                "t_stat": round(stat, 2) if not np.isnan(stat) else np.nan,
                "p_value": round(p, 6),
                "n_conditions": len(o),
            })

    # ── 3. Per category ──────────────────────────────────────────────────────
    cat_rows = []
    for cat in sorted(pcond_orig["category"].unique()):
        o_sub = pcond_orig[pcond_orig["category"] == cat]
        r_sub = pcond_removed[pcond_removed["category"] == cat]
        oc, rc = align_and_compare(agg_by_condition(o_sub), agg_by_condition(r_sub))
        for hc, ml in zip(hit_cols, metric_labels):
            o = oc[hc].values
            r = rc[hc].values
            diff = o - r
            if np.all(diff == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = stats.ttest_rel(o, r)
            cat_rows.append({
                "category": cat,
                "metric": ml,
                "mean_original": round(np.mean(o), 4),
                "mean_removed": round(np.mean(r), 4),
                "mean_diff": round(np.mean(diff), 4),
                "sd_diff": round(np.std(diff, ddof=1), 4),
                "t_stat": round(stat, 2) if not np.isnan(stat) else np.nan,
                "p_value": round(p, 6),
                "n_conditions": len(o),
            })

    # ── 4. Per translation source ────────────────────────────────────────────
    source_rows = []
    # Exclude Original Description (identical in both datasets)
    o_trans = pcond_orig[pcond_orig["category"] != "Original Description"]
    r_trans = pcond_removed[pcond_removed["category"] != "Original Description"]
    for src in sorted(o_trans["translation_source"].unique()):
        o_sub = o_trans[o_trans["translation_source"] == src]
        r_sub = r_trans[r_trans["translation_source"] == src]
        if r_sub.empty:
            continue
        oc, rc = align_and_compare(agg_by_condition(o_sub), agg_by_condition(r_sub))
        for hc, ml in zip(hit_cols, metric_labels):
            o = oc[hc].values
            r = rc[hc].values
            diff = o - r
            if np.all(diff == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = stats.ttest_rel(o, r)
            source_rows.append({
                "translation_source": src,
                "metric": ml,
                "mean_original": round(np.mean(o), 4),
                "mean_removed": round(np.mean(r), 4),
                "mean_diff": round(np.mean(diff), 4),
                "sd_diff": round(np.std(diff, ddof=1), 4),
                "t_stat": round(stat, 2) if not np.isnan(stat) else np.nan,
                "p_value": round(p, 6),
                "n_conditions": len(o),
            })

    # ── 5. Per individual file ───────────────────────────────────────────────
    file_rows = []
    for fname in sorted(pcond_orig["file"].unique()):
        o_sub = pcond_orig[pcond_orig["file"] == fname].sort_values("condition_index")
        r_sub = pcond_removed[pcond_removed["file"] == fname].sort_values("condition_index")
        if r_sub.empty or len(o_sub) != len(r_sub):
            continue
        cat = o_sub["category"].iloc[0]
        model = o_sub["prediction_model"].iloc[0]
        source = o_sub["translation_source"].iloc[0]
        for hc, ml in zip(hit_cols, metric_labels):
            o = o_sub[hc].values
            r = r_sub[hc].values
            diff = o - r
            if np.all(diff == 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = stats.ttest_rel(o, r)
            file_rows.append({
                "file": fname,
                "category": cat,
                "prediction_model": model,
                "translation_source": source,
                "metric": ml,
                "mean_original": round(np.mean(o), 4),
                "mean_removed": round(np.mean(r), 4),
                "mean_diff": round(np.mean(diff), 4),
                "n_changed": int(np.sum(diff != 0)),
                "t_stat": round(stat, 2) if not np.isnan(stat) else np.nan,
                "p_value": round(p, 6),
                "n_conditions": len(o),
            })

    # ── 6. Per-condition detail ──────────────────────────────────────────────
    # Show which conditions changed most between variants
    merged = orig_cond.merge(rem_cond, on="condition_index", suffixes=("_orig", "_rem"))
    # Read correct answer names
    ans_path = os.path.join(BASE, "Greek Data", "Greek Data (Original)",
                            "Greek Predictions", "Prediction_Correct_Answers.xlsx")
    answers_df = pd.read_excel(ans_path)
    cond_names = {int(r["condition_index"]): r["Correct answers"].split(";")[0].strip()
                  for _, r in answers_df.iterrows()}
    pubmed = {int(r["condition_index"]): int(r["Numer of PubMed entries"])
              for _, r in answers_df.iterrows()}

    cond_detail_rows = []
    for _, row in merged.iterrows():
        ci = int(row["condition_index"])
        for hc, ml in zip(hit_cols, metric_labels):
            o = row[f"{hc}_orig"]
            r = row[f"{hc}_rem"]
            cond_detail_rows.append({
                "condition_index": ci,
                "condition_name": cond_names.get(ci, ""),
                "pubmed_entries": pubmed.get(ci, 0),
                "metric": ml,
                "mean_original": round(o, 4),
                "mean_removed": round(r, 4),
                "diff": round(o - r, 4),
            })
    cond_detail_df = pd.DataFrame(cond_detail_rows)

    # ── Write ────────────────────────────────────────────────────────────────
    overall_df = pd.DataFrame(overall_rows)
    model_df = pd.DataFrame(model_rows)
    cat_df = pd.DataFrame(cat_rows)
    source_df = pd.DataFrame(source_rows)
    file_df = pd.DataFrame(file_rows)

    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        overall_df.to_excel(w, sheet_name="Overall", index=False)
        model_df.to_excel(w, sheet_name="By Prediction Model", index=False)
        cat_df.to_excel(w, sheet_name="By Category", index=False)
        source_df.to_excel(w, sheet_name="By Translation Source", index=False)
        file_df.to_excel(w, sheet_name="By File", index=False)
        cond_detail_df.to_excel(w, sheet_name="Per Condition Detail", index=False)

    print(f"  -> {out_path}")

    # Console summary
    print()
    print("  Overall (Original vs English Inclusions Removed):")
    for _, r in overall_df.iterrows():
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"   {sig} {r['metric']:<18}  orig={r['mean_original']:.4f}  "
              f"removed={r['mean_removed']:.4f}  diff={r['mean_diff']:+.4f}  "
              f"p={r['p_value']:.6f}")

    print()
    print("  By prediction model (top1 only):")
    for _, r in pd.DataFrame(model_rows).query("metric == 'top1_accuracy'").iterrows():
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"   {sig} {r['prediction_model']:<25}  orig={r['mean_original']:.4f}  "
              f"removed={r['mean_removed']:.4f}  diff={r['mean_diff']:+.4f}  "
              f"p={r['p_value']:.6f}")

    print()
    print("  By category (top1 only):")
    for _, r in pd.DataFrame(cat_rows).query("metric == 'top1_accuracy'").iterrows():
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"   {sig} {r['category']:<25}  orig={r['mean_original']:.4f}  "
              f"removed={r['mean_removed']:.4f}  diff={r['mean_diff']:+.4f}  "
              f"p={r['p_value']:.6f}")


def main():
    variant_data = {}
    for label, folder in VARIANTS:
        print(f"\n{'=' * 80}")
        print(f"  {label}")
        print(f"{'=' * 80}")
        result = process_variant(label, folder)
        if result is not None:
            variant_data[label] = result

    # Cross-variant comparison
    if len(variant_data) == 2:
        labels = [v[0] for v in VARIANTS]
        pcond_orig, summary_orig = variant_data[labels[0]]
        pcond_removed, summary_removed = variant_data[labels[1]]

        print(f"\n{'=' * 80}")
        print("  Dataset Comparison: Original vs English Inclusions Removed")
        print(f"{'=' * 80}")
        compare_variants(pcond_orig, pcond_removed, summary_orig, summary_removed)


if __name__ == "__main__":
    main()
