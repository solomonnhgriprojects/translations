#!/usr/bin/env python3
"""
Bulk genetic-condition inference with OpenAI chat models.

- Input: CSV (.csv) or Excel (.xlsx/.xls) with sentences in a specified column.
- Output: adds Pred1..Pred5 columns (top-5 conditions in EN), checkpoints every N rows,
          and writes final file. Resume-safe (skips rows with Pred1 already filled).

Models: gpt-4o-mini (recommended for bulk), gpt-3.5-turbo (cheap baseline).
Set your API key as an env var: OPENAI_API_KEY
"""

import os
import sys
import time
import json
import uuid
import math
import argparse
import threading
import random
import re
import concurrent.futures as cf
from typing import List, Union

import pandas as pd
from openai import OpenAI


SYSTEM_MSG = (
    "You are a geneticist. Given a sentence or phrase (which may be in any language), "
    "return EXACTLY five (5) genetic condition names that the text most likely describes, "
    'ordered from most likely to least likely. Your output MUST be a compact JSON array ONLY, e.g. ["A","B","C","D","E"]. '
    "ALWAYS return the condition names in ENGLISH. Do NOT include any extra text. "
    'If fewer than five are identifiable, use "Unknown" so the length is always 5. '
    "Treat each input independently."
)

PRED_COLS = ["Pred1", "Pred2", "Pred3", "Pred4", "Pred5"]


def parse_args():
    ap = argparse.ArgumentParser(
        description="Bulk inference of genetic condition names from sentences."
    )
    ap.add_argument("input_path", help="Path to input file (.csv or .xlsx/.xls).")
    ap.add_argument(
        "--model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"],
        help="OpenAI chat model to use (default: gpt-4o-mini).",
    )
    ap.add_argument(
        "--sentence-col",
        default="3",
        help="Sentence column (0-based index or column name). Default: 3 = Column D.",
    )
    ap.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Parallel workers (threads). 3–4 is a good starting point.",
    )
    ap.add_argument(
        "--target-rps",
        type=float,
        default=2.8,
        help="Global requests/second across all threads (rate limiter).",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=250,
        help="Checkpoint frequency (rows). Writes/overwrites partial output after every N completions.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=40,
        help="Max output tokens per call (keep modest for speed/cost).",
    )
    ap.add_argument(
        "--trim-input",
        type=int,
        default=512,
        help="Trim each input sentence to this many characters before sending.",
    )
    ap.add_argument(
        "--out-dir",
        default=".",
        help="Directory for outputs (partial + final). Default: current directory.",
    )
    ap.add_argument(
        "--excel-output",
        action="store_true",
        help="Write outputs as .xlsx instead of .csv (avoids mojibake in Excel).",
    )
    return ap.parse_args()


def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, header=0)
    elif ext == ".csv":
        # strict UTF-8 read (fail fast on encoding errors)
        return pd.read_csv(path, header=0, encoding="utf-8", encoding_errors="strict")
    else:
        raise ValueError(f"Unsupported input extension: {ext}")


def get_sentence_column(df: pd.DataFrame, spec: str) -> str:
    # spec can be an integer index (as string) or a column name
    if spec.isdigit() or (spec.startswith("-") and spec[1:].isdigit()):
        idx = int(spec)
        if idx < 0 or idx >= len(df.columns):
            raise IndexError(f"sentence-col index {idx} out of range (0..{len(df.columns)-1})")
        return df.columns[idx]
    else:
        if spec not in df.columns:
            raise KeyError(f"sentence-col '{spec}' not in columns: {list(df.columns)}")
        return spec


def ensure_pred_columns(df: pd.DataFrame):
    for c in PRED_COLS:
        if c not in df.columns:
            df[c] = ""


def parse_five_list(out_text: str) -> List[str]:
    """
    Robustly parse EXACTLY 5 strings from model output.
    Primary: strict JSON list of 5 strings.
    Fallbacks: first JSON array substring; lenient delimiter splits (handles 3.5 quirks).
    """
    out_text = (out_text or "").strip()

    # Strict JSON
    try:
        arr = json.loads(out_text)
        if isinstance(arr, list) and len(arr) == 5 and all(isinstance(x, str) for x in arr):
            return [x.strip() for x in arr]
    except Exception:
        pass

    # First JSON array substring
    m = re.search(r"\[.*\]", out_text, flags=re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list) and len(arr) == 5 and all(isinstance(x, str) for x in arr):
                return [x.strip() for x in arr]
        except Exception:
            pass

    # Lenient split (handles numbered lists etc.)
    items = re.split(r"[\n,;|/]+", out_text)
    cleaned = []
    for it in items:
        it = it.strip().strip('"').strip()
        it = re.sub(r"^\s*\d+[\).\s-]*", "", it)  # drop "1) ", "2. " prefixes
        if it:
            cleaned.append(it)
        if len(cleaned) == 5:
            break
    if len(cleaned) == 5:
        return cleaned

    return ["Unknown"] * 5


class RateLimiter:
    """Simple global RPS limiter shared across threads."""
    def __init__(self, rps: float):
        self.min_gap = 1.0 / max(0.01, rps)
        self.last = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            wait = (self.last + self.min_gap) - now
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()


def call_model(
    client: OpenAI,
    model: str,
    system_msg: str,
    sentence: str,
    max_tokens: int,
    limiter: RateLimiter,
    trim_input: int,
    max_retries: int = 5,
) -> List[str]:
    prompt = (str(sentence) or "")[:trim_input]

    for attempt in range(1, max_retries + 1):
        try:
            limiter.wait()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=max_tokens,
                user=str(uuid.uuid4()),
                timeout=30,
            )
            text = resp.choices[0].message.content.strip()
            return parse_five_list(text)
        except Exception:
            if attempt == max_retries:
                return ["Unknown"] * 5
            # Jittered exponential backoff (handles 429/5xx/transient net)
            backoff = min(8.0, 0.5 * (2 ** (attempt - 1)) + random.random() * 0.5)
            time.sleep(backoff)


def save_dataframe(df: pd.DataFrame, path: str, excel: bool):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if excel:
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def main():
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Please set OPENAI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load
    df = load_dataframe(args.input_path)
    sentence_col = get_sentence_column(df, str(args.sentence_col))
    ensure_pred_columns(df)

    base = os.path.splitext(os.path.basename(args.input_path))[0]
    ext = ".xlsx" if args.excel_output else ".csv"
    partial_path = os.path.join(args.out_dir, f"{base}_predictions_partial{ext}")
    final_path = os.path.join(args.out_dir, f"{base}_predictions{ext}")

    # Figure out remaining work (resume-safe)
    need_mask = (~df["Pred1"].astype(str).str.len().astype(bool)) | (df["Pred1"].isna())
    todo_idx = df.index[need_mask].tolist()

    random.seed(42)
    random.shuffle(todo_idx)  # mix easy/hard for smoother perceived speed

    total = len(todo_idx)
    print(f"Columns: {list(df.columns)}")
    print(f"Using sentence column: {sentence_col}")
    print(f"Already processed: {len(df) - total} | Remaining: {total}")
    if total == 0:
        print("Nothing to do. Writing final file (unmodified) …")
        save_dataframe(df, final_path, args.excel_output)
        print(f"Done: {final_path}")
        return

    limiter = RateLimiter(args.target_rps)
    start = time.time()
    completed = 0
    write_lock = threading.Lock()

    def process_one(i: int):
        try:
            preds = call_model(
                client=client,
                model=args.model,
                system_msg=SYSTEM_MSG,
                sentence=df.at[i, sentence_col],
                max_tokens=args.max_tokens,
                limiter=limiter,
                trim_input=args.trim_input,
            )
            df.loc[i, PRED_COLS] = preds
        except Exception:
            df.loc[i, PRED_COLS] = ["Unknown"] * 5

    def save_partial():
        with write_lock:
            save_dataframe(df, partial_path, args.excel_output)

    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(process_one, i) for i in todo_idx]
        for fut in cf.as_completed(futures):
            completed += 1
            # Progress log
            if (completed % 50 == 0) or (completed == total):
                avg = (time.time() - start) / max(1, completed)
                eta = (total - completed) * avg
                print(f"[{completed}/{total}] avg {avg:.2f}s | ETA {int(eta/60)}m")

            # Checkpoint every N rows
            if (completed % args.save_every == 0) or (completed == total):
                save_partial()

    # Final save
    save_dataframe(df, final_path, args.excel_output)
    print(f"\nDone. Final written to: {final_path}")


if __name__ == "__main__":
    main()
