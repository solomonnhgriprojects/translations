#!/usr/bin/env python3
"""
Run genetic-condition predictions on a single file using one of these backends:
  - gpt-4o-mini, gpt-3.5-turbo (OpenAI)
  - gemini-2.0-flash, gemini-3-flash-preview (Google Gemini)
  - llama3.1:8b, ilsp/llama-krikri-8b-instruct:latest (Ollama local)

Supports parallel row processing via --workers flag.

Usage:
  py -3 RunPredictions.py <input.xlsx> --model gemini-2.0-flash --sentence-col forward_translation --out output.xlsx --workers 5
"""

import os
import sys
import time
import json
import re
import argparse
import threading
import concurrent.futures as cf
from typing import List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Prompt (same as Predictions.py) ──────────────────────────────────────────
SYSTEM_MSG = (
    "You are a geneticist. Given a sentence or phrase (which may be in any language), "
    "return EXACTLY five (5) genetic condition names that the text most likely describes, "
    'ordered from most likely to least likely. Your output MUST be a compact JSON array ONLY, e.g. ["A","B","C","D","E"]. '
    "ALWAYS return the condition names in ENGLISH. Do NOT include any extra text. "
    'If fewer than five are identifiable, use "Unknown" so the length is always 5. '
    "Treat each input independently."
)

PRED_COLS = ["Pred1", "Pred2", "Pred3", "Pred4", "Pred5"]


# ── Parsing (reused from Predictions.py) ─────────────────────────────────────
def parse_five_list(out_text: str) -> List[str]:
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

    # Lenient split
    items = re.split(r"[\n,;|/]+", out_text)
    cleaned = []
    for it in items:
        it = it.strip().strip('"').strip()
        it = re.sub(r"^\s*\d+[\).\s-]*", "", it)
        if it:
            cleaned.append(it)
        if len(cleaned) == 5:
            break
    if len(cleaned) == 5:
        return cleaned

    return ["Unknown"] * 5


# ── API callers ──────────────────────────────────────────────────────────────
def call_openai(client, model: str, sentence: str, max_retries: int = 5) -> List[str]:
    prompt = (str(sentence) or "")[:512]
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                top_p=1.0,
                max_tokens=100,
            )
            text = resp.choices[0].message.content.strip()
            return parse_five_list(text)
        except Exception as e:
            if attempt == max_retries:
                print(f"  [WARN] OpenAI failed after {max_retries} retries: {e}", flush=True)
                return ["Unknown"] * 5
            backoff = min(8.0, 0.5 * (2 ** (attempt - 1)))
            time.sleep(backoff)


def strip_code_block(text: str) -> str:
    """Remove ```json ... ``` wrappers that Gemini likes to add."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def call_gemini(client, model_name: str, sentence: str, max_retries: int = 5) -> List[str]:
    prompt = (str(sentence) or "")[:512]
    full_prompt = f"{SYSTEM_MSG}\n\nSentence: {prompt}"
    config = {"temperature": 0.0, "top_p": 1.0, "max_output_tokens": 8192}

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=config,
            )
            text = strip_code_block(response.text)
            return parse_five_list(text)
        except Exception as e:
            if attempt == max_retries:
                print(f"  [WARN] Gemini failed after {max_retries} retries: {e}", flush=True)
                return ["Unknown"] * 5
            backoff = min(8.0, 0.5 * (2 ** (attempt - 1)))
            time.sleep(backoff)


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="Run genetic condition predictions on a single file.")
    ap.add_argument("input_path", help="Path to input .xlsx file.")
    ap.add_argument("--model", required=True, help="Prediction model name")
    ap.add_argument("--sentence-col", required=True, help="Column name containing sentences")
    ap.add_argument("--out", required=True, help="Output .xlsx file path.")
    ap.add_argument("--workers", type=int, default=5, help="Parallel workers for row processing (default: 5)")
    return ap.parse_args()


def main():
    args = parse_args()
    model = args.model

    # Load input
    df = pd.read_excel(args.input_path, header=0)
    sentence_col = args.sentence_col
    if sentence_col not in df.columns:
        print(f"ERROR: column '{sentence_col}' not found. Available: {list(df.columns)}")
        sys.exit(1)

    # Add Pred columns
    for c in PRED_COLS:
        if c not in df.columns:
            df[c] = ""

    # Resume: skip rows that already have Pred1
    need_mask = df["Pred1"].isna() | (df["Pred1"].astype(str).str.strip() == "")

    # If output file exists, load it for resume
    if os.path.exists(args.out):
        existing = pd.read_excel(args.out, header=0)
        if "Pred1" in existing.columns and len(existing) == len(df):
            df = existing
            need_mask = df["Pred1"].isna() | (df["Pred1"].astype(str).str.strip() == "")

    todo_idx = df.index[need_mask].tolist()
    total = len(todo_idx)
    print(f"File: {os.path.basename(args.input_path)} | Model: {model} | Workers: {args.workers}", flush=True)
    print(f"  Rows: {len(df)} | Already done: {len(df) - total} | Remaining: {total}", flush=True)

    if total == 0:
        print("  Nothing to do.", flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.to_excel(args.out, index=False)
        return

    # Initialize API client
    is_gemini = model.startswith("gemini")
    is_ollama = model.startswith("llama") or model.startswith("ilsp/")
    if is_gemini:
        from google import genai
        api_key = os.getenv("GOOGLE_Key")
        if not api_key:
            print("ERROR: GOOGLE_Key not set in .env")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
        call_fn = lambda sentence: call_gemini(client, model, sentence)
    elif is_ollama:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:11434/v1/", api_key="ollama")
        call_fn = lambda sentence: call_openai(client, model, sentence)
    else:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_Key")
        if not api_key:
            print("ERROR: OPENAI_Key not set in .env")
            sys.exit(1)
        client = OpenAI(api_key=api_key)
        call_fn = lambda sentence: call_openai(client, model, sentence)

    # Process rows in parallel
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    start = time.time()
    completed = 0
    lock = threading.Lock()

    def process_row(i):
        sentence = df.at[i, sentence_col]
        return i, call_fn(sentence)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_row, i): i for i in todo_idx}
        for fut in cf.as_completed(futures):
            i, preds = fut.result()
            with lock:
                df.loc[i, PRED_COLS] = preds
                completed += 1
                c = completed

            if c % 10 == 0 or c == total:
                elapsed = time.time() - start
                avg = elapsed / c
                eta = (total - c) * avg
                print(f"  [{c}/{total}] avg {avg:.2f}s/row | ETA {int(eta)}s", flush=True)

            # Checkpoint every 20 rows
            if c % 20 == 0:
                with lock:
                    df.to_excel(args.out, index=False)

    # Final save
    df.to_excel(args.out, index=False)
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
