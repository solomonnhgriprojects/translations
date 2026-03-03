#!/usr/bin/env python3
"""
Translate sentences between English and Greek using LLM backends.

Supports forward translation (English -> Greek) and back-translation (Greek -> English).

Backends:
  - ollama: Local Ollama models via OpenAI-compatible API (multi-threaded)
  - gemini: Google Gemini models via genai API (requires GOOGLE_Key in .env)

Backend is auto-detected from the model name (gemini-* -> gemini, otherwise -> ollama).
Override with --backend.

Usage examples
--------------
# Forward translate (English -> Greek):
python Translate.py forward "data/Original_Sentences.xlsx" --model gemini-2.0-flash
python Translate.py forward "data/Original_Sentences.xlsx" --model llama3.1:8b --eirini

# Back-translate (Greek -> English):
python Translate.py back "data/forward_translation.xlsx" --model gemini-2.0-flash
python Translate.py back "data/forward_translation.xlsx" --model llama3.1:8b
"""

import os
import sys
import time
import uuid
import argparse
import threading
import random
import concurrent.futures as cf

import pandas as pd


# ---------------------------------------------------------------------------
# Prompt definitions
# ---------------------------------------------------------------------------

FORWARD_INSTRUCTION = (
    "Translate the following English text into Greek. "
    "Return ONLY the Greek translation."
)

BACK_INSTRUCTION = (
    "Translate the following Greek text into English. "
    "Return ONLY the English translation."
)

EIRINI_REFERENCE = (
    '''Provided for reference are the following translations:

Additional descriptions

Genetic conditions:

Alport syndrome (MIM #301050)

\u03a3\u03cd\u03bd\u03b4\u03c1\u03bf\u03bc\u03bf Alport


Adolescent with hematuria, sensorineural deafness, and anterior lenticonus

\u0388\u03c6\u03b7\u03b2\u03bf\u03c2 \u03bc\u03b5 \u03b1\u03b9\u03bc\u03b1\u03c4\u03bf\u03c5\u03c1\u03af\u03b1, \u03bd\u03b5\u03c5\u03c1\u03bf\u03b1\u03b9\u03c3\u03b8\u03b7\u03c4\u03ae\u03c1\u03b9\u03b1 \u03b2\u03b1\u03c1\u03b7\u03ba\u03bf\u0390\u03b1 /\u03ba\u03ce\u03c6\u03c9\u03c3\u03b7 \u03ba\u03b1\u03b9 \u03c0\u03c1\u03cc\u03c3\u03b8\u03b9\u03bf \u03c6\u03b1\u03ba\u03cc\u03ba\u03c9\u03bd\u03bf.



Thrombocytopenia-absent radius syndrome (MIM #274000):

\u03a3\u03cd\u03bd\u03b4\u03c1\u03bf\u03bc\u03bf \u03b8\u03c1\u03bf\u03bc\u03b2\u03bf\u03c0\u03b5\u03bd\u03af\u03b1\u03c2 \u03bc\u03b5 \u03b1\u03c0\u03bf\u03c5\u03c3\u03af\u03b1 \u03ba\u03b5\u03c1\u03ba\u03af\u03b4\u03b1\u03c2


Neonate with bilateral radial aplasia but normal thumbs also found to have thrombocytopenia

\u039d\u03b5\u03bf\u03b3\u03bd\u03cc \u03bc\u03b5 \u03ba\u03b5\u03c1\u03ba\u03b9\u03b4\u03b9\u03ba\u03ae \u03b1\u03c0\u03bb\u03b1\u03c3\u03af\u03b1 \u03ac\u03bc\u03c6\u03c9 \u03b1\u03bb\u03bb\u03ac \u03bc\u03b5 \u03c6\u03c5\u03c3\u03b9\u03bf\u03bb\u03bf\u03b3\u03b9\u03ba\u03bf\u03cd\u03c2 (normal) \u03b1\u03bd\u03c4\u03af\u03c7\u03b5\u03b9\u03c1\u03b5\u03c2, \u03ba\u03b1\u03b9 \u03c3\u03c5\u03bd\u03bf\u03b4\u03cc \u03b8\u03c1\u03bf\u03bc\u03b2\u03bf\u03c0\u03b5\u03bd\u03af\u03b1.

\u039d\u03b5\u03bf\u03b3\u03bd\u03cc \u03bc\u03b5 \u03b1\u03bc\u03c6o\u03c4\u03b5\u03c1\u03cc\u03c0\u03bb\u03b5\u03c5\u03c1\u03b7 \u03ba\u03b5\u03c1\u03ba\u03b9\u03b4\u03b9\u03ba\u03ae \u03b1\u03c0\u03bb\u03b1\u03c3\u03af\u03b1, \u03b1\u03bb\u03bb\u03ac \u03c0\u03b1\u03c1\u03cc\u03bd\u03c4\u03b5\u03c2 (present) \u03b1\u03bd\u03c4\u03af\u03c7\u03b5\u03b9\u03c1\u03b5\u03c2, \u03ba\u03b1\u03b9 \u03c3\u03c5\u03bd\u03bf\u03b4\u03cc \u03b8\u03c1\u03bf\u03bc\u03b2\u03bf\u03c0\u03b5\u03bd\u03af\u03b1.



Alternating hemiplegia of childhood (MIM $104290)

\u0395\u03c0\u03b1\u03bb\u03bb\u03ac\u03c3\u03c3\u03bf\u03c5\u03c3\u03b1 \u03b7\u03bc\u03b9\u03c0\u03bb\u03b7\u03b3\u03af\u03b1 \u03c4\u03b7\u03c2 \u03c0\u03b1\u03b9\u03b4\u03b9\u03ba\u03ae\u03c2 \u03b7\u03bb\u03b9\u03ba\u03af\u03b1\u03c2


Young child with episodes of daytime hemiplegia, who also has dystonic movements and choreoathetosis, later noted to have developmental delay and nystagmus

\u039d\u03ae\u03c0\u03b9\u03bf (\u03bd\u03b5\u03b1\u03c1\u03cc \u03c0\u03b1\u03b9\u03b4\u03af) \u03bc\u03b5 \u03c0\u03b1\u03c1\u03bf\u03be\u03c5\u03c3\u03bc\u03b9\u03ba\u03ac \u03b5\u03c0\u03b5\u03b9\u03c3\u03cc\u03b4\u03b9\u03b1 \u03b7\u03bc\u03b9\u03c0\u03bb\u03b7\u03b3\u03af\u03b1\u03c2 \u03ba\u03b1\u03c4\u03ac \u03c4\u03b7 \u03b4\u03b9\u03ac\u03c1\u03ba\u03b5\u03b9\u03b1 \u03c4\u03b7\u03c2 \u03bc\u03ad\u03c1\u03b1\u03c2, \u03c0\u03bf\u03c5 \u03c3\u03c5\u03bd\u03bf\u03b4\u03b5\u03cd\u03bf\u03bd\u03c4\u03b1\u03b9 \u03b1\u03c0\u03cc \u03ba\u03b9\u03bd\u03b7\u03c4\u03b9\u03ba\u03ad\u03c2 \u03b4\u03b9\u03b1\u03c4\u03b1\u03c1\u03b1\u03c7\u03ad\u03c2 (\u03b4\u03c5\u03c3\u03c4\u03bf\u03bd\u03af\u03b1 \u03ae \u03c7\u03bf\u03c1\u03b5\u03b9\u03bf\u03b1\u03b8\u03ad\u03c4\u03c9\u03c3\u03b7), \u03ba\u03b1\u03b9 \u03c3\u03c4\u03b7\u03bd \u03c0\u03bf\u03c1\u03b5\u03af\u03b1 \u03b5\u03bc\u03c6\u03b1\u03bd\u03af\u03b6\u03b5\u03b9 \u03b1\u03bd\u03b1\u03c0\u03c4\u03c5\u03be\u03b9\u03b1\u03ba\u03ae \u03ba\u03b1\u03b8\u03c5\u03c3\u03c4\u03ad\u03c1\u03b7\u03c3\u03b7 (\u03bc\u03b1\u03b8\u03b7\u03c3\u03b9\u03b1\u03ba\u03ad\u03c2 \u03b4\u03c5\u03c3\u03ba\u03bf\u03bb\u03af\u03b5\u03c2) \u03ba\u03b1\u03b9 \u03bd\u03c5\u03c3\u03c4\u03b1\u03b3\u03bc\u03cc. \u0397 \u03b7\u03bc\u03b9\u03c0\u03bb\u03b7\u03b3\u03af\u03b1 \u03bc\u03c0\u03bf\u03c1\u03b5\u03af \u03bd\u03b1 \u03b5\u03bc\u03c6\u03b1\u03bd\u03af\u03b6\u03b5\u03c4\u03b1\u03b9 \u03c3\u03c4\u03b7\u03bd \u03b1\u03c1\u03b9\u03c3\u03c4\u03b5\u03c1\u03ae \u03ae \u03b4\u03b5\u03be\u03b9\u03ac \u03c0\u03bb\u03b5\u03c5\u03c1\u03ac \u03c4\u03bf\u03c5 \u03c3\u03ce\u03bc\u03b1\u03c4\u03bf\u03c2 \u03ba\u03b1\u03b9 \u03bd\u03b1 \u03b5\u03bd\u03b1\u03bb\u03bb\u03ac\u03c3\u03c3\u03b5\u03c4\u03b1\u03b9 \u03bc\u03b5\u03c4\u03b1\u03be\u03cd \u03c4\u03c9\u03bd \u03b5\u03c0\u03b5\u03b9\u03c3\u03bf\u03b4\u03af\u03c9\u03bd.



Bardet-Biedl syndrome (MIM #209900)

\u03a3\u03cd\u03bd\u03b4\u03c1\u03bf\u03bc\u03bf Bardet-Biedl


Young adult with history of retinal dystrophy, learning disabilities, and renal  insufficiency, noted on examination to have obesity and postaxial polydactyly

\u039d\u03b5\u03b1\u03c1\u03cc\u03c2/\u03ae (for young man or woman) \u03b5\u03bd\u03ae\u03bb\u03b9\u03ba\u03b1\u03c2 \u03bc\u03b5 \u03b9\u03c3\u03c4\u03bf\u03c1\u03b9\u03ba\u03cc \u03b4\u03c5\u03c3\u03c4\u03c1\u03bf\u03c6\u03af\u03b1\u03c2 \u03c4\u03bf\u03c5 \u03b1\u03bc\u03c6\u03b9\u03b2\u03bb\u03b7\u03c3\u03c4\u03c1\u03bf\u03b5\u03b9\u03b4\u03bf\u03cd\u03c2 (\u03bc\u03b5\u03bb\u03b1\u03b3\u03c7\u03c1\u03c9\u03c3\u03c4\u03b9\u03ba\u03ae \u03b1\u03bc\u03c6\u03b9\u03b2\u03bb\u03b7\u03c3\u03c4\u03c1\u03bf\u03b5\u03b9\u03b4\u03bf\u03c0\u03ac\u03b8\u03b5\u03b9\u03b1), \u03bc\u03b1\u03b8\u03b7\u03c3\u03b9\u03b1\u03ba\u03ad\u03c2 \u03b4\u03c5\u03c3\u03ba\u03bf\u03bb\u03af\u03b5\u03c2 \u03ba\u03b1\u03b9 \u03bd\u03b5\u03c6\u03c1\u03b9\u03ba\u03ae \u03b4\u03c5\u03c3\u03bb\u03b5\u03b9\u03c4\u03bf\u03c5\u03c1\u03b3\u03af\u03b1/\u03b1\u03bd\u03b5\u03c0\u03ac\u03c1\u03ba\u03b5\u03b9\u03b1, \u03c0\u03bf\u03c5 \u03c3\u03c4\u03b7\u03bd \u03ba\u03bb\u03b9\u03bd\u03b9\u03ba\u03ae \u03b5\u03be\u03ad\u03c4\u03b1\u03c3\u03b7 \u03b5\u03bc\u03c6\u03b1\u03bd\u03af\u03b6\u03b5\u03b9 \u03c0\u03b1\u03c7\u03c5\u03c3\u03b1\u03c1\u03ba\u03af\u03b1 \u03ba\u03b1\u03b9 \u03bc\u03b5\u03c4\u03b1-\u03b1\u03be\u03bf\u03bd\u03b9\u03ba\u03ae \u03c0\u03bf\u03bb\u03c5\u03b4\u03b1\u03ba\u03c4\u03c5\u03bb\u03af\u03b1.



Vascular Ehlers-Danlos syndrome (MIM #130050)

\u0391\u03b3\u03b3\u03b5\u03b9\u03b1\u03ba\u03cc \u03c3\u03cd\u03bd\u03b4\u03c1\u03bf\u03bc\u03bf Ehlers-Danlos


Woman with history of easy bruising who has uterine rupture during pregnancy

\u0393\u03c5\u03bd\u03b1\u03af\u03ba\u03b1 \u03bc\u03b5 \u03b9\u03c3\u03c4\u03bf\u03c1\u03b9\u03ba\u03cc \u03b5\u03cd\u03ba\u03bf\u03bb\u03c9\u03bd \u03b5\u03ba\u03c7\u03c5\u03bc\u03ce\u03c3\u03b5\u03c9\u03bd \u03ba\u03b1\u03b9 \u03c1\u03ae\u03be\u03b7 \u03c4\u03b7\u03c2 \u03bc\u03ae\u03c4\u03c1\u03b1\u03c2 \u03ba\u03b1\u03c4\u03ac \u03c4\u03b7\u03bd \u03b5\u03b3\u03ba\u03c5\u03bc\u03bf\u03c3\u03cd\u03bd\u03b7.


Control descriptions:


Man sends a long email to his relatives to let them know about his upcoming vacation plans

\u0386\u03bd\u03b4\u03c1\u03b1\u03c2 \u03c3\u03c4\u03ad\u03bb\u03bd\u03b5\u03b9 \u03ad\u03bd\u03b1 \u03bc\u03b1\u03ba\u03c1\u03bf\u03c3\u03ba\u03b5\u03bb\u03ad\u03c2 \u03b7\u03bb\u03b5\u03ba\u03c4\u03c1\u03bf\u03bd\u03b9\u03ba\u03cc \u03bc\u03ae\u03bd\u03c5\u03bc\u03b1 \u03c3\u03c4\u03bf\u03c5\u03c2 \u03c3\u03c5\u03b3\u03b3\u03b5\u03bd\u03b5\u03af\u03c2 \u03c4\u03bf\u03c5 \u03b3\u03b9\u03b1 \u03bd\u03b1 \u03c4\u03bf\u03c5\u03c2 \u03b5\u03bd\u03b7\u03bc\u03b5\u03c1\u03ce\u03c3\u03b5\u03b9 \u03b3\u03b9\u03b1 \u03c4\u03b1 \u03c3\u03c7\u03ad\u03b4\u03b9\u03b1 \u03bc\u03b5\u03bb\u03bb\u03bf\u03bd\u03c4\u03b9\u03ba\u03ce\u03bd \u03b4\u03b9\u03b1\u03ba\u03bf\u03c0\u03ce\u03bd \u03c4\u03bf\u03c5.


Toddler with a fever and cough has to stay home from daycare for the rest of the week

\u039d\u03ae\u03c0\u03b9\u03bf \u03bc\u03b5 \u03b5\u03bc\u03c0\u03cd\u03c1\u03b5\u03c4\u03bf \u03ba\u03b1\u03b9 \u03b2\u03ae\u03c7\u03b1 \u03c0\u03c1\u03ad\u03c0\u03b5\u03b9 \u03bd\u03b1 \u03c0\u03b1\u03c1\u03b1\u03bc\u03b5\u03af\u03bd\u03b5\u03b9 \u03c3\u03c4\u03bf \u03c3\u03c0\u03af\u03c4\u03b9 \u03c4\u03bf\u03c5 \u03b1\u03bd\u03c4\u03af \u03b3\u03b9\u03b1 \u03c4\u03bf\u03bd \u03c0\u03b1\u03b9\u03b4\u03b9\u03ba\u03cc \u03c3\u03c4\u03b1\u03b8\u03bc\u03cc \u03b3\u03b9\u03b1 \u03c4\u03bf \u03c5\u03c0\u03cc\u03bb\u03bf\u03b9\u03c0\u03bf \u03c4\u03b7\u03c2 \u03b2\u03b4\u03bf\u03bc\u03ac\u03b4\u03b1\u03c2.


Young woman with a receipt for purchasing office supplies using the company credit card

\u039d\u03b5\u03b1\u03c1\u03ae \u03b3\u03c5\u03bd\u03b1\u03af\u03ba\u03b1 \u03c0\u03c1\u03bf\u03c3\u03ba\u03bf\u03bc\u03af\u03b6\u03b5\u03b9 \u03b1\u03c0\u03cc\u03b4\u03b5\u03b9\u03be\u03b7 \u03b1\u03b3\u03bf\u03c1\u03ac\u03c2 \u03c0\u03c1\u03bf\u03ca\u03cc\u03bd\u03c4\u03c9\u03bd/\u03b5\u03b9\u03b4\u03ce\u03bd \u03b3\u03c1\u03b1\u03c6\u03b5\u03af\u03bf\u03c5 \u03bc\u03b5 \u03c4\u03b7\u03bd \u03c0\u03b9\u03c3\u03c4\u03c9\u03c4\u03b9\u03ba\u03ae \u03ba\u03ac\u03c1\u03c4\u03b1 \u03c4\u03b7\u03c2 \u03b5\u03c4\u03b1\u03b9\u03c1\u03b5\u03af\u03b1\u03c2.


Child who has to return two books to the local library by the end of the month

\u03a0\u03b1\u03b9\u03b4\u03af \u03c0\u03bf\u03c5 \u03c0\u03c1\u03ad\u03c0\u03b5\u03b9 \u03bd\u03b1 \u03b5\u03c0\u03b9\u03c3\u03c4\u03c1\u03ad\u03c8\u03b5\u03b9 \u03b4\u03cd\u03bf \u03b2\u03b9\u03b2\u03bb\u03af\u03b1 \u03c3\u03c4\u03b7\u03bd \u03c4\u03bf\u03c0\u03b9\u03ba\u03ae \u03b2\u03b9\u03b2\u03bb\u03b9\u03bf\u03b8\u03ae\u03ba\u03b7 \u03bc\u03ad\u03c7\u03c1\u03b9 \u03c4\u03bf \u03c4\u03ad\u03bb\u03bf\u03c2 \u03c4\u03bf\u03c5 \u03bc\u03ae\u03bd\u03b1.


Adolescent with glasses goes to the eye doctor to check his prescription

\u0388\u03c6\u03b7\u03b2\u03bf\u03c2/\u03b7 \u03bc\u03b5 \u03b3\u03c5\u03b1\u03bb\u03b9\u03ac \u03c0\u03b7\u03b3\u03b1\u03af\u03bd\u03b5\u03b9 \u03c3\u03c4\u03bf\u03bd \u03bf\u03c6\u03b8\u03b1\u03bb\u03bc\u03af\u03b1\u03c4\u03c1\u03bf \u03bd\u03b1 \u03b5\u03bb\u03ad\u03b3\u03be\u03b5\u03b9 \u03c4\u03b7 \u03c3\u03c5\u03bd\u03c4\u03b1\u03b3\u03ae \u03c4\u03bf\u03c5.'''
)

GEMINI_MODEL_NAMES = {
    "gemini-2.0-flash": "Gemini_2.0_Flash",
    "gemini-3-flash-preview": "Gemini_3_Flash",
}


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Translate sentences between English and Greek via Ollama or Gemini."
    )
    sub = ap.add_subparsers(dest="direction", required=True)

    # Forward translation
    fwd = sub.add_parser("forward", help="Translate English -> Greek")
    fwd.add_argument("input_path", help="Path to input file (.csv or .xlsx/.xls).")
    fwd.add_argument("--model", default="llama3.1:8b", help="Model to use.")
    fwd.add_argument("--backend", choices=["ollama", "gemini"], default=None)
    fwd.add_argument("--eirini", action="store_true",
                      help="Include Eirini expert reference translations in the prompt.")
    fwd.add_argument("--sentence-col", default="original_sentence",
                      help="Column containing English text (default: original_sentence).")
    fwd.add_argument("--out-name", default=None, help="Base name for output file (without extension).")
    fwd.add_argument("--out-dir", default=None, help="Directory for outputs.")
    fwd.add_argument("--max-workers", type=int, default=3)
    fwd.add_argument("--target-rps", type=float, default=2.8)
    fwd.add_argument("--save-every", type=int, default=10)
    fwd.add_argument("--max-tokens", type=int, default=512)
    fwd.add_argument("--trim-input", type=int, default=1024)

    # Back-translation
    bk = sub.add_parser("back", help="Translate Greek -> English")
    bk.add_argument("input_path", help="Path to input file (.csv or .xlsx/.xls).")
    bk.add_argument("--model", default="llama3.1:8b", help="Model to use.")
    bk.add_argument("--backend", choices=["ollama", "gemini"], default=None)
    bk.add_argument("--sentence-col", default="forward_translation",
                     help="Column containing Greek text (default: forward_translation).")
    bk.add_argument("--out-dir", default=None, help="Directory for outputs.")
    bk.add_argument("--max-workers", type=int, default=3)
    bk.add_argument("--target-rps", type=float, default=2.8)
    bk.add_argument("--save-every", type=int, default=10)
    bk.add_argument("--max-tokens", type=int, default=512)
    bk.add_argument("--trim-input", type=int, default=1024)

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, header=0)
    elif ext == ".csv":
        return pd.read_csv(path, header=0, encoding="utf-8", encoding_errors="strict")
    else:
        raise ValueError(f"Unsupported input extension: {ext}")


def save_dataframe(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_excel(path, index=False)


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

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


def call_ollama(
    client,
    model: str,
    sentence: str,
    system_msg: str,
    max_tokens: int,
    limiter: RateLimiter,
    trim_input: int,
    max_retries: int = 5,
) -> str:
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
                timeout=60,
            )
            text = resp.choices[0].message.content.strip()
            text = text.strip('"').strip("'").strip()
            return text
        except Exception as e:
            if attempt == max_retries:
                print(f"  [WARN] Failed after {max_retries} retries: {e}")
                return ""
            backoff = min(8.0, 0.5 * (2 ** (attempt - 1)) + random.random() * 0.5)
            time.sleep(backoff)


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def call_gemini(
    client,
    model_id: str,
    sentence: str,
    system_msg: str,
    max_tokens: int,
    max_retries: int = 5,
) -> str:
    from google.genai import types

    prompt = (str(sentence) or "")[:1024]

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model_id,
                config=types.GenerateContentConfig(
                    system_instruction=system_msg,
                    temperature=0.0,
                    top_p=1.0,
                    max_output_tokens=max_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
                contents=prompt,
            )
            text = (resp.text or "").strip()
            text = text.strip('"').strip("'").strip()
            return text
        except Exception as e:
            if attempt == max_retries:
                print(f"  [WARN] Failed after {max_retries} retries: {e}")
                return ""
            backoff = min(8.0, 0.5 * (2 ** (attempt - 1)) + random.random() * 0.5)
            print(f"  [RETRY {attempt}/{max_retries}] {e} -- waiting {backoff:.1f}s")
            time.sleep(backoff)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    is_forward = args.direction == "forward"

    # Determine backend
    backend = args.backend or ("gemini" if args.model.startswith("gemini-") else "ollama")

    # Build system message
    if is_forward:
        system_msg = FORWARD_INSTRUCTION
        if args.eirini:
            system_msg += "\n\n" + EIRINI_REFERENCE
        output_col = "forward_translation"
    else:
        system_msg = BACK_INSTRUCTION
        output_col = "back_translation_llm"

    # Initialize client
    if backend == "gemini":
        sys.stdout.reconfigure(encoding="utf-8")
        from dotenv import load_dotenv
        from google import genai
        load_dotenv()
        api_key = os.getenv("GOOGLE_Key")
        if not api_key:
            print("ERROR: GOOGLE_Key not found in .env")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
    else:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:11434/v1/",
            api_key="ollama",
        )

    # Load input
    src = load_dataframe(args.input_path)
    sentence_col = args.sentence_col
    if sentence_col not in src.columns:
        raise KeyError(f"Column '{sentence_col}' not found. Available: {list(src.columns)}")

    # Build working DataFrame
    if is_forward:
        if "condition_index" not in src.columns:
            raise KeyError("Input file must contain a 'condition_index' column.")
        df = pd.DataFrame({
            "condition_index": src["condition_index"],
            "target_language_code": "el",
            "target_language_name": "Greek",
            output_col: "",
        })
    else:
        df = src.copy()
        if output_col not in df.columns:
            df[output_col] = ""

    # Determine output paths
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.input_path))

    if is_forward:
        if hasattr(args, "out_name") and args.out_name:
            out_base = args.out_name
        elif backend == "gemini":
            friendly = GEMINI_MODEL_NAMES.get(args.model, args.model.replace("-", "_").replace(":", "_"))
            if args.eirini:
                out_base = f"{friendly}_Eirini_forward_translation_greek"
            else:
                out_base = f"{friendly}_forward_translation_greek"
        else:
            out_base = args.model.replace(":", "_").replace("/", "_") + f".{len(df)}_Greek"
    else:
        input_base = os.path.splitext(os.path.basename(args.input_path))[0]
        if backend == "gemini":
            out_base = input_base.replace("forward_translation", "back_translation")
        else:
            out_base = f"{input_base}_backtranslated"

    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, f"{out_base}_partial.xlsx")
    final_path = os.path.join(out_dir, f"{out_base}.xlsx")

    # Resume from partial file if it exists
    if os.path.exists(partial_path):
        print(f"Resuming from partial file: {partial_path}")
        partial_df = pd.read_excel(partial_path, header=0)
        if output_col in partial_df.columns and "condition_index" in df.columns:
            for i in df.index:
                ci = df.at[i, "condition_index"]
                match = partial_df.loc[partial_df["condition_index"] == ci, output_col]
                if not match.empty:
                    val = str(match.iloc[0]).strip()
                    if val and val != "nan":
                        df.at[i, output_col] = val

    # Figure out remaining work
    need_mask = (
        df[output_col].isna()
        | (~df[output_col].astype(str).str.strip().astype(bool))
    )
    todo_idx = df.index[need_mask].tolist()

    random.seed(42)
    random.shuffle(todo_idx)

    total = len(todo_idx)
    print(f"Direction: {'forward (English -> Greek)' if is_forward else 'back (Greek -> English)'}")
    print(f"Backend: {backend}")
    print(f"Model: {args.model}")
    if is_forward and args.eirini:
        print(f"Eirini: Yes")
    print(f"Input: {args.input_path} ({len(src)} sentences)")
    print(f"Output: {final_path}")
    print(f"Already translated: {len(df) - total} | Remaining: {total}")
    if total == 0:
        print("Nothing to do. Writing final file ...")
        save_dataframe(df, final_path)
        print(f"Done: {final_path}")
        return

    # Map df index -> source sentence
    if is_forward:
        sentence_map = {}
        for i in df.index:
            ci = df.at[i, "condition_index"]
            match = src.loc[src["condition_index"] == ci, sentence_col]
            if not match.empty:
                sentence_map[i] = match.iloc[0]
    else:
        sentence_map = {i: df.at[i, sentence_col] for i in df.index}

    start = time.time()
    completed = 0

    if backend == "gemini":
        for i in todo_idx:
            text = sentence_map.get(i, "")
            translation = call_gemini(
                client=client,
                model_id=args.model,
                sentence=text,
                system_msg=system_msg,
                max_tokens=args.max_tokens,
            )
            df.at[i, output_col] = translation
            completed += 1

            if (completed % 5 == 0) or (completed == total):
                avg = (time.time() - start) / max(1, completed)
                eta = (total - completed) * avg
                print(f"[{completed}/{total}] avg {avg:.2f}s | ETA {int(eta/60)}m{int(eta%60)}s")

            if (completed % args.save_every == 0):
                save_dataframe(df, partial_path)
    else:
        limiter = RateLimiter(args.target_rps)
        write_lock = threading.Lock()

        def process_one(i: int):
            try:
                text = sentence_map.get(i, "")
                translation = call_ollama(
                    client=client,
                    model=args.model,
                    sentence=text,
                    system_msg=system_msg,
                    max_tokens=args.max_tokens,
                    limiter=limiter,
                    trim_input=args.trim_input,
                )
                df.at[i, output_col] = translation
            except Exception:
                df.at[i, output_col] = ""

        def save_partial():
            with write_lock:
                save_dataframe(df, partial_path)

        with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = [ex.submit(process_one, i) for i in todo_idx]
            for fut in cf.as_completed(futures):
                completed += 1
                if (completed % 10 == 0) or (completed == total):
                    avg = (time.time() - start) / max(1, completed)
                    eta = (total - completed) * avg
                    print(f"[{completed}/{total}] avg {avg:.2f}s | ETA {int(eta/60)}m{int(eta%60)}s")
                if (completed % args.save_every == 0) or (completed == total):
                    save_partial()

    # Final save
    save_dataframe(df, final_path)
    elapsed = time.time() - start
    print(f"\nDone in {int(elapsed/60)}m{int(elapsed%60)}s. Final written to: {final_path}")

    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)
        print(f"Removed partial checkpoint: {partial_path}")


if __name__ == "__main__":
    main()
