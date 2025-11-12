# Translation and Genetic Condition Recognition Analyses

This repository contains all code and documentation supporting the study:

**Quantifying the Effects of Translation Methods on Large Language Model Recognition of Genetic Conditions.**  
National Human Genome Research Institute, National Institutes of Health (NIH).

---

## Overview

This repository provides all analysis code and supporting scripts for evaluating how different **translation approaches**—Neural Machine Translation (NMT) and Large Language Model (LLM)-based translation—affect the ability of **LLMs** to recognize descriptions of genetic conditions across nearly 200 languages.

### Included scripts

| Script | Purpose |
|--------|----------|
| **`Jaccard.py`** | Calculates token-level Jaccard similarity between original and back-translated condition descriptions. |
| **`Overlap.py`** | Measures exact and fuzzy phrase retention for demographic and phenotypic phrases highlighted in Word documents. |
| **`Predictions.py`** | Performs bulk inference using OpenAI large language models to predict genetic conditions from translated text. |

---

## ⚙️ Installation

### Requirements
- **Python 3.10+**
- Works on macOS, Linux, and Windows.

### Setup
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r environment/requirements.txt
```

Dependencies include `pandas`, `openpyxl`, `python-docx`, `unidecode`, `metaphone`, `pyspellchecker`, and `openai`.  
For full reproducibility, see `environment/requirements.txt`.

---

## 1. Jaccard Similarity (`Jaccard.py`)

**Purpose:** Quantify lexical similarity between two Excel columns (e.g., original vs. back-translated text).

**Example:**
```bash
python Jaccard.py   --input-dir path/to/folder   --sheet "After Exclusions"   --col-a E --col-b G --col-out H
```

**Options:**
- `--use-metaphone` : enable phonetic matching  
- `--use-spell-correct` : enable spelling normalization  
- `--no-lowercase`, `--no-remove-punct`, etc. : toggle normalization  
- `--out-suffix _jaccard` : add suffix to output files  

**Output:** A new Excel file with similarity scores appended to the specified output column.

---

## 2. Phrase Retention Analysis (`Overlap.py`)

**Purpose:** Evaluate phrase-level retention from a reference Word document to translated Excel text.

**Input:**  
- `.docx` with **green-highlighted** demographic phrases and **yellow-highlighted** phenotypic phrases.  
- Corresponding Excel files with translated condition descriptions.

**Example:**
```bash
python Overlap.py   --word-doc path/to/Descriptions.docx   --input-dir path/to/excel_dir   --sheet "After Exclusions"   --col-a A --col-e E --col-g G   --col-h H --col-i I --col-j J --col-k K --col-l L
```

**Outputs:**

| Column | Metric |
|--------|--------|
| H | Jaccard similarity |
| I | Demographic phrases – exact retention |
| J | Demographic phrases – fuzzy retention |
| K | Phenotypic phrases – exact retention |
| L | Phenotypic phrases – fuzzy retention |

---

## 3. Genetic Condition Prediction (`Predictions.py`)

**Purpose:** Perform large-scale inference using OpenAI models to predict genetic conditions from descriptions.

**Supported models:**
- `gpt-4o-mini` *(recommended for speed/cost balance)*  
- `gpt-4o`  
- `gpt-3.5-turbo`

**Example:**
```bash
python Predictions.py   input_data.xlsx   --model gpt-4o-mini   --sentence-col 3   --max-workers 3   --target-rps 2.8   --save-every 250   --excel-output
```

Before running, set your API key:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

**Outputs:**
- `*_predictions_partial.(csv|xlsx)` – periodic checkpoints  
- `*_predictions.(csv|xlsx)` – final results with predicted conditions (`Pred1`–`Pred5`)  

---

## Workflow and Reproducibility

1. Generate or collect translations.  
2. Use `Jaccard.py` to measure translation fidelity.  
3. Use `Overlap.py` to compute phrase retention.  
4. Use `Predictions.py` to evaluate model recognition accuracy.  
5. Aggregate results for statistical comparison (e.g., in R or Python).

To capture your exact environment:
```bash
pip freeze > environment/requirements.txt
```

---

## Citation

If you use this repository, please cite:

> Solomon, B.D. (2025). *Quantifying the Effects of Translation Methods on Large Language Model Recognition of Genetic Conditions.* Manuscript submitted for review.

A `CITATION.cff` file is provided for automatic citation tools.

---

## License

This software was developed by employees of the **National Institutes of Health (NIH)** as part of their official duties and is a **work of the United States Government** (17 U.S.C. §105).  
Accordingly, it is in the **public domain within the United States** and released under the **Creative Commons Zero (CC0 1.0 Universal)** dedication to ensure broad reuse worldwide.

See the [LICENSE](LICENSE) file for full terms.

---

## Acknowledgments

This research was supported by the Intramural Research Program of the National Human Genome Research Institute of the National Institutes of Health. The contributions of the NIH author(s) were made as part of their official duties as NIH federal employees, are in compliance with agency policy requirements, and are considered Works of the United States Government. However, the findings and conclusions presented in this paper are those of the author(s) and do not necessarily reflect the views of the NIH or the U.S. Department of Health and Human Services.

