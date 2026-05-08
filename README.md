# Hybrid LLM-Based Automated Essay Scoring

Replication and extension of [Atkinson & Palma (2025)](https://doi.org/10.1038/s41598-025-87862-3) using fully open-source LLMs, evaluated on English (ASAP) and Malayalam essay datasets.

---

## What this does

Automatically scores student essays by combining three feature layers:

- **LLM Embeddings** — E5-Mistral-7B-Instruct (with official ASAP rubrics as instruction context) + INSTRUCTOR-large, ensembled 50/50
- **Discourse Coherence** — 16 entity grid transition features + KNN-based essay similarity
- **Shallow Linguistic Features** — readability metrics, lexical diversity, sentence structure (10 features)

Final predictions come from an **XGBoost regressor** trained under strict 10-fold cross-validation with zero data leakage.

---

## Results

### English AES — ASAP Dataset (12,976 essays, 8 prompts)

| System | Avg QWK |
|--------|---------|
| Reference paper (GPT-3 + Random Forest) | 0.840 |
| **This work (E5-Mistral + INSTRUCTOR + XGBoost)** | **0.745** |

*88.7% of GPT-3 performance using entirely free, open-source models.*

### Malayalam AES — Custom Dataset (3,000 essays, 150 prompts)

| Metric | Score |
|--------|-------|
| QWK | **0.943** |
| Exact Agreement | 0.591 |
| Adjacent Agreement | 0.917 |
| Spearman Correlation | 0.945 |

* LLM-based automated grading system for Malayalam.*

---

## Key Contributions

1. **Open-source replication** — Replaced proprietary GPT-3 (175B) with E5-Mistral-7B-Instruct; achieved competitive QWK using only free models
2. **Rubric-aware embeddings** — Official ASAP scoring rubrics injected as instruction context in a 2048-token window, improving score-criterion alignment
3. **Malayalam essay scoring** — `multilingual-e5-large-instruct` embeddings with essay-to-reference cosine similarity as the primary signal
4. **Length bias fix** — Novel `word_ratio` feature (essay word count normalized by reference answer length) outperforms raw word count

---

## Tech Stack

```
Python · HuggingFace Transformers · Sentence Transformers · XGBoost
PyTorch · scikit-learn · NLTK · spaCy · textstat · Kaggle GPU (T4)
```

**Models used:** `intfloat/e5-mistral-7b-instruct` · `hkunlp/instructor-large` · `intfloat/multilingual-e5-large-instruct`

---

## Project Structure

```
├── hybrid-llm.ipynb          # Main notebook — English AES pipeline
│   ├── Step 1  — Data loading (ASAP dataset)
│   ├── Step 2  — Shallow feature extraction (readability, lexical, structural)
│   ├── Step 3  — Entity grid features (discourse coherence)
│   ├── Step 4  — LLM embeddings (E5-Mistral + INSTRUCTOR)
│   ├── Step 5  — Feature matrix assembly
│   └── Step 6  — XGBoost 10-fold CV + ensemble (V3 → V23)
│
└── Malayalam section
    ├── multilingual-e5 embeddings
    ├── essay-reference similarity features
    ├── word_ratio normalization
    └── XGBoost grouped 10-fold CV
```

---

## How to Run

> Designed for Kaggle (T4 GPU). Local runs require ~16GB VRAM for E5-Mistral.

```bash
# 1. Add the ASAP dataset to Kaggle input
# Dataset: Hewlett Foundation: Automated Essay Scoring

# 2. Run cells in order — each step saves outputs to /kaggle/working/
# Steps are idempotent: already-computed features are loaded from disk

# 3. For Malayalam, add the custom dataset (questions_150.csv + mal_dataset.csv)
```

---

## Dataset

- **English:** [ASAP — Kaggle](https://www.kaggle.com/c/asap-aes) — 12,976 essays across 8 prompts (persuasive, narrative, source-dependent)
- **Malayalam:** Custom dataset — 3,000 responses to 150 prompts with manual scores (0–10)

---

## Reference

> Atkinson, J., & Palma, D. (2025). An LLM-based hybrid approach for enhanced automated essay scoring. *Scientific Reports*, 15:14551. https://doi.org/10.1038/s41598-025-87862-3
