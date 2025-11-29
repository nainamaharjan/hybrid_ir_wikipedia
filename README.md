# Hybrid IR Wikipedia (Streamlit App)

This is a simple hybrid information retrieval system over a subset of Wikipedia.

It combines:
- **BM25** (sparse retrieval) using `rank-bm25`
- **SentenceTransformer** embeddings (dense retrieval)
- A **Streamlit** UI

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).
