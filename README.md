---
title: Review Sentiment
emoji: 💬
colorFrom: gray
colorTo: blue
sdk: streamlit
sdk_version: 1.40.2
python_version: "3.11"
app_file: app.py
pinned: false
---

# Review Sentiment & Explainability

3-class (Positive / Neutral / Negative) review classifier with global features and LIME explanations.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy: Hugging Face Spaces (recommended, free CPU has 16 GB RAM)
1. Create a Space at huggingface.co/new-space, SDK **Streamlit**, hardware CPU basic.
2. Upload `app.py`, `sentiment_engine.py`, `requirements.txt`, `README.md` and `.streamlit/config.toml`
   (or `git clone` the Space repo, copy the files in, then `git add . && git commit -m init && git push`).
3. The Space builds automatically. The README front matter above configures it.

## Deploy: Render (native Python)
1. Push this folder to a GitHub repo, then New + > Web Service > select the repo.
2. Runtime **Python 3**, Build: `pip install -r requirements.txt`
3. Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Env var `PYTHON_VERSION=3.11.9`. Use a paid instance with 2 GB+ RAM (free 512 MB is too small for the transformer).

## Deploy: Render / Railway (Docker)
Both detect the `Dockerfile` automatically (Render: runtime Docker; Railway: New Project > Deploy from GitHub).
The container reads `$PORT`. Give it 2 GB+ RAM.

## Notes
- First transformer use downloads model weights (~500 MB for RoBERTa) from Hugging Face.
- Baseline trains on your uploaded file's rating column (1-2 Negative, 3 Neutral, 4-5 Positive).
