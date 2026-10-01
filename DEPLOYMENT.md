# Deployment Guide – TweetSense on Render

This document provides step-by-step instructions for deploying **TweetSense** to [Render](https://render.com/) as a production Web Service.

---

## 🛠️ Render Service Configuration

| Setting Key | Configuration Value |
|---|---|
| **Service Type** | Web Service |
| **Environment / Runtime** | Python |
| **Region** | Oregon, USA (or preferred region) |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python run_streamlit.py` |
| **Python Version** | `3.11.9` |

---

## ⚙️ Environment Variables

Render dynamically binds a port via the `PORT` environment variable. The production entrypoint `run_streamlit.py` reads this environment variable and binds Streamlit server automatically:

```python
import os
import subprocess
import sys

port = os.environ.get("PORT", "10000")
cmd = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    "app.py",
    "--server.address=0.0.0.0",
    f"--server.port={port}",
    "--server.headless=true"
]
subprocess.run(cmd)
```

No custom environment variables are strictly required, but you can set:
- `PYTHONUNBUFFERED=1` (Recommended for clean live logs)

---

## 📋 Pre-Deployment Checklist

1. **No Hardcoded Absolute Paths:** All paths in `src/`, `models/`, and `data/` use relative `os.path.join()` paths compatible across Linux/Render containers.
2. **NLTK Auto-Download:** `src/preprocessing.py` contains `ensure_nltk_resources()` to dynamically fetch NLTK data (`stopwords`, `wordnet`, `punkt`, `punkt_tab`) at runtime without requiring manual server commands.
3. **Model Artifact Inclusion:** Ensure `models/sentiment_model.pkl`, `models/tfidf_vectorizer.pkl`, and `models/model_metadata.json` are committed to your git repository, or run `python train_model.py` as part of your build command (`pip install -r requirements.txt && python train_model.py`).

---

## 🚀 Deployment Steps

1. Push your code to GitHub:
   ```bash
   git add .
   git commit -m "Deploy TweetSense to Render"
   git push origin main
   ```
2. Log into your [Render Dashboard](https://dashboard.render.com).
3. Click **New +** -> **Web Service**.
4. Connect your GitHub repository `TweetSense`.
5. Fill in the deployment details:
   - **Name:** `tweetsense-app`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `python run_streamlit.py`
6. Click **Create Web Service**. Render will build and deploy the app live!
