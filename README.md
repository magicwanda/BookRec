# Book Recommender (Streamlit + FastAPI)

Run a simple book recommender with a Streamlit UI and a FastAPI backend.

---

## 📥 Download the model (required)

The SVD model is too large for the repo. Download it and place it where your code expects it (e.g., project root):

**Model:** https://drive.google.com/file/d/139nni8C2-5pbPhgE65_1cfBSjZ_BpThR/view?usp=sharing

---

## 🧰 Setup

### install dependencies
`pip install -r requirements.txt`

### Start the backend (FastAPI + Uvicorn)
`python3 -m uvicorn app:app --reload --port 8000`

### Start the UI (Streamlit) — in a separate terminal
`python3 -m streamlit run ui.py --server.port 8501`
