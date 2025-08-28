Run this basic recommender system app with:
Download the SVD model from here (too large to upload to git): https://drive.google.com/file/d/139nni8C2-5pbPhgE65_1cfBSjZ_BpThR/view?usp=sharing

The UI: 
python3 -m streamlit run ui.py --server.port 8501
The backend:
python3 -m uvicorn app:app --reload --port 8000

