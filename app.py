# app.py
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from surprise import dump
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import math

# ---- Load core data
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")
books = pd.read_csv("Books.csv")
ratings_filtered = pd.read_csv("ratings_filtered.csv")

algo = dump.load("svd_model.pkl")[1]
pop = pd.read_csv("popularity.csv")
vectorizer = joblib.load("tfidf_vectorizer.joblib")
tfidf = joblib.load("tfidf_matrix.joblib")
books_index = pd.read_csv("books_index.csv")

#Fast lookups
all_isbns = set(books_index["ISBN"])
title_to_rows = (books_index
                 .assign(_t=lambda d: d["Book-Title"].str.lower())
                 .groupby("_t").indices)

app = FastAPI(title="Hybrid Book Recommender")

def safe_float(x, default=0.0):
    try:
        fx = float(x)
        return fx if math.isfinite(fx) else default
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def top_popular(n=10):
    pool = pop.head(max(n*5, n))
    out = []
    for _, row in pool.iterrows():
        out.append({
            "isbn": row["ISBN"],
            "title": row["Book-Title"],
            "author": row["Book-Author"],
            "rating_mean": float(row["rating_mean"]),
            "rating_count": int(row["rating_count"]),
            "why": "popular",
        })
        if len(out) >= n:
            break
    return out

def content_similar_by_text(text, n=10):
    vec = vectorizer.transform([text])
    sims = cosine_similarity(vec, tfidf).ravel()
    order = sims.argsort()[::-1]
    out = []
    used = set()
    for j in order:
        isbn = books_index.iloc[j]["ISBN"]
        if isbn in used:
            continue
        used.add(isbn)
        out.append({
            "isbn": isbn,
            "title": books_index.iloc[j]["Book-Title"],
            "author": books_index.iloc[j]["Book-Author"],
            "why": "content-similar"
        })
        if len(out) >= n:
            break
    return out

def cf_for_user(user_id: int, n=10):
    #books the user already rated
    rated = set(ratings.loc[ratings["User-ID"] == user_id, "ISBN"])
    #candidates: all books not yet rated
    candidates = [isbn for isbn in all_isbns if isbn not in rated]
    if not candidates:
        return []
    #predict scores (SVD)
    preds = []
    for isbn in candidates:
        try:
            est = algo.predict(user_id, isbn).est
        except Exception:
            continue
        preds.append((isbn, est))
    preds.sort(key=lambda x: x[1], reverse=True)
    top = preds[:n]
    #pretty output
    idf = books_index.set_index("ISBN")
    out = []
    for isbn, score in top:
        if isbn in idf.index:
            row = idf.loc[isbn]
            out.append({"isbn": isbn, "title": row["Book-Title"], "author": row["Book-Author"],
                        "score": round(float(score), 2), "why": "cf"})
    return out

# ---- Endpoints

@app.get("/recommend/user/{user_id}")
def recommend_for_user(user_id: int, n: int = Query(10, ge=1, le=50)):
    """Hybrid strategy:
       1) CF if user has ratings; 2) fallback: popularity.
    """
    user_has_ratings = not ratings.loc[ratings["User-ID"] == user_id].empty
    exclude = set(ratings.loc[ratings["User-ID"] == user_id, "ISBN"])

    results = []
    if user_has_ratings:
        results = cf_for_user(user_id, n=n)

    # Fallbacks if CF is empty/weak
    if len(results) < n:
        needed = n - len(results)
        results += top_popular(needed, exclude_isbns=exclude)

    if not results:
        raise HTTPException(status_code=404, detail="No recommendations available for this user.")
    return {"user_id": user_id, "recommendations": results[:n]}

@app.get("/recommend/by_book")
def recommend_by_book(title: str, author: str = "", n: int = Query(10, ge=1, le=50)):
    """Content-only: similar items from TF-IDF(title+author)."""
    query_text = (title or "").strip()
    if author:
        query_text += f" {author}".strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Provide at least a title.")

    key = title.lower().strip()
    if key in title_to_rows:

        row_idx = list(title_to_rows[key])[0]
        base_isbn = books_index.iloc[row_idx]["ISBN"]
        sim = content_similar_by_isbn(base_isbn, n=n+5)  # buffer, then trim
        sim = sim[:n] if sim else content_similar_by_text(query_text, n=n)
    else:
        sim = content_similar_by_text(query_text, n=n)

    if not sim:
        raise HTTPException(status_code=404, detail="No similar books found.")
    return {"query": {"title": title, "author": author}, "recommendations": sim[:n]}

@app.get("/recommend/popular")
def recommend_popular(n_pop: int = 10):
    rows = top_popular(n_pop)
    for r in rows:
        r["rating"] = r.pop("rating_mean", 0.0)          
    return {"results": rows}

@app.get("/user/{user_id}/ratings")
def get_user_ratings(user_id: int, include_zero: bool = False, limit: int = 100):
    if user_id not in ratings_filtered["User-ID"].unique():
        raise HTTPException(status_code=404, detail="User not found")

    df = ratings_filtered[ratings_filtered["User-ID"] == user_id].copy()
    if not include_zero:
        df = df[df["Book-Rating"] > 0]

    #Join titles/authors
    df["ISBN"] = df["ISBN"].astype(str)
    joined = df.merge(books_index, on="ISBN", how="left")

    #Sort by rating desc, then by title
    joined = joined.sort_values(["Book-Rating", "Book-Title"], ascending=[False, True]).head(limit)

    out = []
    for _, r in joined.iterrows():
        out.append({
            "isbn": str(r.get("ISBN", "")),
            "title": str(r.get("Book-Title", "")),
            "author": str(r.get("Book-Author", "")),
            "rating": float(r.get("Book-Rating", 0.0)),
        })
    return {"user_id": user_id, "count": len(out), "ratings": out}