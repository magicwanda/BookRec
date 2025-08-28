# train_model.py
import pandas as pd
from surprise import Dataset, Reader, SVD, dump, accuracy
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from itertools import islice
import pickle

users = pd.read_csv("Users.csv")
ratings = pd.read_csv("ratings.csv")
ratings_filtered = pd.read_csv("ratings_filtered.csv")
books = pd.read_csv("Books.csv")
books_dedup = pd.read_csv("books_dedup.csv")
books_index = pd.read_csv("books_index.csv")
ratings["ISBN"] = ratings["ISBN"].astype(str)
ratings_filtered["ISBN"] = ratings_filtered["ISBN"].astype(str)
books["ISBN"] = books["ISBN"].astype(str)
books_dedup["ISBN"] = books_dedup["ISBN"].astype(str)

# Canonical key on original books (for mapping ALL ISBNs)
def canon_key(df):
    return (df["Book-Title"].astype(str).str.strip().str.lower() + "||" +
            df["Book-Author"].astype(str).str.strip().str.lower())

books["canon_key"] = canon_key(books)
books_dedup["canon_key"] = canon_key(books_dedup)

# Remove rows with missing/empty title in the ORIGINAL books before mapping
valid_titles = books["Book-Title"].notna() & (books["Book-Title"].astype(str).str.strip() != "")
books = books[valid_titles].copy()

# Map ratings → canon_key via original books
ratings_map = ratings.merge(books[["ISBN","canon_key"]], on="ISBN", how="left")

# Drop ratings whose book has no title/author (no canon_key)
ratings_map = ratings_map[ratings_map["canon_key"].notna()].copy()

# remove zero ratings to not pollute the averages
ratings_map = ratings_map[pd.to_numeric(ratings_map["Book-Rating"], errors="coerce") != 0].copy()

# Popularity aggregated by canon_key
pop_key = (ratings_map.groupby("canon_key")["Book-Rating"]
           .agg(count="count", mean="mean")
           .rename(columns={"count":"rating_count","mean":"rating_mean"})
           .reset_index())

# Join to dedup to get canonical ISBN/title/author (inner join drops anything not in dedup)
pop = pop_key.merge(
    books_dedup[["canon_key","ISBN","Book-Title","Book-Author"]],
    on="canon_key", how="inner"
)

# Sanitize and sort
pop["rating_mean"]  = pop["rating_mean"].replace([np.inf,-np.inf], np.nan).fillna(0.0).astype(float)
pop["rating_count"] = pop["rating_count"].fillna(0).astype(int)
#pop = pop.sort_values(["rating_count","rating_mean"], ascending=[False, False]).reset_index(drop=True)

pop["popularity_score"] = pop["rating_mean"] * np.log1p(pop["rating_count"])
pop = pop.sort_values("popularity_score", ascending=False)

# Save for the API
pop.to_csv("popularity.csv", index=False)

# ---- Train SVD on explicit filetered ratings - more than 2 per user
reader = Reader(rating_scale=(1, 10))

# Remove zero-only ratings
ratings_filtered = ratings_filtered[ratings_filtered["Book-Rating"] > 0].copy()

data = Dataset.load_from_df(ratings_filtered[["User-ID", "ISBN", "Book-Rating"]], reader)
# split the data into training and testing sets (80% train, 20% test)
trainset, testset = train_test_split(data, test_size=.2)
#trainset = data.build_full_trainset()
svd = SVD()
svd.fit(trainset)
dump.dump("svd_model.pkl", algo=svd)
predictions = svd.test(testset)
accuracy.rmse(predictions)
print("✅ Saved SVD model → svd_model.pkl")

# ---- Content-based: TF-IDF over (title + author)
books_dedup["__text__"] = (books_dedup["Book-Title"].astype(str) + " " +
                     books_dedup["Book-Author"].astype(str))
vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
tfidf = vectorizer.fit_transform(books_dedup["__text__"].fillna(""))

joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(tfidf, "tfidf_matrix.joblib")
books[["ISBN","Book-Title","Book-Author"]].to_csv("books_index.csv", index=False)
print("✅ Saved TF-IDF vectorizer/matrix and books index")



# --- Content (TF-IDF) ---
def eval_content(k=10, users_sample=None):
    users = list(test_pos_by_user.keys()) if users_sample is None else users_sample
    scores = {"precision":[], "recall":[], "ndcg":[]}
    for u in users:
        truth = test_pos_by_user.get(u, set())
        if not truth:
            continue
        # pick one held-out positive (leave-one-out setup)
        fav = list(truth)[0]
        ranked = content_similar_by_text(fav, n=k*10)  # get candidate ISBNs
        ranked = [isbn for isbn in ranked if isbn not in seen_by_user.get(u,set())][:k]
        scores["precision"].append(precision_at_k(ranked, truth, k))
        scores["recall"].append(recall_at_k(ranked, truth, k))
        scores["ndcg"].append(ndcg_at_k(ranked, truth, k))
    return summarize(scores)



# ---- Metric functions ----
def precision_at_k(predicted, ground_truth, k=10):
    if not ground_truth:
        return None
    top_k = predicted[:k]
    hits = len(set(top_k) & ground_truth)
    return hits / k

def recall_at_k(predicted, ground_truth, k=10):
    if not ground_truth:
        return None
    top_k = predicted[:k]
    hits = len(set(top_k) & ground_truth)
    return hits / len(ground_truth)

def ndcg_at_k(predicted, ground_truth, k=10):
    if not ground_truth:
        return None
    top_k = predicted[:k]
    dcg = 0.0
    for i, isbn in enumerate(top_k):
        if isbn in ground_truth:
            dcg += 1 / np.log2(i+2)  # rank position i (1-based)
    idcg = sum(1 / np.log2(i+2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ----  Eval SVD -----
# Keep only positive ratings
ratings_filtered = ratings_filtered[ratings_filtered["Book-Rating"] > 0].copy()

# Ground-truth positives dict: user -> set of held-out items
test_pos_by_user = defaultdict(set)
for _, row in test.iterrows():
    test_pos_by_user[row["User-ID"]].add(str(row["ISBN"]))

# Precompute user->seen set (train) for fast filtering
seen_by_user = train.groupby("User-ID")["ISBN"].apply(set).to_dict()

# pick one item per user as test
test_idx = ratings_filtered.groupby("User-ID", group_keys=False).apply(lambda g: g.sample(1, random_state=42)).index
test = ratings_filtered.loc[test_idx]
train = ratings_filtered.drop(index=test_idx)

# take first 50000 entries from the dict
sample_users = dict(islice(test_pos_by_user.items(), 50000))

with open('sample_users_50000.pkl', 'wb') as f:
    pickle.dump(sample_users, f)

all_items = set(map(str, train["ISBN"].unique()))

Ks = [10]
scores = {k: {"precision":[], "recall":[], "ndcg":[]} for k in Ks}
#very slow, need to refactor
for uid, truth in sample_users.items():
    seen = seen_by_user.get(uid, set())
    candidates = all_items - seen
    preds = [(iid, svd.predict(uid, iid).est) for iid in candidates]
    ranked = [iid for iid, _ in sorted(preds, key=lambda x: -x[1])]
    
    for K in Ks:
        scores[K]["precision"].append(precision_at_k(ranked, truth, K))
        scores[K]["recall"].append(recall_at_k(ranked, truth, K))
        scores[K]["ndcg"].append(ndcg_at_k(ranked, truth, K))

# average results
for K in Ks:
    print(f"K={K}: "
          f"Precision={np.mean(scores[K]['precision']):.4f}, "
          f"Recall={np.mean(scores[K]['recall']):.4f}, "
          f"NDCG={np.mean(scores[K]['ndcg']):.4f}")


