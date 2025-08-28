import pandas as pd

ratings = pd.read_csv("Ratings.csv")

# Ensure numeric rating
ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")

# For each user: count total ratings, count nonzero ratings
agg = ratings.groupby("User-ID")["Book-Rating"].agg(
    total_ratings="count",
    nonzero_ratings=lambda x: (x > 0).sum()
)

# Users with all ratings = 0
users_all_zero = agg[agg["nonzero_ratings"] == 0]
print("Users with only 0 ratings:", users_all_zero.shape[0])

# Users with exactly one nonzero rating (and rest 0)
users_one_nonzero = agg[agg["nonzero_ratings"] == 1]
print("Users with exactly one nonzero rating:", users_one_nonzero.shape[0])

# --- Clean dataset by removing these users (best practice in recommender literature) ---
filtered_users = agg[(agg["nonzero_ratings"] > 1)]
ratings_filtered = ratings[ratings["User-ID"].isin(filtered_users.index)]

print("Original ratings:", ratings.shape)
print("Filtered ratings:", ratings_filtered.shape)

# ratings_filtered is now ready for training/evaluation
