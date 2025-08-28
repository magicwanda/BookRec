# ui.py
import streamlit as st
import requests
import pandas as pd

#st.set_page_config(page_title="📚 Hybrid Book Recommender", layout="centered")
st.set_page_config(page_title="📚 Hybrid Book Recommender", layout="wide")

st.caption("Backend expected at /recommend/user/{user_id}, /recommend/by_book?title=..., /recommend/popular")
api_base = "http://127.0.0.1:8000"

st.markdown("""
<style>
.block-container {
    max-width: 100%;
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)
st.title("📚 Hybrid Book Recommender (CF + Content + Popular)")

def render_recs(items, title="Recommendations"):
    if not items:
        st.info("No recommendations returned.")
        return
    # Normalize dicts to a neat table
    df = pd.DataFrame(items)
    # Order columns nicely if present
    cols = [c for c in ["title", "author", "score", "why", "isbn"] if c in df.columns] + [c for c in df.columns if c not in ["title","author","score","why","isbn"]]
    df = df[cols]
    st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_table(items, title="Items"):
    if not items:
        st.info("Nothing to show.")
        return
    df = pd.DataFrame(items)
    cols = [c for c in ["title","author","rating","isbn"] if c in df.columns] + \
           [c for c in df.columns if c not in ["title","author","rating","isbn"]]
    df = df[cols]
    st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=True)

tab_user, tab_book, tab_pop = st.tabs(["👤 By User", "📖 By Book", "🔥 Popular"])

with tab_user:
    st.write("Get personalized recommendations and see this user’s rated books.")
    #st.write("Examples of users with diff amounts of ratings: user 8 - 7 ratings, user 99 - 8 ratings, user 242 - 33 ratings, 254 - 58 ratings")
    st.markdown(
    "<span style='color:grey'><i>Examples of users with diff amounts of ratings: user <u>8</u> - 7 ratings, user <u>99</u> - 8 ratings, user <u>242</u> - 33 ratings, user <u>254</u> - 58 ratings</i></span>",
    unsafe_allow_html=True
)
    user_id = st.number_input("User ID", min_value=1, step=1, value=99)
    n_user = st.slider("How many recommendations?", 1, 50, 10)

    # Two columns: left (recs), right (rated books)
    col_left, col_right = st.columns([3, 2])

    if st.button("Get recommendations for user"):
        # LEFT: recommendations
        with col_left:
            try:
                with st.spinner("Fetching recommendations..."):
                    url = f"{api_base}/recommend/user/{user_id}"
                    r = requests.get(url, params={"n": n_user}, timeout=30)
                    if r.ok:
                        data = r.json()
                        render_recs(data.get("recommendations", []), f"Top {n_user} for user {user_id}")
                    else:
                        st.error(f"API error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

        # RIGHT: rated books
        with col_right:
            try:
                with st.spinner("Loading user’s rated books..."):
                    r2 = requests.get(f"{api_base}/user/{user_id}/ratings",
                                      params={"include_zero": False, "limit": 200},
                                      timeout=30)
                    if r2.ok:
                        data2 = r2.json()
                        render_table(data2.get("ratings", []), "User’s rated books")
                    else:
                        st.error(f"API error {r2.status_code}: {r2.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
                st.markdown("---")
        

with tab_book:
    st.write("Content-based fallback using TF-IDF on title + author (works even for unrated books).")
    title = st.text_input("Book title (required)", placeholder="Examples: Harry Potter, Pride and Prejudice, Bronze Mirror")
    author = st.text_input("Author (optional)", placeholder="Example: J. K. Rowling, Jane Austen, Jeanne Larsen")
    n_book = st.slider("How many similar books?", 1, 50, 10, key="n_book")
    if st.button("Find similar books"):
        if not title.strip():
            st.warning("Please enter a title.")
        else:
            try:
                with st.spinner("Searching similar books..."):
                    url = f"{api_base}/recommend/by_book"
                    r = requests.get(url, params={"title": title, "author": author, "n": n_book}, timeout=30)
                    if r.ok:
                        data = r.json()
                        render_recs(data.get("recommendations", []), f"Similar to “{data.get('query', {}).get('title', title)}”")
                    else:
                        st.error(f"API error {r.status_code}: {r.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

with tab_pop:
    st.write("Most popular books (weighted ratings count by mean rating - IMDB approach).")
    n_pop = st.slider("How many popular books?", 1, 50, 10, key="n_pop")
    if st.button("Show popular"):
        try:
            with st.spinner("Loading popular books..."):
                url = f"{api_base}/recommend/popular"
                r = requests.get(url, params={"n": n_pop}, timeout=30)
                if r.ok:
                    data = r.json()
                    #render_recs(data.get("recommendations", []), f"Top {n_pop} popular")
                    render_recs(data["results"], f"Top {n_pop} popular")
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Request failed: {e}")
