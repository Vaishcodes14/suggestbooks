import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("books_50k.csv")
    df.fillna("", inplace=True)
    return df

df = load_data()

# Precompute TF-IDF
@st.cache_resource
def compute_tfidf(df):
    corpus = (df['Title'] + " " + df['Subjects'] + " " + df['Bookshelves']).fillna("")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = compute_tfidf(df)

st.title("📚 Mood-Based Book Recommender")

user_input = st.text_input("Tell me how you’re feeling or your situation:", "")

if user_input:
    query_vec = vectorizer.transform([user_input])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:5]
    st.subheader("Recommended Books:")
    for i in top_idx:
        st.write(f"**{df.iloc[i]['Title']}** — {df.iloc[i]['Authors']}")
