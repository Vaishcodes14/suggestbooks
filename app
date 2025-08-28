import streamlit as st
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_resource
def load_data():
    df = pd.read_csv("books_50k.csv")
    # Fallback: if no 'summary' column, create a pseudo-summary
    if "summary" not in df.columns:
        df["summary"] = (
            df["Title"].fillna("")
            + " " + df["Subjects"].fillna("")
            + " " + df["Bookshelves"].fillna("")
        )
    return df

books = load_data()

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # small + fast + semantic
    emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    return embed_model, emotion_model

embed_model, emotion_model = load_models()

# ---------------------------
# Precompute Book Embeddings
# ---------------------------
@st.cache_resource
def compute_embeddings(texts):
    return embed_model.encode(texts.tolist(), convert_to_tensor=True, show_progress_bar=True)

book_embeddings = compute_embeddings(books["summary"])

# ---------------------------
# Map emotions to preferred genres
# ---------------------------
emotion_to_genres = {
    "joy": ["Humor", "Romance", "Adventure", "Fairy Tales"],
    "sadness": ["Poetry", "Philosophy", "Self-Help"],
    "anger": ["Politics", "History", "Social Science"],
    "fear": ["Thriller", "Mystery", "Gothic"],
    "love": ["Romance", "Drama", "Poetry"],
    "surprise": ["Science Fiction", "Fantasy", "Adventure"],
    "disgust": ["Satire", "Social Criticism", "Philosophy"]
}

# ---------------------------
# Recommend Books
# ---------------------------
def recommend_books(user_text, top_n=5):
    # Step 1: detect emotion
    emo_scores = emotion_model(user_text)[0]
    detected_emotion = max(emo_scores, key=lambda x: x["score"])["label"].lower()

    # Step 2: Encode user query
    query_embedding = embed_model.encode(user_text, convert_to_tensor=True)

    # Step 3: Compute cosine similarities
    cos_scores = util.cos_sim(query_embedding, book_embeddings)[0]

    # Step 4: Filter by emotion → genre mapping
    preferred_genres = emotion_to_genres.get(detected_emotion, [])
    filtered_idx = books[
        books["Bookshelves"].apply(
            lambda x: any(g in str(x) for g in preferred_genres)
        )
    ].index.tolist()

    if len(filtered_idx) == 0:  # fallback if no match
        filtered_idx = list(range(len(books)))

    # Get top matches
    ranked = torch.topk(cos_scores[filtered_idx], k=min(top_n*5, len(filtered_idx)))
    indices = [filtered_idx[i.item()] for i in ranked.indices]

    # Pick top_n but shuffle slightly for diversity
    np.random.shuffle(indices)
    indices = indices[:top_n]

    return detected_emotion, books.iloc[indices]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📚 Sentiment-Driven Book Recommender")
st.write("Discover books based on your **life events & emotions** (Bibliotherapy).")

user_input = st.text_area("💭 Describe your mood or situation:")

if st.button("Recommend Books") and user_input.strip():
    with st.spinner("Analyzing your emotion and finding the best matches..."):
        emotion, recs = recommend_books(user_input, top_n=5)

    st.subheader(f"Detected Emotion: `{emotion}`")
    st.write("### Recommended Books:")

    for _, row in recs.iterrows():
        st.markdown(f"**{row['Title']}** by *{row['Authors']}*")
        st.caption(f"📚 {row['Bookshelves']} | 🗓 {row['Issued']}")
        st.write(f"➡️ {row['summary'][:300]}...")  # show snippet
        st.write("---")
