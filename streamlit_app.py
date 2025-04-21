import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from difflib import get_close_matches
from collections import Counter

# --- SAYFA AYARI ---
st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")

# --- VERİ YÜKLEME ---
df = pd.read_csv("netflix_titles.csv")
df = df.dropna(subset=['description']).copy()
df['combined_text'] = df['listed_in'].fillna('') + ' ' + df['description'].fillna('')
df = df.reset_index(drop=True)

# --- TF-IDF & KMEANS & COSINE ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_text'])
kmeans = KMeans(n_clusters=8, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
titles = df['title'].tolist()

# --- ÖNERİ FONKSİYONU ---
def get_recommendations(title, top_n=5):
    idx = df[df['title'] == title].index[0]
    target_cluster = df.loc[idx, 'cluster']
    same_cluster_indices = df[df['cluster'] == target_cluster].index
    sim_scores = [(i, cosine_sim[idx][i]) for i in same_cluster_indices if i != idx]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:top_n]
    recommended_indices = [i[0] for i in sim_scores]
    return df.iloc[recommended_indices][['title', 'listed_in', 'description']]

# --- STİL VE BAŞLIK ---
st.markdown("""
    <style>
    .main { background-color: #141414; color: white; font-family: 'Arial'; }
    h1, h2, h3 { color: #E50914; }
    </style>
""", unsafe_allow_html=True)

st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Netflix_icon.svg", width=100)
st.title("🍿 Netflix İçerik Öneri Sistemi")
st.markdown("🎯 3 farklı yolla içerik önerisi alabilirsiniz: rastgele içerikler, popüler türler veya başlık arama.")

# --- INITIAL STATE TANIMI ---
for key in ['genre_recommendations', 'search_recommendations', 'random_recommendations', 'genre_selected']:
    if key not in st.session_state:
        st.session_state[key] = None

# --- 🎲 RASTGELE 5 İÇERİK (default seçimsiz) ---
st.subheader("🎲 Bugünlük Rastgele Seçimler")
random_titles = ["Seçiniz..."] + df['title'].sample(5, random_state=42).tolist()
selected_random = st.radio("Bir başlık seçin ve önerileri görün:", random_titles, horizontal=True)

if selected_random != "Seçiniz...":
    st.session_state.random_recommendations = get_recommendations(selected_random)[['title', 'description']].values.tolist()
    st.session_state.genre_recommendations = None
    st.session_state.search_recommendations = None
    st.session_state.genre_selected = None

# --- 🔥 POPÜLER TÜR BUTONLARI ---
st.subheader("🔥 Popüler Türlere Göre İçerikler")
genres = [g.strip() for sublist in df['listed_in'].str.split(',') for g in sublist]
top_genres = [g for g, _ in Counter(genres).most_common(6)]
genre_col1, genre_col2, genre_col3 = st.columns(3)

for i, genre in enumerate(top_genres):
    with [genre_col1, genre_col2, genre_col3][i % 3]:
        if st.button(f"🎈 {genre}", key=f"genre_{genre}"):
            genre_df = df[df['listed_in'].str.contains(genre, case=False, na=False)].sample(5)
            st.session_state.genre_recommendations = genre_df[['title', 'description']].values.tolist()
            st.session_state.genre_selected = genre
            st.session_state.search_recommendations = None
            st.session_state.random_recommendations = None

# --- 🔎 FİLM/DİZİ ADIYLA ARAMA ---
st.subheader("🔎 Film/Dizi Adı ile Öneri Al")
user_input = st.text_input("İçerik başlığını girin (örneğin: Narcos):")

if user_input:
    if user_input in titles:
        results = get_recommendations(user_input)
        st.session_state.search_recommendations = results[['title', 'description']].values.tolist()
        st.session_state.genre_recommendations = None
        st.session_state.random_recommendations = None
        st.session_state.genre_selected = None
    else:
        st.session_state.search_recommendations = None
        close_matches = get_close_matches(user_input, titles, n=5, cutoff=0.5)
        st.warning(f"❌ '{user_input}' adlı içerik bulunamadı.")
        if close_matches:
            st.markdown("🔍 Belki şunları demek istediniz:")
            for match in close_matches:
                st.markdown(f"➤ **{match}**")
        else:
            st.markdown("🚫 Benzer içerik bulunamadı.")

# --- SONUÇLARI GÖSTER ---
if st.session_state.random_recommendations:
    st.success("🎲 Rastgele seçilen içerik için öneriler:")
    for title, desc in st.session_state.random_recommendations:
        st.markdown(f"**🎬 {title}**")
        st.markdown(f"📄 {desc}")
        st.markdown("---")

if st.session_state.genre_recommendations:
    st.info(f"🔍 '{st.session_state.genre_selected}' türünden öneriler:")
    for title, desc in st.session_state.genre_recommendations:
        st.markdown(f"**🎬 {title}**")
        st.markdown(f"📄 {desc}")
        st.markdown("---")

if st.session_state.search_recommendations:
    st.success("🔎 Arama sonucuna göre öneriler:")
    for title, desc in st.session_state.search_recommendations:
        st.markdown(f"**🎬 {title}**")
        st.markdown(f"📄 {desc}")
        st.markdown("---")
