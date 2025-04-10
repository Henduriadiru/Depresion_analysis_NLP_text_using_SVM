import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns
import pickle

# Load data (for demo, replace this with actual data path)
@st.cache_data
def load_data():
    # Sample data loading, replace with your real dataset
    df = pd.read_csv("sample_depresion_analysis.csv")  # make sure the CSV contains 'tweet', 'cleaned_tweet', and 'label'
    return df

df = load_data()

st.title("📊 Depression Text Analysis App")
st.write("This app performs PCA/t-SNE visualization and predicts depression sentiment from input text.")

# Train Word2Vec model and PCA
@st.cache_resource
def train_model(df):
    model = Word2Vec(df["cleaned_tweet"], vector_size=100, window=5, min_count=1, workers=4)

    def text_to_vector(clean_text):
        vectors = [model.wv[token] for token in clean_text if token in model.wv]
        if len(vectors) == 0:
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)

    X_vectors = np.array([text_to_vector(tokens) for tokens in df["cleaned_tweet"]])
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_vectors)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    return model, X_pca, X_tsne

model, X_pca, X_tsne = train_model(df)

# Tabs for visualization and prediction
visual_tab, predict_tab = st.tabs(["📈 Visualization", "🧠 Predict Sentiment"])

with visual_tab:
    plot_type = st.selectbox("Choose a plot", ["PCA", "t-SNE"])
    if plot_type == "PCA":
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df['label'], cmap='viridis')
        ax.set_title("PCA Visualisation")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['label'], cmap='viridis')
        ax.set_title("t-SNE Visualisation")
        st.pyplot(fig)

with predict_tab:
    user_input = st.text_area("Enter a tweet or sentence to analyze:")

    if user_input:
        import re
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        stop_words = set(stopwords.words('english'))

        def clean_and_tokenize(text):
            text = re.sub(r"[^a-zA-Z]", " ", text.lower())
            tokens = word_tokenize(text)
            return [word for word in tokens if word not in stop_words]

        cleaned = clean_and_tokenize(user_input)
        vector = np.mean([model.wv[token] for token in cleaned if token in model.wv] or [np.zeros(model.vector_size)], axis=0)

        # Dummy classifier (replace with trained SVM or load from pickle)
        st.warning("Note: This is a dummy prediction. Replace with a trained classifier.")
        dummy_label = int(np.random.choice([0, 1]))
        st.write("Predicted Label:", "Depressed" if dummy_label == 1 else "Not Depressed")


