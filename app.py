import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# -------------------------------
# Load Dataset
# -------------------------------
data = pd.read_csv("dataset.csv")

# Make sure column name is correct
texts = data["source_text"]

# Transform dataset texts
dataset_vectors = vectorizer.transform(texts)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Plagiarism Detector")

st.title("🕵 Plagiarism Detection System")
st.write("Enter text below to check if it is plagiarized.")

user_input = st.text_area("Enter your text here")

if st.button("Check Plagiarism"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Convert user input to vector
        user_vector = vectorizer.transform([user_input])

        # Calculate similarity with dataset
        similarities = cosine_similarity(user_vector, dataset_vectors)

        max_similarity = np.max(similarities)
        similarity_percent = round(max_similarity * 100, 2)

        # Also predict using ML model
        prediction = model.predict(user_vector)[0]

        st.subheader("🔎 Analysis Result")
        st.write(f"Highest Similarity in Dataset: **{similarity_percent}%**")

        if prediction == 1:
            st.error("⚠ Plagiarized Content Detected!")
        else:
            st.success("✅ Content Appears Original")