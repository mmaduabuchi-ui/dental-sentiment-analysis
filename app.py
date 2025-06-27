import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

st.title("🦷 Dental Clinic Sentiment Analyzer")
st.write("Enter a dental review to predict whether it's Positive, Negative, or Neutral.")

review = st.text_area("Review Text:")

if st.button("Analyze Sentiment"):
    if review.strip():
        transformed = vectorizer.transform([review])
        prediction = model.predict(transformed)[0]
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter a review.")
