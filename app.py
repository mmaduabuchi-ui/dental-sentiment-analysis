import gradio as gr
import joblib

# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Define prediction function
def predict_sentiment(text):
    if not text.strip():
        return "Please enter a review"
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)[0]
    return f"Predicted Sentiment: {prediction}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter dental review..."),
    outputs="text",
    title="🦷 Dental Clinic Sentiment Analyzer",
    description="Paste a review and this AI will tell you if it's Positive, Negative, or Neutral."
)

# Launch the app
interface.launch()
