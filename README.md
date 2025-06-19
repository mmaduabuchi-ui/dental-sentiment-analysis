# dental-sentiment-analysis
# Dental Clinic Sentiment Analysis (Nigeria)

This project builds a machine learning model to classify patient reviews into `positive`, `neutral`, and `negative` sentiment categories using real reviews from dental clinics.

## 🧠 Model
- Trained using Logistic Regression with TF-IDF vectorizer
- Balanced neutral class using upsampling
- Accuracy: 89%
- Handles real Nigerian dental review texts

## 🧪 Files
- `sentiment_model.pkl`: Trained sentiment classifier
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer used for input text
- `dental_reviews_with_sentiment.csv`: Review dataset
- `notebook.ipynb`: Full training and evaluation pipeline

## 🛠️ How to Use
```python
import joblib

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

text = ["I loved the service at the clinic"]
vector = vectorizer.transform(text)
print(model.predict(vector))
