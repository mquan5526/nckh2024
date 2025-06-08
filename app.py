from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import pdfplumber
import docx
import os
import re
import string
import nltk
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)  # Enable CORS for frontend access

# Load models and vectorizers
email_svm = joblib.load("email_svm_sentiment_model.pkl")
sms_svm = joblib.load("sms_svm_sentiment_model.pkl")
email_vectorizer = joblib.load("email_vectorizer.pkl")
sms_vectorizer = joblib.load("sms_vectorizer.pkl")
tfidf_transformer_email = joblib.load("email_tfidf_transformer.pkl")
tfidf_transformer_sms = joblib.load("sms_tfidf_transformer.pkl")
sentiment_lstm = load_model("sentiment_lstm_model.h5")

# Load Tokenizers
email_tokenizer = joblib.load("email_tokenizer.pkl")
sms_tokenizer = joblib.load("sms_tokenizer.pkl")


# Serve frontend
@app.route("/")
def home():
    return render_template("index.html")


# Health check route
@app.route("/health")
def health_check():
    return jsonify({"status": "API is running", "routes": ["/analyze"]})


# Text preprocessing function
def preprocess_text(text, vectorizer, tokenizer, tfidf_transformer):
    if not isinstance(text, str):
        return "", np.zeros((1, 200))

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    # Remove numbers (e.g., phone numbers)
    text = re.sub(r'\d{10,}', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    words = nltk.word_tokenize(text.lower())
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word.isalpha()]

    cleaned_text = ' '.join(words)

    # TF-IDF Transformation
    text_series = pd.Series([cleaned_text])
    bow = vectorizer.transform(text_series)
    tfidf = tfidf_transformer.transform(bow)

    # Tokenization for LSTM
    try:
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=200)
    except Exception as e:
        print(f"Tokenizer error: {e}")
        padded = np.zeros((1, 200))

    return tfidf, padded


@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.json
    text = data.get("text", "")
    message_type = data.get("type", "email")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        if message_type == "email":
            vectorizer = email_vectorizer
            tokenizer = email_tokenizer
            model = email_svm
            tfidf_transformer = tfidf_transformer_email
        else:
            vectorizer = sms_vectorizer
            tokenizer = sms_tokenizer
            model = sms_svm
            tfidf_transformer = tfidf_transformer_sms

        processed_text, padded_seq = preprocess_text(text, vectorizer, tokenizer, tfidf_transformer)
        sentiment_score = sentiment_lstm.predict(padded_seq)[0][0]
        combined_features = np.hstack((processed_text.toarray(), [[sentiment_score]]))

        # Ensure correct feature size
        num_features_expected = model.coef_.shape[1]  # Model expects 5004 features
        num_features_actual = combined_features.shape[1]

        if num_features_actual < num_features_expected:
            padding = np.zeros((1, num_features_expected - num_features_actual))
            combined_features = np.hstack((combined_features, padding))

        prediction = model.predict(combined_features)[0]
        probability = model.decision_function(combined_features)[0]

        return jsonify({
            "result": "Spam" if prediction == 1 else "Ham",
            "score": float(probability),
            "sentiment": float(sentiment_score)
        })
    except Exception as e:
        return jsonify({"error": "Error processing SMS", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
