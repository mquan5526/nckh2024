import os
import string
import warnings
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from joblib import parallel_backend
import re
import requests

warnings.simplefilter(action='ignore', category=FutureWarning)

# Ensure necessary nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data_path = "email_dataset.csv"
spam_df = pd.read_csv(data_path, encoding='latin-1')
spam_df.columns = ["spam", "text"]
spam_df.drop_duplicates(inplace=True)

# Handle NaN values in 'text' column
spam_df["text"] = spam_df["text"].astype(str).fillna("")
spam_df["spam"] = pd.to_numeric(spam_df["spam"], errors='coerce').fillna(0).astype(int)

# Preprocessing function
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d{10,}', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word.isalpha()]
    return ' '.join(words)

spam_df["preprocessed_text"] = spam_df["text"].apply(preprocess_text)

# Tokenization for LSTM
max_words = 5000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(spam_df["preprocessed_text"])
sequences = tokenizer.texts_to_sequences(spam_df["preprocessed_text"])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Save tokenizer
joblib.dump(tokenizer, "email_tokenizer.pkl")

# Train and Save LSTM Model
sentiment_model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    LSTM(128, return_sequences=True),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

sentiment_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
sentiment_model.fit(padded_sequences, spam_df["spam"], epochs=5, batch_size=32, validation_split=0.2)

# Save the trained sentiment model
sentiment_model.save("./sentiment_lstm_model.h5")

# Predict sentiment scores
sentiment_predictions = sentiment_model.predict(padded_sequences).astype(np.float32)
spam_df["sentiment_score"] = sentiment_predictions[:, 0]

# Feature Extraction for SVM
vectorizer = CountVectorizer(max_features=5002)
tfidf_transformer = TfidfTransformer()
bow = vectorizer.fit_transform(spam_df["preprocessed_text"])
tfidf = tfidf_transformer.fit_transform(bow)

# Save vectorizer
joblib.dump(vectorizer, "./email_vectorizer.pkl")
joblib.dump(tfidf_transformer, "./email_tfidf_transformer.pkl")

# Combine Features
X_combined = np.hstack((tfidf.toarray().astype(np.float32), sentiment_predictions))
X_train, X_test, y_train, y_test = train_test_split(X_combined, spam_df["spam"], test_size=0.2, random_state=42)

# Train SVM Model
svm_model = SGDClassifier(loss="hinge", random_state=42, max_iter=1000, tol=1e-3)
with parallel_backend('threading', n_jobs=-1):
    svm_model.fit(X_train, y_train)

# Save trained SVM model
joblib.dump(svm_model, "./email_svm_sentiment_model.pkl")

# Evaluate Model
y_pred = svm_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix For Spam Email Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
