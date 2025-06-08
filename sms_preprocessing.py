import os
import string
import warnings
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from joblib import parallel_backend
import re

warnings.simplefilter(action='ignore', category=FutureWarning)

# Ensure necessary nltk resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
data_path = "sms_dataset.csv"
spam_df = pd.read_csv(data_path, encoding='latin-1', usecols=['v1', 'v2'])
spam_df.columns = ["label", "text"]
spam_df.drop_duplicates(inplace=True)

# Convert labels to binary format
spam_df["label"] = spam_df["label"].map({"ham": 0, "spam": 1})

# Handle NaN values in 'label' and 'text' columns
spam_df["label"] = spam_df["label"].fillna(0).astype(int)
spam_df["text"] = spam_df["text"].astype(str).fillna("")

# Function to extract URLs and analyze them
def analyze_urls(text):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(urls)

# Preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    url_count = analyze_urls(text)
    text = re.sub(r'http\\S+|www\\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\d{10,}', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word.isalpha()]
    return ' '.join(words), url_count

# Apply preprocessing and URL analysis
spam_df[["preprocessed_text", "url_count"]] = spam_df["text"].apply(lambda x: pd.Series(preprocess_text(x)))

# Tokenization for LSTM
max_words = 5000
max_len = 200
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(spam_df["preprocessed_text"])

# Lưu tokenizer
joblib.dump(tokenizer, "sms_tokenizer.pkl")
print("Tokenizer saved successfully!")

sequences = tokenizer.texts_to_sequences(spam_df["preprocessed_text"])
padded_sequences = pad_sequences(sequences, maxlen=max_len)

# Train and Save LSTM Model
sentiment_model = Sequential([
    Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dense(1, activation='sigmoid')
])

sentiment_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
sentiment_model.fit(padded_sequences, spam_df["label"], epochs=5, batch_size=32, validation_split=0.2)

# Save LSTM Model
sentiment_model.save("./sentiment_lstm_model.keras")

# Predict Sentiment Scores
sentiment_predictions = sentiment_model.predict(padded_sequences).astype(np.float32)
spam_df["sentiment_score"] = sentiment_predictions[:, 0]

# Feature Extraction for SVM
vectorizer = CountVectorizer(max_features=5002)
tfidf_transformer = TfidfTransformer()
bow = vectorizer.fit_transform(spam_df["preprocessed_text"])
tfidf = tfidf_transformer.fit_transform(bow)

# Save vectorizer for future inference
joblib.dump(vectorizer, "./sms_vectorizer.pkl")
joblib.dump(tfidf_transformer, "./sms_tfidf_transformer.pkl")

# Combine TF-IDF, Sentiment Scores, and URL Analysis
X_combined = np.hstack((tfidf.toarray().astype(np.float32), sentiment_predictions, spam_df[["url_count"]].to_numpy().astype(np.float32)))

# Remove NaN values before applying SMOTE
X_combined = np.nan_to_num(X_combined)

# Train-Test Split with SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_combined, spam_df["label"], test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train SVM Model
svm_model = SGDClassifier(loss="hinge", random_state=42, max_iter=1000, tol=1e-3)
with parallel_backend('threading', n_jobs=-1):
    svm_model.fit(X_train, y_train)

# Save trained SVM model
joblib.dump(svm_model, "./sms_svm_sentiment_model.pkl")

# Evaluate Model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for SMS Spam Detection Model")
plt.show()

print("Hybrid SMS Spam Detection Model Trained and Saved Successfully!")
print("\nEVALUATION METRICS:")
print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1-score   : {f1:.4f}")

# Kiểm tra số lượng đặc trưng trong vectorizer
print(f"SMS Vectorizer Features: {len(vectorizer.get_feature_names_out())}")

# Kiểm tra số lượng đặc trưng trong tfidf
print(f"TF-IDF Shape: {tfidf.shape}")

# Kiểm tra số lượng đặc trưng của X_combined
print(f"X_combined Shape: {X_combined.shape}")
sms_svm = joblib.load("./sms_svm_sentiment_model.pkl")

# Kiểm tra số lượng đặc trưng mà SVM mong đợi
print(f"SVM Expected Features: {sms_svm.coef_.shape[1]}")