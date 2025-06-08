# 🔍 Dự án Phát hiện Spam bằng Machine Learning (SVM) & Deep Learning (LSTM/Bi-LSTM)

Dự án này áp dụng kết hợp các kỹ thuật học máy (SVM) và học sâu (LSTM, BiLSTM) để phát hiện spam trong email và SMS.

## 📂 Mô tả dự án

- **Spam Detection**: Sử dụng SVM để phân loại văn bản là spam hoặc không spam.
- **Email Sentiment Analysis**: Sử dụng mô hình LSTM để phân tích cảm xúc trong email.
- **SMS Sentiment Analysis**: Sử dụng mô hình BiLSTM để phân tích cảm xúc trong tin nhắn SMS.

## 📥 Dữ liệu

1. Truy cập liên kết sau để tải về dữ liệu:
   👉 [Tải dataset tại đây (Google Drive)](https://drive.google.com/drive/u/1/folders/1ZeUBZVLjXZJ48s7nv3xNmGSnO9a_vv-N)

2. Sau khi tải xong, thêm **2 file dataset** đó vào thư mục chính của dự án.

## ⚙️ Cài đặt và chạy dự án

1. **Tạo mô hình từ dữ liệu:**

```bash
# Tạo mô hình phát hiện spam SMS
python sms_preprocessing.py

# Tạo mô hình phát hiện spam email
python email_preprocessing.py
````

2. **Chạy API FastAPI:**

```bash
uvicorn app:app --reload
```

Sau khi chạy thành công, truy cập [http://127.0.0.1:8000/](http://127.0.0.1:8000/) để xem giao diện Swagger UI và test các API.

## 🧠 Mô hình sử dụng

* `SVM` cho phân loại spam.
* `LSTM` cho phân tích cảm xúc email.
* `Bi-LSTM` cho phân tích cảm xúc SMS.
* `TF-IDF` cho trích xuất đặc trưng từ dữ liệu văn bản.

## 🧰 Công nghệ

* Python
* FastAPI
* Scikit-learn
* TensorFlow / Keras
* Pandas, NumPy

---

**Tác giả**: Quan Nguyen Duong Minh
📧 Email: [mquan5526@gmail.com](mailto:mquan5526@gmail.com)
📍 Thành phố Hồ Chí Minh, Việt Nam
