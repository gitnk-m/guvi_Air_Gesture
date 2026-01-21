# ✋ Air Gesture Digit Recognition

An end-to-end deep learning application that recognizes **digits drawn in the air using smartphone motion sensors**. Sensor data is collected via an API, preprocessed into time-series sequences, and classified using multiple neural network architectures. The system is deployed as an **interactive Streamlit application**.

---

## 📌 Project Overview

This project enables users to draw digits (0–9) in the air by moving a mobile phone. Motion sensor data (accelerometer and gyroscope) is captured, truncated, normalized, and passed to trained deep learning models to predict the digit drawn. Multiple models are evaluated to identify the most effective architecture for air-gesture recognition.

---

## 🚀 Features

- 📱 Mobile motion sensor data via API (e.g., Phyphox)
- 🔄 Automatic preprocessing and truncation
- 🧠 Deep learning models:
  - CNN
  - LSTM
  - CNN + LSTM
- 📊 Detailed evaluation (accuracy, precision, recall, F1-score)
- 📉 Confusion matrices and learning curves
- 🌐 Streamlit-based interactive UI

---

## 🧠 Models Used

### CNN (Best Model)
- Strong spatial feature extraction
- Fast convergence and high generalization

### LSTM
- Temporal modeling
- Performed poorly on short gesture sequences

### CNN + LSTM
- Combined spatial + temporal modeling
- Slightly lower performance than CNN-only

---

## 📊 Results & Evaluation

Evaluation performed on a balanced test set of **600 samples (60 per digit)**.

| Model | Accuracy | Weighted F1-score |
|-----|---------|------------------|
| CNN Only | ~99.7% | 0.9966 |
| CNN + LSTM | ~99% | 0.9899 |
| LSTM Only | ~15% | 0.0544 |

### Key Conclusion
**CNN-only outperformed both LSTM-only and CNN+LSTM models.**

---

## 🗂️ Project Structure

```
Air-Gesture/
├── data/
├── models/
├── src/
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
streamlit run src/app_cnn_multi.py
```

---

## 👤 Author

**Nirmalkumar Marimuthu**
