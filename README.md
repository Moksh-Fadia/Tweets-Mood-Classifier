# 🧠 Twitter Mental Health Classification

This project aims to detect **signs of depression** in tweets using Natural Language Processing (NLP) and Machine Learning, enhanced with social metadata (e.g., followers, tweet time).

---

## 📂 Dataset

- **Source**: [Mental Health Social Media Dataset – Kaggle](https://www.kaggle.com/datasets/infamouscoder/mental-health-social-media)
- **Size**: ~8,000 tweets with features like `post_text`, `user_id`, `followers`, `friends`, `statuses`, `retweets`, and a binary `label`.

---

## 📌 Objective

> Build a model to classify tweets as **depressed** or **non-depressed** using tweet content and user-level metadata, while ensuring fairness and interpretability.

---

## 🧪 Dataset Overview

The dataset contains:

- `post_text`: Raw tweet content
- `label`: 1 (depressed), 0 (not depressed)
- Metadata:
  - `followers`, `friends`, `favourites`, `statuses`, `retweets`
  - `post_created`: timestamp (used to extract hour and weekday)
  - `user_id`: to ensure group-based splitting

---

## ⚙️ Key Steps

### 1. Preprocessing

- Lowercasing, removing URLs, mentions, emojis, punctuation
- Lemmatization using `spaCy`
- Stopword removal
- Handling null values and formatting timestamps

### 2. Text Representation

- TF-IDF vectorization
- N-gram tuning (`unigram`, `bigram`)
- `max_features`, `min_df`, `max_df`, and `sublinear_tf` hyperparameters

### 🧾 Feature Engineering

- Columns used (User-based metadata):
  - `followers`, `friends`, `favourites`, `statuses`, `retweets`
  - `hour`, `weekday` (from `post_created`)
- Scaled using `StandardScaler`  

---

## 🧪 Modeling Pipeline

- **Train/Test Split**:
  - Done using `GroupShuffleSplit` on `user_id` (grouping tweets) to prevent data leakage
- **Random Forest**
- Combined sparse matrix of TF-IDF + scaled metadata  

### 🧮 Evaluation

- Accuracy: ~**88.83%**
- Metrics:
  - Precision, Recall, F1-score
  - Confusion Matrix
  - Feature Importance ranking

---

## 📊 Visual Exploratory Data Analysis (EDA)

- **Tweet Time Plot**: Tweet frequency across hours split by label (shows late-night patterns)
- **Top Words per Class**: Bar plots of most frequent tokens for each label
- **KDE Plots**: Distributions of `followers`, `favourites`, and `statuses` by label
- **Mean Metadata Values**: Bar plot showing average values across metadata features by label
- **Feature Importances**: Top 20 features from Random Forest model


