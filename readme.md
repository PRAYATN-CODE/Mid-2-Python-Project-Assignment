# 📂 Machine Learning & AI Portfolio Hub

![Python](https://img.shields.io/badge/Python-3.12-blue.svg) 
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg) 
![Machine Learning](https://img.shields.io/badge/ML-Supervised%20%26%20Unsupervised-orange.svg)
![Deep Learning](https://img.shields.io/badge/DL-LSTM%20%2F%20GRU-green.svg)

Welcome to the **Machine Learning Assignment Suite**. This repository contains five industrial-grade data science projects, each featuring a robust training pipeline and an interactive **Streamlit** dashboard.

---

## 🗺️ Project Navigation
1. [Fake News Detection](#1-fake-news-detection)
2. [Customer Segmentation](#2-customer-segmentation)
3. [Credit Card Fraud Detection](#3-credit-card-fraud-detection)
4. [Stock Price Prediction](#4-stock-price-prediction)
5. [Movie Recommender System](#5-movie-recommender-system)

---

## 📰 1. Fake News Detection
**Objective:** Classify global news articles as "Real" or "Fake" using NLP.
* **Algorithms:** Passive Aggressive Classifier, Naive Bayes, Logistic Regression.
* **Preprocessing:** Lemmatization, Stopword removal, and N-gram (1,2) TF-IDF Vectorization.
* **Key Achievement:** Developed a "Publisher-Strip" regex to remove data leakage (e.g., "Reuters" tags) ensuring the model learns actual context rather than source names.

## 📊 2. Customer Segmentation
**Objective:** Group customers based on purchasing behavior using the UK Online Retail dataset.
* **Methodology:** **RFM Analysis** (Recency, Frequency, Monetary).
* **Clustering:** K-Means (Elbow Method), DBSCAN, and Hierarchical Clustering.
* **UI:** Interactive **3D Scatter Plots** allowing users to rotate and inspect global customer clusters.

## 🛡️ 3. Credit Card Fraud Detection
**Objective:** Identify fraudulent transactions using a dataset of 1 million records.
* **Techniques:** **SMOTE** for handling extreme class imbalance.
* **Explainability:** Integrated **SHAP** values to provide transparency on why a transaction was flagged as high-risk.
* **Design:** Professional "Banking Terminal" UI with custom CSS.

## 📈 4. Stock Price Prediction
**Objective:** Forecast future closing prices for global assets (Bitcoin, Ethereum, S&P 500).
* **Data:** Live market data fetched via `yfinance` API.
* **Models:** Comparison between **LSTM** (Long Short-Term Memory) and **GRU** neural networks.
* **UI:** Real-time forecasting charts with automated data scaling.

## 🍿 5. Movie Recommender System
**Objective:** A hybrid recommendation engine using the `mymoviedb.csv` dataset.
* **Hybrid Logic:** * **Collaborative:** SVD Matrix Factorization for existing profiles.
    * **Content-Based:** TF-IDF Similarity on movie overviews and genres.
* **Cold-Start Solution:** Popularity-based demographic filtering for new users.
* **Visuals:** Card-based UI displaying actual TMDB posters and metadata.

---

## 📦 Data Management & `.gitignore`

To ensure optimal repository performance and adhere to GitHub's file size limitations, this project utilizes a `.gitignore` file. 

### What is ignored?
* **Large Datasets:** All `.csv` and `.xlsx` files are ignored. You must place your own data files (e.g., `card_transdata.csv`, `mymoviedb.csv`) into the project folders locally.
* **Model Artifacts:** Trained models (`.pkl`, `.h5`) are not pushed to the cloud. You must run the `train_model.py` scripts locally to generate these files before launching the Streamlit apps.
* **Python Environment:** Virtual environments and cache files are excluded.

### How to use with your own data:
1. Download the required datasets mentioned in each project section.
2. Place them in their respective project directories.
3. If you have renamed a file, ensure the filename in `train_model.py` matches your local file.

## ⚙️ Installation & Setup

1. **Clone the project:**
   ```bash
   git clone [https://github.com/PRAYATN-CODE/Mid-2-Python-Project-Assignment](https://github.com/PRAYATN-CODE/Mid-2-Python-Project-Assignment.git)