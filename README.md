# Credit Card Fraud Detection API 💳

## 📌 Project Overview
This project was developed as part of my Machine Learning Internship at CodSoft. It is an end-to-end Machine Learning pipeline designed to identify fraudulent credit card transactions. 

Because fraudulent transactions are incredibly rare compared to normal ones, the primary challenge of this project was dealing with a highly imbalanced dataset. This repository showcases my ability to handle real-world data issues, train robust AI/ML classification models, and deploy them as a live web service.

## ✨ Key Highlights
* **Data Balancing:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic examples of fraudulent transactions, preventing the model from becoming biased toward normal transactions.
* **Model Selection:** Built and evaluated multiple classification models (Logistic Regression, Decision Trees). The **Random Forest** classifier was selected for its high accuracy and robustness.
* **Live Deployment:** Packaged the optimized machine learning model into a fully functional REST API using **FastAPI**, allowing users to send transaction details and receive instant fraud predictions.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, Imbalanced-learn (SMOTE)
* **API Deployment:** FastAPI, Uvicorn
