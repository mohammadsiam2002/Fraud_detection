# Fraud Detection System
**Overview

This project implements an intelligent machine learning–based system for detecting fraudulent financial transactions.
The system analyzes transaction details such as transaction type, amount, and account balances to automatically classify transactions as fraudulent or legitimate.
The project covers the complete machine learning pipeline, including:

>Data analysis and preprocessing
>Feature engineering
>Model training and evaluation
>Model persistence (saving/loading)
>Deployment as a web application using Streamlit

**Objectives

>Automatically detect fraudulent transactions in financial systems.
>Reduce financial losses caused by fraud.
>Apply machine learning techniques to real-world transaction data.
>Provide a simple and interactive web interface for predictions.


**Dataset
Due to GitHub file size limits, the dataset is not stored in this repository.

You can download it from:
🔗 Dataset Link:
https://drive.google.com/file/d/1RcJRtQYILDjapaUG2okVZvq7i7x47EZA/view?usp=sharing


**Project Workflow

Dataset (CSV)
      ↓
Data Cleaning & Feature Engineering
      ↓
Preprocessing (Scaling & Encoding)
      ↓
Model Training (Logistic Regression)
      ↓
Model Evaluation
      ↓
Model Saved to Disk (.pkl)
      ↓
Streamlit Web App Loads Model
      ↓
User Inputs Transaction → Prediction (Fraud / Not Fraud)


**Dataset Description

The dataset contains financial transaction records, where each row represents one transaction.

Main Features:
type —> Transaction type (TRANSFER, CASH_OUT, PAYMENT, etc.)
amount —> Transaction amount
oldbalanceOrg -> Sender balance before transaction
newbalanceOrig —> Sender balance after transaction
oldbalanceDest —> Receiver balance before transaction
newbalanceDest —> Receiver balance after transaction
isFraud —> Target label (1 = Fraud, 0 = Normal)

Engineered Features:

balanceDiffOrig = oldbalanceOrg − newbalanceOrig

balanceDiffDest = newbalanceDest − oldbalanceDest


**Machine Learning Model

Algorithm: Logistic Regression
Type: Supervised Binary Classification.
Preprocessing:
Numerical features scaled using StandardScaler.
Categorical features encoded using OneHotEncoder.
Full pipeline built using ColumnTransformer and Pipeline.


**Model Evaluation

The model is evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix


**Web Application

#The trained model is deployed using Streamlit.
#The web app allows the user to:
1-Enter transaction details
2-Click Predict
Instantly see whether the transaction is:
✅ Legitimate
❌ Fraudulent


**Technologies Used

Python
Pandas, NumPy
Scikit-learn
Matplotlib, Seaborn
Streamlit


**Project Structure

fraud-detection/
│
├── fraud_detection.py                     # Streamlit web application
├── analysis_model.py             # Model training and saving script
├── fraud_detection_pipeline.pkl
├── AIML Dataset (2).csv
└── README.md
