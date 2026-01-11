# -*- coding: utf-8 -*-
"""Fraud detection"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set(style='whitegrid')

MODEL_PATH = 'fraud_detection_pipeline.pkl'

def train_and_save(df, model_path=MODEL_PATH):
    df = df.copy()
    df['balanceDiff0rig'] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df['balanceDiffDest'] = df["newbalanceDest"] - df["oldbalanceDest"]

    df_model = df.drop(columns=['nameOrig','nameDest'], errors='ignore')

    categorical = ['type']
    numeric = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    y = df_model['isFraud']
    X = df_model.drop('isFraud', axis=1)

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(), categorical)
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    import joblib
    joblib.dump(pipeline, model_path)
    return pipeline


def load_or_train_model():
    import joblib
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    candidates = [
        'AIML Dataset (2).csv',
        'AIML Dataset.csv',
        'dataset.csv',
        'data.csv'
    ]
    for c in candidates:
        if os.path.exists(c):
            df = pd.read_csv(c)
            return train_and_save(df)

    try:
        import streamlit as st
        st.warning('No dataset found in project folder. Upload CSV to train the model.')
        uploaded = st.file_uploader('Upload dataset CSV (training only)', type='csv')
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            return train_and_save(df)
        st.stop()
    except Exception:
        raise FileNotFoundError("Dataset not found")


if __name__ == '__main__':
    import joblib
    import streamlit as st

    model = None
    try:
        model = load_or_train_model()
    except FileNotFoundError as e:
        raise

    st.title('Fraud Detection App')
    st.markdown('Enter the transaction details and press Predict')
    transaction_type = st.selectbox('Transaction Type', ['CASH_OUT','TRANSFER','PAYMENT','CASH_IN','DEBIT'])
    amount = st.number_input('Amount', min_value=0.0, value=1000.0)
    old_balance_org = st.number_input('Old Balance (Sender)', min_value=0.0, value=1000.0)
    new_balance_org = st.number_input('New Balance (Sender)', min_value=0.0, value=1000.0)
    old_balance_dest = st.number_input('Old Balance (Receiver)', min_value=0.0, value=0.0)
    new_balance_dest = st.number_input('New Balance (Receiver)', min_value=0.0, value=0.0)

    if st.button('Predict'):
        input_data = pd.DataFrame([{
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': old_balance_org,
            'newbalanceOrig': new_balance_org,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'isFlaggedFraud': 0,
            'balanceDiff0rig': old_balance_org - new_balance_org,
            'balanceDiffDest': new_balance_dest - old_balance_dest
        }])

        prediction = model.predict(input_data)[0]
        st.subheader(f'Prediction: {int(prediction)}')
        if prediction == 1:
            st.error('Fraudulent Transaction')
        else:
            st.success('Non-Fraudulent Transaction')
