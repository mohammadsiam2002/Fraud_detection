import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

df=pd.read_csv("AIML Dataset (2).csv")

df.head()

df.info()

df.columns

df['isFraud'].value_counts()

df['isFlaggedFraud'].value_counts()

df.isnull().sum()

df['type'].value_counts().plot(kind='bar',title='Transaction Types',color='skyblue')
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.show()

fraud_by_type=df.groupby('type')['isFraud'].mean().sort_values(ascending=False)
fraud_by_type.plot(kind='bar',title="Fraud Rate by Type",color='salmon')
plt.ylabel("Fraud Rate")
plt.show()

df['amount'].describe().astype(int)

sns.histplot(np.log1p(df['amount']),bins=100,kde=True,color='green')
plt.title("Transaction Amount Distribution (Log Scale)")
plt.xlabel("Log(Amount+1)")
plt.show()

df.columns

df['balanceDiff0rig']=df["oldbalanceOrg"]-df["newbalanceOrig"]
df["balanceDiffDest"]=df["newbalanceDest"]-df["oldbalanceDest"]

(df['balanceDiff0rig']<0).sum()

(df['balanceDiffDest']<0).sum()

df.head(2)

df.drop(columns='step',inplace=True)

df.head()

top_senders=df['nameOrig'].value_counts().head(10)
top_receivers=df['nameDest'].value_counts().head(10)

top_senders

top_receivers

fraud_users=df[df['isFraud']==1]['nameDest'].value_counts().head(10)
fraud_users

fraud_types=df[df['type'].isin(["TRANSFER","CASH_OUT"])]
fraud_types["type"].value_counts()

sns.countplot(data=fraud_types,x="type",hue="isFraud")
plt.title("Fraud Distribution in Transfer & Cash Out")
plt.show()

corr=df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']].corr()

corr

sns.heatmap(corr,annot=True,cmap='coolwarm',fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

zero_after_transfer=df[
    (df['oldbalanceOrg']>0)&
    (df['newbalanceOrig']==0)&
    (df['type'].isin(["TRANSFER","CASH_OUT"]))
]

len(zero_after_transfer)

df['isFraud'].value_counts()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df.head()

df_model=df.drop(columns=['nameOrig','nameDest'],axis=1)

df_model.head()

categorical=['type']
numeric=['amount',"oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]

y=df_model["isFraud"]
X=df_model.drop("isFraud",axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,stratify=y)

preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numeric),
        ('cat',OneHotEncoder(),categorical)
    ],
    remainder='drop'
)

pipeline=Pipeline([('prep',preprocessor),('clf',LogisticRegression(class_weight='balanced',max_iter=1000))
])

pipeline.fit(X_train,y_train)

y_pred=pipeline.predict(X_test)

print(classification_report(y_test,y_pred))

confusion_matrix(y_test,y_pred)

pipeline.score(X_test,y_test)*100
