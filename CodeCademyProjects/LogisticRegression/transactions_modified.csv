import seaborn
import pandas as pd
import numpy as np
import codecademylib3
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import codecademylib3

# Load the data
transactions = pd.read_csv('transactions_modified.csv')
print(transactions.head())
print(transactions.info())

# How many fraudulent transactions?
print(transactions.isFraud.sum())

# Summary statistics on amount column
median_amount=transactions.amount.describe()
print(median_amount)
# Create isPayment field
transactions.isPayment=transactions['type'].isin(['PAYMENT', 'DEBIT']).astype(int)

# Create isMovement field
transactions.isMovement=transactions['type'].isin(['CASH_OUT', 'TRANSFER']).astype(int)
print(transactions.head())
# Create accountDiff field
transactions.accountDiff=(transactions.oldbalanceOrg-transactions.oldbalanceDest).abs()

# Create features and label variables
features=transactions[['amount','isPayment','isMovement','accountDiff']]
label=transactions.isFraud

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features,label,test_size=0.3)


# Normalize the features variables
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# Fit the model to the training data
lrm=LogisticRegression()
lrm.fit(X_train,y_train,0.5)
y_predict=lrm.predict(X_test)
print(lrm.score(X_test,y_test))
print(lrm.score(X_train,y_train))
# Score the model on the training data


# Score the model on the test data


# Print the model coefficients

print(lrm.coef_)
# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])
transaction4= np.array([123678.31, 1.0, 0.0, 3.5])
# Create a new transaction
sample_transactions=np.array([transaction1,transaction2,transaction3,transaction4])
sample_transactions=scaler.transform(sample_transactions)

# Combine new transactions into a single array


# Normalize the new transactions


# Predict fraud on the new transactions
sample_predict=lrm.predict(sample_transactions)
print(sample_predict)
# Show probabilities on the new transactions
print(lrm.predict_proba(sample_transactions))