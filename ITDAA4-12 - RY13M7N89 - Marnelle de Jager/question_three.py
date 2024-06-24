"""
Author: Marnelle de Jager
Date: 24/06/2024
Name: question_three.py
Description:
This script performs classification on heart disease data using Logistic Regression,
Decision Tree, and Random Forest classifiers. It loads the dataset, handles missing values,
standardizes numeric variables, encodes categorical variables, splits the data into
training and testing sets, trains the models, evaluates their performance, and saves
the best model along with necessary preprocessing transformers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
file_path = 'heart_disease_data.csv'
heart_disease_data = pd.read_csv(file_path, delimiter=';')

# Handle missing values (if any)
heart_disease_data = heart_disease_data.dropna()

# Convert categorical variables to appropriate formats
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
heart_disease_data[categorical_columns] = heart_disease_data[categorical_columns].astype('category')

# Normalize/scale numeric variables
numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
scaler = StandardScaler()
heart_disease_data[numeric_columns] = scaler.fit_transform(heart_disease_data[numeric_columns])

# Split the data into training and testing sets
X = heart_disease_data.drop('target', axis=1)
y = heart_disease_data['target'].astype('int')  # Ensure target is integer

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode categorical variables
encoder = OneHotEncoder(sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train[categorical_columns[:-1]])  # Exclude 'target'
X_test_encoded = encoder.transform(X_test[categorical_columns[:-1]])

# Combine with numeric columns
X_train_encoded = np.hstack((X_train_encoded, X_train[numeric_columns]))
X_test_encoded = np.hstack((X_test_encoded, X_test[numeric_columns]))

# Initialize models
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

# Fit models
lr.fit(X_train_encoded, y_train)
dt.fit(X_train_encoded, y_train)
rf.fit(X_train_encoded, y_train)

# Predict and evaluate
lr_pred = lr.predict(X_test_encoded)
dt_pred = dt.predict(X_test_encoded)
rf_pred = rf.predict(X_test_encoded)

# Calculate accuracy
lr_acc = accuracy_score(y_test, lr_pred)
dt_acc = accuracy_score(y_test, dt_pred)
rf_acc = accuracy_score(y_test, rf_pred)

print(f'Logistic Regression Accuracy: {lr_acc}')
print(f'Decision Tree Accuracy: {dt_acc}')
print(f'Random Forest Accuracy: {rf_acc}')

# Save the best model 
best_model = rf
joblib.dump(best_model, 'heart_disease_model.pkl')

# Save the encoder and scaler
joblib.dump(encoder, 'onehot_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
