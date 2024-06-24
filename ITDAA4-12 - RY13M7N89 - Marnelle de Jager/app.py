"""
Author: Marnelle de Jager
Date: 24/06/2024
Name: app.py
Description:
This script creates a web-based decision support system using Streamlit for predicting
heart disease based on user-input data. It loads a pre-trained machine learning model,
encoder, and scaler, allows users to input their health parameters through a user-friendly
interface, preprocesses the input data, makes predictions, and displays the prediction
result ('likely to have heart disease' or 'unlikely to have heart disease').
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the model, encoder, and scaler used during training
model = joblib.load('heart_disease_model.pkl')
encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Define user-friendly labels
chest_pain_types = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
restecg_types = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
sex_labels = {0: 'Female', 1: 'Male'}
fbs_labels = {0: 'No', 1: 'Yes'}
slope_labels = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
thal_labels = {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}

# Layout the input fields in three columns
st.title('Heart Disease Prediction Decision Support System')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age', min_value=1, max_value=100, value=25)
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: sex_labels[x])
    cp = st.selectbox('Chest Pain Type', options=list(chest_pain_types.keys()), format_func=lambda x: chest_pain_types[x])
    trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=200, value=120)
    
with col2:
    chol = st.number_input('Serum Cholesterol in mg/dl', min_value=100, max_value=500, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: fbs_labels[x])
    restecg = st.selectbox('Resting Electrocardiographic Results', options=list(restecg_types.keys()), format_func=lambda x: restecg_types[x])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)

with col3:
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2], format_func=lambda x: slope_labels[x])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', options=[1, 2, 3], format_func=lambda x: thal_labels[x])





# Create a prediction button
if st.button('Predict', key='predict_button', help="Click to make a prediction"):
    # Create a DataFrame for the input data
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    input_df = pd.DataFrame(input_data, columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])

    # Encode and scale the input data
    categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    input_encoded = encoder.transform(input_df[categorical_columns])
    input_scaled = scaler.transform(input_df[numeric_columns])

    # Combine encoded and scaled features
    input_processed = np.hstack((input_encoded, input_scaled))

    # Make a prediction
    prediction = model.predict(input_processed)
    
    # Display an image based on the prediction
    if prediction[0] == 1:
        st.write('The patient is likely to have heart disease.')
    else:
        st.write('The patient is unlikely to have heart disease.')

# To run the app, use the following command:
# streamlit run app.py

# Sample Data Table
st.subheader('Sample Data Table')
st.dataframe(pd.DataFrame({
    'Age': [63, 37, 41],
    'Sex': ['Male', 'Male', 'Female'],
    'Chest Pain Type': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain'],
    'Resting Blood Pressure': [145, 130, 130],
    'Serum Cholesterol': [233, 250, 204],
    'Fasting Blood Sugar': [1, 0, 0],
    'Resting ECG': ['Normal', 'ST-T Wave Abnormality', 'Normal'],
    'Maximum Heart Rate': [150, 187, 172],
    'Exercise Induced Angina': [0, 0, 0],
    'ST Depression': [2.3, 3.5, 1.4],
    'Slope': [0, 0, 2],
    'Major Vessels': [0, 0, 0],
    'Thalassemia': [1, 2, 2]
}))