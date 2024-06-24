"""
Author: Marnelle de Jager
Date: 24/06/2024
Name: question_two.py
Description:
This script analyzes heart disease data through exploratory data analysis and modeling. 
It visualizes the distribution of categorical and numeric variables related to heart 
disease, utilizing seaborn for visualizations and pandas for data manipulation. The 
script loads the dataset from a CSV file, handles missing values, standardizes numeric 
variables, and encodes categorical variables. It then plots the distribution of classes 
for categorical variables and the density distribution for numeric variables. This 
analysis helps in understanding the data patterns before performing further modeling 
steps such as classification using Logistic Regression, Decision Tree, and Random Forest 
classifiers. The script concludes by closing any connections and saving necessary 
preprocessing transformers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from CSV file
file_path = 'heart_disease_data.csv'
heart_disease_data = pd.read_csv(file_path, delimiter=';')

# Handle missing values (if any)
heart_disease_data = heart_disease_data.dropna()

# Convert categorical variables to appropriate formats
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
heart_disease_data[categorical_columns] = heart_disease_data[categorical_columns].astype('category')

# Define mapping for categorical variables for easy reading
category_labels = {
    'sex': {0: 'Female', 1: 'Male'},
    'cp': {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'},
    'fbs': {0: 'FBS < 120 mg/dl', 1: 'FBS > 120 mg/dl'},
    'restecg': {0: 'Normal', 1: 'ST-T wave abnormality', 2: 'Left ventricular hypertrophy'},
    'exang': {0: 'No', 1: 'Yes'},
    'slope': {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'},
    'ca': {0: '0 vessels', 1: '1 vessel', 2: '2 vessels', 3: '3 vessels'},
    'thal': {1: 'Normal', 2: 'Fixed defect', 3: 'Reversible defect'},
    'target': {'0': 'No Disease', '1': 'Disease'}
}

# Define custom labels for the x-axis
x_axis_labels = {
    'sex': 'Sex',
    'cp': 'Chest Pain Type',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG',
    'exang': 'Exercise Induced Angina',
    'slope': 'Slope of ST Segment',
    'ca': 'Number of Major Vessels',
    'thal': 'Thalassemia'
}

# Function to add counts on the bars
def add_counts(ax):
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    fontsize=11, color='black', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', linewidth=0.5))

# Plot the distribution of classes for each categorical variable based on the target variable
for var in categorical_columns[:-1]:  # Exclude 'target' column
    plt.figure(figsize=(10, 6))
    sns.countplot(x=var, hue='target', data=heart_disease_data, palette={0: 'blue', 1: 'red'})
    plt.title(f'Distribution of Heart Disease by {x_axis_labels.get(var, var.replace("_", " ").title())}')
    plt.xlabel(x_axis_labels.get(var, var.replace('_', ' ').title()))
    plt.ylabel('Number of Patients')
    
    # Relabel the x-axis with more descriptive names
    if var in category_labels:
        ax = plt.gca()
        try:
            ax.set_xticks(ax.get_xticks())  # Ensure the ticks are fixed
            ax.set_xticklabels([category_labels[var][int(label.get_text())] for label in ax.get_xticklabels()])
        except KeyError as e:
            print(f"KeyError for variable {var}: {e}")
    
    plt.legend(title='Heart Disease', labels=['No', 'Yes'])
    add_counts(ax)
    
    plt.tight_layout()
    plt.show()
    plt.close()

print("Distribution of classes for categorical variables plotted successfully")

# Plot the distribution of each numeric variable based on the target variable
numericVariables = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for var in numericVariables:
    plt.figure(figsize=(10, 6))

    # Filter data based on target
    data_no_disease = heart_disease_data[heart_disease_data['target'] == 0]
    data_disease = heart_disease_data[heart_disease_data['target'] == 1]

    # Plot KDE plots
    sns.kdeplot(data=data_no_disease[var], fill=True, color='blue', label='No Disease')
    sns.kdeplot(data=data_disease[var], fill=True, color='red', label='Disease')

    # Set custom title and x-axis label using x_axis_labels_numeric
    plt.title(f'Distribution of Heart Disease by {x_axis_labels.get(var, var.replace("_", " ").title())}', fontsize=16)
    plt.xlabel(x_axis_labels.get(var, var.replace('_', ' ').title()), fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='Heart Disease', labels=['No Disease', 'Disease'], fontsize=12)
    plt.show()
    plt.close()

print("Distribution of classes for numeric variables plotted successfully")
