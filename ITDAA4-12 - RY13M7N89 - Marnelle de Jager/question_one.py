"""
Author: Marnelle de Jager
Date: 24/06/2024
Name: question_one.py
Description:
This script loads heart disease data from a CSV file into a SQLite database,
creates a table, inserts the data into the table, and verifies the number of records loaded.
"""

import sqlite3
import pandas as pd

# Load the dataset
file_path = 'heart_disease_data.csv'
heart_disease_data = pd.read_csv(file_path, delimiter=';')

# Create a SQLite database connection
conn = sqlite3.connect('heart_disease.db')
cursor = conn.cursor()

# Create a table in the database
create_table_query = """
CREATE TABLE heart_disease (
    age INTEGER,
    sex INTEGER,
    cp INTEGER,
    trestbps INTEGER,
    chol INTEGER,
    fbs INTEGER,
    restecg INTEGER,
    thalach INTEGER,
    exang INTEGER,
    oldpeak REAL,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    target INTEGER
)
"""
cursor.execute(create_table_query)

# Load the CSV data into the database
heart_disease_data.to_sql('heart_disease', conn, if_exists='replace', index=False)

# Verify that the data has been loaded successfully
cursor.execute("SELECT COUNT(*) FROM heart_disease")
record_count = cursor.fetchone()[0]

# Close the connection
conn.close()

print(f'Record count: {record_count}')
