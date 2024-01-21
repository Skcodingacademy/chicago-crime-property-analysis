import streamlit as st
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Connect to the database
conn = sqlite3.connect('chicago-crime-property.db')

# Read data from the database into a DataFrame
query = 'SELECT * FROM property_with_crime'
property_with_crime_df = pd.read_sql(query, conn)

# Close the database connection
conn.close()

# Replace -1 with NaN in the 'year_built' column
property_with_crime_df['year_built'].replace(-1, pd.NA, inplace=True)

# Convert the entire DataFrame to numeric
property_with_crime_df = property_with_crime_df.apply(pd.to_numeric, errors='ignore')

# Assuming X and y are your feature matrix and target variable
X = property_with_crime_df[['longitude', 'latitude', 'sold_price', 'year_built']]
y = property_with_crime_df['crime_count']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Impute missing values in the 'year_built' column
imputer = SimpleImputer(strategy='mean')
X_train_scaled[:, 3:4] = imputer.fit_transform(X_train_scaled[:, 3:4])
X_test_scaled[:, 3:4] = imputer.transform(X_test_scaled[:, 3:4])

# Initialize Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the model
nb_classifier.fit(X_train_scaled, y_train)

# Streamlit App
st.title("Crime Prediction Streamlit App")

# Sidebar for user input
st.sidebar.header("User Input Features")

# Get user input for features
longitude = st.sidebar.slider("Longitude", float(X['longitude'].min()), float(X['longitude'].max()), float(X['longitude'].mean()))
latitude = st.sidebar.slider("Latitude", float(X['latitude'].min()), float(X['latitude'].max()), float(X['latitude'].mean()))
sold_price = st.sidebar.slider("Sold Price", float(X['sold_price'].min()), float(X['sold_price'].max()), float(X['sold_price'].mean()))
year_built = st.sidebar.slider("Year Built", float(X['year_built'].min()), float(X['year_built'].max()), float(X['year_built'].mean()))

# Make predictions
user_data = np.array([[longitude, latitude, sold_price, year_built]])
user_data_scaled = scaler.transform(user_data)
user_prediction = nb_classifier.predict(user_data_scaled)

# Display prediction
st.subheader("Prediction")
st.write(f"The predicted crime count is: {user_prediction[0]}")

# Additional functionalities can be added as needed
