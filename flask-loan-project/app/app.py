from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model/random_forest_model.pkl')

# Preprocessing function
def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Define all purpose categories as used in model training
    purpose_categories = ['credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'major_purchase', 'small_business']

    # Initialize all purpose columns to False
    for category in purpose_categories:
        input_df[f'purpose_{category}'] = False

    # Set the selected purpose to True
    selected_purpose = input_data.get('purpose', '')
    if selected_purpose in purpose_categories:
        input_df[f'purpose_{selected_purpose}'] = True

    # Drop the original 'purpose' column
    input_df.drop('purpose', axis=1, inplace=True)

    # StandardScaler for numerical columns
    numeric_columns = ['credit.policy', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
    scaler = StandardScaler()
    input_df[numeric_columns] = scaler.fit_transform(input_df[numeric_columns])

    return input_df


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve data from form and preprocess
        form_data = request.form.to_dict()
        preprocessed_data = preprocess_input(form_data)
        prediction = model.predict(preprocessed_data)
        output = 'Loan will be fully paid' if prediction[0] == 0 else 'Loan will not be fully paid'
        return render_template('index.html', prediction_text=output)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)

