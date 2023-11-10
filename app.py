# Import necessary libraries and modules
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np  # For handling NaN values
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)

# Load your machine learning model and other necessary data
model = joblib.load("RFC.joblib")
ct = joblib.load('feature_values2')

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['GET','POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        # Retrieve form data
        data = {
            'discharge_disposition_id': request.form["discharge_disposition_id"],
            'admission_source_id': request.form["admission_source_id"],
            'time_in_hospital': request.form["time_in_hospital"],
            'num_medications': request.form["num_medications"],
            'number_emergency': request.form["number_emergency"],
            'number_inpatient': request.form["number_inpatient"],
            'diag_1': request.form["diag_1"],
            'diag_2': request.form["diag_2"],
            'max_glu_serum': request.form["max_glu_serum"],
            'glimepiride': request.form["glimepiride"],
            'diabetesMed': request.form["diabetesMed"]
        }
        
        # Perform data preprocessing and transformation
        feature_cols = ['discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
                        'num_medications', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'max_glu_serum', 'glimepiride', 'diabetesMed']

        data_df = pd.DataFrame([data], columns=feature_cols)
        data_df['max_glu_serum'] = data_df['max_glu_serum'].apply(lambda x: 1 if x == '>200' or x == '>300' else (0 if x == 'Norm' else -99))

# Map 'glimepiride' to numeric values
        data_df['glimepiride'] = data_df['glimepiride'].apply(lambda x: 1 if x == 'Yes' else 0)
        data_df['diabetesMed'] = data_df['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Convert specific columns to float values
        columns_to_convert_to_float = ['discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
                        'num_medications', 'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'max_glu_serum', 'glimepiride', 'diabetesMed']

        for column in columns_to_convert_to_float:
           data_df[column] = data_df[column].astype(float)
        data=ct.fit(data_df)
        
        pred = model.predict(ct.transform(data_df))

        if pred[0] == 1:
            prediction = "This patient will be readmitted"
        else:
            prediction = "This patient will not be readmitted"

    return render_template("prediction.html",prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
