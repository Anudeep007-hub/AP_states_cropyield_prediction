import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import LabelEncoder
from Model3 import EnhancedRandomForestRegressor
from Model3 import EnhancedDecisionTreeRegressor

# Load the saved model and label encoders
with open('enhanced_random_forest_regressor.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Preprocessing function
def preprocess_user_input(user_input, label_encoders):
    categorical_input = user_input[:3]
    numerical_input = user_input[3:]

    # Use the transform method of LabelEncoder
    categorical_input = [
        label_encoders['Crop'].transform([categorical_input[0]])[0],  # Encodes the 'Crop' value
        label_encoders['District'].transform([categorical_input[1]])[0],  # Encodes the 'District' value
        label_encoders['Season'].transform([categorical_input[2]])[0],  # Encodes the 'Season' value
    ]

    processed_input = np.array(categorical_input + numerical_input)

    area, annual_temp, fertilizer, annual_rainfall = processed_input[3:]
    rainfall_fertilizer = annual_rainfall * fertilizer  # Only derived feature

    processed_input = np.concatenate((processed_input, [rainfall_fertilizer]))

    final_input = processed_input.reshape(1, -1)
    return final_input

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form data
        crop = request.form['crop']
        district = request.form['district']
        season = request.form['season']
        area = float(request.form['area'])
        annual_temp = float(request.form['annual_temp'])
        fertilizer = float(request.form['fertilizer'])
        rainfall = float(request.form['rainfall'])

        # Prepare user input for prediction
        user_input = [crop, district, season, area, annual_temp, fertilizer, rainfall]
        processed_input = preprocess_user_input(user_input, label_encoders)

        # Predict yield using the loaded model
        yield_prediction = rf_model.predict(processed_input)

        # Render result
        return render_template('index.html', prediction=yield_prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
