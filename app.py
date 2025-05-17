from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extracting 20 yes/no inputs in the same order as in training
    input_values = [request.form.get(f'feature{i}') for i in range(1, 21)]

    # Convert 'yes'/'no' to 1/0
    try:
        processed_input = [1 if val.strip().lower() == 'yes' else 0 for val in input_values]
    except Exception as e:
        return render_template('index.html', prediction_text='Invalid input! Please enter Yes or No only.')

    # Scale the input
    features_scaled = scaler.transform([processed_input])

    # Make prediction
    prediction = model.predict(features_scaled)

    result = "Positive" if prediction[0] == 1 else "Negative"
    return render_template('index.html', prediction_text=f'COVID-19 Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)