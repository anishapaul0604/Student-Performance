from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import numpy as np
import os
import joblib

app = Flask(__name__, static_url_path='/static')

# Load the model bundle
try:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "student_pass_predictor.pkl")
    model_bundle = joblib.load(MODEL_PATH)
    rf_model = model_bundle['rf_model']
    model_columns = model_bundle['columns']
    feature_names = model_bundle['feature_names']
    
    print("Model loaded successfully")
    print("Features required:", feature_names)
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'gender': request.form.get('gender'),
            'race/ethnicity': request.form.get('race_ethnicity'),
            'parental level of education': request.form.get('parental_level_of_education'),
            'lunch': request.form.get('lunch'),
            'test preparation course': request.form.get('test_preparation_course')
        }

        # Validate input data
        if None in input_data.values() or "" in input_data.values():
            raise ValueError("All fields are required")

        # Create DataFrame with single row
        input_df = pd.DataFrame([input_data])
        
        # Convert to dummy variables
        input_encoded = pd.get_dummies(input_df)
        
        # Align input features with model columns
        final_input = pd.DataFrame(columns=model_columns)
        for col in model_columns:
            if col in input_encoded.columns:
                final_input[col] = input_encoded[col]
            else:
                final_input[col] = 0

        # Make prediction
        prediction = rf_model.predict(final_input)[0]
        prediction_proba = rf_model.predict_proba(final_input)[0]
        
        # Get confidence score
        confidence = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        # Create result message
        result = {
            'success': True,
            'prediction': 'Pass' if prediction == 1 else 'Fail',
            'confidence': f"{confidence * 100:.2f}%",
            'message': f"Student predicted to {'Pass' if prediction == 1 else 'Fail'} with {confidence * 100:.2f}% confidence"
        }
        
        print("Prediction details:", result)
        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/')
def home():
    return render_template('predict_form.html')

@app.route('/home')
def home_page():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)