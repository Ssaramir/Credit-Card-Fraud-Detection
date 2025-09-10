from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


models_dir = Path("python_projects\\fraud_detection_system\\notebooks\\models")
models_dir.mkdir(parents=True, exist_ok=True) 

# loading the saved model and its components
model = joblib.load(models_dir / 'fraud_model.pkl')
feature_names = joblib.load(models_dir / 'feature_names.pkl')

app = Flask(__name__)

# Update your fraud_api.py file with this change:

@app.route('/predict', methods=['POST'])
def predict_fraud():
    try:
        # Get data from request
        data = request.json
        
        # Convert to DataFrame with correct feature names
        input_df = pd.DataFrame([data], columns=feature_names)
        
        # Make prediction with optimal threshold
        prediction_proba = model.predict_proba(input_df)[0]
        fraud_probability = float(prediction_proba[1])
        
        # Apply optimal threshold (0.9 instead of default 0.5)
        OPTIMAL_THRESHOLD = 0.9
        prediction = 1 if fraud_probability >= OPTIMAL_THRESHOLD else 0
        
        result = {
            'prediction': int(prediction),
            'fraud_probability': fraud_probability,
            'threshold_used': OPTIMAL_THRESHOLD,
            'status': 'fraud' if prediction == 1 else 'legitimate',
            'confidence': 'high' if fraud_probability > 0.8 else 'medium' if fraud_probability > 0.3 else 'low'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'API is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)