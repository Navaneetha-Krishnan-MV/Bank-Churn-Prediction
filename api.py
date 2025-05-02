from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
model = model_data['model']
scaler = model_data['scaler']

# Get expected feature names from the model
expected_features = model.feature_names_in_

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Convert to DataFrame
    input_data = pd.DataFrame([data])
    
    # Ensure all expected Geography columns exist
    for geo_col in ['Geography_Germany', 'Geography_Spain']:
        if geo_col not in input_data.columns:
            input_data[geo_col] = 0  # Set to 0 if not provided
    
    # Preprocess
    input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
    
    # Scale numerical features
    numerical_features = ['CreditScore', 'Age', 'Balance', 'NumOfProducts']
    input_data[numerical_features] = scaler.transform(input_data[numerical_features])
    
    # Reorder columns to match training data
    input_data = input_data.reindex(columns=expected_features, fill_value=0)
    
    # Predict
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    
    return jsonify({
        'prediction': prediction,
        'probability': probability
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)