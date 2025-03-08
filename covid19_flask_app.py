from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to your COVID-19 prediction model
model_path = "./models/Random_Forest.pkl"

# Load the model with error handling
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Updated feature list for COVID-19 symptoms
features = [
    'Chest pain', 'Chills or sweats', 'Confused or disoriented', 'Cough',
    'Diarrhea', 'Difficulty breathing or Dyspnea', 'Cough with sputum',
    'Cough with heamoptysis', 'Wheezing'
]

# Root route to render the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # Ensure 'index.html' is inside the templates folder

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json  # Receive JSON data
    input_data = {feature: data.get(feature, 0) for feature in features}  # Prepare data for prediction
    input_df = pd.DataFrame([input_data])  # Convert to DataFrame
    prediction = model.predict(input_df)[0]  # Get prediction

    # Convert numerical prediction to text result
    result_text = "Positive" if prediction == 1 else "Negative"

    return jsonify({"prediction": result_text})

if __name__ == '__main__':
    app.run(debug=True)
