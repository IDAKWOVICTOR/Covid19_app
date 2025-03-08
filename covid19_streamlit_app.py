import streamlit as st
import joblib
import pandas as pd

# Path to your COVID-19 prediction model
model_path = "./models/Random_Forest.pkl"

# Load the model with error handling
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

# Updated feature list for COVID-19 symptoms
features = [
    'Chest pain', 'Chills or sweats', 'Confused or disoriented', 'Cough',
    'Diarrhea', 'Difficulty breathing or Dyspnea', 'Cough with sputum',
    'Cough with heamoptysis', 'Wheezing'
]

# Add company logo at the top
st.image("./static/images/eHA-logo-blue_320x132.png", width=320)

# Title
st.title("COVID-19 Symptom Prediction")

# Instructions
st.write("Please select the symptoms you are experiencing.")

# Create user input form using buttons for Yes/No
symptoms = {}

for feature in features:
    # Display buttons for each symptom
    st.write(feature)
    symptom = st.radio(f"Do you have {feature}?", ("No", "Yes"))
    
    # Map response to 0 (No) and 1 (Yes)
    symptoms[feature] = 1 if symptom == "Yes" else 0

# Predict button
if st.button("Submit Prediction"):
    if model is None:
        st.error("Model not loaded")
    else:
        # Prepare data for prediction
        input_df = pd.DataFrame([symptoms])

        # Make prediction
        prediction = model.predict(input_df)[0]

        # Convert numerical prediction to text result
        result_text = "Positive" if prediction == 1 else "Negative"

        # Display the result
        st.subheader(f"Prediction: {result_text}")
