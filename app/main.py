import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# -----------------------------
# Load Model + Scaler
# -----------------------------
BASE = Path(__file__).parent  # path to /app
MODEL_PATH = BASE / "ml" / "model.pkl"
SCALER_PATH = BASE / "ml" / "scaler.pkl"

# Load model
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

# Load scaler
with open(SCALER_PATH, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Salary Prediction App")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
education = st.selectbox(
    "Education Level",
    ["High School or less", "Intermediate", "Graduation", "PG"]
)
experience_months = st.number_input(
    "Months of Experience",
    min_value=0,
    max_value=600,
    value=60
)

# Encode education
education_mapping = {
    "High School or less": 0,
    "Intermediate": 1,
    "Graduation": 2,
    "PG": 3
}
education_encoded = education_mapping[education]

# Prepare feature vector
features = np.array([[age, education_encoded, experience_months]], dtype=np.float64)

# Scale
features_scaled = scaler.transform(features)

# Predict
predicted_salary = model.predict(features_scaled)

# Display
st.write(f"### Predicted Salary: **₹{predicted_salary[0]:,.2f}**")
