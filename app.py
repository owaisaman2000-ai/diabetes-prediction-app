
import pickle
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load the saved model, scaler, and medians
@st.cache_resource
def load_artifacts():
    with open("diabetes_deployment.pkl", "rb") as file:
        artifacts = pickle.load(file)
    return artifacts

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
medians = artifacts["medians"]

# Feature order must match model training
feature_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# App heading
st.title("🩺 Diabetes Prediction App")
st.write(
    "Enter the details below to estimate the model's "
    "prediction for diabetes."
)

st.sidebar.header("Enter Patient Details")

pregnancies = st.sidebar.number_input(
    "Pregnancies", min_value=0, max_value=20, value=1
)
glucose = st.sidebar.number_input(
    "Glucose", min_value=0, max_value=300, value=120
)
blood_pressure = st.sidebar.number_input(
    "Blood Pressure", min_value=0, max_value=200, value=70
)
skin_thickness = st.sidebar.number_input(
    "Skin Thickness", min_value=0, max_value=100, value=20
)
insulin = st.sidebar.number_input(
    "Insulin", min_value=0, max_value=900, value=80
)
bmi = st.sidebar.number_input(
    "BMI", min_value=0.0, max_value=80.0, value=25.0
)
diabetes_pedigree = st.sidebar.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0, max_value=3.0, value=0.5
)
age = st.sidebar.number_input(
    "Age", min_value=1, max_value=120, value=30
)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=feature_names
    )

    # Replace zero values in measurement fields with saved medians
    measurement_columns = [
        "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI"
    ]

    for column in measurement_columns:
        if input_data.loc[0, column] == 0:
            input_data.loc[0, column] = medians[column]

    # Apply the same scaling used during training
    scaled_input = scaler.transform(input_data)

    # Generate prediction and probability
     prediction = model.predict(scaled_input)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Yes — the model predicts diabetes.")
    else:
        st.success("No — the model does not predict diabetes.")
