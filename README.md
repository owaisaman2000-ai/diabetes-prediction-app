
# Diabetes Prediction Using Logistic Regression

## Project Overview
This project uses a Logistic Regression machine learning model to predict whether a person is likely to have diabetes based on medical measurements.

The model is deployed as an interactive web application using Streamlit.

## Live Application
[Click here to open the Diabetes Prediction App](https://diabetes-prediction-app-5blmyyjcdaasxqvudlcxqz.streamlit.app/)

## Dataset
The project uses a diabetes dataset containing the following features:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Target variable:** Outcome
- 0 = No diabetes prediction
- 1 = Diabetes prediction

## Data Preprocessing
The following preprocessing steps were used:

- Checked the dataset for missing values.
- Treated zero values in selected medical measurement columns as missing.
- Filled missing values using median imputation.
- Split the data into training and testing sets.
- Applied StandardScaler to scale the input features.

## Machine Learning Model
Logistic Regression was used for this binary classification problem.

The trained model and scaler were saved and used in the Streamlit application to generate predictions for user-provided inputs.

## Model Evaluation
The model was evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- ROC curve

Add the actual evaluation scores from your notebook here.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- GitHub

## How to Run Locally
1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the application using `streamlit run app.py`.

## Disclaimer
This project is for educational purposes only. Its predictions are not a medical diagnosis and should not be used as a substitute for professional medical advice.
