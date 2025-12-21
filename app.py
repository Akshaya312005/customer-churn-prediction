import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# App Config
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Analytics Dashboard")
st.write(
    "This application predicts the likelihood of a telecom customer "
    "leaving the service based on account and usage details."
)

st.caption("Developed by Akshaya")

# ----------------------------
# Load Trained Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

# ----------------------------
# User Input Form
# ----------------------------
st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet_service = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
online_backup = st.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
device_protection = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
tech_support = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
streaming_tv = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
streaming_movies = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)
total_charges = st.number_input(
    "Total Charges", min_value=0.0, value=1000.0
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Churn Risk"):
    input_df = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    })

    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(
                f"High risk of churn\n\nEstimated probability: {probability:.2%}"
            )
        else:
            st.success(
                f"Low risk of churn\n\nEstimated probability: {probability:.2%}"
            )

        st.caption(
            "Probability indicates the model's confidence, "
            "not a medical or financial guarantee."
        )

    except Exception as e:
        st.error("Prediction failed due to input or model mismatch.")
        st.code(str(e))
