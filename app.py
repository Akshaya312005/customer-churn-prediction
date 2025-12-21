import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <h1 style="text-align:center;">Customer Churn Analytics Dashboard</h1>
    <p style="text-align:center; font-size:16px;">
    Predict whether a telecom customer is likely to leave the service
    </p>
    <p style="text-align:center; font-size:13px;">
    Developed by <b>Akshaya</b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])

tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)

st.sidebar.header("Services")

phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
online_backup = st.sidebar.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
device_protection = st.sidebar.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
tech_support = st.sidebar.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
streaming_tv = st.sidebar.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
streaming_movies = st.sidebar.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

st.sidebar.header("Billing Details")

contract = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.sidebar.number_input(
    "Monthly Charges", min_value=0.0, value=70.0
)
total_charges = st.sidebar.number_input(
    "Total Charges", min_value=0.0, value=1000.0
)

# ----------------------------
# Main Panel
# ----------------------------
st.subheader("Prediction Output")

if st.sidebar.button("Predict Churn Risk"):
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

        st.markdown("### Risk Assessment")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("High Risk Customer")
            else:
                st.success("Low Risk Customer")

        with col2:
            st.metric(
                label="Estimated Churn Probability",
                value=f"{probability:.2%}"
            )

        st.info(
            "This probability reflects the model's confidence based on historical patterns. "
            "It is not a guarantee of customer behavior."
        )

    except Exception as e:
        st.error("Prediction failed due to input or model mismatch.")
        st.code(str(e))
