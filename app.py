import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("Customer Churn Prediction")
st.caption("Developed by Akshaya")

# -------------------- MODEL TRAINING --------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert TotalCharges safely
    if "totalcharges" in df.columns:
        df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
        df["totalcharges"].fillna(df["totalcharges"].median(), inplace=True)

    # Target
    y = df["churn"].astype(str).str.lower().map({"yes": 1, "no": 0})

    # Features
    X = df.drop(columns=["customerid", "churn"])

    # Identify column types
    categorical_cols = X.select_dtypes(include="object").columns
    numerical_cols = X.select_dtypes(exclude="object").columns

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    # Model pipeline
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, auc, X.columns.tolist()

model, auc_score, feature_cols = train_model()

st.success(f"Model trained successfully | ROC-AUC: {auc_score:.3f}")

# -------------------- USER INPUT UI --------------------
st.subheader("Enter Customer Details")

user_input = {}

# Categorical selections
user_input["gender"] = st.selectbox("Gender", ["Male", "Female"])
user_input["seniorcitizen"] = st.selectbox("Senior Citizen", [0, 1])
user_input["partner"] = st.selectbox("Has Partner?", ["Yes", "No"])
user_input["dependents"] = st.selectbox("Has Dependents?", ["Yes", "No"])
user_input["phoneservice"] = st.selectbox("Phone Service", ["Yes", "No"])
user_input["multiplelines"] = st.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)

user_input["internetservice"] = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

user_input["onlinesecurity"] = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
user_input["onlinebackup"] = st.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
user_input["deviceprotection"] = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
user_input["techsupport"] = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
user_input["streamingtv"] = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
user_input["streamingmovies"] = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

user_input["contract"] = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)

user_input["paperlessbilling"] = st.selectbox(
    "Paperless Billing", ["Yes", "No"]
)

user_input["paymentmethod"] = st.selectbox(
=======
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
>>>>>>> 8cf95e191dfce5abc135bc03839f5ce962ebd9f2
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
<<<<<<< HEAD
        "Credit card (automatic)",
    ],
)

# Numerical inputs
user_input["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
user_input["monthlycharges"] = st.number_input(
    "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
)
user_input["totalcharges"] = st.number_input(
    "Total Charges", min_value=0.0, max_value=10000.0, value=1000.0
)

# -------------------- PREDICTION --------------------
if st.button("Predict Churn Risk"):
    input_df = pd.DataFrame([user_input])
    probability = model.predict_proba(input_df)[0][1]

    if probability > 0.5:
        st.error(f"High risk of churn ({probability*100:.2f}%)")
    else:
        st.success(f"Low risk of churn ({(1 - probability)*100:.2f}%)")
=======
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
>>>>>>> 8cf95e191dfce5abc135bc03839f5ce962ebd9f2
