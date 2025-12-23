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
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
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
