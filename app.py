import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.caption("Developed by Akshaya")

# ---------------- MODEL TRAINING ----------------
@st.cache_resource
def train_model():
    # Load dataset
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert TotalCharges safely
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")
    df["totalcharges"].fillna(df["totalcharges"].median(), inplace=True)

    # Target variable
    y = df["churn"].map({"Yes": 1, "No": 0})

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluation
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    return model, auc

model, auc_score = train_model()

st.success(f"Model trained successfully | ROC-AUC: {auc_score:.3f}")

# ---------------- USER INPUT UI ----------------
st.subheader("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
seniorcitizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
multiplelines = st.selectbox(
    "Multiple Lines", ["Yes", "No", "No phone service"]
)

internetservice = st.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

onlinesecurity = st.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
onlinebackup = st.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
deviceprotection = st.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
techsupport = st.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
streamingtv = st.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
streamingmovies = st.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)

paperlessbilling = st.selectbox(
    "Paperless Billing", ["Yes", "No"]
)

paymentmethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)

tenure = st.slider("Tenure (months)", 0, 72, 12)
monthlycharges = st.number_input(
    "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
)
totalcharges = st.number_input(
    "Total Charges", min_value=0.0, max_value=10000.0, value=1000.0
)

# ---------------- PREDICTION ----------------
if st.button("Predict Churn Risk"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges
    }])

    probability = model.predict_proba(input_df)[0][1]

    if probability > 0.5:
        st.error(f"High risk of churn ({probability * 100:.2f}%)")
    else:
        st.success(f"Low risk of churn ({(1 - probability) * 100:.2f}%)")
