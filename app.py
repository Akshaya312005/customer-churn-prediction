import streamlit as st
import pandas as pd
import joblib

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Customer Churn Analytics",
    layout="wide",
)

st.title("📊 Customer Churn Analytics Dashboard")
st.markdown(
    """
    This application estimates the likelihood of a customer leaving a telecom service 
    based on usage patterns and account details.
    """
)

# ---------------------------------
# Load model
# ---------------------------------
@st.cache_resource
def train_and_load_model():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df.columns=df.columns.str.strip()

    if 'customerID' in df.columns:
        df.drop("customerID",axis=1,inplace=True)

    if 'TotalCharges' in df.columns:
        df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    assert 'Churn' in df.columns, "Target column 'Churn' missing"
    X=df.drop(columns=['Churn'])
    y=df['Churn']

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols)
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced"
            ))
        ]
    )

    model.fit(X_train, y_train)

    return model


model = train_and_load_model()


# ---------------------------------
# Sidebar Inputs
# ---------------------------------
st.sidebar.header("📁 Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

st.sidebar.header("📄 Account Information")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

st.sidebar.header("🌐 Services Used")
PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

st.sidebar.header("💳 Billing")
MonthlyCharges = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges", min_value=0.0, value=500.0)

# ---------------------------------
# Prepare input dataframe
# ---------------------------------
input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}])

# ---------------------------------
# Main Dashboard Output
# ---------------------------------
st.markdown("---")
st.subheader("📈 Retention Risk Assessment")

if st.button("Run Churn Analysis"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Churn Probability",
            value=f"{probability:.2%}"
        )

    with col2:
        if prediction == 1:
            st.metric(
                label="Customer Status",
                value="High Risk",
                delta="⚠️ Action Required"
            )
        else:
            st.metric(
                label="Customer Status",
                value="Low Risk",
                delta="✅ Stable"
            )

    st.markdown("---")

    if prediction == 1:
        st.warning(
            "🔸 This customer shows a **high likelihood of churn.**\n\n"
            "Recommended actions:\n"
            "- Offer personalized retention deals\n"
            "- Check for service complaints\n"
            "- Provide loyalty rewards"
        )
    else:
        st.success(
            "✅ This customer is **likely to stay.**\n\n"
            "Maintain engagement with quality service and regular communication."
        )

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 14px;'>
        Developed by <b>Akshaya</b> — Biomedical Engineering Student<br>
        © 2025 Customer Churn Prediction Project
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "Note: The prediction reflects model confidence based on historical data. "
    "It is intended for decision-support purposes, not a guarantee of behavior."
)


