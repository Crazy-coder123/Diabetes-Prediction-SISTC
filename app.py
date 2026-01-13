import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Diabetes Prediction Web App (Deployment-ready)
# ------------------------------------------------------------
# This app:
# 1) Loads a trained ML pipeline (preprocessing + XGBoost model)
# 2) Collects user inputs through a clean UI form
# 3) Predicts diabetes probability and applies a fixed threshold
# 4) Displays a simple, professional outcome message
# ------------------------------------------------------------

# ---------------------------
# App page configuration
# ---------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
    page_icon="🩺"
)

# ---------------------------
# Light UI styling
# - Keeps layout clean
# - Makes the primary button feel more polished
# ---------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.1rem;
    }
    h1, h2, h3 { letter-spacing: -0.5px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Diabetes Prediction (XGBoost)")
st.caption("Enter health details to get a prediction. This is a demo project, not medical advice.")

# ---------------------------
# Configuration
# ---------------------------
# Final threshold chosen during model evaluation (kept fixed for consistent predictions in production)
FINAL_THRESHOLD = 0.3

# File names expected to be in the same folder as app.py in deployment (Streamlit Cloud)
MODEL_PATH = "diabetes_xgb_pipeline.joblib"
DATASET_PATH = "diabetes_prediction_dataset.csv"

# ---------------------------
# Load the trained pipeline
# ---------------------------
# st.cache_resource caches heavy objects (like ML models) so they are loaded only once per app restart.
@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

# Load the model pipeline (includes preprocessing + XGBoost classifier)
pipe = load_model(MODEL_PATH)

# ---------------------------
# Load categories for dropdowns
# ---------------------------
# st.cache_data caches data reads so the CSV is not reloaded on every interaction.
@st.cache_data
def load_categories(csv_path: str):
    df = pd.read_csv(csv_path)
    gender_categories = sorted(df["gender"].dropna().unique().tolist())
    smoking_categories = sorted(df["smoking_history"].dropna().unique().tolist())
    return gender_categories, smoking_categories

gender_categories, smoking_categories = load_categories(DATASET_PATH)

# ---------------------------
# Input form
# ---------------------------
st.subheader("Patient details")

with st.form("diabetes_form"):
    col1, col2, col3 = st.columns(3)

    # Column 1 inputs
    with col1:
        gender = st.selectbox("Gender", gender_categories)
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        hypertension = st.selectbox(
            "Hypertension (High BP)",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    # Column 2 inputs
    with col2:
        heart_disease = st.selectbox(
            "Heart disease",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        smoking_history = st.selectbox("Smoking history", smoking_categories)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)

    # Column 3 inputs
    with col3:
        hba1c = st.number_input("HbA1c level", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
        # Short explanation placed directly under the HbA1c input for clarity
        st.caption("HbA1c = average blood sugar over the last ~2–3 months.")
        glucose = st.number_input("Blood glucose level", min_value=0.0, max_value=400.0, value=120.0, step=1.0)

    # Submit button triggers prediction
    submitted = st.form_submit_button("Predict")

# ---------------------------
# Prediction + Output
# ---------------------------
if submitted:
    # Build a single-row DataFrame with the exact feature names used during training
    user_df = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose
    }])

    # Predict probability of class 1 (diabetes)
    prob = float(pipe.predict_proba(user_df)[0, 1])

    # Apply fixed decision threshold to convert probability into a class label
    pred = int(prob >= FINAL_THRESHOLD)

    st.divider()
    st.subheader("Result")

    # Simple, professional output message (no extra wording)
    if pred == 1:
        st.error("This person is diabetic")
    else:
        st.success("This person is not diabetic")

    # Optional technical details for transparency/debugging (kept hidden by default)
    with st.expander("More details (optional)"):
        st.write(f"Predicted diabetes probability: `{prob:.3f}`")
        st.write(f"Decision threshold: `{FINAL_THRESHOLD}`")
        st.dataframe(user_df, use_container_width=True)

# ---------------------------
# Footer disclaimer
# ---------------------------
st.caption("Disclaimer: This is a student ML project and not a medical diagnosis tool.")
