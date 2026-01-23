import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap

# ------------------------------------------------------------
# Diabetes Prediction Web App (XGBoost + SHAP)
# ------------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    layout="wide",
    page_icon="🩺"
)

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

st.title("🩺 Diabetes Risk Predictor")
st.caption("Demo application — not a substitute for professional medical advice.")

# ---------------------------
# Configuration
# ---------------------------
FINAL_THRESHOLD = 0.3
MODEL_PATH = "diabetes_xgb_pipeline.joblib"
DATASET_PATH = "diabetes_prediction_dataset.csv"

# ---------------------------
# Load model & data
# ---------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_categories(path):
    df = pd.read_csv(path)
    return (
        sorted(df["gender"].dropna().unique().tolist()),
        sorted(df["smoking_history"].dropna().unique().tolist())
    )

pipe = load_model(MODEL_PATH)
gender_categories, smoking_categories = load_categories(DATASET_PATH)

# Extract preprocessor + model
preprocessor = pipe.named_steps[list(pipe.named_steps.keys())[0]]
model = pipe.named_steps[list(pipe.named_steps.keys())[-1]]

# SHAP explainer
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model)

# ---------------------------
# Helper functions
# ---------------------------
def pretty_indicator(name: str) -> str:
    n = name.lower()
    if "hba1c" in n: return "HbA1c"
    if "blood_glucose" in n: return "Blood glucose"
    if "bmi" in n: return "BMI"
    if "hypertension" in n: return "Hypertension"
    if "heart_disease" in n: return "Heart disease"
    if "smoking" in n: return "Smoking history"
    if "age" in n: return "Age"
    if "gender" in n: return "Gender"
    return name

def tip_for(indicator: str) -> str:
    tips = {
        "HbA1c": "Focus on consistent meals, regular activity, and routine health checks.",
        "Blood glucose": "Reduce sugary foods and drinks, increase fiber, and stay active.",
        "BMI": "Maintain balanced eating habits and regular physical activity.",
        "Hypertension": "Limit excess salt, manage stress, and do regular cardio exercise.",
        "Smoking history": "Reducing or quitting smoking can significantly lower long-term risk.",
        "Heart disease": "Follow heart-healthy lifestyle habits and regular checkups.",
        "Age": "Age is non-modifiable; focus on controllable lifestyle factors.",
        "Gender": "Gender is non-modifiable; focus on controllable lifestyle factors."
    }
    return tips.get(indicator, "Adopt healthy lifestyle habits and seek professional guidance if needed.")

# ---------------------------
# Input form
# ---------------------------
st.subheader("Patient details")

with st.form("diabetes_form"):
    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", gender_categories)
        age = st.number_input("Age", 0, 120, 30)
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with c2:
        heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
        smoking_history = st.selectbox("Smoking history", smoking_categories)
        bmi = st.number_input("BMI", 0.0, 80.0, 25.0, 0.1)

    with c3:
        hba1c = st.number_input("HbA1c level", 0.0, 20.0, 5.5, 0.1)
        st.caption("HbA1c = average blood sugar over the last ~2–3 months.")
        glucose = st.number_input("Blood glucose level", 0.0, 400.0, 120.0)

    submitted = st.form_submit_button("Predict")

# ---------------------------
# Prediction + Explanation
# ---------------------------
if submitted:
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

    prob = float(pipe.predict_proba(user_df)[0, 1])
    risk_pct = prob * 100
    pred = int(prob >= FINAL_THRESHOLD)

    st.divider()
    st.subheader("Outcome")

    if pred == 1:
        st.error("This person is diabetic")
    else:
        st.success(f"This person has {risk_pct:.1f}% risk of diabetes")

    # ---------------------------
    # Show indicators ONLY if risk > 0.0%
    # ---------------------------
    if risk_pct > 0.0:
        st.subheader("Key contributing indicators")

        try:
            X_trans = preprocessor.transform(user_df)
            feature_names = preprocessor.get_feature_names_out()

            if hasattr(X_trans, "toarray"):
                X_dense = X_trans.toarray()
            else:
                X_dense = np.asarray(X_trans)

            shap_values = explainer.shap_values(X_dense)
            shap_row = shap_values[0]

            top_idx = np.argsort(np.abs(shap_row))[::-1][:3]
            shown = set()

            for idx in top_idx:
                indicator = pretty_indicator(feature_names[idx])
                if indicator in shown:
                    continue
                shown.add(indicator)
                st.markdown(f"**• {indicator}**")
                st.caption(tip_for(indicator))

        except Exception:
            st.info("Key indicators are currently unavailable.")

# ---------------------------
# Final disclaimer
# ---------------------------
st.caption("Not for medical diagnostic, just for educational purposes.")
