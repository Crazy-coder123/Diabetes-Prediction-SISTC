import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap

# ------------------------------------------------------------
# Diabetes Prediction Web App (XGBoost + SHAP explanations)
# ------------------------------------------------------------
# This app:
# 1) Loads a trained sklearn Pipeline (preprocessing + XGBoost model)
# 2) Takes a single user input row
# 3) Outputs diabetes probability as "risk %"
# 4) Uses SHAP to explain the top contributing indicators for THIS user
# 5) Shows simple, general tips (not medical advice)
# ------------------------------------------------------------

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide", page_icon="🩺")

# Light UI polish (kept minimal to avoid breaking deployments)
st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; }
    .stButton > button { border-radius: 10px; font-weight: 600; padding: 0.55rem 1.1rem; }
    h1, h2, h3 { letter-spacing: -0.5px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Diabetes Risk Predictor")
st.caption("Demo project only — not medical advice. If you’re concerned, talk to a qualified clinician.")

# ---------------------------
# Config
# ---------------------------
FINAL_THRESHOLD = 0.3  # threshold you selected during tuning
MODEL_PATH = "diabetes_xgb_pipeline.joblib"
DATASET_PATH = "diabetes_prediction_dataset.csv"

# ---------------------------
# Helpers: load model + categories
# ---------------------------
@st.cache_resource
def load_pipeline(model_path: str):
    return joblib.load(model_path)

@st.cache_data
def load_categories(csv_path: str):
    df = pd.read_csv(csv_path)
    genders = sorted(df["gender"].dropna().unique().tolist())
    smokings = sorted(df["smoking_history"].dropna().unique().tolist())
    return genders, smokings

pipe = load_pipeline(MODEL_PATH)
gender_categories, smoking_categories = load_categories(DATASET_PATH)

# ---------------------------
# Helpers: robustly get preprocessor + model from the pipeline
# ---------------------------
def get_preprocessor_and_model(pipeline):
    """
    Attempts to extract:
    - preprocessor (ColumnTransformer / transformer)
    - model (XGBClassifier)
    from a sklearn Pipeline saved via joblib.
    """
    # Most common naming convention
    if hasattr(pipeline, "named_steps"):
        steps = pipeline.named_steps
        # Try common keys
        pre = steps.get("preprocess") or steps.get("preprocessor") or steps.get("prep")
        model = steps.get("model") or steps.get("clf") or steps.get("classifier")

        # Fallback: assume last step is model, earlier transformer is preprocessor
        if model is None:
            last_key = list(steps.keys())[-1]
            model = steps[last_key]
        if pre is None and len(steps) >= 2:
            # take the first step that is not the last as preprocessor
            keys = list(steps.keys())
            pre = steps[keys[0]]

        return pre, model

    # If not a Pipeline, assume it is directly a model (no preprocessing)
    return None, pipeline

preprocessor, model = get_preprocessor_and_model(pipe)

# ---------------------------
# Build SHAP explainer (cached)
# ---------------------------
@st.cache_resource
def build_shap_explainer(_model):
    """
    TreeExplainer works for tree-based models like XGBoost.
    """
    return shap.TreeExplainer(_model)

explainer = build_shap_explainer(model)

# ---------------------------
# Turn engineered feature names into human-friendly indicator labels
# ---------------------------
def pretty_indicator(feature_name: str) -> str:
    """
    Maps one-hot / transformed feature names to a readable indicator label.
    This keeps explanations clean even when the model uses one-hot columns.
    """
    f = feature_name.lower()

    if "hba1c" in f:
        return "HbA1c"
    if "blood_glucose" in f:
        return "Blood glucose"
    if f.endswith("bmi") or "bmi" in f:
        return "BMI"
    if "hypertension" in f:
        return "Hypertension"
    if "heart_disease" in f:
        return "Heart disease"
    if "smoking_history" in f or "smoking" in f:
        return "Smoking history"
    if f.endswith("age") or "age" in f:
        return "Age"
    if "gender" in f:
        return "Gender"

    # fallback
    return feature_name

def tip_for_indicator(indicator: str) -> str:
    """
    Safe, general wellness tips (not medical advice).
    Avoids prescribing medication/dosages.
    """
    tips = {
        "HbA1c": "General tips: consistent balanced meals, regular movement, and follow-up testing with a clinician if concerned.",
        "Blood glucose": "General tips: reduce sugary drinks/snacks, increase fiber (veg/whole grains), and stay active after meals.",
        "BMI": "General tips: focus on sustainable habits—balanced portions, strength + cardio, and steady routines (not crash dieting).",
        "Hypertension": "General tips: reduce excess salt, manage stress/sleep, and do regular cardio; check BP with a professional if high.",
        "Heart disease": "General tips: prioritize heart-healthy foods (fiber, healthy fats), don’t smoke, and do regular low-impact cardio.",
        "Smoking history": "General tips: cutting down or quitting helps long-term risk; support from a GP/counsellor can make it easier.",
        "Age": "Age can’t be changed. Focus on controllable factors: activity, diet, sleep, and regular health checks.",
        "Gender": "Gender is not something you change for risk. Focus on controllable health factors instead."
    }
    return tips.get(indicator, "General tip: focus on consistent lifestyle habits and get professional advice if you’re concerned.")

# ---------------------------
# UI: Input form
# ---------------------------
st.subheader("Patient details")

with st.form("diabetes_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", gender_categories)
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        hypertension = st.selectbox("Hypertension (High BP)", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col2:
        heart_disease = st.selectbox("Heart disease", [0, 1], format_func=lambda x: "Yes" if x else "No")
        smoking_history = st.selectbox("Smoking history", smoking_categories)
        bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)

    with col3:
        hba1c = st.number_input("HbA1c level", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
        st.caption("HbA1c = average blood sugar over the last ~2–3 months.")
        glucose = st.number_input("Blood glucose level", min_value=0.0, max_value=400.0, value=120.0, step=1.0)

    submitted = st.form_submit_button("Predict")

# ---------------------------
# Prediction + SHAP explanations
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

    # Probability prediction (risk score)
    prob = float(pipe.predict_proba(user_df)[0, 1])
    risk_pct = prob * 100.0
    pred = int(prob >= FINAL_THRESHOLD)

    st.divider()
    st.subheader("Outcome")

    # Requirement:
    # - If predicted diabetic: show clean diabetic statement
    # - Otherwise: show risk percentage statement
    if pred == 1:
        st.error("This person is diabetic")
    else:
        st.success(f"This person has {risk_pct:.1f}% risk of diabetes")

    # ---------
    # SHAP: explain the top contributing indicators for THIS user
    # ---------
    st.subheader("Key contributing indicators")

    try:
        if preprocessor is not None:
            # Transform user row into model input space
            X_trans = preprocessor.transform(user_df)

            # Get feature names after preprocessing (e.g., one-hot columns)
            # Works on sklearn >= 1.0
            feature_names = preprocessor.get_feature_names_out()
        else:
            # No preprocessing: use raw columns
            X_trans = user_df.values
            feature_names = np.array(user_df.columns)

        # SHAP expects dense for some configurations
        if hasattr(X_trans, "toarray"):
            X_dense = X_trans.toarray()
        else:
            X_dense = np.asarray(X_trans)

        # SHAP values for a single row
        shap_vals = explainer.shap_values(X_dense)
        shap_row = shap_vals[0] if isinstance(shap_vals, np.ndarray) else shap_vals

        # Rank by absolute contribution
        abs_contrib = np.abs(shap_row)
        top_idx = np.argsort(abs_contrib)[::-1][:3]

        # Build readable top indicators with tips
        shown = set()
        for idx in top_idx:
            raw_name = str(feature_names[idx])
            indicator = pretty_indicator(raw_name)

            # Avoid repeating the same indicator due to one-hot columns
            if indicator in shown:
                continue
            shown.add(indicator)

            st.markdown(f"**• {indicator}**")
            st.caption(tip_for_indicator(indicator))

        if len(shown) == 0:
            st.info("No clear top indicators could be extracted for this input.")

    except Exception:
        # If SHAP fails for any reason, keep the app usable with a graceful message.
        st.info("Explanation is temporarily unavailable for this deployment build.")

    st.caption("Note: Tips are general wellbeing guidance and not a medical treatment plan.")
