import streamlit as st
import pandas as pd
import joblib

# ===============================
# Page config
# ===============================
st.set_page_config(
page_title="Diabetes Risk Predictor",
layout="wide",
page_icon="🩺"
)

st.title("🩺 Diabetes Prediction (XGBoost)")
st.write("Enter your health details to get a prediction. This is a project demo, not medical advice.")

# ===============================
# Load model (cached)
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("diabetes_xgb_pipeline.joblib")


pipe = load_model()

# Fixed threshold selected from tuning
FINAL_THRESHOLD = 0.3

# ===============================
# Load dropdown categories from dataset (cached)
# ===============================
@st.cache_data
def load_categories():
df = pd.read_csv("diabetes_prediction_dataset.csv")
gender_categories = sorted(df["gender"].dropna().unique().tolist())
smoking_categories = sorted(df["smoking_history"].dropna().unique().tolist())
return gender_categories, smoking_categories

gender_categories, smoking_categories = load_categories()

# ===============================
# Input form
# ===============================
with st.form("diabetes_form"):
col1, col2, col3 = st.columns(3)

with col1:
gender = st.selectbox("Gender", gender_categories)
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
hypertension = st.selectbox(
"Hypertension (High BP)",
[0, 1],
format_func=lambda x: "Yes" if x == 1 else "No"
)

with col2:
heart_disease = st.selectbox(
"Heart disease",
[0, 1],
format_func=lambda x: "Yes" if x == 1 else "No"
)
smoking_history = st.selectbox("Smoking history", smoking_categories)
bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)

with col3:
hba1c = st.number_input("HbA1c level", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
glucose = st.number_input("Blood glucose level", min_value=0.0, max_value=400.0, value=120.0, step=1.0)

submitted = st.form_submit_button("Predict")

# ===============================
# Prediction
# ===============================
if submitted:
# Build a 1-row DataFrame with EXACT training column names
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
pred = int(prob >= FINAL_THRESHOLD)

st.subheader("Result")
st.write(f"**Predicted diabetes probability:** `{prob:.3f}`")
st.write(f"**Decision threshold:** `{FINAL_THRESHOLD}`")

if pred == 1:
st.error("Prediction: **Diabetic (Positive)**")
else:
st.success("Prediction: **Not Diabetic (Negative)**")

# Optional: show input summary
with st.expander("Show my input values"):
st.dataframe(user_df, use_container_width=True)

st.caption("Disclaimer: This is a student ML project and not a medical diagnosis tool.")

