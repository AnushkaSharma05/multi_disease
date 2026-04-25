import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Multi-Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models and scalers
models_path = Path(__file__).parent / "models"

@st.cache_resource
def load_models():
    """Load all trained models and scalers"""
    diabetes_model = joblib.load(models_path / "diabetes_model.pkl")
    diabetes_scaler = joblib.load(models_path / "diabetes_scaler.pkl")
    
    heart_model = joblib.load(models_path / "heart_model.pkl")
    
    parkinsons_model = joblib.load(models_path / "parkinsons_model.pkl")
    parkinsons_scaler = joblib.load(models_path / "parkinsons_scaler.pkl")
    
    return {
        "diabetes": {"model": diabetes_model, "scaler": diabetes_scaler},
        "heart": {"model": heart_model},
        "parkinsons": {"model": parkinsons_model, "scaler": parkinsons_scaler}
    }

# Load models
models = load_models()

# Title and intro
st.title("🏥 Multi-Disease Prediction System")
st.markdown("---")

# Sidebar for disease selection
disease = st.sidebar.selectbox(
    "Select Disease for Prediction",
    ["Diabetes", "Heart Disease", "Parkinson's"]
)

# Main content
if disease == "Diabetes":
    st.header("🩺 Diabetes Prediction")
    st.markdown("Enter the patient's health metrics for diabetes prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=150, value=70)
    
    with col2:
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    
    with col3:
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
    
    if st.button("Predict Diabetes Risk", key="diabetes_predict"):
        # Prepare input
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                               insulin, bmi, diabetes_pedigree, age]])
        
        # Scale input
        scaled_input = models["diabetes"]["scaler"].transform(input_data)
        
        # Make prediction
        prediction = models["diabetes"]["model"].predict(scaled_input)
        
        # Get decision function score for confidence (SVC doesn't have predict_proba by default)
        decision_score = models["diabetes"]["model"].decision_function(scaled_input)[0]
        confidence = 1 / (1 + np.exp(-abs(decision_score)))  # Convert to probability-like score
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 0:
                st.success("✅ **No Diabetes Detected**")
            else:
                st.warning("⚠️ **Diabetes Risk Detected**")
        
        with col2:
            st.info(f"**Model Confidence**: {confidence * 100:.2f}%")

elif disease == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")
    st.markdown("Enter the patient's health metrics for heart disease prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age (years)", min_value=29, max_value=77, value=50, key="heart_age")
        sex = st.selectbox("Sex", ["M", "F"])
        cp = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
    
    with col2:
        trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=90, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl (0/1)", min_value=0, max_value=1, value=0)
    
    with col3:
        restecg = st.selectbox("Resting ECG", ["LVH", "Normal", "ST"])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["N", "Y"])
    
    col4, col5 = st.columns(2)
    with col4:
        oldpeak = st.number_input("ST Depression Induced", min_value=0.0, max_value=10.0, value=1.0)
        slope = st.selectbox("Slope of ST Segment", ["Down", "Flat", "Up"])
    
    with col5:
        pass  # placeholder
    
    if st.button("Predict Heart Disease Risk", key="heart_predict"):
        # Create DataFrame with proper column names and types
        input_data = pd.DataFrame({
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [cp],
            "RestingBP": [trestbps],
            "Cholesterol": [chol],
            "FastingBS": [fbs],
            "RestingECG": [restecg],
            "MaxHR": [thalach],
            "ExerciseAngina": [exang],
            "Oldpeak": [oldpeak],
            "ST_Slope": [slope],
            "AgeGroup": ["Young" if age < 40 else "Mid" if age < 55 else "Old"]
        })
        
        # Make prediction (Pipeline handles preprocessing)
        prediction = models["heart"]["model"].predict(input_data)
        prediction_proba = models["heart"]["model"].predict_proba(input_data)
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 0:
                st.success("✅ **No Heart Disease Detected**")
            else:
                st.error("🚨 **Heart Disease Risk Detected**")
        
        with col2:
            st.info(f"**Confidence Score**: {max(prediction_proba[0]) * 100:.2f}%")

elif disease == "Parkinson's":
    st.header("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter the patient's voice and health metrics for Parkinson's prediction")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fo = st.number_input("MDVP:Fo (Hz)", min_value=50.0, max_value=300.0, value=150.0)
        fhi = st.number_input("MDVP:Fhi (Hz)", min_value=50.0, max_value=600.0, value=300.0)
        flo = st.number_input("MDVP:Flo (Hz)", min_value=50.0, max_value=300.0, value=100.0)
        jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=1.0, value=0.005)
    
    with col2:
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0, value=0.03)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=2.0, value=0.3)
        apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=1.0, value=0.04)
        nhr = st.number_input("NHR", min_value=0.0, max_value=1.0, value=0.1)
    
    with col3:
        hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, value=20.0)
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.5)
        dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.7)
        spread1 = st.number_input("Spread1", min_value=-10.0, max_value=0.0, value=-5.0)
    
    with col4:
        spread2 = st.number_input("Spread2", min_value=0.0, max_value=1.0, value=0.3)
        d2 = st.number_input("D2", min_value=0.0, max_value=10.0, value=2.5)
        ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, value=0.15)
    
    if st.button("Predict Parkinson's Disease Risk", key="parkinsons_predict"):
        # Prepare input with the 15 features used in the model
        # (7 correlated features were dropped: Jitter(Abs), RAP, PPQ, DDP, APQ3, APQ5, DDA)
        input_data = np.array([[fo, fhi, flo, jitter_percent, shimmer, shimmer_db, apq,
                               nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        
        # Scale input
        scaled_input = models["parkinsons"]["scaler"].transform(input_data)
        
        # Make prediction
        prediction = models["parkinsons"]["model"].predict(scaled_input)
        prediction_proba = models["parkinsons"]["model"].predict_proba(scaled_input)
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if prediction[0] == 0:
                st.success("✅ **No Parkinson's Disease Detected**")
            else:
                st.error("🚨 **Parkinson's Disease Risk Detected**")
        
        with col2:
            st.info(f"**Confidence Score**: {max(prediction_proba[0]) * 100:.2f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><small>⚠️ Disclaimer: This system is for educational purposes only and should not be used as a substitute for professional medical diagnosis.</small></p>
</div>
""", unsafe_allow_html=True)
