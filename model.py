import streamlit as st
import numpy as np
import joblib

# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_model()

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f0fe 100%);
    }

    /* Header */
    .main-header {
        background: linear-gradient(90deg, #0f766e, #2563eb);
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .main-header h1 {
        margin: 0;
        font-size: 38px;
        color:white;
    }

    .main-header p {
        margin-top: 8px;
        font-size: 17px;
        color: white;
    }

    /* Section cards */
    .section-card {
        background: white;
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }

    .section-title {
        color: #0f766e;
        font-size: 22px;
        font-weight: 700;
        margin-bottom: 15px;
    }
    label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Input text */
    input,
    textarea {
        color: #000000 !important;
    }

    /* Number input */
    [data-testid="stNumberInput"] input {
        color: #000000 !important;
    }

    /* Selectbox */
    [data-baseweb="select"] * {
        color: #000000 !important;
    }

    [data-testid="stSelectbox"] label {
        color: #000000 !important;
    }

    [data-testid="stNumberInput"] label {
        color: #000000 !important;
    }
    /* Result card */
    .result-card {
        padding: 25px;
        border-radius: 18px;
        background: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.10);
        text-align: center;
        margin-top: 20px;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 18px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #2563eb;
    }

    .metric-label {
        font-size: 14px;
        color: #555;
    }

    /* Predict button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #0f766e, #2563eb);
        color: white;
        border: none;
        padding: 14px;
        border-radius: 12px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(37,99,235,0.35);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🩺 Diabetes Prediction System</h1>
    <p>Machine Learning Based Diabetes Risk Assessment</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:

    st.markdown("## 🩺 About System")

    st.write("""
    This application uses a Machine Learning model
    to estimate the risk of diabetes based on
    patient health parameters.
    """)

    st.markdown("---")

    st.markdown("### 📊 Model Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Preprocessing:** StandardScaler")
    st.write("**Dataset:** Diabetes Dataset")

    st.markdown("---")

    st.info(
        "⚠️ This application is for educational purposes "
        "and should not be used as a medical diagnosis."
    )


# ---------------------------------------------------
# PATIENT INFORMATION
# ---------------------------------------------------
st.markdown("""
<div class="section-card">
<div class="section-title">👤 Patient Information</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25,
        step=1
    )

with col3:
    if gender == "Female":
        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            step=1
        )
    else:
        pregnancies = 0
        st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=0,
            disabled=True
        )


# ---------------------------------------------------
# HEALTH PARAMETERS
# ---------------------------------------------------
st.markdown("""
<div class="section-card">
<div class="section-title">🧪 Health Parameters</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    glucose = st.number_input(
        "🩸 Glucose Level (mg/dL)",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

    insulin = st.number_input(
        "💉 Insulin (μU/mL)",
        min_value=0,
        max_value=900,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=70.0,
        value=23.5,
        step=0.1
    )

with col2:

    # ------------------------------------------------
    # BLOOD PRESSURE
    # ------------------------------------------------
    st.markdown("### 🩸 Blood Pressure")

    bp_col1, bp_col2 = st.columns(2)

    with bp_col1:
        systolic = st.number_input(
            "Upper / Systolic (mmHg)",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )

    with bp_col2:
        diastolic = st.number_input(
            "Lower / Diastolic (mmHg)",
            min_value=30,
            max_value=150,
            value=80,
            step=1
        )

    skin_thickness = st.number_input(
        "📏 Skin Thickness (mm)",
        min_value=0,
        max_value=100,
        value=22,
        step=1
    )

    dpf = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.25,
        step=0.01
    )


# ---------------------------------------------------
# BLOOD PRESSURE DISPLAY
# ---------------------------------------------------
st.markdown("### ❤️ Blood Pressure Summary")

bp1, bp2, bp3 = st.columns(3)

with bp1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{systolic}</div>
        <div class="metric-label">Upper / Systolic (mmHg)</div>
    </div>
    """, unsafe_allow_html=True)

with bp2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{diastolic}</div>
        <div class="metric-label">Lower / Diastolic (mmHg)</div>
    </div>
    """, unsafe_allow_html=True)

with bp3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{systolic}/{diastolic}</div>
        <div class="metric-label">Blood Pressure</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------
# BP CHECK
# ---------------------------------------------------
if systolic < 90 or diastolic < 60:
    st.warning("⚠️ Blood pressure appears to be low.")

elif systolic >= 140 or diastolic >= 90:
    st.warning("⚠️ Blood pressure appears to be high.")

else:
    st.success("✅ Blood pressure is within a commonly used normal range.")


# ---------------------------------------------------
# BMI INFORMATION
# ---------------------------------------------------
if bmi < 18.5:
    bmi_status = "Underweight"
elif bmi < 25:
    bmi_status = "Normal"
elif bmi < 30:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"

st.info(f"⚖️ BMI Category: **{bmi_status}**")


# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------
st.markdown("---")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:

    predict = st.button(
        "🔍 Predict Diabetes Risk"
    )


# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if predict:

    # IMPORTANT:
    # The original Pima dataset contains only ONE
    # BloodPressure feature. It represents diastolic BP.
    #
    # Therefore we use diastolic BP for the existing
    # model and keep systolic BP as additional
    # information displayed in the interface.

    data = np.array([
        [
            pregnancies,
            glucose,
            diastolic,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
    ])

    # Scale input
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)[0]

    probability = model.predict_proba(data_scaled)[0]

    no_diabetes_probability = probability[0] * 100
    diabetes_probability = probability[1] * 100


    # ------------------------------------------------
    # RESULT
    # ------------------------------------------------
    st.markdown("## 📊 Prediction Result")

    if prediction == 1:

        st.markdown("""
        <div class="result-card">
            <h2>⚠️ High Risk of Diabetes</h2>
            <p>The model predicts a higher probability of diabetes.</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="result-card">
            <h2>✅ Low Risk of Diabetes</h2>
            <p>The model predicts a lower probability of diabetes.</p>
        </div>
        """, unsafe_allow_html=True)


    # ------------------------------------------------
    # PROBABILITY
    # ------------------------------------------------
    st.markdown("### 📈 Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "No Diabetes",
            f"{no_diabetes_probability:.2f}%"
        )

        st.progress(
            int(no_diabetes_probability)
        )

    with col2:
        st.metric(
            "Diabetes",
            f"{diabetes_probability:.2f}%"
        )

        st.progress(
            int(diabetes_probability)
        )


    # ------------------------------------------------
    # PATIENT SUMMARY
    # ------------------------------------------------
    st.markdown("### 📋 Patient Summary")

    summary1, summary2, summary3, summary4 = st.columns(4)

    with summary1:
        st.metric("Age", f"{age} years")

    with summary2:
        st.metric("Glucose", f"{glucose} mg/dL")

    with summary3:
        st.metric("BMI", f"{bmi:.1f}")

    with summary4:
        st.metric(
            "Blood Pressure",
            f"{systolic}/{diastolic}"
        )


    # ------------------------------------------------
    # DISCLAIMER
    # ------------------------------------------------
    st.warning(
        "⚠️ This prediction is generated by a machine learning "
        "model and is intended for educational purposes only. "
        "Please consult a qualified healthcare professional "
        "for medical evaluation."
    )


# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")

st.markdown("""
<div style="text-align:center; color:#666; padding:15px;">
    <b>🩺 Diabetes Prediction System</b><br>
    Machine Learning Project | Logistic Regression
</div>
""", unsafe_allow_html=True)