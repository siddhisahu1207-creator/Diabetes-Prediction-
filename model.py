import streamlit as st
import numpy as np
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error("Unable to load the model files.")
    st.code(str(e))


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   MAIN PAGE
========================================================= */

.stApp {
    background: linear-gradient(
        135deg,
        #f5f9ff 0%,
        #eef7f6 100%
    );
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* =========================================================
   HEADER
========================================================= */

.header-box {
    background: linear-gradient(
        135deg,
        #0f766e,
        #2563eb
    );

    padding: 35px 30px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 30px;

    box-shadow:
        0 10px 30px rgba(15, 118, 110, 0.20);
}

.header-box h1 {
    color: white !important;
    font-size: 38px;
    font-weight: 800;
    margin: 0;
}

.header-box p {
    color: #eaf6ff !important;
    font-size: 17px;
    margin-top: 10px;
    margin-bottom: 0;
}


/* =========================================================
   SECTION HEADER
========================================================= */

.section-header {
    background: white;
    border-left: 5px solid #0f766e;

    padding: 14px 18px;
    border-radius: 12px;

    margin-top: 22px;
    margin-bottom: 18px;

    box-shadow:
        0 4px 15px rgba(0, 0, 0, 0.06);
}

.section-header h2 {
    color: #111827 !important;
    font-size: 21px;
    margin: 0;
}

.section-header p {
    color: #6b7280 !important;
    font-size: 14px;
    margin: 5px 0 0 0;
}


/* =========================================================
   LABELS
========================================================= */

[data-testid="stWidgetLabel"] p {
    color: #111827 !important;
    font-weight: 600 !important;
    font-size: 15px !important;
}


/* =========================================================
   INPUT BOXES
========================================================= */

[data-testid="stNumberInput"] input {
    color: #111827 !important;
    background-color: white !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}

[data-testid="stNumberInput"] > div {
    background-color: white !important;
    border-radius: 10px;
}


/* =========================================================
   SELECTBOX
========================================================= */

[data-baseweb="select"] {
    background-color: white !important;
    border-radius: 10px;
}

[data-baseweb="select"] div {
    color: #111827 !important;
}

[data-baseweb="select"] span {
    color: #111827 !important;
}


/* Dropdown */

[role="option"] {
    color: #111827 !important;
    background-color: white !important;
}


/* =========================================================
   BUTTON
========================================================= */

.stButton > button {
    width: 100%;
    min-height: 52px;

    border-radius: 12px;

    background: linear-gradient(
        90deg,
        #0f766e,
        #2563eb
    );

    color: white !important;

    border: none;

    font-size: 17px;
    font-weight: 700;

    box-shadow:
        0 6px 18px rgba(37, 99, 235, 0.20);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 9px 22px rgba(37, 99, 235, 0.30);
}

.stButton > button p {
    color: white !important;
}


/* =========================================================
   INFORMATION CARDS
========================================================= */

.info-card {
    background: white;

    border-radius: 15px;

    padding: 18px;

    text-align: center;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 5px 15px rgba(0, 0, 0, 0.06);
}

.info-card .value {
    color: #2563eb !important;

    font-size: 26px;

    font-weight: 800;
}

.info-card .label {
    color: #374151 !important;

    font-size: 13px;

    font-weight: 600;

    margin-top: 4px;
}


/* =========================================================
   RESULT
========================================================= */

.result-box {
    background: white;

    border-radius: 20px;

    padding: 28px;

    text-align: center;

    margin-top: 25px;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.08);
}

.result-box h2 {
    color: #111827 !important;

    font-size: 28px;

    margin-bottom: 8px;
}

.result-box p {
    color: #4b5563 !important;

    font-size: 15px;
}


/* =========================================================
   SIDEBAR
========================================================= */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a,
        #1e293b
    );
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: white !important;
}


/* =========================================================
   ALERTS
========================================================= */

[data-testid="stAlert"] p {
    color: #111827 !important;
}


/* =========================================================
   FOOTER
========================================================= */

.footer {
    text-align: center;

    color: #6b7280 !important;

    padding: 25px;

    font-size: 13px;
}

.footer strong {
    color: #0f766e !important;

    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# IMPORTANT: HTML STARTS AT COLUMN 1
# ============================================================

st.markdown("""
<div class="header-box">
<h1>🩺 Diabetes Prediction System</h1>
<p>Machine Learning Based Diabetes Risk Assessment</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 Diabetes Predictor")

    st.write(
        "Enter the patient's health information "
        "to estimate diabetes risk using a trained "
        "machine learning model."
    )

    st.markdown("---")

    st.markdown("### 📊 Model Details")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Preprocessing:** StandardScaler")
    st.write("**Features:** 8")

    st.markdown("---")

    st.markdown("### 📋 Input Features")

    st.write("• Pregnancies")
    st.write("• Glucose")
    st.write("• Blood Pressure")
    st.write("• Skin Thickness")
    st.write("• Insulin")
    st.write("• BMI")
    st.write("• Diabetes Pedigree")
    st.write("• Age")

    st.markdown("---")

    st.warning(
        "This application is for educational purposes "
        "and is not a medical diagnosis."
    )


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown("""
<div class="section-header">
<h2>👤 Patient Information</h2>
<p>Enter basic information about the patient.</p>
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
        "Age (years)",
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
            step=1,
            disabled=True
        )


# ============================================================
# HEALTH PARAMETERS
# ============================================================

st.markdown("""
<div class="section-header">
<h2>🧪 Health Parameters</h2>
<p>Enter the patient's medical and physical measurements.</p>
</div>
""", unsafe_allow_html=True)


left, right = st.columns(2)


# ============================================================
# LEFT SIDE
# ============================================================

with left:

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


# ============================================================
# RIGHT SIDE
# ============================================================

with right:

    st.markdown("### 🩸 Blood Pressure")

    bp1, bp2 = st.columns(2)

    with bp1:

        systolic = st.number_input(
            "Upper / Systolic (mmHg)",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )

    with bp2:

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


# ============================================================
# BLOOD PRESSURE SUMMARY
# ============================================================

st.markdown("""
<div class="section-header">
<h2>❤️ Blood Pressure Summary</h2>
<p>Your entered blood pressure values.</p>
</div>
""", unsafe_allow_html=True)


bp_col1, bp_col2, bp_col3 = st.columns(3)


with bp_col1:

    st.markdown(
        f"""
<div class="info-card">
<div class="value">{systolic}</div>
<div class="label">Upper / Systolic (mmHg)</div>
</div>
""",
        unsafe_allow_html=True
    )


with bp_col2:

    st.markdown(
        f"""
<div class="info-card">
<div class="value">{diastolic}</div>
<div class="label">Lower / Diastolic (mmHg)</div>
</div>
""",
        unsafe_allow_html=True
    )


with bp_col3:

    st.markdown(
        f"""
<div class="info-card">
<div class="value">{systolic}/{diastolic}</div>
<div class="label">Blood Pressure</div>
</div>
""",
        unsafe_allow_html=True
    )


# ============================================================
# BLOOD PRESSURE STATUS
# ============================================================

if systolic < 90 or diastolic < 60:

    st.warning(
        "⚠️ Your entered blood pressure appears to be low."
    )

elif systolic >= 140 or diastolic >= 90:

    st.warning(
        "⚠️ Your entered blood pressure appears to be high."
    )

else:

    st.success(
        "✅ Your entered blood pressure is within a commonly "
        "used normal range."
    )


# ============================================================
# BMI STATUS
# ============================================================

if bmi < 18.5:

    bmi_status = "Underweight"

elif bmi < 25:

    bmi_status = "Normal"

elif bmi < 30:

    bmi_status = "Overweight"

else:

    bmi_status = "Obese"


st.info(
    f"⚖️ BMI Category: **{bmi_status}**"
)


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown("")

button_left, button_center, button_right = st.columns(
    [1, 2, 1]
)

with button_center:

    predict = st.button(
        "🔍  Predict Diabetes Risk",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict:

    if not model_loaded:

        st.error(
            "Model files could not be loaded. "
            "Please check diabetes_model.pkl and scaler.pkl."
        )

    else:

        # ----------------------------------------------------
        # EXISTING MODEL HAS 8 FEATURES
        #
        # BloodPressure in the original dataset is represented
        # by one value. We use diastolic BP here so that the
        # existing model remains compatible.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SCALE
        # ----------------------------------------------------

        data_scaled = scaler.transform(data)


        # ----------------------------------------------------
        # PREDICT
        # ----------------------------------------------------

        prediction = model.predict(data_scaled)[0]


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        probability = model.predict_proba(data_scaled)[0]

        no_diabetes_probability = probability[0] * 100

        diabetes_probability = probability[1] * 100


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown("""
<div class="section-header">
<h2>📊 Prediction Result</h2>
<p>Estimated result generated by the machine learning model.</p>
</div>
""", unsafe_allow_html=True)


        if prediction == 1:

            st.markdown("""
<div class="result-box">
<h2>⚠️ Higher Risk of Diabetes</h2>
<p>The model estimates a higher probability of diabetes based on the information entered.</p>
</div>
""", unsafe_allow_html=True)

        else:

            st.markdown("""
<div class="result-box">
<h2>✅ Lower Risk of Diabetes</h2>
<p>The model estimates a lower probability of diabetes based on the information entered.</p>
</div>
""", unsafe_allow_html=True)


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.markdown("### 📈 Prediction Probability")


        probability_col1, probability_col2 = st.columns(2)


        with probability_col1:

            st.metric(
                "No Diabetes",
                f"{no_diabetes_probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        int(no_diabetes_probability),
                        0
                    ),
                    100
                )
            )


        with probability_col2:

            st.metric(
                "Diabetes",
                f"{diabetes_probability:.2f}%"
            )

            st.progress(
                min(
                    max(
                        int(diabetes_probability),
                        0
                    ),
                    100
                )
            )


        # ====================================================
        # PATIENT SUMMARY
        # ====================================================

        st.markdown("### 📋 Patient Summary")


        summary1, summary2, summary3, summary4 = st.columns(4)


        with summary1:

            st.metric(
                "Age",
                f"{age} years"
            )


        with summary2:

            st.metric(
                "Glucose",
                f"{glucose} mg/dL"
            )


        with summary3:

            st.metric(
                "BMI",
                f"{bmi:.1f}"
            )


        with summary4:

            st.metric(
                "Blood Pressure",
                f"{systolic}/{diastolic}"
            )


        # ====================================================
        # HEALTH SUMMARY
        # ====================================================

        st.markdown("### 💡 Health Summary")


        health1, health2 = st.columns(2)


        with health1:

            st.info(
                f"""
**Blood Pressure:** {systolic}/{diastolic} mmHg

**BMI:** {bmi:.1f} ({bmi_status})

**Glucose:** {glucose} mg/dL
"""
            )


        with health2:

            if prediction == 1:

                st.warning(
                    """
The model indicates a higher estimated risk.
Consider discussing the result with a qualified
healthcare professional.
"""
                )

            else:

                st.success(
                    """
The model indicates a lower estimated risk based
on the information provided.
"""
                )


        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.warning(
            """
⚠️ **Important:** This prediction is generated by a
machine learning model for educational purposes only.
It is not a medical diagnosis and should not replace
professional medical advice.
"""
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown("""
<div class="footer">
<strong>🩺 Diabetes Prediction System</strong><br>
Machine Learning Project • Logistic Regression<br>
Educational Purpose Only
</div>
""", unsafe_allow_html=True)