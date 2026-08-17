import streamlit as st
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
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
   GLOBAL PAGE
========================================================= */

.stApp {
    background: linear-gradient(
        135deg,
        #f5f9ff 0%,
        #eef7f6 100%
    );
}


/* =========================================================
   MAIN CONTENT
========================================================= */

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* =========================================================
   ALL NORMAL MAIN TEXT
========================================================= */

.main p {
    color: #111827 !important;
}

.main h1,
.main h2,
.main h3,
.main h4,
.main h5,
.main h6 {
    color: #111827 !important;
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
    color: #4b5563 !important;

    font-size: 14px;

    margin: 5px 0 0 0;
}


/* =========================================================
   INPUT LABELS
========================================================= */

[data-testid="stWidgetLabel"] {
    color: #111827 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #111827 !important;

    font-weight: 600 !important;

    font-size: 15px !important;
}


/* =========================================================
   NUMBER INPUT
========================================================= */

[data-testid="stNumberInput"] {
    color: #111827 !important;
}

[data-testid="stNumberInput"] label {
    color: #111827 !important;
}

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
   SELECT BOX
========================================================= */

[data-testid="stSelectbox"] {
    color: #111827 !important;
}

[data-testid="stSelectbox"] label {
    color: #111827 !important;
}

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


/* =========================================================
   DROPDOWN
========================================================= */

[role="listbox"] {
    background-color: white !important;
}

[role="option"] {
    color: #111827 !important;

    background-color: white !important;
}

[role="option"]:hover {
    background-color: #e5e7eb !important;

    color: #111827 !important;
}


/* =========================================================
   BLOOD PRESSURE / INFORMATION CARDS
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
   PREDICTION RESULT
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
    color: #374151 !important;

    font-size: 15px;
}


/* =========================================================
   PREDICTION HEADINGS
========================================================= */

.prediction-heading {
    color: #111827 !important;

    font-size: 23px;

    font-weight: 750;

    margin-top: 25px;

    margin-bottom: 15px;
}


/* =========================================================
   STREAMLIT METRICS
========================================================= */

/* Metric label */

[data-testid="stMetricLabel"] {
    color: #374151 !important;
}

[data-testid="stMetricLabel"] p {
    color: #374151 !important;

    font-weight: 600 !important;
}


/* Metric value */

[data-testid="stMetricValue"] {
    color: #111827 !important;
}

[data-testid="stMetricValue"] div {
    color: #111827 !important;
}


/* Metric delta */

[data-testid="stMetricDelta"] {
    color: #374151 !important;
}


/* =========================================================
   PROGRESS BAR
========================================================= */

[data-testid="stProgressBar"] {
    margin-top: 5px;

    margin-bottom: 15px;
}


/* =========================================================
   ALERTS
========================================================= */

[data-testid="stAlert"] {
    color: #111827 !important;
}

[data-testid="stAlert"] p {
    color: #111827 !important;
}

[data-testid="stAlert"] span {
    color: #111827 !important;
}

[data-testid="stAlert"] strong {
    color: #111827 !important;
}


/* =========================================================
   HEALTH SUMMARY
========================================================= */

.health-box {
    background: white;

    border-radius: 15px;

    padding: 20px;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 5px 15px rgba(0, 0, 0, 0.06);
}

.health-box h4 {
    color: #0f766e !important;

    margin-top: 0;
}

.health-box p {
    color: #111827 !important;

    margin: 8px 0;
}


/* =========================================================
   PREDICTION BUTTON
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

.stButton > button span {
    color: white !important;
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
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    color: white !important;
}


/* =========================================================
   FOOTER
========================================================= */

.footer {
    text-align: center;

    color: #374151 !important;

    padding: 25px;

    font-size: 13px;
}

.footer strong {
    color: #0f766e !important;

    font-size: 16px;
}

.footer span {
    color: #374151 !important;
}


/* =========================================================
   DIVIDER
========================================================= */

hr {
    border-color: #d1d5db !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
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


# ============================================================
# GENDER
# ============================================================

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


# ============================================================
# AGE
# ============================================================

with col2:

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=25,
        step=1
    )


# ============================================================
# PREGNANCIES
# ============================================================

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
# LEFT COLUMN
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
# RIGHT COLUMN
# ============================================================

with right:

    st.markdown(
        '<h3 style="color:#111827 !important;">🩸 Blood Pressure</h3>',
        unsafe_allow_html=True
    )

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

st.markdown(
    '<h3 class="prediction-heading">❤️ Blood Pressure Summary</h3>',
    unsafe_allow_html=True
)

st.caption(
    "Your entered blood pressure values."
)


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
# PREDICTION BUTTON
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

        # ====================================================
        # EXISTING MODEL HAS 8 FEATURES
        #
        # 1. Pregnancies
        # 2. Glucose
        # 3. BloodPressure
        # 4. SkinThickness
        # 5. Insulin
        # 6. BMI
        # 7. DiabetesPedigreeFunction
        # 8. Age
        #
        # The original dataset contains one BP feature.
        # Diastolic BP is used for model compatibility.
        # ====================================================

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


        # ====================================================
        # SCALE DATA
        # ====================================================

        data_scaled = scaler.transform(data)


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model.predict(data_scaled)[0]


        # ====================================================
        # PREDICTION PROBABILITY
        # ====================================================

        probability = model.predict_proba(data_scaled)[0]

        no_diabetes_probability = probability[0] * 100

        diabetes_probability = probability[1] * 100


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<h2 class="prediction-heading">📊 Prediction Result</h2>',
            unsafe_allow_html=True
        )

        st.caption(
            "Estimated result generated by the machine learning model."
        )


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
        # PREDICTION PROBABILITY
        # ====================================================

        st.markdown(
            '<h3 class="prediction-heading">📈 Prediction Probability</h3>',
            unsafe_allow_html=True
        )


        probability_col1, probability_col2 = st.columns(2)


        with probability_col1:

            st.metric(
                label="No Diabetes",
                value=f"{no_diabetes_probability:.2f}%"
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
                label="Diabetes",
                value=f"{diabetes_probability:.2f}%"
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

        st.markdown(
            '<h3 class="prediction-heading">📋 Patient Summary</h3>',
            unsafe_allow_html=True
        )


        summary1, summary2, summary3, summary4 = st.columns(4)


        with summary1:

            st.metric(
                label="Age",
                value=f"{age} years"
            )


        with summary2:

            st.metric(
                label="Glucose",
                value=f"{glucose} mg/dL"
            )


        with summary3:

            st.metric(
                label="BMI",
                value=f"{bmi:.1f}"
            )


        with summary4:

            st.metric(
                label="Blood Pressure",
                value=f"{systolic}/{diastolic}"
            )


        # ====================================================
        # HEALTH SUMMARY
        # ====================================================

        st.markdown(
            '<h3 class="prediction-heading">💡 Health Summary</h3>',
            unsafe_allow_html=True
        )


        health1, health2 = st.columns(2)


        with health1:

            st.markdown(
                f"""
<div class="health-box">

<h4>📋 Entered Health Information</h4>

<p><strong>Blood Pressure:</strong> {systolic}/{diastolic} mmHg</p>

<p><strong>BMI:</strong> {bmi:.1f} ({bmi_status})</p>

<p><strong>Glucose:</strong> {glucose} mg/dL</p>

<p><strong>Insulin:</strong> {insulin} μU/mL</p>

<p><strong>Age:</strong> {age} years</p>

</div>
""",
                unsafe_allow_html=True
            )


        with health2:

            if prediction == 1:

                st.warning(
                    """
The model indicates a **higher estimated risk**.

Consider discussing the result and your health
measurements with a qualified healthcare professional.
"""
                )

            else:

                st.success(
                    """
The model indicates a **lower estimated risk**
based on the information provided.
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