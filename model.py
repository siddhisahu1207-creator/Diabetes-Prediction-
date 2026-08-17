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


model, scaler = load_model()


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ==========================================================
   GLOBAL APP
========================================================== */

.stApp {
    background: linear-gradient(
        135deg,
        #f8fafc 0%,
        #eef4ff 50%,
        #e8f5f3 100%
    );

    color: #111827 !important;
}


/* ==========================================================
   MAIN CONTENT TEXT
========================================================== */

.main {
    color: #111827 !important;
}

.main p {
    color: #111827 !important;
}

.main span {
    color: #111827;
}

.main label {
    color: #111827 !important;
}


/* ==========================================================
   HEADINGS
========================================================== */

.main h1,
.main h2,
.main h3,
.main h4,
.main h5,
.main h6 {
    color: #111827 !important;
}


/* ==========================================================
   HEADER
========================================================== */

.main-header {
    background: linear-gradient(
        90deg,
        #0f766e,
        #2563eb
    );

    padding: 30px;

    border-radius: 20px;

    text-align: center;

    color: white !important;

    margin-bottom: 28px;

    box-shadow:
        0 10px 30px rgba(0,0,0,0.12);
}

.main-header h1 {
    color: white !important;

    font-size: 40px;

    font-weight: 800;

    margin: 0;
}

.main-header p {
    color: white !important;

    font-size: 18px;

    margin-top: 10px;
}


/* ==========================================================
   SECTION CARD
========================================================== */

.section-card {
    background: white;

    padding: 20px;

    border-radius: 18px;

    margin-bottom: 20px;

    box-shadow:
        0 5px 20px rgba(0,0,0,0.08);

    border: 1px solid #e5e7eb;
}

.section-title {
    color: #0f766e !important;

    font-size: 23px;

    font-weight: 750;

    margin-bottom: 10px;
}


/* ==========================================================
   STREAMLIT WIDGET LABELS
========================================================== */

[data-testid="stWidgetLabel"] {
    color: #111827 !important;
}

[data-testid="stWidgetLabel"] p {
    color: #111827 !important;

    font-size: 15px !important;

    font-weight: 650 !important;
}


/* ==========================================================
   NUMBER INPUT
========================================================== */

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


/* Number input container */

[data-testid="stNumberInput"] > div {
    background-color: white !important;

    border-radius: 10px;
}


/* ==========================================================
   SELECTBOX
========================================================== */

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


/* ==========================================================
   SELECTBOX DROPDOWN
========================================================== */

[role="listbox"] {
    background-color: white !important;
}

[role="option"] {
    color: #111827 !important;

    background-color: white !important;
}

[role="option"]:hover {
    background-color: #e5e7eb !important;
}


/* ==========================================================
   INPUT PLACEHOLDER
========================================================== */

input::placeholder {
    color: #6b7280 !important;

    opacity: 1 !important;
}


/* ==========================================================
   TEXT INPUT
========================================================== */

input {
    color: #111827 !important;

    background-color: white !important;
}


/* ==========================================================
   BLOOD PRESSURE CARDS
========================================================== */

.bp-card {
    background: white;

    padding: 20px;

    border-radius: 16px;

    text-align: center;

    box-shadow:
        0 5px 18px rgba(0,0,0,0.08);

    border: 1px solid #e5e7eb;
}

.bp-value {
    font-size: 28px;

    font-weight: 800;

    color: #2563eb !important;
}

.bp-label {
    font-size: 14px;

    font-weight: 600;

    color: #374151 !important;

    margin-top: 5px;
}


/* ==========================================================
   METRIC CARDS
========================================================== */

.metric-card {
    background: white;

    padding: 20px;

    border-radius: 16px;

    text-align: center;

    box-shadow:
        0 5px 18px rgba(0,0,0,0.08);

    border: 1px solid #e5e7eb;
}

.metric-value {
    font-size: 28px;

    font-weight: 800;

    color: #2563eb !important;
}

.metric-label {
    font-size: 14px;

    color: #374151 !important;

    font-weight: 600;
}


/* ==========================================================
   BUTTON
========================================================== */

.stButton > button {
    width: 100%;

    background: linear-gradient(
        90deg,
        #0f766e,
        #2563eb
    );

    color: white !important;

    border: none;

    padding: 15px;

    border-radius: 12px;

    font-size: 18px;

    font-weight: 700;

    transition: all 0.3s ease;

    box-shadow:
        0 5px 15px rgba(37,99,235,0.20);
}

.stButton > button p {
    color: white !important;
}

.stButton > button span {
    color: white !important;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 20px rgba(37,99,235,0.30);
}


/* ==========================================================
   RESULT CARD
========================================================== */

.result-card {
    background: white;

    padding: 30px;

    border-radius: 20px;

    text-align: center;

    margin-top: 20px;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.10);

    border: 1px solid #e5e7eb;
}

.result-card h2 {
    color: #111827 !important;

    font-size: 28px;
}

.result-card p {
    color: #374151 !important;

    font-size: 16px;
}


/* ==========================================================
   ALERTS
========================================================== */

[data-testid="stAlert"] {
    color: #111827 !important;
}

[data-testid="stAlert"] p {
    color: #111827 !important;
}

[data-testid="stAlert"] span {
    color: #111827 !important;
}


/* ==========================================================
   PROGRESS BAR
========================================================== */

[data-testid="stProgressBar"] {
    margin-bottom: 10px;
}


/* ==========================================================
   SIDEBAR
========================================================== */

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a,
        #1e293b
    );
}


/* Sidebar text should remain white */

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] p {
    color: white !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: white !important;
}


/* ==========================================================
   DIVIDER
========================================================== */

hr {
    border-color: #d1d5db !important;
}


/* ==========================================================
   FOOTER
========================================================== */

.footer {
    text-align: center;

    color: #374151 !important;

    padding: 20px;

    font-size: 14px;
}

.footer b {
    color: #0f766e !important;

    font-size: 16px;
}

.footer span {
    color: #374151 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">

    <h1>🩺 Diabetes Prediction System</h1>

    <p>
        Machine Learning Based Diabetes Risk Assessment
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🩺 About System")

    st.write(
        """
        This application uses a Machine Learning model
        to estimate the risk of diabetes based on
        important health parameters.
        """
    )

    st.markdown("---")

    st.markdown("### 📊 Model Information")

    st.write("**Algorithm:** Logistic Regression")

    st.write("**Preprocessing:** StandardScaler")

    st.write("**Dataset:** Diabetes Dataset")

    st.markdown("---")

    st.markdown("### 📌 Features")

    st.write("• Pregnancies")

    st.write("• Glucose")

    st.write("• Blood Pressure")

    st.write("• Skin Thickness")

    st.write("• Insulin")

    st.write("• BMI")

    st.write("• Diabetes Pedigree Function")

    st.write("• Age")

    st.markdown("---")

    st.info(
        "⚠️ This application is for educational purposes "
        "and should not be used as a medical diagnosis."
    )


# ============================================================
# PATIENT INFORMATION
# ============================================================

st.markdown("""
<div class="section-card">

    <div class="section-title">
        👤 Patient Information
    </div>

</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Gender
# ------------------------------------------------------------

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )


# ------------------------------------------------------------
# Age
# ------------------------------------------------------------

with col2:

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=25,
        step=1
    )


# ------------------------------------------------------------
# Pregnancies
# ------------------------------------------------------------

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
<div class="section-card">

    <div class="section-title">
        🧪 Health Parameters
    </div>

</div>
""", unsafe_allow_html=True)


left_col, right_col = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_col:

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

with right_col:

    st.markdown("### 🩸 Blood Pressure")

    bp_col1, bp_col2 = st.columns(2)


    # --------------------------------------------------------
    # SYSTOLIC
    # --------------------------------------------------------

    with bp_col1:

        systolic = st.number_input(
            "Upper / Systolic (mmHg)",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )


    # --------------------------------------------------------
    # DIASTOLIC
    # --------------------------------------------------------

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


# ============================================================
# BLOOD PRESSURE SUMMARY
# ============================================================

st.markdown("### ❤️ Blood Pressure Summary")


bp1, bp2, bp3 = st.columns(3)


with bp1:

    st.markdown(
        f"""
        <div class="bp-card">

            <div class="bp-value">
                {systolic}
            </div>

            <div class="bp-label">
                Upper / Systolic (mmHg)
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with bp2:

    st.markdown(
        f"""
        <div class="bp-card">

            <div class="bp-value">
                {diastolic}
            </div>

            <div class="bp-label">
                Lower / Diastolic (mmHg)
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with bp3:

    st.markdown(
        f"""
        <div class="bp-card">

            <div class="bp-value">
                {systolic}/{diastolic}
            </div>

            <div class="bp-label">
                Blood Pressure
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# BLOOD PRESSURE STATUS
# ============================================================

if systolic < 90 or diastolic < 60:

    st.warning(
        "⚠️ Blood pressure appears to be low."
    )

elif systolic >= 140 or diastolic >= 90:

    st.warning(
        "⚠️ Blood pressure appears to be high."
    )

else:

    st.success(
        "✅ Blood pressure is within a commonly used normal range."
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

st.markdown("---")


predict_col1, predict_col2, predict_col3 = st.columns(
    [1, 2, 1]
)


with predict_col2:

    predict = st.button(
        "🔍 Predict Diabetes Risk"
    )


# ============================================================
# MACHINE LEARNING PREDICTION
# ============================================================

if predict:

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Existing Pima-style model contains 8 features:
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
    # Therefore, we use DIastolic BP for the existing
    # BloodPressure feature.
    #
    # Systolic BP is displayed separately in the application.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SCALE DATA
    # --------------------------------------------------------

    data_scaled = scaler.transform(data)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(data_scaled)[0]


    # --------------------------------------------------------
    # PREDICTION PROBABILITY
    # --------------------------------------------------------

    probability = model.predict_proba(data_scaled)[0]


    no_diabetes_probability = probability[0] * 100

    diabetes_probability = probability[1] * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("## 📊 Prediction Result")


    if prediction == 1:

        st.markdown(
            """
            <div class="result-card">

                <h2>
                    ⚠️ High Risk of Diabetes
                </h2>

                <p>
                    The machine learning model predicts
                    a higher probability of diabetes.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-card">

                <h2>
                    ✅ Low Risk of Diabetes
                </h2>

                <p>
                    The machine learning model predicts
                    a lower probability of diabetes.
                </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.markdown("### 📈 Prediction Probability")


    prob_col1, prob_col2 = st.columns(2)


    # --------------------------------------------------------
    # NO DIABETES
    # --------------------------------------------------------

    with prob_col1:

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


    # --------------------------------------------------------
    # DIABETES
    # --------------------------------------------------------

    with prob_col2:

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


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

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


    # ========================================================
    # ADDITIONAL INFORMATION
    # ========================================================

    st.markdown("### 📌 Health Information")


    info_col1, info_col2 = st.columns(2)


    with info_col1:

        st.info(
            f"""
            **Blood Pressure:** {systolic}/{diastolic} mmHg

            **BMI:** {bmi:.1f}

            **Glucose:** {glucose} mg/dL
            """
        )


    with info_col2:

        if prediction == 1:

            st.warning(
                """
                The prediction indicates a higher
                estimated risk. Consider discussing
                the result with a healthcare professional.
                """
            )

        else:

            st.success(
                """
                The prediction indicates a lower
                estimated risk based on the entered values.
                """
            )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.warning(
        """
        ⚠️ This prediction is generated by a machine
        learning model and is intended for educational
        purposes only. It is not a medical diagnosis.
        Please consult a qualified healthcare professional
        for medical evaluation.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

        <b>🩺 Diabetes Prediction System</b>

        <br>

        Machine Learning Project | Logistic Regression

        <br><br>

        Developed for educational purposes

    </div>
    """,
    unsafe_allow_html=True
)