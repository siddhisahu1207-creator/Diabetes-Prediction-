import streamlit as st
import numpy as np
import joblib

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction System")

st.write("Enter Patient Details")
gender = st.radio("Gender", ("Male", "Female"))
if gender == "Female":
         pregnancies = st.number_input("Pregnancies", 0, 20)

else:
    pregnancies = 0
glucose = st.number_input("Glucose Level", 0, 300)

blood_pressure = st.number_input("Blood Pressure", 0, 200)

skin_thickness = st.number_input("Skin Thickness", 0, 100)

insulin = st.number_input("Insulin", 0, 900)

bmi = st.number_input("BMI", 0.0, 70.0)

dpf = st.number_input("Diabetes Pedigree Function",0.0,3.0)

age = st.number_input("Age", 1, 120)

if st.button("Predict"):

    data = np.array([
        [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0]

    if prediction == 1:
        st.error("⚠ High Risk of Diabetes")

    else:
        st.success("✅ Low Risk of Diabetes")

    st.write(f"Probability of No Diabetes: {probability[0]*100:.2f}%")
    st.write(f"Probability of Diabetes: {probability[1]*100:.2f}%")