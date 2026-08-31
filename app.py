import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.title("🩺 Diabetes Risk Predictor")

model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")
feature_names = joblib.load("feature_names.pkl")
zero_cols = joblib.load("zero_cols.pkl")

pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 70)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    input_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]],
                             columns=feature_names)
    input_df[zero_cols] = input_df[zero_cols].replace(0, np.nan)
    input_df[zero_cols] = imputer.transform(input_df[zero_cols])

    proba = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error(f"High risk: Diabetic ({proba:.1%} probability)")
    else:
        st.success(f"Lower risk: Non-Diabetic ({proba:.1%} probability of diabetes)")