import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("👤 Customer Risk Scoring")

model = joblib.load("lightgbm_credit_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = joblib.load("feature_names.pkl")

LIMIT_BAL = st.number_input("Credit Limit",200000)
AGE = st.number_input("Age",35)
PAY_0 = st.slider("PAY_0",-2,8,0)
BILL_AMT1 = st.number_input("Bill Amount",50000)
PAY_AMT1 = st.number_input("Payment Amount",10000)

if st.button("Predict Risk"):

    row = pd.DataFrame(
        np.zeros((1,len(FEATURES))),
        columns=FEATURES
    )

    row["LIMIT_BAL"] = LIMIT_BAL
    row["AGE"] = AGE
    row["PAY_0"] = PAY_0
    row["BILL_AMT1"] = BILL_AMT1
    row["PAY_AMT1"] = PAY_AMT1

    row["utilization_rate"] = BILL_AMT1/(LIMIT_BAL+1)
    row["latest_pay_ratio"] = PAY_AMT1/(BILL_AMT1+1)

    scaled = scaler.transform(row)

    prob = model.predict_proba(scaled)[0][1]

    st.metric(
        "Default Probability",
        f"{prob:.2%}"
    )

    if prob < 0.3:
        st.success("🟢 Low Risk")

    elif prob < 0.6:
        st.warning("🟡 Medium Risk")

    else:
        st.error("🔴 High Risk")