import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("👤 Customer Risk Scoring")

# 載入模型
model = joblib.load("lightgbm_credit_model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
'LIMIT_BAL','AGE','PAY_0','BILL_AMT1','PAY_AMT1',
'consecutive_delay','utilization_rate','avg_pay_ratio',
'bill_trend','pay_trend','pay_0_vs_hist_avg',
'util_velocity','rollover_debt_ratio','latest_pay_ratio',
'bill_std_normalized','pay_std_normalized',
'is_high_risk_trigger','edu_target_enc',
'SEX_Male','EDUCATION_High_School',
'EDUCATION_Other','EDUCATION_University',
'MARRIAGE_Other','MARRIAGE_Single'
]

st.subheader("Single Customer Prediction")

LIMIT_BAL = st.number_input("Credit Limit", value=200000)
AGE = st.number_input("Age", value=35)
PAY_0 = st.slider("Latest Repayment Status", -2, 8, 0)
BILL_AMT1 = st.number_input("Latest Bill Amount", value=50000)
PAY_AMT1 = st.number_input("Latest Payment Amount", value=10000)

if st.button("Predict Risk"):

    row = pd.DataFrame(
        np.zeros((1,24)),
        columns=FEATURES
    )

    row["LIMIT_BAL"] = LIMIT_BAL
    row["AGE"] = AGE
    row["PAY_0"] = PAY_0
    row["BILL_AMT1"] = BILL_AMT1
    row["PAY_AMT1"] = PAY_AMT1

    # Demo版特徵工程
    row["utilization_rate"] = BILL_AMT1 / LIMIT_BAL
    row["latest_pay_ratio"] = PAY_AMT1 / (BILL_AMT1 + 1)

    scaled = scaler.transform(row)

    prob = model.predict_proba(scaled)[0][1]

    st.metric(
        "Default Probability",
        f"{prob:.2%}"
    )

    if prob < 0.30:
        st.success("🟢 Low Risk")
    elif prob < 0.60:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")