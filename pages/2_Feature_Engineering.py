import streamlit as st
import pandas as pd

st.title("🔧 Feature Engineering")

feature_df = pd.DataFrame({
    "Feature":[
        "consecutive_delay",
        "utilization_rate",
        "avg_pay_ratio",
        "util_velocity"
    ],
    "Meaning":[
        "Delay Count",
        "Credit Utilization",
        "Repayment Ratio",
        "Risk Trend"
    ]
})

st.dataframe(feature_df)