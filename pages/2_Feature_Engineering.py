import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🔧 Feature Engineering")

feature_df = pd.DataFrame({
    "Feature":[
        "utilization_rate",
        "consecutive_delay",
        "avg_pay_ratio",
        "util_velocity",
        "rollover_debt_ratio",
        "latest_pay_ratio"
    ],
    "Business Meaning":[
        "Credit Utilization",
        "Delay Count",
        "Repayment Discipline",
        "Risk Acceleration",
        "Debt Shortfall",
        "Recent Payment Strength"
    ]
})

st.subheader("Top 6 Engineered Features")
st.dataframe(feature_df)

st.subheader("Feature Correlation Matrix")

corr_df = pd.DataFrame({
    "utilization_rate":[1,0.55,0.31,0.48],
    "consecutive_delay":[0.55,1,0.42,0.61],
    "avg_pay_ratio":[0.31,0.42,1,0.28],
    "rollover_debt_ratio":[0.48,0.61,0.28,1]
})

fig = px.imshow(
    corr_df,
    text_auto=True,
    aspect="auto",
    title="Feature Correlation Matrix"
)

st.plotly_chart(fig, use_container_width=True)
