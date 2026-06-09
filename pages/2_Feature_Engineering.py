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

    "Meaning":[
        "Credit Utilization",
        "Delay Count",
        "Repayment Ratio",
        "Risk Acceleration",
        "Debt Shortfall",
        "Recent Repayment Strength"
    ]
})

st.dataframe(feature_df)

corr = pd.DataFrame({
    "utilization_rate":[1,0.45,0.32,0.60],
    "consecutive_delay":[0.45,1,0.39,0.52],
    "avg_pay_ratio":[0.32,0.39,1,0.31],
    "rollover_debt_ratio":[0.60,0.52,0.31,1]
})

fig = px.imshow(
    corr,
    text_auto=True,
    title="Feature Correlation Matrix"
)

st.plotly_chart(fig,use_container_width=True)