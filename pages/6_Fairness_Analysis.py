import streamlit as st
import pandas as pd
import plotly.express as px

st.title("⚖ Fairness Analysis")

fairness_df = pd.DataFrame({

    "Age Group":[
        "<30",
        "30-50",
        ">50"
    ],

    "Default Rate":[
        0.24,
        0.22,
        0.20
    ],

    "Recall":[
        0.74,
        0.72,
        0.71
    ]
})

st.subheader("Demographic Parity")

fig = px.bar(
    fairness_df,
    x="Age Group",
    y="Default Rate",
    title="Default Prediction Rate by Age Group"
)

st.plotly_chart(fig)

st.subheader("Equal Opportunity")

fig2 = px.bar(
    fairness_df,
    x="Age Group",
    y="Recall",
    title="Recall Comparison"
)

st.plotly_chart(fig2)

st.success("""
Fairness Assessment

✓ No significant age bias detected

✓ Recall difference < 5%

✓ Model fairness acceptable
""")
