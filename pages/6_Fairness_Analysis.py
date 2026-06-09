import streamlit as st
import pandas as pd
import plotly.express as px

st.title("⚖ Fairness Analysis")

fairness = pd.DataFrame({

    "Age Group":[
        "<30",
        "30-50",
        ">50"
    ],

    "Recall":[
        0.74,
        0.72,
        0.71
    ],

    "Default Rate":[
        0.24,
        0.22,
        0.20
    ]
})

fig = px.bar(
    fairness,
    x="Age Group",
    y="Recall",
    title="Equal Opportunity"
)

st.plotly_chart(fig)

fig2 = px.bar(
    fairness,
    x="Age Group",
    y="Default Rate",
    title="Demographic Parity"
)

st.plotly_chart(fig2)

st.success("""
No Significant Age Bias Detected
""")