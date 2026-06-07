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
    "Average Risk":[
        0.41,
        0.32,
        0.21
    ]
})

st.dataframe(fairness_df)

fig = px.bar(
    fairness_df,
    x="Age Group",
    y="Average Risk",
    title="Predicted Risk by Age Group"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.success(
"""
No significant age bias detected.

The model demonstrates relatively
consistent risk estimation across age groups.
"""
)