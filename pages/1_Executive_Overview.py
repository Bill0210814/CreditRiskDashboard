import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Executive Overview")

results = pd.DataFrame({
    "Model":[
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "LightGBM"
    ],
    "ROC-AUC":[
        0.765,
        0.780,
        0.748,
        0.783
    ]
})

st.dataframe(results)

fig = px.bar(
    results,
    x="Model",
    y="ROC-AUC"
)

st.plotly_chart(fig)