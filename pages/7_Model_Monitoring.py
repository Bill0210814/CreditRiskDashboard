import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Model Monitoring")

monitor = pd.DataFrame({

    "Model":[
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "LightGBM"
    ],

    "Train AUC":[
        0.788,
        0.818,
        0.762,
        0.816
    ],

    "Test AUC":[
        0.765,
        0.780,
        0.748,
        0.783
    ]
})

monitor["Gap"] = (
    monitor["Train AUC"]
    - monitor["Test AUC"]
)

st.dataframe(monitor)

fig = px.bar(
    monitor,
    x="Model",
    y="Gap",
    title="Overfitting Monitoring"
)

st.plotly_chart(fig)

st.success("""
Champion Model

LightGBM

ROC-AUC = 0.7827
""")