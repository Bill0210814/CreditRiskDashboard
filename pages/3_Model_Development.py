import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("🤖 Model Development")

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
    ],
    "F1":[
        0.519,
        0.538,
        0.517,
        0.530
    ]
})

st.dataframe(results)

st.subheader("Model Performance Radar Chart")

categories = ["ROC-AUC","F1"]

fig = go.Figure()

for i,row in results.iterrows():

    fig.add_trace(
        go.Scatterpolar(
            r=[row["ROC-AUC"],row["F1"]],
            theta=categories,
            fill='toself',
            name=row["Model"]
        )
    )

st.plotly_chart(fig, use_container_width=True)