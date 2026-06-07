import streamlit as st
import pandas as pd

st.title("📈 Model Monitoring")

monitor_df = pd.DataFrame({

    "Model":[
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "LightGBM"
    ],

    "Train AUC":[
        0.788035,
        0.818386,
        0.761962,
        0.816073
    ],

    "Test AUC":[
        0.765193,
        0.779810,
        0.748273,
        0.782697
    ],

    "AUC Gap":[
        0.022842,
        0.038575,
        0.013690,
        0.033376
    ]
})

st.dataframe(monitor_df)

st.subheader("Champion Model")

st.success(
"""
LightGBM

Train AUC = 0.816

Test AUC = 0.783

AUC Gap = 0.033

Model Health = Healthy
"""
)