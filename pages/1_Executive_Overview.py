import streamlit as st

st.title("📊 Executive Overview")

c1,c2,c3,c4 = st.columns(4)

c1.metric("Dataset Size","30,000")
c2.metric("Features","24")
c3.metric("Champion","LightGBM")
c4.metric("ROC-AUC","0.7827")

st.markdown("---")

st.subheader("Project Workflow")

st.markdown("""
Raw Data

⬇

Feature Engineering

⬇

Model Training

⬇

Hyperparameter Tuning

⬇

SHAP Explainability

⬇

Fairness Testing

⬇

Dashboard Deployment
""")