import streamlit as st

st.set_page_config(
    page_title="AI Credit Risk Intelligence",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 AI Credit Risk Intelligence Platform")

st.markdown("""
### Financial Machine Learning Final Project

Dataset:
- UCI Credit Card Default Dataset

Champion Model:
- LightGBM

Best ROC-AUC:
- 0.7827
""")