import streamlit as st
import pandas as pd

# ====================================
# PAGE CONFIG & SIDEBAR
# ====================================
st.set_page_config(page_title="AI Credit Risk System", page_icon="🏦", layout="wide")

with st.sidebar:
    st.markdown("## 🏦 AI-Powered Credit Risk & Default Intelligence System")
    st.caption("DSF504 The Practice of Big Data and Analysis in the Financial Industry Semester Project • Team 8")
    st.markdown("---")
    st.info("💡 **Tip:** Navigate through the sidebar to explore data insights, model performance, and interactive risk scoring.")

# ====================================
# 1. 首頁大標題與專案背景
# ====================================
st.title("🏦 AI-Powered Credit Risk & Default Intelligence System")

st.markdown("""
### DSF504 The Practice of Big Data and Analysis in the Financial Industry

**TEAM 8** 
* M14B020009_Bill Lai_賴柏諺  
* M14B020026_Darryl Lin_林詰寶  
* M14B020045_Jerry Kuo_郭翔鈞  

---

## 🎯 Problem & Solution Overview

In consumer finance, **credit default risk prediction** is one of the most critical drivers of profitability and risk management for commercial banks.

Traditional credit risk models often rely heavily on static historical credit information and therefore struggle to capture sudden changes in customer financial conditions and emerging default signals.

To address this challenge, this project utilizes the **UCI Credit Card Default Dataset (30,000 customer records)** and develops an AI-powered early warning system through:

* **Advanced Feature Engineering** (Deriving behavioral risk indicators)
* **Machine Learning Modeling** (Tree-based & Linear algorithms)
* **Automated Hyperparameter Optimization** (Using Optuna)
* **Explainable AI (XAI)** (Integrating SHAP for transparency)

After comparing multiple algorithms, **LightGBM** was selected as the champion model, achieving a **Test ROC-AUC of 0.7827**, demonstrating strong predictive performance and generalization ability.

Furthermore, SHAP explainability techniques were integrated to transform the traditional black-box model into a transparent decision-support system, enabling financial institutions to understand the key drivers behind default risk and make informed decisions.
""")

st.markdown("---")

# ====================================
# 2. 視覺化系統架構圖 (Project Workflow)
# ====================================
st.subheader("🗺️ System Architecture & Workflow")
st.markdown("An end-to-end AI deployment architecture from data processing to explainable risk scoring:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style='background-color:#f0f2f6; padding:20px; border-radius:10px; min-height:240px;'>
        <h4 style='margin-top:0px;'>📥 1. Data Preparation</h4>
        <p style='font-size:13px; color:#555; line-height:1.6;'>
        Cleaned raw customer data, applied strict category mapping,
        and implemented leakage-free target encoding techniques.
        </p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style='background-color:#e1f5fe; padding:20px; border-radius:10px; min-height:240px;'>
        <h4 style='margin-top:0px;'>⚙️ 2. Feature Engineering</h4>
        <p style='font-size:13px; color:#555; line-height:1.6;'>
        Developed early-warning indicators including utilization rate,
        repayment behavior, risk acceleration, and behavioral volatility.
        </p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style='background-color:#e8f5e9; padding:20px; border-radius:10px; min-height:240px;'>
        <h4 style='margin-top:0px;'>🧠 3. Model Development</h4>
        <p style='font-size:13px; color:#555; line-height:1.6;'>
        Optimized multiple machine learning models using Optuna,
        cross-validation, and overfitting control mechanisms.
        </p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div style='background-color:#fff3e0; padding:20px; border-radius:10px; min-height:240px;'>
        <h4 style='margin-top:0px;'>🔍 4. Explainability</h4>
        <p style='font-size:13px; color:#555; line-height:1.6;'>
        Integrated SHAP explainability to identify key risk drivers
        and support transparent credit decision-making via dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
# ====================================
# 3. 專案亮點與評分導覽指南
# ====================================
st.subheader("📊 Project Highlights & Navigation")

# 上方放置 KPI 亮點
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Dataset Size", "30,000")
with c2: st.metric("Features Engineered", "24")  # 更新為您最新的特徵數量
with c3: st.metric("Champion Model", "LightGBM")
with c4: st.metric("Test ROC-AUC", "0.7827")

st.markdown("<br>", unsafe_allow_html=True)

