import streamlit as st


with st.sidebar:
    st.markdown("## 🏦 AI-Powered Credit Risk & Default Intelligence System")
    st.caption(" DSF504 The Practice of Big Data and Analysis in the Financial Industry Semester Project • Team 8")
    st.markdown("---") 
# ====================================
# 1. 首頁大標題與專案背景
# ====================================
st.title("🏦 AI-Powered Credit Risk & Default Intelligence System")

st.markdown("""
### DSF504 The Practice of Big Data and Analysis in the Financial Industry Semester Project

TEAM 8  · M14B020009_Bill Lai_賴柏諺 M14B020026_Darryl Lin_林詰寶 M14B020045_Jerry Kuo_郭翔鈞
### 🎯 Problem & Solution Overview

在消費金融實務中，**信用卡違約風險預警**是商業銀行核心的獲利關鍵。傳統風控模型往往依賴靜態的歷史信用評分，難以捕捉客戶在「近期財務惡化」時的動態波動。

本專案基於 **UCI Credit Card 數據集 (30,000 筆客戶樣本)**，透過**機器學習輔助特徵工程**與 **Optuna 自動化超參數調參**，成功打造了高精準度的預警型 **LightGBM 冠軍模型**（測試集 ROC-AUC 達 **0.7827**）。本系統更導入了 **SHAP 歸因分析**，讓傳統的黑盒子模型具備企業級的可解釋性，協助管理階層進行動態的風險決策。
""")

st.markdown("---")

# ====================================
# 2. 視覺化系統架構圖 (Project Workflow)
# ====================================
st.subheader("🗺️ System Architecture & Workflow")
st.markdown("本專案從小規模實驗到最終部署的標準 AI 落地架構：")

# 利用 st.columns 建立精美的橫向流程圖
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style='background-color:#f0f2f6; padding:15px; border-radius:10px; height:180px;'>
        <h4>📥 1. 數據清洗</h4>
        <p style='font-size:13px; color:#555;'>對齊真實世界比例，執行嚴格映射與 Target Encoding 防洩漏機制。</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color:#e1f5fe; padding:15px; border-radius:10px; height:180px;'>
        <h4>⚙️ 2. 預警特徵</h4>
        <p style='font-size:13px; color:#555;'>衍生近期風險加速、行為波動度、以及關鍵交互特徵 <code>delay_x_pay0</code>。</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background-color:#e8f5e9; padding:15px; border-radius:10px; height:180px;'>
        <h4>🧠 3. 模型調參</h4>
        <p style='font-size:13px; color:#555;'>經由 Optuna 100 次迭代與驗證集監控，全面壓制 Gap 揪出過擬合。</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style='background-color:#fff3e0; padding:15px; border-radius:10px; height:180px;'>
        <h4>🔍 4. 落地解釋</h4>
        <p style='font-size:13px; color:#555;'>整合全母體 SHAP 圖表，將信用違約因素透明化、具體化。</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ====================================
