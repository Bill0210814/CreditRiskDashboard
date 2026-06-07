import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.title("🧠 SHAP 違約風險驅動因素分析")

st.info(
"""
透過 SHAP (SHapley Additive exPlanations) 機器學習可解釋性框架，
我們得以打開 LightGBM 的黑盒子，深度剖析哪些財務特徵最容易導致客戶違約。
"""
)

# ==========================================
# 1. 載入模型
# ==========================================
@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_path = os.path.join(parent_dir, "lightgbm_credit_model.pkl")
    return joblib.load(model_path)

# ==========================================
# 2. 載入資料與計算 SHAP (包含防呆機制)
# ==========================================
@st.cache_data
def get_shap_values():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    data_path = os.path.join(parent_dir, "UCI_Credit_Card.csv.xlsx")
    
    # 讀取資料 (解碼器防呆)
    try:
        sample_df = pd.read_csv(data_path)
    except UnicodeDecodeError:
        try:
            sample_df = pd.read_csv(data_path, encoding='big5')
        except Exception:
            sample_df = pd.read_excel(data_path)

    # 切分與強制轉型
    X = sample_df.iloc[:, 1:-1]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.dropna()

    # 特徵對齊 (補齊大腦需要的欄位)
    model = load_model()
    expected_features = model.feature_name_
    missing_features = [f for f in expected_features if f not in X.columns]
    
    if missing_features:
        # 重建衍生特徵 (若原始 CSV 缺少這些您自行開發的特徵，系統會自動補上預設值防呆)
        if 'utilization_rate' in missing_features and 'BILL_AMT1' in X.columns and 'LIMIT_BAL' in X.columns:
            X['utilization_rate'] = (X['BILL_AMT1'] / X['LIMIT_BAL']).replace([np.inf, -np.inf], 0).fillna(0)
            
        for f in missing_features:
            if f not in X.columns:
                X[f] = 0

    # 嚴格按照模型記憶的順序排列欄位
    X = X[expected_features]

    # 抽樣 800 筆以提升網頁繪圖速度
    sample_size = min(800, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    
    # 計算 SHAP
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    
    # 精準還原您的 Jupyter 邏輯：處理 LightGBM 的 List 輸出
    shap_values = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw
    
    return X_sample, shap_values

# ==========================================
# 3. 繪製圖表與呈現商業洞察
# ==========================================
try:
    with st.spinner("正在解析 LightGBM 模型決策邏輯 (SHAP)... 這可能需要幾秒鐘的時間 ☕"):
        X_sample, shap_values = get_shap_values()
        
    st.subheader("📊 SHAP Feature Importance (Impact on Default Risk)")
    
    # 還原您的 Jupyter 繪圖設定
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
    plt.title("SHAP Feature Importance (Impact on Default Risk)", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # 將您的 Jupyter print 內容轉化為高質感的商業洞察區塊
    st.markdown("### 💡 風險驅動核心結論")
    st.success("""
    **1. 核心風險指標：**
    `PAY_0` 與 `consecutive_delay` 是最強力的風險訊號。當客戶當月延遲還款，且過去 6 個月內有多次延遲紀錄時，違約機率呈指數級上升。
    
    **2. 交互作用效果：**
    模型成功捕捉了 `delay_x_pay0` 的特徵交互作用，精準鎖定「長期延遲且當月仍未還款」的極端高風險族群。
    
    **3. 財務健康度：**
    `utilization_rate`（額度使用率）與 `latest_pay_ratio`（最新還款比例）是次要關鍵，反映了客戶潛在的資金流動性枯竭風險。
    """)

except Exception as e:
    st.error(f"🚨 繪圖時發生錯誤。詳細錯誤訊息：{str(e)}")