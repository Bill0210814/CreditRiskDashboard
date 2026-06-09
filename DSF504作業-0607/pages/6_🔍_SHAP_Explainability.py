import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
with st.sidebar:
    st.markdown("## 🏦 AI-Powered Credit Risk & Default Intelligence System")
    st.caption(" DSF504 The Practice of Big Data and Analysis in the Financial Industry Semester Project • Team 8")
    st.markdown("---") 
st.title("🧠 SHAP Default Risk Driver Analysis")

# ==========================================
# 🛡️ 防護罩 1：載入資料快取
# ==========================================
@st.cache_resource
def load_data():
    try:
        m = joblib.load("lightgbm_credit_model.pkl")
        x = joblib.load("X_train_scaled.pkl") # 讀取全母體資料
        return m, x
    except FileNotFoundError:
        st.error("🚨 找不到檔案！請確認 X_train_scaled.pkl 已放入專案中。")
        st.stop()

model, X_train_scaled = load_data()

# ==========================================
# 🛡️ 防護罩 2：SHAP 運算快取 (超級重要！)
# ==========================================
# 加上 @st.cache_data，讓這好幾萬筆的運算只在伺服器啟動時跑一次，後續直接拿算好的結果
@st.cache_data
def calculate_shap(_model, _X_data):
    explainer = shap.TreeExplainer(_model)
    shap_vals_raw = explainer.shap_values(_X_data)
    return shap_vals_raw[1] if isinstance(shap_vals_raw, list) else shap_vals_raw

# 顯示載入動畫，因為全母體運算會需要一點時間
with st.spinner("⏳ 正在運算全母體大數據的 SHAP 影響力，請耐心等候... (僅初次載入需等待)"):
    shap_values = calculate_shap(model, X_train_scaled)

# ==========================================
# 繪製與 Notebook 100% 相同的 SHAP 圖
# ==========================================
st.subheader("📊 SHAP Feature Importance (Impact on Default Risk)")

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values, 
    X_train_scaled,  # 使用全母體畫圖
    max_display=15, 
    show=False
)
import pandas as pd
import numpy as np

mean_abs_shap = pd.DataFrame({
    "Feature": X_train_scaled.columns,
    "Importance": np.abs(shap_values).mean(axis=0)
})

mean_abs_shap = (
    mean_abs_shap
    .sort_values(
        "Importance",
        ascending=False
    )
    .head(15)
)

st.subheader("🏆 Top 15 Most Important Features")

st.dataframe(
    mean_abs_shap,
    use_container_width=True
)
plt.title("SHAP Feature Importance (Impact on Default Risk)", fontsize=14, pad=20)
st.pyplot(plt.gcf(), bbox_inches='tight')
plt.clf()

st.markdown("---")

st.success("""
### Key Risk Driver Findings

#### 1. Core Risk Indicators
**PAY_0** and **consecutive_delay** are the most influential predictors of default risk.

When a customer has a recent repayment delay and a history of multiple delinquent months,
the probability of default increases significantly.

---

#### 2. Behavioral Interaction Effects

The engineered interaction feature successfully captures customers who exhibit:

• Persistent repayment problems

• Ongoing delinquency behavior

• Deteriorating financial conditions

These customers represent the highest-risk segment in the portfolio.

---

#### 3. Financial Health Indicators

Two additional risk drivers are:

• **utilization_rate**
(Credit Utilization Ratio)

• **latest_pay_ratio**
(Recent Repayment Coverage Ratio)

Customers with:

- high credit utilization
- low repayment ratios

are substantially more likely to default in the following billing cycle.

---

#### Business Interpretation

The results indicate that customer default risk is primarily driven by
behavioral repayment patterns rather than demographic characteristics.

This finding supports the use of behavior-based credit risk monitoring
and early warning systems in modern financial institutions.
""")