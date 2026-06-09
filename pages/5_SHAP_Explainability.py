import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

st.title("🧠 SHAP 違約風險驅動因素分析 (全母體分析)")

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

plt.title("SHAP Feature Importance (Impact on Default Risk)", fontsize=14, pad=20)
st.pyplot(plt.gcf(), bbox_inches='tight')
plt.clf()

st.markdown("---")
st.success("""
**【風險驅動核心結論】：**
1. **核心風險指標**：`PAY_0` 與 `consecutive_delay` 是最強力的風險訊號。當客戶當月延遲還款，且過去 6 個月內有多次延遲紀錄時，違約機率呈指數級上升。
2. **交互作用效果**：您建立的 `delay_x_pay0` 成功捕捉了「長期延遲且當月仍未還款」的極端高風險族群。
3. **財務健康度**：`utilization_rate`（額度使用率）與 `latest_pay_ratio`（最新還款比例）是次要關鍵。
""")