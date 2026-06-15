import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 新增 plotly 與其他所需 metrics
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# 設定 matplotlib 顯示字型，避免亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ====================================
# SIDEBAR
# ====================================
with st.sidebar:
    st.markdown("## 🏦 AI-Powered Credit Risk & Default Intelligence System")
    st.caption("DSF504 The Practice of Big Data and Analysis in the Financial Industry\n\nSemester Project • Team 8")
    st.markdown("---") 

st.title("📈 Model Performance & Evaluation")
st.markdown("Evaluate the champion model's performance on the unseen test dataset with a strict focus on highly imbalanced data.")

# ====================================
# 0. 安全載入資料與模型
# ====================================
@st.cache_resource
def load_eval_data():
    try:
        model = joblib.load("lightgbm_credit_model.pkl")
        X_test = joblib.load("X_test_scaled.pkl")
        y_test = joblib.load("y_test.pkl")
        return model, X_test, y_test
    except FileNotFoundError:
        st.error("🚨 Missing .pkl files! Ensure model and test data are uploaded.")
        st.stop()

model, X_test_scaled, y_test = load_eval_data()

# 即時計算預測機率
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ====================================
# 1. 動態計算核心指標 (AP, Lift, F1, Threshold)
# ====================================
# 計算 Average Precision 與 Lift
ap_score = average_precision_score(y_test, y_prob)
baseline = y_test.mean()  # 約等於 0.221
lift = ap_score / baseline

# 計算最佳 Threshold 與 F1
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# 建立靜態比較表 (確保與圖片上的數字一致)
comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Logistic Regression", "LightGBM"],
    "Average Precision": [0.5058, 0.5445, 0.4976, 0.5554],
    "F1 Score": [0.5191, 0.5377, 0.5168, 0.5297]
})

# ====================================
# 2. KPI DASHBOARD
# ====================================
st.subheader("🏆 Champion Model KPI Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Champion Model", "LightGBM")
c2.metric("Average Precision", f"{ap_score:.4f}", delta="Strict Imbalance Focus")
c3.metric("Lift vs Baseline", f"{lift:.2f}x", delta=f"Base: {baseline:.3f}")
c4.metric("Optimal Threshold", f"{best_threshold:.4f}", delta=f"Max F1: {best_f1:.4f}", delta_color="off")

st.markdown("---")

# ==================================================
# 3. AVERAGE PRECISION & PR CURVES (團隊官方圖表)
# ==================================================
st.subheader("📊 Model Performance: Average Precision & PR Curves")

# 匯入團隊製作的精美靜態圖表
st.image("images/pr_curves.jpg", caption="各模型 Average Precision 效能比較與 PR 曲線疊加圖", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# 保留數據表格
st.subheader("📋 Model Metrics Board")
st.dataframe(
    comparison.style.highlight_max(subset=["Average Precision", "F1 Score"], color="#d4edda")
                    .format({"Average Precision": "{:.4f}", "F1 Score": "{:.4f}"}),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ==================================================
# 4. LIFT GAUGE
# ==================================================
st.subheader("🚀 Model Lift vs Random Guess")

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=lift,
    number={"suffix": "x"},
    title={"text": "Model Lift"},
    gauge={
        "axis": {"range": [1, 3]},
        "steps": [
            {"range": [1, 1.5], "color": "lightgray"},
            {"range": [1.5, 2], "color": "khaki"},
            {"range": [2, 3], "color": "lightgreen"}
        ]
    }
))

gauge.update_layout(height=350)
st.plotly_chart(gauge, use_container_width=True)

st.info(f"""
**Random Baseline** = {baseline:.3f} | **LightGBM AP** = {ap_score:.4f} | **Lift** = {lift:.2f}x

The model identifies high-risk customers over **{lift:.2f} times better** than random selection.
""")

st.markdown("---")

# ====================================
# 5. CONFUSION MATRIX
# ====================================
st.subheader("🧮 Confusion Matrix at Optimal Threshold")

y_pred_opt = (y_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_opt)
TN, FP, FN, TP = cm.ravel()

col_cm, col_kpi = st.columns([1.2, 1])

with col_cm:
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Default", "Default"])
    disp.plot(cmap="Blues", ax=ax_cm, values_format='d')
    plt.title(f"Threshold = {best_threshold:.4f}")
    st.pyplot(fig_cm)
    plt.clf()

with col_kpi:
    st.markdown("### 📌 Matrix KPIs")
    st.info(f"**True Positive (TP):** {TP}\n\n*Correctly flagged as risky.*")
    st.success(f"**True Negative (TN):** {TN}\n\n*Correctly identified as safe.*")
    st.warning(f"**False Positive (FP):** {FP}\n\n*Safe customers wrongly flagged (Friction).*")
    st.error(f"**False Negative (FN):** {FN}\n\n*Risky customers missed (Credit Loss).*")

st.markdown("---")

# ====================================
# 6. BUSINESS INTERPRETATION
# ====================================
st.subheader("💼 Business Interpretation")

st.success(f"""
### Why Average Precision & Threshold Optimization Matters

Credit default prediction is a highly imbalanced classification problem. Only **{baseline:.1%}** of customers belong to the default class. Therefore, **Average Precision (AP)** provides a much more realistic assessment than ROC-AUC, as it strictly evaluates the model's ability to catch true defaulters without being inflated by the large volume of safe customers.

Furthermore, instead of using the default 0.50 threshold, we selected an optimal threshold (**{best_threshold:.4f}**) based on the Precision-Recall curve. 

This helps balance:
* **Recall** → Capture more risky customers (Minimize FN / Credit Loss)
* **Precision** → Reduce unnecessary rejections (Minimize FP / Customer Friction)

By shifting the threshold, the bank can dynamically adjust its risk appetite depending on macroeconomic conditions.
""")
