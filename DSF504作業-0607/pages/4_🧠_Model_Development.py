import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 設定 matplotlib 顯示字型，避免亂碼
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
with st.sidebar:
    st.markdown("## 🏦 AI-Powered Credit Risk & Default Intelligence System")
    st.caption(" DSF504 The Practice of Big Data and Analysis in the Financial Industry Semester Project • Team 8")
    st.markdown("---") 
st.title("📈 Model Performance")
st.markdown("Evaluate the champion model's performance on the unseen test dataset.")

# ====================================
# 0. 安全載入資料與模型 (取代 session_state)
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
# 1. 靜態模型比較表
# ====================================
results = pd.DataFrame({
    "Model":[
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "LightGBM"
    ],
    "ROC-AUC":[0.7652, 0.7798, 0.7483, 0.7827],
    "F1":[0.5191, 0.5377, 0.5168, 0.5297]
})

st.subheader("🏆 Model Comparison")
st.dataframe(results, use_container_width=True)

# ====================================
# 2. 動態計算最佳閾值與 F1
# ====================================
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# ====================================
# 3. 核心指標 (KPIs)
# ====================================
st.subheader("📊 Key Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Champion Model", "LightGBM")
c2.metric("Test ROC-AUC", f"{auc(*roc_curve(y_test, y_prob)[:2]):.4f}")
c3.metric("Optimized F1", f"{best_f1:.4f}")
c4.metric("Optimal Threshold", f"{best_threshold:.4f}")

st.markdown("---")

# ====================================
# 4. 評估曲線 (ROC & PR 並排顯示)
# ====================================
st.subheader("📉 Evaluation Curves")

fig_curves, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

ax_roc.plot(fpr, tpr, linewidth=2, color='darkorange', label=f"AUC = {roc_auc:.4f}")
ax_roc.plot([0,1], [0,1], linestyle="--", color='navy')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend(loc="lower right")

# --- PR Curve ---
ax_pr.plot(recall, precision, linewidth=2, color='blue', label='PR Curve')
ax_pr.scatter(recall[best_idx], precision[best_idx], color='red', s=100, zorder=5, 
              label=f"Best Threshold ({best_threshold:.4f})")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title("Precision-Recall Curve")
ax_pr.legend(loc="lower left")

st.pyplot(fig_curves)
plt.clf()

st.markdown("---")

# ====================================
# 5. 混淆矩陣與詳細 KPI
# ====================================
st.subheader("🧮 Confusion Matrix")

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
    st.warning(f"**False Positive (FP):** {FP}\n\n*Safe customers wrongly flagged.*")
    st.error(f"**False Negative (FN):** {FN}\n\n*Risky customers missed.*")

st.markdown("---")

# ====================================
# 6. 商業解讀
# ====================================
st.subheader("💼 Business Interpretation")

st.info("""
### Why Threshold Optimization Matters

Instead of using the default 0.50 threshold, we selected an optimal threshold based on the Precision-Recall curve.

This helps balance:
* **Recall** → Capture more risky customers (Minimize FN)
* **Precision** → Reduce unnecessary rejections (Minimize FP)

By shifting the threshold, the bank can dynamically adjust its risk appetite depending on economic conditions.
""")