import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score, # 新增計算 AP 的套件
    confusion_matrix,
    ConfusionMatrixDisplay
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

# ====================================
# TITLE & STRATEGY
# ====================================
st.title("📈 Model Development & Training")
st.markdown("Explore the algorithm configuration, hyperparameter tuning, and threshold optimization process for the champion model.")

st.subheader("⚙️ Algorithm Configuration & Imbalance Handling")
st.info("""
**Data Imbalance Strategy (22% Default Rate):**
由於資料集存在極度的類別不平衡，我們在 LightGBM 模型開發階段採取了以下策略：
1. **Hyperparameter Tuning (Optuna):** 設定 `is_unbalance=True` (或調整 `scale_pos_weight`)，強制演算法提高對少數類別（高風險違約客）的關注度與懲罰權重。
2. **Dual-Metric Tracking:** 在交叉驗證 (Cross-Validation) 階段，我們除了追蹤傳統的 ROC-AUC 以確保整體排序能力外，更**同步追蹤 PR-AUC (Average Precision)** 作為衡量少數類別精準度的核心指標。
""")

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
# 1. 靜態模型比較表 (加入 PR-AUC 伏筆)
# ====================================
results = pd.DataFrame({
    "Model":[
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "LightGBM"
    ],
    "Val ROC-AUC":[0.7652, 0.7798, 0.7483, 0.7827],
    "Val PR-AUC (AP)": [0.5058, 0.5445, 0.4976, 0.5554], # 加入 PR-AUC
    "F1 Score":[0.5191, 0.5377, 0.5168, 0.5297]
})

st.subheader("🏆 Model Tuning Comparison (Validation Set)")
st.dataframe(
    results.style.highlight_max(subset=["Val ROC-AUC", "Val PR-AUC (AP)", "F1 Score"], color="#d4edda"),
    use_container_width=True,
    hide_index=True
)
st.caption("📌 **Dev Note:** Optuna successfully maximized global ranking (ROC-AUC) while significantly pushing the precision boundary (PR-AUC) for the minority class.")

# ====================================
# 2. 動態計算最佳閾值、F1 與 AP
# ====================================
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

# 計算 ROC-AUC 與 PR-AUC (Average Precision)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
ap_score = average_precision_score(y_test, y_prob)

# ====================================
# 3. 核心指標 (KPIs) - 雙指標並列
# ====================================
st.subheader("📊 Champion Model Test Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Champion Model", "LightGBM")
c2.metric("Test ROC-AUC", f"{roc_auc:.4f}")
c3.metric("Test PR-AUC (AP)", f"{ap_score:.4f}", delta="Imbalance Focus")
c4.metric("Optimized F1", f"{best_f1:.4f}")

st.markdown("---")

# ====================================
# 4. 評估曲線 (ROC & PR 並排顯示)
# ====================================
st.subheader("📉 Evaluation Curves")

fig_curves, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5))

# --- ROC Curve ---
ax_roc.plot(fpr, tpr, linewidth=2, color='darkorange', label=f"AUC = {roc_auc:.4f}")
ax_roc.plot([0,1], [0,1], linestyle="--", color='navy')
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC Curve")
ax_roc.legend(loc="lower right")
ax_roc.grid(True, linestyle='--', alpha=0.5)

# --- PR Curve ---
ax_pr.plot(recall, precision, linewidth=2, color='green', label=f'AP = {ap_score:.4f}')
ax_pr.scatter(recall[best_idx], precision[best_idx], color='red', s=100, zorder=5, 
              label=f"Best F1 Threshold ({best_threshold:.4f})")
# 畫上 PR 的 Baseline (22.1%)
ax_pr.axhline(y=0.221, color='gray', linestyle='--', label='Baseline (0.221)')
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.set_title(f"Precision-Recall Curve (AP={ap_score:.4f})")
ax_pr.legend(loc="lower left")
ax_pr.grid(True, linestyle='--', alpha=0.5)

st.pyplot(fig_curves)
plt.clf()

st.markdown("---")

# ====================================
# 5. 混淆矩陣與詳細 KPI
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
# 6. 商業解讀與下一步轉場
# ====================================
st.subheader("💼 Business Interpretation & Next Steps")

st.success("""
### 🎯 From Threshold Tuning to Business Value

In the development phase, calculating the **Optimal Threshold** based on the Precision-Recall curve allows us to balance:
* **Recall:** Capturing more risky customers to minimize credit loss (FN).
* **Precision:** Reducing false alarms to prevent unnecessary rejection of good customers (FP).

### ➡️ Next Step: The "Honest Metric" Monitoring
While **ROC-AUC** is excellent for global model training, the sheer volume of True Negatives (78%) inflates its value. Therefore, as we transition to the **Model Monitoring & Selection** phase, we will pivot entirely to **Average Precision (PR-AUC)** and the dynamic business cost matrix to finalize our deployment strategy.
""")
