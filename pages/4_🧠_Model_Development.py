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

# ==================================================
# 引入必要的套件 (請確保這些放在檔案最上方)
# ==================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# 假設在此之前已經執行了:
# model, X_test_scaled, y_test = load_eval_data()
# y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ==================================================
# 1. MODEL COMPARISON (建立靜態比較表)
# ==================================================
comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Logistic Regression", "LightGBM"],
    "Average Precision": [0.5058, 0.5445, 0.4976, 0.5554],
    "F1": [0.5191, 0.5377, 0.5168, 0.5297]
})

# ==================================================
# 2. AP / LIFT (動態計算指標)
# ==================================================
ap_score = average_precision_score(y_test, y_prob)
baseline = y_test.mean()
lift = ap_score / baseline

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores[:-1])
best_threshold = thresholds[best_idx]

# ==================================================
# 3. KPI DASHBOARD
# ==================================================
st.subheader("🏆 Champion Model KPI Dashboard")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Champion Model", "LightGBM")
with k2:
    st.metric("Average Precision", f"{ap_score:.4f}")
with k3:
    st.metric("Lift vs Baseline", f"{lift:.2f}x")
with k4:
    st.metric("Optimal Threshold", f"{best_threshold:.3f}")

st.markdown("---")

# ==================================================
# 4. MODEL COMPARISON BOARD
# ==================================================
left, right = st.columns([1.4, 1])

with left:
    st.subheader("📊 Average Precision Comparison")
    fig_ap = px.bar(
        comparison,
        x="Average Precision",
        y="Model",
        orientation="h",
        text="Average Precision",
        color="Average Precision",
        color_continuous_scale="Viridis"
    )
    fig_ap.add_vline(
        x=baseline,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Random Baseline ({baseline:.3f})"
    )
    fig_ap.update_layout(height=450)
    st.plotly_chart(fig_ap, use_container_width=True)

with right:
    st.subheader("📋 Model Metrics Board")
    st.dataframe(
        comparison.style.highlight_max(subset=["Average Precision"], color="lightgreen")
                        .format({"Average Precision": "{:.4f}", "F1": "{:.4f}"}),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==================================================
# 5. PR CURVE
# ==================================================
st.subheader("🎯 Precision-Recall Curve")

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"LightGBM (AP={ap_score:.4f})"))
fig_pr.add_hline(y=baseline, line_dash="dash", line_color="red", annotation_text=f"Random Baseline ({baseline:.3f})")
fig_pr.add_trace(go.Scatter(
    x=[recall[best_idx]], 
    y=[precision[best_idx]], 
    mode="markers", 
    marker=dict(size=12), 
    name="Best Threshold"
))

fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=500)
st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("---")

# ==================================================
# 6. LIFT GAUGE
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
**Random Baseline** = {baseline:.3f}  |  **LightGBM AP** = {ap_score:.4f}  |  **Lift** = {lift:.2f}x

The model identifies high-risk customers over **{lift:.2f} times better** than random selection.
""")

st.markdown("---")

# ==================================================
# 7. CONFUSION MATRIX
# ==================================================
st.subheader("🧮 Confusion Matrix")

y_pred = (y_prob >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

precision_val = precision_score(y_test, y_pred)
recall_val = recall_score(y_test, y_pred)
f1_val = f1_score(y_test, y_pred)
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

col1, col2 = st.columns([1.3, 1])

with col1:
    fig_cm, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Default", "Default"])
    disp.plot(cmap="Blues", ax=ax)
    plt.title(f"Threshold = {best_threshold:.3f}")
    st.pyplot(fig_cm)
    plt.clf()

with col2:
    st.metric("Recall", f"{recall_val:.2%}")
    st.metric("Precision", f"{precision_val:.2%}")
    st.metric("Specificity", f"{specificity:.2%}")
    st.metric("F1 Score", f"{f1_val:.2%}")

st.markdown("---")

# ==================================================
# 8. BUSINESS INTERPRETATION
# ==================================================
st.subheader("💼 Business Interpretation")

st.success(f"""
### Why Average Precision Matters
Credit default prediction is a highly imbalanced classification problem.
Only **{baseline:.1%}** of customers belong to the default class.
Therefore, Average Precision (AP) provides a more realistic assessment than ROC-AUC.

### Key Findings
• Average Precision = **{ap_score:.4f}**
• Random Baseline = **{baseline:.3f}**
• Lift = **{lift:.2f}x**

### Business Impact
The LightGBM model identifies high-risk customers more than **{lift:.2f} times better** than random selection, making it highly suitable for credit risk screening and early warning systems.
""")
