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



# ==================================================
# 4. AVERAGE PRECISION & PR CURVES (團隊官方圖表)
# ==================================================
st.subheader("📊 Model Performance: Average Precision & PR Curves")

# 匯入團隊製作的精美靜態圖表
# 請確保圖片路徑與檔名正確
st.image("images/pr_curves.jpg", caption="各模型 Average Precision 效能比較與 PR 曲線疊加圖", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# 依然保留數據表格，讓教授可以清楚看到精確數值
st.subheader("📋 Model Metrics Board")
st.dataframe(
    comparison.style.highlight_max(subset=["Average Precision"], color="lightgreen")
                    .format({"Average Precision": "{:.4f}", "F1": "{:.4f}"}),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# (原本的第 5 區塊 PR CURVE 已經與第 4 區塊的圖片合併，所以這裡直接接第 6 區塊的 LIFT GAUGE)

# ==================================================
# 5. LIFT GAUGE (原本的第 6 區塊，編號往前推)
# ==================================================
st.subheader("🚀 Model Lift vs Random Guess")
# ... (下方保留你原本畫 Gauge 儀表板的程式碼) ...
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
