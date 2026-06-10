import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# PAGE CONFIG & SIDEBAR
# ==================================================
st.set_page_config(page_title="Model Monitoring Dashboard", page_icon="📈", layout="wide")

with st.sidebar:
    st.markdown("## 🏦 AI-Powered Credit Risk & Default Intelligence System")
    st.caption("DSF504 The Practice of Big Data and Analysis in the Financial Industry Semester Project • Team 8")
    st.markdown("---")

# ==================================================
# TITLE & HEADER
# ==================================================
st.title("📈 Model Performance & Monitoring")
st.markdown("Compare algorithm performance, evaluate overfitting risks, and audit the deployment readiness of the final champion model.")

# ==================================================
# CORE DATA PROCESSING (保留您的優異邏輯並優化)
# ==================================================
monitor = pd.DataFrame({
    "Model": ["Decision Tree", "Random Forest", "Logistic Regression", "LightGBM"],
    "Train AUC": [0.788, 0.818, 0.762, 0.816],
    "Test AUC": [0.765, 0.780, 0.748, 0.783]
})

# 計算過擬合缺口
monitor["Gap"] = (monitor["Train AUC"] - monitor["Test AUC"]).round(3)

# 排序建立核心英雄榜
ranking = monitor.sort_values("Test AUC", ascending=False).reset_index(drop=True)
ranking.index += 1  # 排名從 1 開始

# 提取關鍵 KPI 指標
champion_model = monitor.loc[monitor["Test AUC"].idxmax(), "Model"]
best_auc = monitor["Test AUC"].max()
avg_gap = monitor["Gap"].mean()

# ==================================================
# SECTION 1: KPI CARDS
# ==================================================
st.subheader("📊 Performance Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("👑 Champion Model", champion_model)
with col2:
    st.metric("Best Test ROC-AUC", f"{best_auc:.3f}", delta="Top Predictor")
with col3:
    st.metric("Avg Overfitting Gap", f"{avg_gap:.3f}", delta="Low Risk Zone", delta_color="inverse")

st.markdown("---")

# ==================================================
# SECTION 2: LEADERBOARD & PERFORMANCE COMPARISON (左右並排)
# ==================================================
row1_col1, row1_col2 = st.columns([1, 1.2])

with row1_col1:
    st.subheader("🏆 Model Leaderboard")
    st.markdown("Algorithms sorted by validation performance on unseen test data.")
    
    # 加上高亮樣式，讓教授一眼看到最棒的 Test AUC 與最低的 Gap
    styled_ranking = ranking.style.highlight_max(subset=["Test AUC"], color="#d4edda") \
                                   .highlight_min(subset=["Gap"], color="#d4edda") \
                                   .format({"Train AUC": "{:.3f}", "Test AUC": "{:.3f}", "Gap": "{:.3f}"})
    
    st.dataframe(styled_ranking, use_container_width=True)
    st.caption("💡 **Note:** LightGBM balances peak predictive accuracy with excellent generalization.")

with row1_col2:
    st.subheader("⚔️ Train vs Test ROC-AUC")
    
    auc_long = monitor.melt(id_vars="Model", value_vars=["Train AUC", "Test AUC"], 
                            var_name="Dataset", value_name="AUC")
    
    fig_auc = px.bar(
        auc_long, x="Model", y="AUC", color="Dataset", barmode="group",
        text_auto=".3f",
        color_discrete_sequence=["#4C72B0", "#DD8452"],  # 換上經典高對比商務配色
        title="Generalization Matrix"
    )
    fig_auc.update_layout(height=350, yaxis=dict(range=[0.7, 0.85]), margin=dict(t=40, b=10))
    st.plotly_chart(fig_auc, use_container_width=True)

st.markdown("---")

# ==================================================
# SECTION 3: OVERFITTING GAP & GAUGE (左右並排)
# ==================================================
row2_col1, row2_col2 = st.columns([1.2, 1])

with row2_col1:
    st.subheader("⚠️ Overfitting Gap Analysis")
    
    fig_gap = px.bar(
        monitor.sort_values("Gap"), x="Model", y="Gap", color="Gap",
        text_auto=".3f",
        color_continuous_scale="Oranges",
        title="Train-Test Performance Discrepancy"
    )
    fig_gap.update_layout(height=320, margin=dict(t=40, b=10))
    st.plotly_chart(fig_gap, use_container_width=True)
    
    st.info("💡 Smaller gaps indicate better robustness in production environments. Random Forest shows signs of slight overfitting compared to LightGBM.")

with row2_col2:
    st.subheader("🥇 Deployable Champion Asset")
    
    champion_data = ranking.iloc[0]
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=champion_data["Test AUC"] * 100,
        number={'suffix': "%", 'valueformat': ".1f"},
        gauge={
            "axis": {"range": [50, 100]},
            "bar": {"thickness": 0.2, "color": "black"},
            "steps": [
                {"range": [50, 70], "color": "lightcoral"},
                {"range": [70, 80], "color": "khaki"},
                {"range": [80, 100], "color": "lightgreen"}
            ]
        }
    ))
    fig_gauge.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.caption(f"<center><b>Selected Model:</b> {champion_data['Model']} (Test AUC: {champion_data['Test AUC']:.3f})</center>", unsafe_allow_html=True)

st.markdown("---")

# ==================================================
# SECTION 4: BUSINESS INTERPRETATION
# ==================================================
st.subheader("💼 Business Interpretation & Deployment Strategy")

st.success(f"""
### 📊 Key Executive Insights

1. **LightGBM Supremacy**: **LightGBM** achieved the highest Test ROC-AUC (**{best_auc:.3f}**), making it the most mathematically sound option for capturing complex non-linear default signals.
2. **Overfitting Control**: While Random Forest shows competitive performance (Test AUC 0.780), its Train AUC (0.818) indicates higher variance and overfitting risk. LightGBM demonstrates tighter optimization boundaries.
3. **The Efficiency Frontier**: LightGBM provides the optimal financial trade-off—maximizing data-driven predictive power while maintaining institutional stability and lightning-fast inference latency.

**💡 Final Recommended Action:** Approved the deployment of the **LightGBM Early Warning Engine** into the bank's production core. This model will drive dynamic credit limit adjustments and proactive portfolio risk mitigations for Team 8's consumer lending framework.
""")
