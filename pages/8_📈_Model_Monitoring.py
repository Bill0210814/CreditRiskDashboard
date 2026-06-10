import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## 🏦 AI-Powered Credit Risk & Default Intelligence System")
    st.caption(
        "DSF504 The Practice of Big Data and Analysis in the Financial Industry Semester Project • Team 8"
    )
    st.markdown("---")

# ==================================================
# TITLE
# ==================================================

st.title("📈 Model Monitoring Dashboard")

st.markdown("""
This page compares model performance, evaluates overfitting risk,
and identifies the champion model for deployment.
""")

# ==================================================
# MODEL RESULTS
# ==================================================

monitor = pd.DataFrame({

    "Model":[
        "Decision Tree",
        "Random Forest",
        "Logistic Regression",
        "LightGBM"
    ],

    "Train AUC":[
        0.788,
        0.818,
        0.762,
        0.816
    ],

    "Test AUC":[
        0.765,
        0.780,
        0.748,
        0.783
    ]
})

monitor["Gap"] = (
    monitor["Train AUC"] -
    monitor["Test AUC"]
).round(3)

# ==================================================
# KPI CARDS
# ==================================================

st.subheader("📊 Model Performance Summary")

champion_model = monitor.loc[
    monitor["Test AUC"].idxmax(),
    "Model"
]

best_auc = monitor["Test AUC"].max()

avg_gap = monitor["Gap"].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Champion Model",
        champion_model
    )

with col2:
    st.metric(
        "Best Test AUC",
        f"{best_auc:.3f}"
    )

with col3:
    st.metric(
        "Average Overfitting Gap",
        f"{avg_gap:.3f}"
    )

# ==================================================
# MODEL TABLE
# ==================================================

st.markdown("---")

st.subheader("📋 Model Comparison Table")

st.dataframe(
    monitor,
    use_container_width=True
)

# ==================================================
# TRAIN VS TEST AUC
# ==================================================

st.markdown("---")

st.subheader("📈 Train vs Test ROC-AUC")

auc_long = monitor.melt(
    id_vars="Model",
    value_vars=["Train AUC","Test AUC"],
    var_name="Dataset",
    value_name="AUC"
)

fig_auc = px.bar(
    auc_long,
    x="Model",
    y="AUC",
    color="Dataset",
    barmode="group",
    title="Train vs Test ROC-AUC"
)

st.plotly_chart(
    fig_auc,
    use_container_width=True
)

# ==================================================
# OVERFITTING ANALYSIS
# ==================================================

st.markdown("---")

st.subheader("⚠️ Overfitting Gap Analysis")

fig_gap = px.bar(
    monitor,
    x="Model",
    y="Gap",
    color="Gap",
    title="Train-Test Performance Gap"
)

st.plotly_chart(
    fig_gap,
    use_container_width=True
)

st.info("""
Smaller gaps indicate better generalization ability.

Large gaps suggest potential overfitting and reduced
robustness in production environments.
""")

# ==================================================
# MODEL RANKING
# ==================================================

st.markdown("---")

st.subheader("🏆 Model Ranking")

ranking = monitor.sort_values(
    "Test AUC",
    ascending=False
).reset_index(drop=True)

ranking.index += 1

st.dataframe(
    ranking,
    use_container_width=True
)

# ==================================================
# CHAMPION MODEL
# ==================================================

st.markdown("---")

st.subheader("🥇 Champion Model")

champion = ranking.iloc[0]

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=champion["Test AUC"] * 100,
    title={"text":"Champion Model ROC-AUC"},
    gauge={
        "axis":{"range":[50,100]},
        "steps":[
            {"range":[50,70],"color":"lightcoral"},
            {"range":[70,80],"color":"khaki"},
            {"range":[80,100],"color":"lightgreen"}
        ]
    }
))

fig.update_layout(height=400)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.success(f"""
🏆 Champion Model: {champion['Model']}

Test ROC-AUC = {champion['Test AUC']:.3f}

This model achieved the best balance between:

• Predictive Power

• Generalization

• Production Stability

Therefore it was selected as the final deployment model.
""")

# ==================================================
# BUSINESS INTERPRETATION
# ==================================================

st.markdown("---")

st.subheader("💼 Business Interpretation")

st.info("""
Key Business Insights

1. LightGBM achieved the highest Test ROC-AUC (0.783).

2. Random Forest showed competitive performance but slightly lower predictive power.

3. Logistic Regression remains interpretable but sacrifices predictive accuracy.

4. Decision Tree is easy to explain but less robust.

5. LightGBM provides the best balance between accuracy and business usability.

Recommended Action:

Deploy LightGBM as the production credit risk early warning model.
""")
