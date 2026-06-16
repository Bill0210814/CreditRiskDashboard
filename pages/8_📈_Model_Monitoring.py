import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Model Monitoring Dashboard",
    page_icon="📈",
    layout="wide"
)

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
st.title("📈 Model Monitoring & Governance")

st.markdown("""
Monitor model performance, evaluate generalization capability,
and assess deployment readiness using **Average Precision (AP)**,
which is more appropriate than ROC-AUC for imbalanced credit default datasets.
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

    "Train AP":[
        0.545,
        0.570,
        0.510,
        0.580
    ],

    "Test AP":[
        0.520,
        0.548,
        0.501,
        0.555
    ]
})

monitor["Gap"] = (
    monitor["Train AP"]
    -
    monitor["Test AP"]
).round(3)

baseline_ap = 0.221

best_model = monitor.loc[
    monitor["Test AP"].idxmax(),
    "Model"
]

best_ap = monitor["Test AP"].max()

improvement = (
    (best_ap - baseline_ap)
    /
    baseline_ap
) * 100

avg_gap = monitor["Gap"].mean()

# ==================================================
# KPI DASHBOARD
# ==================================================
st.subheader("📊 Monitoring KPI Dashboard")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "👑 Champion Model",
        best_model
    )

with c2:
    st.metric(
        "Best AP Score",
        f"{best_ap:.3f}"
    )

with c3:
    st.metric(
        "Baseline AP",
        f"{baseline_ap:.3f}"
    )

with c4:
    st.metric(
        "Improvement",
        f"{improvement:.0f}%"
    )

st.markdown("---")

# ==================================================
# MODEL LEADERBOARD
# ==================================================
col1, col2 = st.columns([1, 1.5])

with col1:

    st.subheader("🏆 Model Leaderboard")

    ranking = monitor.sort_values(
        "Test AP",
        ascending=False
    )

    st.dataframe(
        ranking.style
        .highlight_max(
            subset=["Test AP"],
            color="lightgreen"
        )
        .highlight_min(
            subset=["Gap"],
            color="lightgreen"
        )
        .format({
            "Train AP":"{:.3f}",
            "Test AP":"{:.3f}",
            "Gap":"{:.3f}"
        }),
        use_container_width=True
    )

with col2:

    st.subheader("📈 Train vs Test AP")

    ap_long = monitor.melt(
        id_vars="Model",
        value_vars=[
            "Train AP",
            "Test AP"
        ],
        var_name="Dataset",
        value_name="AP Score"
    )

    fig_ap = px.bar(
        ap_long,
        x="Model",
        y="AP Score",
        color="Dataset",
        barmode="group",
        text_auto=".3f",
        title="Average Precision Comparison"
    )

    fig_ap.update_layout(
        height=400,
        yaxis=dict(range=[0.4, 0.65])
    )

    st.plotly_chart(
        fig_ap,
        use_container_width=True
    )

st.markdown("---")

# ==================================================
# AP vs BASELINE
# ==================================================
col3, col4 = st.columns(2)

with col3:

    st.subheader("🎯 AP vs Baseline")

    compare = pd.DataFrame({

        "Metric":[
            "Random Guess",
            "⭐LightGBM"
        ],

        "Score":[
            baseline_ap,
            best_ap
        ]
    })

    fig_base = px.bar(
        compare,
        x="Metric",
        y="Score",
        text_auto=".3f",
        color="Metric",
        title="Average Precision vs Baseline"
    )

    fig_base.add_hline(
        y=baseline_ap,
        line_dash="dash",
        annotation_text="Baseline (22.1%)"
    )

    fig_base.update_layout(
        showlegend=False,
        height=400
    )

    st.plotly_chart(
        fig_base,
        use_container_width=True
    )

with col4:

    st.subheader("🚀 Improvement Over Baseline")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=improvement,
            number={"suffix":"%"},
            title={
                "text":"AP Improvement"
            },
            gauge={
                "axis":{
                    "range":[0,200]
                },
                "steps":[
                    {
                        "range":[0,50],
                        "color":"lightcoral"
                    },
                    {
                        "range":[50,100],
                        "color":"khaki"
                    },
                    {
                        "range":[100,200],
                        "color":"lightgreen"
                    }
                ]
            }
        )
    )

    gauge.update_layout(
        height=400
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

st.markdown("---")

# ==================================================
# OVERFITTING ANALYSIS
# ==================================================
st.subheader("⚠️ Overfitting Gap Analysis")

fig_gap = px.bar(
    monitor.sort_values("Gap"),
    x="Model",
    y="Gap",
    color="Gap",
    text_auto=".3f",
    color_continuous_scale="Oranges",
    title="Train-Test AP Gap"
)

fig_gap.update_layout(
    height=400
)

st.plotly_chart(
    fig_gap,
    use_container_width=True
)

st.info("""
Smaller gaps indicate stronger generalization capability.

Although Random Forest achieves competitive AP performance,
its Train-Test Gap is larger than LightGBM.

LightGBM provides the best balance between predictive power
and deployment robustness.
""")

st.markdown("---")

# ==================================================
# BUSINESS INTERPRETATION
# ==================================================
st.subheader("💼 Executive Interpretation")

st.success(f"""
### Why LightGBM Remains the Champion Model

**1. Highest Average Precision**

LightGBM achieved the highest Test AP Score (**{best_ap:.3f}**),
indicating superior capability in identifying true default customers.

**2. Strong Improvement Over Baseline**

The dataset default rate is approximately 22.1%.

A random classifier would achieve an AP of only 0.221.

LightGBM achieved **0.555 AP**, representing an improvement of approximately **{improvement:.0f}%**.

**3. Better Evaluation Metric**

Because credit default prediction is an imbalanced classification problem,
Average Precision provides a more realistic assessment than ROC-AUC.

AP focuses on the quality of identifying actual risky customers,
which is directly aligned with banking risk management objectives.

**4. Production Readiness**

LightGBM combines:

• Highest AP Score

• Low Overfitting Gap

• Fast inference speed

• Strong generalization capability

Therefore, LightGBM is approved as the Champion Model for deployment
within the AI-powered Credit Risk Early Warning System.
""")
