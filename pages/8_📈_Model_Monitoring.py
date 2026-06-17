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
        "DSF504 The Practice of Big Data and Analysis in the Financial Industry\n\nSemester Project • Team 8"
    )
    st.markdown("---")

# ==================================================
# TITLE
# ==================================================
st.title("📈 Model Monitoring & Governance")

st.markdown("""
Monitor model performance, evaluate generalization capability,
and assess deployment readiness using **Average Precision (AP)**.

Because credit default prediction is a highly imbalanced classification problem,
Average Precision provides a more realistic assessment than ROC-AUC.
""")

st.markdown("---")

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
        0.5424,
        0.6550,
        0.5126,
        0.5920
    ],

    "Test AP":[
        0.5058,
        0.5445,
        0.4976,
        0.5554
    ]
})

monitor["Gap"] = (
    monitor["Train AP"]
    -
    monitor["Test AP"]
).round(4)

# ==================================================
# CORE KPI
# ==================================================
baseline_ap = 0.2210

best_model = monitor.loc[
    monitor["Test AP"].idxmax(),
    "Model"
]

best_ap = monitor["Test AP"].max()

lift = best_ap / baseline_ap

improvement = (
    (best_ap - baseline_ap)
    /
    baseline_ap
) * 100

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
        f"{best_ap:.4f}"
    )

with c3:
    st.metric(
        "Baseline AP",
        f"{baseline_ap:.4f}"
    )

with c4:
    st.metric(
        "Model Lift",
        f"{lift:.2f}x"
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
            "Train AP":"{:.4f}",
            "Test AP":"{:.4f}",
            "Gap":"{:.4f}"
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
        text_auto=".4f",
        title="Average Precision Comparison",
        color_discrete_sequence=[
            "#4C72B0",
            "#DD8452"
        ]
    )

    fig_ap.update_layout(
        height=400,
        yaxis=dict(range=[0.45,0.70])
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

    st.subheader("🎯 AP vs Random Baseline")

    compare = pd.DataFrame({

        "Metric":[
            "Random Baseline",
            "LightGBM"
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
        color="Metric",
        text_auto=".4f",
        title="Average Precision vs Baseline"
    )

    fig_base.add_hline(
        y=baseline_ap,
        line_dash="dash",
        annotation_text="Default Rate Baseline = 22.1%"
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

    st.subheader("🚀 Improvement Gauge")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=improvement,
            number={"suffix":"%"},
            title={"text":"Improvement vs Baseline"},
            gauge={
                "axis":{"range":[0,200]},
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

    gauge.update_layout(height=400)

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

st.markdown("---")

# ==================================================
# OVERFITTING ANALYSIS
# ==================================================
st.subheader("⚠️ Generalization Audit")

fig_gap = px.bar(
    monitor.sort_values("Gap"),
    x="Model",
    y="Gap",
    color="Gap",
    text_auto=".4f",
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

Random Forest achieves competitive predictive power but exhibits
a larger Train-Test gap.

LightGBM delivers the strongest balance between predictive performance
and robustness.
""")

st.markdown("---")

# ==================================================
# WHY AP?
# ==================================================
st.subheader("🎯 Why Average Precision Instead of ROC-AUC?")

st.warning(f"""
### 📌Credit Risk Perspective

The dataset contains only **22.1% default customers**.

For highly imbalanced classification problems:

• ROC-AUC can sometimes appear optimistic because it evaluates ranking performance across all thresholds.

• Average Precision (AP) focuses directly on identifying the minority default class.

• AP therefore provides a more realistic measure of risk detection quality.

### 📌Performance Comparison

Random Baseline AP = **{baseline_ap:.4f}**

LightGBM AP = **{best_ap:.4f}**

Absolute Improvement = **{best_ap - baseline_ap:.4f}**

Model Lift = **{lift:.2f}x**

This means the model identifies high-risk borrowers more than
2.5 times better than random selection.
""")

st.markdown("---")

# ==================================================
# EXECUTIVE SUMMARY
# ==================================================
st.subheader("💼 Executive Interpretation")

st.success(f"""
### 💡Champion Model Approval

After comprehensive validation, LightGBM remains the preferred model.

✅ Highest Test AP Score (**{best_ap:.4f}**)

✅ Strong Generalization Capability

✅ Low Overfitting Risk

✅ Fast Inference Performance

✅ Suitable for Real-Time Credit Risk Scoring

###💡 Final Conclusion

For imbalanced credit default prediction problems,
Average Precision is a more appropriate performance metric than ROC-AUC.

LightGBM achieved:

• AP = {best_ap:.4f}

• Baseline = {baseline_ap:.4f}

• Improvement = {improvement:.1f}%

• Lift = {lift:.2f}x

The model is therefore approved as the Champion Model
for deployment within the AI-Powered Credit Risk Early Warning System.
""")
