import streamlit as st
import pandas as pd
import plotly.express as px

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Data Explorer & Feature Engineering",
    page_icon="📊",
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

st.title("📊 Data Explorer & Feature Engineering")

st.markdown("""
This page presents dataset exploration, customer behavioral patterns,
feature engineering strategies, and correlation analysis used to build the
AI-powered Credit Risk Early Warning System.
""")

# ==================================================
# TABS
# ==================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "👤 Customer Profile",
    "🔧 Feature Engineering",
    "🔗 Correlation Analysis"
])

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("1. Dataset Overview")

    st.write("""
The UCI Credit Card Default Dataset contains approximately 30,000 customer records,
including demographic information, credit limits, repayment behavior,
monthly bills, and payment history.
""")

    summary_df = pd.DataFrame({
        "Metric": [
            "Total Customers",
            "Total Features",
            "Target Variable",
            "Default Rate"
        ],
        "Value": [
            "30,000",
            "24",
            "Default Payment Next Month",
            "22%"
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True
    )

    st.info("""
💡 **Business Objective**

The goal is to predict the probability of customer default and support:

• Credit Risk Management

• Early Warning Monitoring

• Customer Segmentation

• Credit Limit Adjustment

• Portfolio Risk Reduction
""")

    st.warning("""
⚠️ **Class Imbalance Notice**

Default customers represent approximately 22% of the dataset.

Therefore, ROC-AUC, F1 Score, Precision, and Recall are more meaningful
than Accuracy when evaluating model performance.
""")

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("2. Demographic & Behavioral Insights")

    st.write("""
Static demographic variables such as gender, education level,
and marital status provide limited predictive power compared
with behavioral repayment features.
""")

    demo_df = pd.DataFrame({
        "Feature": [
            "Gender",
            "Education",
            "Marital Status",
            "Age"
        ],
        "Business Insight": [
            "Minor impact",
            "Moderate impact",
            "Minor impact",
            "Weak predictor"
        ]
    })

    st.dataframe(
        demo_df,
        use_container_width=True
    )

    st.warning("""
📌 **Key Finding**

Male customers, lower education groups,
and married customers show slightly higher default rates.

However, behavioral variables remain significantly more predictive.
""")

    st.success("""
📌 **Financial Health Indicators**

Non-default customers typically exhibit:

• Higher credit limits

• Lower utilization rates

• Higher repayment coverage

• More stable payment behavior
""")

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.subheader("3. Engineered Features")

    feature_df = pd.DataFrame({

        "Feature":[
            "utilization_rate",
            "consecutive_delay",
            "avg_pay_ratio",
            "util_velocity",
            "rollover_debt_ratio",
            "latest_pay_ratio"
        ],

        "Business Meaning":[
            "Credit Utilization",
            "Delay Count",
            "Repayment Ratio",
            "Risk Acceleration",
            "Debt Shortfall",
            "Recent Repayment Strength"
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )

    st.info("""
### Why Feature Engineering Matters

These engineered features transform raw financial transactions into
actionable behavioral risk indicators.

They significantly improve model performance and early warning capability.
""")

    st.success("""
📌 **Top Risk Indicators**

1. Consecutive Delay

2. Utilization Rate

3. Debt Shortfall

4. Recent Repayment Strength

5. Risk Acceleration

These features form the foundation of the LightGBM risk prediction model.
""")

# ==================================================
# TAB 4
# ==================================================

with tab4:

    st.subheader("4. Feature Correlation Matrix")

    corr = pd.DataFrame({

        "utilization_rate":[1,0.45,0.32,0.60],
        "consecutive_delay":[0.45,1,0.39,0.52],
        "avg_pay_ratio":[0.32,0.39,1,0.31],
        "rollover_debt_ratio":[0.60,0.52,0.31,1]

    },

    index=[
        "utilization_rate",
        "consecutive_delay",
        "avg_pay_ratio",
        "rollover_debt_ratio"
    ])

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info("""
📌 **Correlation Insights**

The strongest relationships are concentrated among financial behavior variables:

• Utilization Rate

• Consecutive Delay

• Debt Shortfall

• Repayment Ratio

These findings validate our feature engineering strategy and explain why
behavioral indicators dominate demographic variables in default prediction.
""")

    st.success("""
### Business Interpretation

Customer default risk is primarily driven by:

• Repayment behavior

• Credit utilization

• Payment consistency

rather than demographic characteristics.

This supports the adoption of behavior-based credit risk models.
""")