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

    st.markdown(
        "## 🏦 AI-Powered Credit Risk & Default Intelligence System"
    )

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
feature engineering strategies, and correlation analysis used to build
the AI-powered Credit Risk Early Warning System.
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

    st.subheader("⭐ Dataset Overview")



    st.write("""
The UCI Credit Card Default Dataset contains approximately 30,000 customer records,
including demographic information, credit limits, repayment behavior,
monthly bills, and payment history.
""")

    summary_df = pd.DataFrame({
        "Metric":[
            "Total Customers",
            "Total Features",
            "Target Variable",
            "Default Rate"
        ],
        "Value":[
            "30,000",
            "24",
            "Default Payment Next Month",
            "22.1%"
        ]
    })
    st.image(
        "images/dataset_overview.jpg",
        caption="Target Variable Distribution",
        use_container_width=True
    )
    st.dataframe(
        summary_df,
        use_container_width=True
    )

    st.info("""

💡 Business Objective

The goal is to predict customer default probability and support:

• Credit Risk Management

• Early Warning Monitoring

• Customer Segmentation

• Credit Limit Adjustment

• Portfolio Risk Reduction
""")

    st.warning("""
⚠️ Class Imbalance Notice

Default customers account for approximately 22% of the dataset.

Therefore ROC-AUC, Precision, Recall and F1 Score are more informative than Accuracy.
""")

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("2. Customer Profile")

    st.image(
        "images/customer_profile.jpg",
        caption="Default Rates by Categorical Variable Groups",
        use_container_width=True
    )

    demo_df = pd.DataFrame({
        "Feature":[
            "Gender",
            "Education",
            "Marital Status",
            "Age"
        ],
        "Business Insight":[
            "Minor Impact",
            "Moderate Impact",
            "Minor Impact",
            "Weak Predictor"
        ]
    })

    st.dataframe(
        demo_df,
        use_container_width=True
    )

    st.warning("""
📌 Key Findings

• Male customers show slightly higher default rates

• Lower education groups exhibit higher risk

• Married customers show moderately higher default rates

However, repayment behavior remains significantly more predictive.
""")

    st.success("""
📌 Financial Health Indicators

Non-default customers generally demonstrate:

• Higher Credit Limits

• Lower Utilization Rates

• Higher Repayment Ratios

• More Stable Payment Behavior
""")

# ==================================================
# TAB 3
# ==================================================

with tab3:

    st.subheader("3. Feature Engineering Strategy")

    st.image(
        "images/feature_engineering.jpg",
        caption="Feature Engineering Framework",
        use_container_width=True
    )

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

These engineered variables transform raw financial records into meaningful behavioral indicators.

They substantially improve prediction accuracy and early warning capability.
""")

    st.success("""
📌 Core Engineered Features

1. Consecutive Delay

2. Utilization Rate

3. Debt Shortfall

4. Recent Repayment Strength

5. Risk Acceleration

These variables became the key drivers of the final LightGBM model.
""")

# ==================================================
# TAB 4
# ==================================================

with tab4:

    st.subheader("4. Correlation Analysis")

    st.image(
        "images/correlation_matrix1.jpg",
        caption="Feature Correlation Matrix",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("5. Feature Importance Analysis")

    st.image(
        "images/correlation_matrix2.jpg",
        caption="Feature Importance Ranking",
        use_container_width=True
    )

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
📌 Correlation Insights

The strongest relationships are concentrated among behavioral financial variables:

• Utilization Rate

• Consecutive Delay

• Debt Shortfall

• Repayment Ratio

These findings validate our feature engineering strategy.
""")

    st.success("""
### Business Interpretation

Customer default risk is primarily driven by:

• Repayment Behavior

• Credit Utilization

• Payment Consistency

rather than demographic characteristics.

This strongly supports the use of behavior-based credit risk models.
""")
