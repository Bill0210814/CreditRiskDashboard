import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# PAGE CONFIG & SIDEBAR
# =====================================================
st.set_page_config(page_title="Customer Risk Scoring", page_icon="👤", layout="wide")

with st.sidebar:
    st.markdown("## 🏦 AI-Powered Credit Risk & Default Intelligence System")
    st.caption("DSF504 The Practice of Big Data and Analysis in the Financial Industry Semester Project • Team 8")
    st.markdown("---")

# =====================================================
# LOAD MODEL & ASSETS (絕對防禦機制)
# =====================================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("lightgbm_credit_model.pkl")
        scaler = joblib.load("scaler.pkl")
        FEATURES = joblib.load("feature_names.pkl")
        return model, scaler, FEATURES
    except FileNotFoundError:
        st.error("🚨 找不到模型檔案，請確認 .pkl 檔已上傳。")
        st.stop()

model, scaler, FEATURES = load_assets()

st.title("👤 Customer Risk Scoring Dashboard")
st.markdown("---")

# =====================================================
# Section 1 — Customer Input Panel
# =====================================================
st.subheader("📥 Input Panel")

col1, col2 = st.columns(2)
with col1:
    LIMIT_BAL = st.number_input("Credit Limit", min_value=10000, max_value=1000000, value=200000, step=10000)
    AGE = st.number_input("Age", min_value=20, max_value=80, value=35)
    PAY_0 = st.slider("PAY_0 (Latest Repayment Status)", min_value=-2, max_value=8, value=0)
with col2:
    BILL_AMT1 = st.number_input("Bill Amount", min_value=0, max_value=1000000, value=50000, step=5000)
    PAY_AMT1 = st.number_input("Payment Amount", min_value=0, max_value=1000000, value=10000, step=1000)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PREDICT BUTTON TRIGGER
# =====================================================
if st.button("🚀 Predict Risk", use_container_width=True):

    # --- 1. 資料建立與特徵工程 ---
    row = pd.DataFrame(np.zeros((1, len(FEATURES))), columns=FEATURES)

    if "LIMIT_BAL" in row.columns: row["LIMIT_BAL"] = LIMIT_BAL
    if "AGE" in row.columns: row["AGE"] = AGE
    if "PAY_0" in row.columns: row["PAY_0"] = PAY_0
    if "BILL_AMT1" in row.columns: row["BILL_AMT1"] = BILL_AMT1
    if "PAY_AMT1" in row.columns: row["PAY_AMT1"] = PAY_AMT1

    utilization = BILL_AMT1 / (LIMIT_BAL + 1e-5)
    pay_ratio = PAY_AMT1 / (BILL_AMT1 + 1e-5)
    
    if "utilization_rate" in row.columns: row["utilization_rate"] = utilization
    if "latest_pay_ratio" in row.columns: row["latest_pay_ratio"] = pay_ratio
    if "delay_x_pay0" in row.columns: row["delay_x_pay0"] = 0 * PAY_0

    # --- 2. 核心預測 ---
    scaled = scaler.transform(row)
    scaled_df = pd.DataFrame(scaled, columns=FEATURES)
    prob = model.predict_proba(scaled_df)[0][1]

    st.markdown("---")
    st.subheader("🎯 Default Risk Prediction")

    # ==========================================
    # Business KPI Cards
    # ==========================================
    if prob < 0.30:
        risk_level = "Low Risk 🟢"
    elif prob < 0.60:
        risk_level = "Medium Risk 🟡"
    else:
        risk_level = "High Risk 🔴"

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric("Default Probability", f"{prob:.2%}")
    with kpi2:
        st.metric("Risk Category", risk_level)
    with kpi3:
        st.metric("Utilization Rate", f"{utilization:.1%}")

    # =====================================================
    # Section 2 — Dashboard Layout (Gauge & Radar)
    # =====================================================
    dash_col1, dash_col2 = st.columns(2)

    with dash_col1:
        # Risk Gauge
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': "%", 'valueformat': ".1f"},
            title={"text": "Default Risk Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3, "color": "black"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "khaki"},
                    {"range": [60, 100], "color": "lightcoral"}
                ]
            }
        ))
        gauge_fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

    with dash_col2:
        # Customer Financial Radar
        categories = ["Credit Limit", "Repayment", "Utilization", "Age", "Payment"]
        values = [
            min(LIMIT_BAL / 500000, 1),
            max((8 - PAY_0) / 10, 0), 
            max(1 - min(utilization, 1), 0), 
            AGE / 80,
            min(PAY_AMT1 / 50000, 1)
        ]
        radar_fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself', name='Customer', fillcolor='rgba(76, 114, 176, 0.5)', line=dict(color='#4C72B0')
        ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
            showlegend=False, height=350, margin=dict(l=40, r=40, t=50, b=20)
        )
        st.plotly_chart(radar_fig, use_container_width=True)

    # =====================================================
    # Section 3 — What-If Analysis (動態決策模擬)
    # =====================================================
    st.markdown("---")
    st.header("📈 What-If Analysis")
    st.write("Simulate how different payment amounts affect default probability.")

    # 建立動態模擬區間
    payment_range = np.arange(0, int(max(BILL_AMT1 * 2, 50000)), 2000)
    what_if_probs = []

    for payment in payment_range:
        temp_row = row.copy()
        if "PAY_AMT1" in temp_row.columns: temp_row["PAY_AMT1"] = payment
        if "latest_pay_ratio" in temp_row.columns: temp_row["latest_pay_ratio"] = payment / (BILL_AMT1 + 1e-5)
        
        # 確保 DataFrame 格式避免 LightGBM 報錯
        scaled_temp = scaler.transform(temp_row)
        scaled_temp_df = pd.DataFrame(scaled_temp, columns=FEATURES)
        temp_prob = model.predict_proba(scaled_temp_df)[0][1]
        what_if_probs.append(temp_prob)

    whatif_df = pd.DataFrame({
        "Payment Amount": payment_range,
        "Default Probability": what_if_probs
    })

    # 繪製 What-If 折線圖
    line_fig = px.line(whatif_df, x="Payment Amount", y="Default Probability", markers=True,
                       title="Impact of Payment Amount on Default Probability")
    
    line_fig.add_hline(y=0.60, line_dash="dash", line_color="tomato", annotation_text="High Risk (60%)")
    line_fig.add_hline(y=0.30, line_dash="dash", line_color="lightgreen", annotation_text="Low Risk (30%)")
    
    # 標示目前所在的落點
    line_fig.add_scatter(
        x=[PAY_AMT1],
        y=[prob],
        mode="markers+text",
        text=["Current Customer"],
        textposition="top center",
        marker=dict(size=15, color="red")
    )
    st.plotly_chart(line_fig, use_container_width=True)

    # ==========================================
    # Risk Reduction Analysis
    # ==========================================
    best_prob = min(what_if_probs)
    reduction = prob - best_prob

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Current Risk", f"{prob:.2%}")
    with col_b:
        st.metric("Best Scenario", f"{best_prob:.2%}")
    with col_c:
        st.metric("Maximum Risk Reduction", f"{reduction:.2%}")
        
    st.info(f"**Current Scenario:** Payment Amount = {PAY_AMT1:,.0f} | Predicted Default Probability = {prob:.2%}")

    # =====================================================
    # Section 4 & 5 — Recommendation & Risk Drivers
    # =====================================================
    st.markdown("---")
    rec_col, driver_col = st.columns([1.2, 1])

    with rec_col:
        st.header("💼 Business Recommendation")
        if prob < 0.30:
            st.success("Recommended Action:\n\n• Maintain current credit limit\n• Eligible for promotional offers\n• Low monitoring frequency")
        elif prob < 0.60:
            st.warning("Recommended Action:\n\n• Monitor repayment behavior\n• Consider temporary limit freeze\n• Monthly risk review")
        else:
            st.error("Recommended Action:\n\n• High default warning\n• Immediate intervention\n• Credit line reduction\n• Escalate to risk management team")

    with driver_col:
        st.header("📊 Key Risk Drivers")
        risk_df = pd.DataFrame({
            "Feature": ["PAY_0", "Utilization Rate", "Latest Pay Ratio", "Consecutive Delay"],
            "Current Value": [
                PAY_0, 
                round(utilization, 3), 
                round(pay_ratio, 3), 
                row.get("consecutive_delay", pd.Series([0])).iloc[0] # 若無此特徵則顯示 0
            ]
        })
        
        driver_fig = px.bar(
            risk_df,
            x="Current Value",
            y="Feature",
            orientation="h",
            title="Current Risk Driver Profile",
            color="Current Value", # 加上顏色漸層讓長條圖更有質感
            color_continuous_scale="Blues"
        )
        driver_fig.update_layout(height=350, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(driver_fig, use_container_width=True)