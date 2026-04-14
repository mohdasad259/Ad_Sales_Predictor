import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ==============================
# Load Model & Scaler
# ==============================
model = joblib.load("ridge_model.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="Ad Sales Predictor",
    page_icon="📊",
    layout="wide"
)

# ==============================
# MODERN DARK UI
# ==============================
st.markdown("""
<style>

/* Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background-color: #0f172a;
}

/* Cards */
.card {
    background: #111827;
    padding: 24px;
    border-radius: 14px;
    border: 1px solid #1f2937;
}

/* Title */
.title {
    font-size: 36px;
    font-weight: 600;
    color: #f9fafb;
}

/* Subtitle */
.subtitle {
    color: #9ca3af;
    margin-bottom: 25px;
}

/* Button */
.stButton>button {
    background: #6366f1;
    color: white;
    border-radius: 10px;
    height: 42px;
    font-weight: 500;
}

/* Labels */
label {
    color: #e5e7eb !important;
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #f9fafb;
    font-weight: 600;
}

/* Divider */
hr {
    border: 1px solid #1f2937;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# Header
# ==============================
st.markdown('<div class="title">📊 Advertisement Sales Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict sales using advertising budgets</div>', unsafe_allow_html=True)

# ==============================
# Layout
# ==============================
col1, col2 = st.columns([1, 1.3])

# ==============================
# INPUT
# ==============================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### Input")

    tv = st.slider("TV Ad Budget ($)", 0, 300, 120)
    radio = st.slider("Radio Ad Budget ($)", 0, 200, 60)
    newspaper = st.slider("Newspaper Ad Budget ($)", 0, 150, 40)

    predict = st.button("Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Data
# ==============================
input_df = pd.DataFrame({
    "Unnamed: 0": [0],
    "TV Ad Budget ($)": [tv],
    "Radio Ad Budget ($)": [radio],
    "Newspaper Ad Budget ($)": [newspaper]
})

scaled = scaler.transform(input_df)

# ==============================
# OUTPUT
# ==============================
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### Result")

    if predict:
        result = model.predict(scaled)[0]

        st.markdown(f"""
        <h1 style='font-size:42px; color:#6366f1;'>$ {result:.2f}</h1>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Budget")

    c1, c2, c3 = st.columns(3)
    c1.metric("TV", f"${tv}")
    c2.metric("Radio", f"${radio}")
    c3.metric("News", f"${newspaper}")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.caption("Modern UI • Clean Design • ML Powered")