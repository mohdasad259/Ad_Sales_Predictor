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
    page_icon="💎",
    layout="wide"
)

# ==============================
# CHAMKEELA UI - Vibrant & Dazzling
# ==============================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ─── Root Palette ─── */
:root {
    --gold:        #FFD700;
    --gold-light:  #FFF0A0;
    --electric:    #00F5FF;
    --violet:      #BF5FFF;
    --hot-pink:    #FF2D87;
    --lime:        #B8FF3F;
    --deep:        #0D1F0D;
    --surface:     #122212;
    --surface2:    #162A16;
    --border:      rgba(184,255,63,0.18);
}

/* ─── Global Reset ─── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--deep) !important;
}

.stApp {
    background: var(--deep) !important;
    background-image:
        radial-gradient(ellipse 80% 60% at 10% 0%, rgba(184,255,63,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 100%, rgba(0,245,255,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(29,158,117,0.08) 0%, transparent 70%);
    background-attachment: fixed !important;
}

/* ─── Hide Streamlit chrome ─── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px !important; }

/* ─── Hero Title ─── */
.hero-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--gold);
    background: rgba(255,215,0,0.08);
    border: 1px solid rgba(255,215,0,0.3);
    padding: 6px 16px;
    border-radius: 50px;
    margin-bottom: 14px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-weight: 900;
    font-size: clamp(38px, 5vw, 60px);
    line-height: 1.08;
    background: linear-gradient(135deg, #FFD700 0%, #FF8C00 30%, #FF2D87 60%, #BF5FFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px;
}

.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 17px;
    font-weight: 300;
    color: rgba(255,255,255,0.5);
    letter-spacing: 0.02em;
    margin-bottom: 42px;
}

/* ─── Cards ─── */
.card-glow {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px;
    position: relative;
    overflow: hidden;
}

.card-glow::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--lime), var(--electric), var(--gold), transparent);
    opacity: 0.7;
}

.card-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 20px;
}

/* ─── Section Headings ─── */
.section-heading {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #fff;
    margin: 0 0 20px;
}

/* ─── Sliders ─── */
.stSlider > label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.75) !important;
    letter-spacing: 0.02em;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background: linear-gradient(135deg, var(--gold), var(--hot-pink)) !important;
    border: 2px solid var(--gold-light) !important;
    box-shadow: 0 0 12px rgba(255,215,0,0.5) !important;
}

.stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
.stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
    color: rgba(255,255,255,0.3) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}

/* Track filled part */
.stSlider [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg, var(--violet), var(--hot-pink), var(--gold)) !important;
}

/* ─── Button ─── */
.stButton > button {
    width: 100%;
    font-family: 'Playfair Display', serif !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    color: #0A0A1A !important;
    background: linear-gradient(135deg, #FFD700, #FF8C00, #FF2D87) !important;
    border: none !important;
    border-radius: 14px !important;
    height: 52px !important;
    margin-top: 12px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(255,45,135,0.35) !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255,215,0,0.45) !important;
}

.stButton > button:active {
    transform: scale(0.98) translateY(0);
}

/* ─── Prediction Result ─── */
.result-amount {
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    font-weight: 900;
    background: linear-gradient(135deg, #FFD700 0%, #FFEF80 50%, #FFD700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    text-shadow: none;
    filter: drop-shadow(0 0 20px rgba(255,215,0,0.4));
}

.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    margin-top: 6px;
}

.result-waiting {
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: rgba(255,255,255,0.25);
    font-style: italic;
    padding: 20px 0;
}

/* ─── Metric Cards ─── */
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #fff !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    color: rgba(255,255,255,0.45) !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

[data-testid="stMetricDelta"] {
    display: none;
}

div[data-testid="metric-container"] {
    background: var(--surface2) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}

/* ─── Divider ─── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,215,0,0.12) !important;
    margin: 24px 0 !important;
}

/* ─── Budget Pill Labels ─── */
.budget-header {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--electric);
    margin: 22px 0 14px;
}

/* ─── Footer ─── */
.footer-text {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.15);
    text-align: center;
    margin-top: 40px;
}

/* ─── Column gap fix ─── */
[data-testid="stHorizontalBlock"] {
    gap: 28px !important;
}

</style>
""", unsafe_allow_html=True)

# ==============================
# Hero Header
# ==============================
st.markdown('<div class="hero-badge">💎 ML Powered Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Ad Sales Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Enter your advertising budgets and discover your predicted revenue</div>', unsafe_allow_html=True)

# ==============================
# Layout
# ==============================
col1, col2 = st.columns([1, 1.3])

# ==============================
# INPUT CARD
# ==============================
with col1:
    st.markdown('<div class="card-glow">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">⚡ Budget Inputs</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Set Your Budgets</div>', unsafe_allow_html=True)

    tv = st.slider("📺  TV Ad Budget ($)", 0, 300, 120)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    radio = st.slider("📻  Radio Ad Budget ($)", 0, 200, 60)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    newspaper = st.slider("📰  Newspaper Ad Budget ($)", 0, 150, 40)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    predict = st.button("✦ Predict My Sales")

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================
# Data Preparation
# ==============================
input_df = pd.DataFrame({
    "Unnamed: 0": [0],
    "TV Ad Budget ($)": [tv],
    "Radio Ad Budget ($)": [radio],
    "Newspaper Ad Budget ($)": [newspaper]
})

scaled = scaler.transform(input_df)

# ==============================
# OUTPUT CARD
# ==============================
with col2:
    st.markdown('<div class="card-glow">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">🏆 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Predicted Sales</div>', unsafe_allow_html=True)

    if predict:
        result = model.predict(scaled)[0]
        st.markdown(f"""
        <div style='padding: 10px 0 20px;'>
            <div class='result-amount'>${result:,.2f}</div>
            <div class='result-label'>Estimated Revenue</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-waiting'>Adjust your budgets and hit Predict ↖</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<div class="budget-header">◈ Budget Breakdown</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("📺 TV", f"${tv:,}")
    c2.metric("📻 Radio", f"${radio:,}")
    c3.metric("📰 News", f"${newspaper:,}")

    total = tv + radio + newspaper
    st.markdown(f"""
    <div style='margin-top:18px; padding:14px 18px; background:rgba(255,215,0,0.06);
                border:1px solid rgba(255,215,0,0.2); border-radius:12px;
                display:flex; justify-content:space-between; align-items:center;'>
        <span style='font-family:"DM Sans",sans-serif; font-size:13px; color:rgba(255,255,255,0.45);
                     text-transform:uppercase; letter-spacing:0.1em;'>Total Budget</span>
        <span style='font-family:"Space Mono",monospace; font-size:20px; font-weight:700; color:#FFD700;'>
            ${total:,}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)