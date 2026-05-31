import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ — AI Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load Model & Preprocessor ──────────────────────────────
@st.cache_resource
def load_models():
    model        = joblib.load('loan_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_models()

# ── Custom CSS — Premium Dark Banking Aesthetic ────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --gold:       #C9A84C;
    --gold-light: #E8C96A;
    --dark:       #0A0C10;
    --dark-2:     #10141C;
    --dark-3:     #181E2A;
    --dark-4:     #1E2738;
    --border:     rgba(201,168,76,0.18);
    --text:       #E8E6E0;
    --text-muted: #7A8399;
    --green:      #2ECC71;
    --red:        #E74C3C;
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--dark) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(201,168,76,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(201,168,76,0.05) 0%, transparent 55%),
        var(--dark) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3rem 2rem 2rem;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(201,168,76,0.04) 0%, transparent 100%);
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
}

.hero-badge {
    display: inline-block;
    background: rgba(201,168,76,0.12);
    border: 1px solid var(--border);
    color: var(--gold);
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.3rem 1rem;
    border-radius: 2rem;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 700;
    color: var(--text);
    line-height: 1.15;
    margin-bottom: 0.8rem;
}

.hero-title span {
    background: linear-gradient(135deg, var(--gold), var(--gold-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-size: 1rem;
    color: var(--text-muted);
    font-weight: 300;
    max-width: 500px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Stats Bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border);
    background: var(--dark-2);
}

.stat-item { text-align: center; }

.stat-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--gold);
    display: block;
}

.stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Main Layout ── */
.main-wrapper {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 0;
    min-height: calc(100vh - 280px);
}

.form-panel {
    padding: 2.5rem 3rem;
    border-right: 1px solid var(--border);
}

.result-panel {
    padding: 2.5rem 2rem;
    background: var(--dark-2);
    position: sticky;
    top: 0;
    height: fit-content;
}

/* ── Section Headers ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-title::before {
    content: '';
    width: 3px;
    height: 1.1rem;
    background: var(--gold);
    border-radius: 2px;
    display: inline-block;
}

/* ── Input Grid ── */
.input-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;
}

/* ── Streamlit Input Overrides ── */
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background: var(--dark-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stNumberInput"] input:focus,
[data-testid="stSelectbox"] > div > div:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.15) !important;
}

div[data-testid="stSlider"] > div > div > div {
    background: var(--gold) !important;
}

label[data-testid="stWidgetLabel"] p {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #B8943A, var(--gold), #E8C96A) !important;
    color: #0A0C10 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    margin-top: 0.5rem !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(201,168,76,0.35) !important;
}

/* ── Result Card ── */
.result-card {
    border-radius: 12px;
    padding: 2rem 1.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.result-card.approved {
    background: linear-gradient(135deg, rgba(46,204,113,0.08), rgba(46,204,113,0.03));
    border: 1px solid rgba(46,204,113,0.25);
}

.result-card.rejected {
    background: linear-gradient(135deg, rgba(231,76,60,0.08), rgba(231,76,60,0.03));
    border: 1px solid rgba(231,76,60,0.25);
}

.result-card.waiting {
    background: var(--dark-3);
    border: 1px solid var(--border);
}

.result-icon {
    font-size: 3rem;
    margin-bottom: 0.8rem;
    display: block;
}

.result-status {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}

.result-status.approved { color: var(--green); }
.result-status.rejected { color: var(--red); }
.result-status.waiting  { color: var(--text-muted); }

.result-confidence {
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 300;
}

/* ── Probability Bars ── */
.prob-section { margin-top: 1.5rem; }

.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 0.35rem;
    margin-top: 0.8rem;
}

.prob-bar-bg {
    height: 6px;
    background: var(--dark-4);
    border-radius: 3px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.8s ease;
}

.prob-bar-fill.approved { background: linear-gradient(90deg, #27AE60, var(--green)); }
.prob-bar-fill.rejected { background: linear-gradient(90deg, #C0392B, var(--red)); }

/* ── Risk Indicator ── */
.risk-section {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

.risk-title {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
}

.risk-meter {
    height: 8px;
    background: linear-gradient(90deg, #2ECC71, #F39C12, #E74C3C);
    border-radius: 4px;
    position: relative;
    margin-bottom: 0.4rem;
}

.risk-needle {
    position: absolute;
    top: -4px;
    width: 16px; height: 16px;
    background: white;
    border: 2px solid var(--dark);
    border-radius: 50%;
    transform: translateX(-50%);
    transition: left 0.8s ease;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}

.risk-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: var(--text-muted);
}

/* ── Factor Pills ── */
.factors-section {
    margin-top: 1.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
}

.factor-pill {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 2rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0.2rem;
}

.factor-pill.positive {
    background: rgba(46,204,113,0.12);
    color: #2ECC71;
    border: 1px solid rgba(46,204,113,0.2);
}

.factor-pill.negative {
    background: rgba(231,76,60,0.12);
    color: #E74C3C;
    border: 1px solid rgba(231,76,60,0.2);
}

.factor-pill.neutral {
    background: rgba(201,168,76,0.1);
    color: var(--gold);
    border: 1px solid var(--border);
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem;
    border-top: 1px solid var(--border);
    font-size: 0.75rem;
    color: var(--text-muted);
    background: var(--dark-2);
}

/* ── Divider ── */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    margin: 1.5rem 0;
    opacity: 0.4;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .main-wrapper { grid-template-columns: 1fr; }
    .stats-bar { gap: 1.5rem; flex-wrap: wrap; }
    .form-panel { padding: 1.5rem; }
    .result-panel { padding: 1.5rem; }
}

/* ── Streamlit element cleanup ── */
[data-testid="stVerticalBlock"] { gap: 0.6rem; }
.stSpinner { color: var(--gold) !important; }
div[data-testid="stMarkdownContainer"] p { color: var(--text); }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ────────────────────────────────────────

def get_cibil_label(score):
    if score >= 750:   return ("Excellent", "positive")
    elif score >= 650: return ("Good", "neutral")
    elif score >= 500: return ("Fair", "negative")
    else:              return ("Poor", "negative")

def get_factors(cibil, income, loan_amount, loan_term,
                dependents, education, self_employed):
    factors = []
    if cibil >= 750:
        factors.append(("✦ Excellent CIBIL Score", "positive"))
    elif cibil >= 650:
        factors.append(("◈ Moderate CIBIL Score", "neutral"))
    else:
        factors.append(("✗ Low CIBIL Score", "negative"))

    ratio = loan_amount / max(income, 1)
    if ratio < 3:
        factors.append(("✦ Low Loan-to-Income Ratio", "positive"))
    elif ratio < 6:
        factors.append(("◈ Moderate Loan-to-Income Ratio", "neutral"))
    else:
        factors.append(("✗ High Loan-to-Income Ratio", "negative"))

    if education == "Graduate":
        factors.append(("✦ Graduate Education", "positive"))
    if dependents <= 1:
        factors.append(("✦ Few Dependents", "positive"))
    elif dependents >= 4:
        factors.append(("✗ High Dependents", "negative"))
    if loan_term <= 8:
        factors.append(("✦ Short Loan Term", "positive"))
    elif loan_term >= 16:
        factors.append(("◈ Long Loan Term", "neutral"))
    if self_employed == "Yes":
        factors.append(("◈ Self Employed", "neutral"))

    return factors


# ── HERO SECTION ────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · Random Forest · 98.1% Accuracy</div>
    <div class="hero-title">Loan<span>IQ</span> Approval Engine</div>
    <div class="hero-sub">
        Instant loan eligibility analysis powered by machine learning.
        Enter applicant details to receive a real-time decision.
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATS BAR ───────────────────────────────────────────────
st.markdown("""
<div class="stats-bar">
    <div class="stat-item">
        <span class="stat-value">98.1%</span>
        <span class="stat-label">Model Accuracy</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">4,269</span>
        <span class="stat-label">Training Records</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">11</span>
        <span class="stat-label">Input Features</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">0.99</span>
        <span class="stat-label">ROC-AUC Score</span>
    </div>
    <div class="stat-item">
        <span class="stat-value">RF</span>
        <span class="stat-label">Algorithm</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── MAIN LAYOUT ─────────────────────────────────────────────
left_col, right_col = st.columns([1.8, 1])

# ════════════════════════════════════════
# LEFT COLUMN — INPUT FORM
# ════════════════════════════════════════
with left_col:
    st.markdown('<div class="form-panel">', unsafe_allow_html=True)

    # ── Section 1: Credit Profile ──
    st.markdown('<div class="section-title">Credit Profile</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        cibil_score = st.slider(
            "CIBIL Score",
            min_value=300, max_value=900,
            value=650, step=1,
            help="Credit score ranging from 300 (poor) to 900 (excellent)"
        )
        cibil_label, cibil_type = get_cibil_label(cibil_score)
        st.markdown(
            f'<span class="factor-pill {cibil_type}">{cibil_label} Credit</span>',
            unsafe_allow_html=True
        )
    with c2:
        education = st.selectbox(
            "Education Level",
            options=["Graduate", "Not Graduate"],
            index=0
        )
        self_employed = st.selectbox(
            "Employment Type",
            options=["No", "Yes"],
            format_func=lambda x: "Salaried" if x == "No" else "Self Employed"
        )

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Section 2: Financial Details ──
    st.markdown('<div class="section-title">Financial Details</div>',
                unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        income_annum = st.number_input(
            "Annual Income (₹)",
            min_value=200_000,
            max_value=9_900_000,
            value=500_000,
            step=50_000,
            format="%d"
        )
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=300_000,
            max_value=39_500_000,
            value=1_000_000,
            step=100_000,
            format="%d"
        )
    with c4:
        loan_term = st.selectbox(
            "Loan Term (Years)",
            options=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            index=4
        )
        no_of_dependents = st.selectbox(
            "Number of Dependents",
            options=[0, 1, 2, 3, 4, 5],
            index=2
        )

    # Loan-to-income live indicator
    lti_ratio = loan_amount / max(income_annum, 1)
    lti_color = "#2ECC71" if lti_ratio < 3 else ("#F39C12" if lti_ratio < 6 else "#E74C3C")
    st.markdown(
        f'<div style="font-size:0.8rem; color:{lti_color}; margin-top:0.3rem;">'
        f'Loan-to-Income Ratio: <strong>{lti_ratio:.1f}x</strong>'
        f'{"  ✦ Healthy" if lti_ratio < 3 else ("  ◈ Moderate" if lti_ratio < 6 else "  ✗ High")}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Section 3: Asset Portfolio ──
    st.markdown('<div class="section-title">Asset Portfolio</div>',
                unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        residential_val = st.number_input(
            "Residential Assets (₹)",
            min_value=0,
            max_value=30_000_000,
            value=2_000_000,
            step=100_000,
            format="%d"
        )
        commercial_val = st.number_input(
            "Commercial Assets (₹)",
            min_value=0,
            max_value=20_000_000,
            value=1_000_000,
            step=100_000,
            format="%d"
        )
    with c6:
        luxury_val = st.number_input(
            "Luxury Assets (₹)",
            min_value=0,
            max_value=40_000_000,
            value=5_000_000,
            step=100_000,
            format="%d"
        )
        bank_val = st.number_input(
            "Bank Assets (₹)",
            min_value=0,
            max_value=15_000_000,
            value=2_000_000,
            step=100_000,
            format="%d"
        )

    total_assets = residential_val + commercial_val + luxury_val + bank_val
    st.markdown(
        f'<div style="font-size:0.82rem; color:var(--gold); margin-top:0.3rem;">'
        f'Total Asset Value: <strong>₹{total_assets:,.0f}</strong></div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)

    # ── Predict Button ──
    predict_clicked = st.button(
        "⬡  ANALYSE & PREDICT LOAN STATUS",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════
# RIGHT COLUMN — RESULT PANEL
# ════════════════════════════════════════
with right_col:
    st.markdown('<div class="result-panel">', unsafe_allow_html=True)

    if not predict_clicked:
        # ── Waiting State ──
        st.markdown("""
        <div class="result-card waiting">
            <span class="result-icon">◈</span>
            <div class="result-status waiting">Awaiting Input</div>
            <div class="result-confidence">
                Complete the form and click<br>Analyse to get a prediction
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:1.5rem;">
            <div class="risk-title">How It Works</div>
            <div style="font-size:0.82rem; color:#7A8399; line-height:1.8;">
                1. Enter applicant financial details<br>
                2. AI analyses 11 key features<br>
                3. Random Forest model predicts<br>
                4. Get instant decision + risk score
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Run Prediction ──
        with st.spinner("Analysing application..."):
            time.sleep(0.6)   # slight delay for UX feel

            input_data = pd.DataFrame([{
                'no_of_dependents':         no_of_dependents,
                'education':                education,
                'self_employed':            self_employed,
                'income_annum':             income_annum,
                'loan_amount':              loan_amount,
                'loan_term':                loan_term,
                'cibil_score':              cibil_score,
                'residential_assets_value': residential_val,
                'commercial_assets_value':  commercial_val,
                'luxury_assets_value':      luxury_val,
                'bank_asset_value':         bank_val,
            }])

            input_processed = preprocessor.transform(input_data)
            prediction      = model.predict(input_processed)[0].strip()
            probabilities   = model.predict_proba(input_processed)[0]

            # Map probabilities to labels
            classes       = [c.strip() for c in model.classes_]
            prob_dict     = dict(zip(classes, probabilities))
            prob_approved = prob_dict.get('Approved', 0)
            prob_rejected = prob_dict.get('Rejected', 0)
            confidence    = max(prob_approved, prob_rejected) * 100
            risk_position = (1 - prob_approved) * 100   # 0=safe, 100=risky

        is_approved = prediction == 'Approved'

        # ── Result Card ──
        card_class  = "approved" if is_approved else "rejected"
        icon        = "✦" if is_approved else "✗"
        status_text = "APPROVED" if is_approved else "REJECTED"

        st.markdown(f"""
        <div class="result-card {card_class}">
            <span class="result-icon">{icon}</span>
            <div class="result-status {card_class}">{status_text}</div>
            <div class="result-confidence">
                Confidence: <strong>{confidence:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability Bars ──
        prob_html = (
            f'<div class="prob-section">'
            f'<div class="risk-title">Prediction Probabilities</div>'
            f'<div class="prob-label">'
            f'<span>✦ Approved</span>'
            f'<span>{prob_approved*100:.1f}%</span>'
            f'</div>'
            f'<div class="prob-bar-bg">'
            f'<div class="prob-bar-fill approved" '
            f'style="width:{prob_approved*100:.1f}%"></div>'
            f'</div>'
            f'<div class="prob-label">'
            f'<span>✗ Rejected</span>'
            f'<span>{prob_rejected*100:.1f}%</span>'
            f'</div>'
            f'<div class="prob-bar-bg">'
            f'<div class="prob-bar-fill rejected" '
            f'style="width:{prob_rejected*100:.1f}%"></div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(prob_html, unsafe_allow_html=True)

        # ── Risk Meter ──
        st.markdown(f"""
        <div class="risk-section">
            <div class="risk-title">Risk Position</div>
            <div class="risk-meter">
                <div class="risk-needle"
                     style="left:{risk_position:.1f}%"></div>
            </div>
            <div class="risk-labels">
                <span>Low Risk</span>
                <span>High Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Key Factors ──
        factors = get_factors(
            cibil_score, income_annum, loan_amount,
            loan_term, no_of_dependents, education, self_employed
        )

        pills_html = "".join([
            f'<span class="factor-pill {ftype}">{fname}</span>'
            for fname, ftype in factors
        ])

        st.markdown(f"""
        <div class="factors-section">
            <div class="risk-title">Key Decision Factors</div>
            <div style="margin-top:0.5rem;">{pills_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Summary Metrics ──
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#7A8399; line-height:2;">
            <div style="display:flex; justify-content:space-between;">
                <span>Annual Income</span>
                <span style="color:#E8E6E0;">₹{income_annum:,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Loan Amount</span>
                <span style="color:#E8E6E0;">₹{loan_amount:,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>CIBIL Score</span>
                <span style="color:#E8E6E0;">{cibil_score}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>Total Assets</span>
                <span style="color:#E8E6E0;">₹{total_assets:,.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span>LTI Ratio</span>
                <span style="color:#E8E6E0;">{lti_ratio:.2f}x</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ──────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    LoanIQ · Built with Random Forest ML · Trained on 4,269 loan records ·
    For academic demonstration purposes only
</div>
""", unsafe_allow_html=True)
