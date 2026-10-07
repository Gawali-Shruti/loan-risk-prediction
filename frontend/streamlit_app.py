import json
import os
import textwrap
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LoanGuard AI | Credit Risk Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "models" / "metrics.json"

DEFAULT_API_URL = "http://127.0.0.1:8000"

try:
    API_URL = st.secrets.get("API_BASE_URL", DEFAULT_API_URL)
except Exception:
    API_URL = os.getenv("API_BASE_URL", DEFAULT_API_URL)

API_URL = str(API_URL).rstrip("/")

if "api_url" not in st.session_state:
    st.session_state.api_url = API_URL

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "applicant_form_data" not in st.session_state:
    st.session_state.applicant_form_data = {
        "Age": 30,
        "Education": "Graduate",
        "Dependents": 0,
        "Employment_Type": "Salaried",
        "Years_Employed": 5.0,
        "Annual_Income": 600000.0,
        "Monthly_Income": 50000.0,
        "Existing_Debt": 150000.0,
        "DTI_Ratio": 25.0,
        "Savings": 500000.0,
        "Credit_Score": 720.0,
        "Credit_History": "Good",
        "Loan_Amount": 700000.0,
        "Loan_Term": 60,
        "Property_Value": 1800000.0,
        "Assets": 2300000.0,
        "Previous_Defaults": 0,
        "Existing_Loans": 1,
    }

FEATURE_NAME_MAP = {
    "Credit_Score": "Credit Score",
    "DTI_Ratio": "Debt-to-Income Ratio",
    "Monthly_Income": "Monthly Income",
    "Annual_Income": "Annual Income",
    "Existing_Debt": "Existing Debt",
    "Loan_Amount": "Loan Amount",
    "Years_Employed": "Years Employed",
    "Property_Value": "Property Value",
    "Savings": "Savings",
    "Assets": "Total Assets",
    "Previous_Defaults": "Previous Defaults",
    "Existing_Loans": "Existing Loans",
    "Age": "Age",
    "Education": "Education",
    "Dependents": "Dependents",
    "Employment_Type": "Employment Type",
    "Credit_History": "Credit History",
    "Loan_Term": "Loan Term",
}


def format_feature_name(raw_name: str) -> str:
    clean = raw_name.replace("num__", "").replace("cat__", "")
    return FEATURE_NAME_MAP.get(clean, clean.replace("_", " ").title())


# ============================================================
# SAFE HTML RENDERER
# ============================================================

def render_html(markup: str):
    """Renders dedented HTML block safely without Markdown code block escaping."""
    cleaned = textwrap.dedent(markup).strip()
    st.markdown(cleaned, unsafe_allow_html=True)


# ============================================================
# SINGLE COHERENT DESIGN SYSTEM & CSS (EXPLICIT HIGH CONTRAST)
# ============================================================

render_html(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #111827;
    }

    .stApp {
        background-color: #F7F8FC !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
    }

    /* Force all main content area headings to Dark Navy (#172554) */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6,
    [data-testid="stMainBlockContainer"] h1,
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3,
    [data-testid="stMainBlockContainer"] h4,
    [data-testid="stMainBlockContainer"] h5,
    [data-testid="stMainBlockContainer"] h6 {
        color: #172554 !important;
        font-weight: 700 !important;
    }

    /* Main body text & captions */
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] span {
        color: #334155;
    }

    [data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p {
        color: #64748B !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #F8FAFC !important;
    }

    /* Sidebar Navigation Buttons */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
        border: 1px solid #1E293B !important;
        text-align: left !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border-color: #334155 !important;
    }

    [data-testid="stSidebar"] button[kind="primary"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #2563EB !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.3) !important;
    }

    .nav-category {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #64748B;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
        padding-left: 0.2rem;
    }

    /* Inputs (NumberInput, TextInput, Selectbox) High Contrast Styling */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] span {
        background-color: #FFFFFF !important;
        color: #111827 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }

    label[data-testid="stWidgetLabel"] p {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Form Container */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04) !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2) !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1D4ED8 !important;
    }

    /* Header Component */
    .page-header {
        margin-bottom: 1.2rem;
    }

    .page-header h1 {
        font-size: 1.8rem;
        font-weight: 800;
        color: #172554 !important;
        letter-spacing: -0.03em;
        margin: 0 0 0.25rem 0;
    }

    .page-header p {
        font-size: 0.95rem;
        color: #64748B !important;
        margin: 0;
    }

    /* Cards */
    .fin-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 1rem;
    }

    .fin-metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    .fin-metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
        margin-bottom: 0.3rem;
    }

    .fin-metric-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #172554;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .fin-metric-sub {
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 0.25rem;
    }

    /* Pipeline Workflow */
    .workflow-container {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.4rem;
        padding: 0.9rem 1.1rem;
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
    }

    .wf-step {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.45rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        color: #172554;
    }

    .wf-arrow {
        color: #94A3B8;
        font-weight: 700;
        font-size: 0.85rem;
    }

    /* Status Pills */
    .status-pill-online {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.76rem;
        font-weight: 600;
        color: #059669;
        background: rgba(5, 150, 105, 0.12);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid rgba(5, 150, 105, 0.25);
    }

    .status-pill-offline {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.76rem;
        font-weight: 600;
        color: #DC2626;
        background: rgba(220, 38, 38, 0.12);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        border: 1px solid rgba(220, 38, 38, 0.25);
    }

    /* Result Box Cards */
    .result-box-approved {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 5px solid #059669;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0 1.25rem 0;
    }

    .result-box-rejected {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 5px solid #DC2626;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0 1.25rem 0;
    }

    .result-box-medium {
        background-color: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 5px solid #D97706;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0 1.25rem 0;
    }

    .disclaimer-box {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        font-size: 0.8rem;
        color: #64748B;
        line-height: 1.45;
        margin-top: 1.5rem;
    }
    </style>
    """
)


# ============================================================
# DATA & API HELPERS
# ============================================================

def load_metrics():
    """Reads existing metrics.json if available."""
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def check_api_status(api_url: str) -> bool:
    """Verifies connection with FastAPI backend via GET /health."""
    try:
        res = requests.get(f"{api_url}/health", timeout=1.5)
        return res.status_code == 200
    except Exception:
        return False


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    render_html(
        """
        <div style="padding: 0.4rem 0 0.8rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.3rem;">💳</span>
                <span style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">LoanGuard AI</span>
            </div>
            <div style="font-size: 0.76rem; color: #94A3B8; margin-top: 0.15rem;">
                AI-Powered Credit Risk Platform
            </div>
        </div>
        """
    )

    is_online = check_api_status(st.session_state.api_url)
    if is_online:
        render_html(
            """
            <div style="margin-bottom: 1rem;">
                <span class="status-pill-online">● API Connected</span>
            </div>
            """
        )
    else:
        render_html(
            """
            <div style="margin-bottom: 1rem;">
                <span class="status-pill-offline">● API Offline</span>
            </div>
            """
        )

    render_html("<div class='nav-category'>OVERVIEW</div>")
    if st.button(
        "Home",
        use_container_width=True,
        key="nav_home",
        type="primary" if st.session_state.current_page == "Home" else "secondary",
    ):
        st.session_state.current_page = "Home"
        st.rerun()

    render_html("<div class='nav-category'>APPLICATION</div>")
    if st.button(
        "Loan Prediction",
        use_container_width=True,
        key="nav_pred",
        type="primary" if st.session_state.current_page == "Loan Prediction" else "secondary",
    ):
        st.session_state.current_page = "Loan Prediction"
        st.rerun()

    if st.button(
        "Applicant Risk Analysis",
        use_container_width=True,
        key="nav_risk",
        type="primary" if st.session_state.current_page == "Applicant Risk Analysis" else "secondary",
    ):
        st.session_state.current_page = "Applicant Risk Analysis"
        st.rerun()

    render_html("<div class='nav-category'>ANALYTICS</div>")
    if st.button(
        "Model Performance",
        use_container_width=True,
        key="nav_perf",
        type="primary" if st.session_state.current_page == "Model Performance" else "secondary",
    ):
        st.session_state.current_page = "Model Performance"
        st.rerun()

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    with st.expander("API Settings", expanded=False):
        new_url = st.text_input(
            "Backend URL",
            value=st.session_state.api_url,
            help="FastAPI server endpoint",
        )
        if new_url and new_url.strip() != st.session_state.api_url:
            st.session_state.api_url = new_url.strip().rstrip("/")
            st.rerun()


# ============================================================
# PAGE 1: HOME
# ============================================================

if st.session_state.current_page == "Home":
    col_hdr, col_btn = st.columns([3.5, 1.2])
    with col_hdr:
        render_html(
            """
            <div class="page-header">
                <h1>Loan Risk Intelligence</h1>
                <p>AI-powered loan eligibility and credit risk analysis.</p>
            </div>
            """
        )
    with col_btn:
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        if st.button("New Loan Assessment", use_container_width=True, type="primary"):
            st.session_state.current_page = "Loan Prediction"
            st.rerun()

    metrics_data = load_metrics()
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        acc_val = f"{metrics_data.get('accuracy', 0.80):.1%}"
        render_html(
            f"""
            <div class="fin-metric-card">
                <div class="fin-metric-label">Model Accuracy</div>
                <div class="fin-metric-val">{acc_val}</div>
                <div class="fin-metric-sub">Cross-validation benchmark</div>
            </div>
            """
        )

    with k2:
        prec_val = f"{metrics_data.get('precision', 0.852):.1%}"
        render_html(
            f"""
            <div class="fin-metric-card">
                <div class="fin-metric-label">Precision</div>
                <div class="fin-metric-val">{prec_val}</div>
                <div class="fin-metric-sub">Positive predictive value</div>
            </div>
            """
        )

    with k3:
        rec_val = f"{metrics_data.get('recall', 0.770):.1%}"
        render_html(
            f"""
            <div class="fin-metric-card">
                <div class="fin-metric-label">Recall</div>
                <div class="fin-metric-val">{rec_val}</div>
                <div class="fin-metric-sub">True approval sensitivity</div>
            </div>
            """
        )

    with k4:
        roc_val = f"{metrics_data.get('roc_auc', 0.889):.3f}"
        render_html(
            f"""
            <div class="fin-metric-card">
                <div class="fin-metric-label">ROC-AUC</div>
                <div class="fin-metric-val">{roc_val}</div>
                <div class="fin-metric-sub">Discrimination capability</div>
            </div>
            """
        )

    st.subheader("AI Credit Decisioning")
    render_html(
        """
        <div class="workflow-container">
            <div class="wf-step">Applicant Data</div>
            <div class="wf-arrow">→</div>
            <div class="wf-step">Validation</div>
            <div class="wf-arrow">→</div>
            <div class="wf-step">Preprocessing</div>
            <div class="wf-arrow">→</div>
            <div class="wf-step">ML Model</div>
            <div class="wf-arrow">→</div>
            <div class="wf-step">Probability</div>
            <div class="wf-arrow">→</div>
            <div class="wf-step">Risk Assessment</div>
            <div class="wf-arrow">→</div>
            <div class="wf-step">Decision</div>
        </div>
        """
    )

    st.subheader("How LoanGuard AI Works")
    w1, w2, w3 = st.columns(3)

    with w1:
        render_html(
            """
            <div class="fin-card">
                <div style="font-size: 0.7rem; font-weight: 700; color: #2563EB; letter-spacing: 0.08em;">01 / INTAKE</div>
                <div style="font-size: 1rem; font-weight: 700; color: #172554; margin: 0.3rem 0;">Applicant Assessment</div>
                <div style="font-size: 0.85rem; color: #334155; line-height: 1.5;">
                    Collect financial and credit information across income, debt obligations, employment history, and asset collateral.
                </div>
            </div>
            """
        )

    with w2:
        render_html(
            """
            <div class="fin-card">
                <div style="font-size: 0.7rem; font-weight: 700; color: #2563EB; letter-spacing: 0.08em;">02 / MODELING</div>
                <div style="font-size: 1rem; font-weight: 700; color: #172554; margin: 0.3rem 0;">Risk Intelligence</div>
                <div style="font-size: 0.85rem; color: #334155; line-height: 1.5;">
                    Analyze the applicant using the trained classification model with standardized feature transformation and ensemble scoring.
                </div>
            </div>
            """
        )

    with w3:
        render_html(
            """
            <div class="fin-card">
                <div style="font-size: 0.7rem; font-weight: 700; color: #2563EB; letter-spacing: 0.08em;">03 / EXECUTION</div>
                <div style="font-size: 1rem; font-weight: 700; color: #172554; margin: 0.3rem 0;">Decision Support</div>
                <div style="font-size: 0.85rem; color: #334155; line-height: 1.5;">
                    Return approval probability, decision, and risk level in milliseconds along with transparent feature attribution.
                </div>
            </div>
            """
        )

    render_html(
        """
        <div class="disclaimer-box">
            <b>Notice:</b> This system is a machine-learning demonstration and should not be used as the sole basis for real-world lending decisions.
        </div>
        """
    )


# ============================================================
# PAGE 2: LOAN PREDICTION
# ============================================================

elif st.session_state.current_page == "Loan Prediction":
    render_html(
        """
        <div class="page-header">
            <h1>Loan Prediction</h1>
            <p>Evaluate an applicant's eligibility using the trained credit risk model.</p>
        </div>
        """
    )

    if not is_online:
        st.warning(f"FastAPI backend is currently offline at {st.session_state.api_url}. Please start the backend to run live predictions.")

    with st.form("loan_prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            render_html(
                """
                <div style="margin-bottom: 0.6rem;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #2563EB; letter-spacing: 0.05em;">01</span>
                    <div style="font-size: 1rem; font-weight: 700; color: #172554;">PERSONAL INFORMATION</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Basic applicant profile and employment details.</div>
                </div>
                """
            )
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=int(st.session_state.applicant_form_data.get("Age", 30)),
                step=1,
            )
            education = st.selectbox(
                "Education",
                ["High School", "Graduate", "Postgraduate"],
                index=["High School", "Graduate", "Postgraduate"].index(
                    st.session_state.applicant_form_data.get("Education", "Graduate")
                ),
            )
            dependents = st.number_input(
                "Dependents",
                min_value=0,
                max_value=20,
                value=int(st.session_state.applicant_form_data.get("Dependents", 0)),
                step=1,
            )
            employment_type = st.selectbox(
                "Employment Type",
                ["Salaried", "Self-Employed", "Business", "Contract"],
                index=["Salaried", "Self-Employed", "Business", "Contract"].index(
                    st.session_state.applicant_form_data.get("Employment_Type", "Salaried")
                ),
            )
            years_employed = st.number_input(
                "Years Employed",
                min_value=0.0,
                max_value=80.0,
                value=float(st.session_state.applicant_form_data.get("Years_Employed", 5.0)),
                step=0.5,
            )

        with c2:
            render_html(
                """
                <div style="margin-bottom: 0.6rem;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #2563EB; letter-spacing: 0.05em;">02</span>
                    <div style="font-size: 1rem; font-weight: 700; color: #172554;">FINANCIAL PROFILE</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Income streams, cash flow, and debt ratios.</div>
                </div>
                """
            )
            annual_income = st.number_input(
                "Annual Income (₹)",
                min_value=1.0,
                max_value=100000000.0,
                value=float(st.session_state.applicant_form_data.get("Annual_Income", 600000.0)),
                step=10000.0,
            )
            monthly_income = st.number_input(
                "Monthly Income (₹)",
                min_value=1.0,
                max_value=10000000.0,
                value=float(st.session_state.applicant_form_data.get("Monthly_Income", 50000.0)),
                step=1000.0,
            )
            existing_debt = st.number_input(
                "Existing Debt (₹)",
                min_value=0.0,
                max_value=100000000.0,
                value=float(st.session_state.applicant_form_data.get("Existing_Debt", 150000.0)),
                step=5000.0,
            )
            dti_ratio = st.number_input(
                "Debt-to-Income Ratio (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.applicant_form_data.get("DTI_Ratio", 25.0)),
                step=0.5,
            )
            savings = st.number_input(
                "Savings (₹)",
                min_value=0.0,
                max_value=100000000.0,
                value=float(st.session_state.applicant_form_data.get("Savings", 500000.0)),
                step=10000.0,
            )

        with c3:
            render_html(
                """
                <div style="margin-bottom: 0.6rem;">
                    <span style="font-size: 0.75rem; font-weight: 700; color: #2563EB; letter-spacing: 0.05em;">03</span>
                    <div style="font-size: 1rem; font-weight: 700; color: #172554;">CREDIT & ASSETS</div>
                    <div style="font-size: 0.8rem; color: #64748B;">Credit history, requested term, and assets.</div>
                </div>
                """
            )
            credit_score = st.number_input(
                "Credit Score",
                min_value=300.0,
                max_value=900.0,
                value=float(st.session_state.applicant_form_data.get("Credit_Score", 720.0)),
                step=5.0,
            )
            credit_history = st.selectbox(
                "Credit History",
                ["Good", "Fair", "Poor"],
                index=["Good", "Fair", "Poor"].index(
                    st.session_state.applicant_form_data.get("Credit_History", "Good")
                ),
            )
            loan_amount = st.number_input(
                "Loan Amount (₹)",
                min_value=1.0,
                max_value=100000000.0,
                value=float(st.session_state.applicant_form_data.get("Loan_Amount", 700000.0)),
                step=10000.0,
            )
            loan_term = st.selectbox(
                "Loan Term (Months)",
                [12, 24, 36, 48, 60, 72, 84, 120],
                index=[12, 24, 36, 48, 60, 72, 84, 120].index(
                    st.session_state.applicant_form_data.get("Loan_Term", 60)
                ),
            )
            property_value = st.number_input(
                "Property Value (₹)",
                min_value=0.0,
                max_value=200000000.0,
                value=float(st.session_state.applicant_form_data.get("Property_Value", 1800000.0)),
                step=25000.0,
            )
            assets = st.number_input(
                "Total Assets (₹)",
                min_value=0.0,
                max_value=300000000.0,
                value=float(st.session_state.applicant_form_data.get("Assets", 2300000.0)),
                step=25000.0,
            )
            previous_defaults = st.number_input(
                "Previous Defaults",
                min_value=0,
                max_value=50,
                value=int(st.session_state.applicant_form_data.get("Previous_Defaults", 0)),
                step=1,
            )
            existing_loans = st.number_input(
                "Existing Loans",
                min_value=0,
                max_value=50,
                value=int(st.session_state.applicant_form_data.get("Existing_Loans", 1)),
                step=1,
            )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        predict_submitted = st.form_submit_button(
            "Predict Loan Eligibility",
            use_container_width=True,
        )

    if predict_submitted:
        # Input validation checks
        val_errors = []
        if age < 18 or age > 100:
            val_errors.append("Age must be between 18 and 100.")
        if credit_score < 300 or credit_score > 900:
            val_errors.append("Credit Score must be between 300 and 900.")
        if annual_income <= 0:
            val_errors.append("Annual Income must be greater than 0.")
        if monthly_income <= 0:
            val_errors.append("Monthly Income must be greater than 0.")
        if loan_amount <= 0:
            val_errors.append("Loan Amount must be greater than 0.")
        if dti_ratio < 0 or dti_ratio > 100:
            val_errors.append("Debt-to-Income Ratio must be between 0% and 100%.")

        if val_errors:
            for err in val_errors:
                st.error(f"⚠️ Validation Error: {err}")
        else:
            payload = {
                "Age": int(age),
                "Education": str(education),
                "Dependents": int(dependents),
                "Employment_Type": str(employment_type),
                "Years_Employed": float(years_employed),
                "Annual_Income": float(annual_income),
                "Monthly_Income": float(monthly_income),
                "Credit_Score": float(credit_score),
                "Credit_History": str(credit_history),
                "Loan_Amount": float(loan_amount),
                "Loan_Term": int(loan_term),
                "Existing_Debt": float(existing_debt),
                "DTI_Ratio": float(dti_ratio),
                "Property_Value": float(property_value),
                "Savings": float(savings),
                "Assets": float(assets),
                "Previous_Defaults": int(previous_defaults),
                "Existing_Loans": int(existing_loans),
            }
            st.session_state.applicant_form_data = payload

            try:
                with st.spinner("Evaluating model decision..."):
                    resp = requests.post(
                        f"{st.session_state.api_url}/predict",
                        json=payload,
                        timeout=10,
                    )
                    resp.raise_for_status()
                    st.session_state.last_prediction = resp.json()
            except Exception as e:
                st.error(f"Prediction request failed: {e}")
                st.info(f"Ensure backend is running at {st.session_state.api_url}")

    result = st.session_state.last_prediction
    if result:
        decision = result.get("prediction", "Rejected")
        prob = result.get("approval_probability", 0.0)
        risk = result.get("risk_level", "Medium")
        latency = result.get("latency_ms", 0.0)

        dec_lower = str(decision).strip().lower()
        risk_lower = str(risk).strip().lower()

        if "approved" in dec_lower and "low" in risk_lower:
            box_class = "result-box-approved"
            title_color = "#059669"
            icon = "APPROVED"
        elif "rejected" in dec_lower or "high" in risk_lower:
            box_class = "result-box-rejected"
            title_color = "#DC2626"
            icon = "REJECTED"
        else:
            box_class = "result-box-medium"
            title_color = "#D97706"
            icon = "MEDIUM RISK"

        render_html(
            f"""
            <div class="{box_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase;">MODEL DECISION</div>
                        <div style="font-size: 1.8rem; font-weight: 800; color: {title_color}; margin: 0.2rem 0;">{icon}</div>
                        <div style="font-size: 0.9rem; color: #334155;">
                            Approval Probability: <b>{prob}%</b> &nbsp;•&nbsp; Assessment: <b>{risk.upper()} RISK</b>
                        </div>
                    </div>
                    <div style="font-size: 0.78rem; color: #64748B; background: #FFFFFF; padding: 0.35rem 0.75rem; border-radius: 6px; border: 1px solid #E2E8F0;">
                        Latency: {latency} ms
                    </div>
                </div>
            </div>
            """
        )

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Decision</div>
                    <div class="fin-metric-val" style="color: {title_color};">{decision}</div>
                    <div class="fin-metric-sub">Classification output</div>
                </div>
                """
            )
        with r2:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Approval Probability</div>
                    <div class="fin-metric-val">{prob}%</div>
                    <div class="fin-metric-sub">Random Forest confidence</div>
                </div>
                """
            )
        with r3:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Risk Level</div>
                    <div class="fin-metric-val">{risk.upper()}</div>
                    <div class="fin-metric-sub">Calibrated risk tier</div>
                </div>
                """
            )
        with r4:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Inference Latency</div>
                    <div class="fin-metric-val">{latency} <span style="font-size: 0.9rem;">ms</span></div>
                    <div class="fin-metric-sub">Pipeline execution time</div>
                </div>
                """
            )

        # STREAMLIT-NATIVE KEY MODEL FACTORS
        top_factors = result.get("top_factors", [])
        st.markdown("<div style='margin-top: 1.8rem;'></div>", unsafe_allow_html=True)
        st.subheader("Key Model Factors")
        st.caption("Top influential variables evaluated by the trained model.")

        if top_factors:
            max_imp = max([f.get("importance", 0.0) for f in top_factors] or [1.0])
            if max_imp <= 0:
                max_imp = 1.0

            with st.container(border=True):
                for factor in top_factors:
                    raw_feature = factor.get("feature", "")
                    clean_name = format_feature_name(raw_feature)
                    imp_val = factor.get("importance", 0.0)
                    normalized_imp = min(1.0, max(0.0, imp_val / max_imp))

                    f_col1, f_col2, f_col3 = st.columns([2.8, 5, 1.2])
                    with f_col1:
                        st.markdown(f"**{clean_name}**")
                    with f_col2:
                        st.progress(normalized_imp)
                    with f_col3:
                        st.markdown(f"`{imp_val:.5f}`")
        else:
            st.info("No model factor information available.")


# ============================================================
# PAGE 3: APPLICANT RISK ANALYSIS
# ============================================================

elif st.session_state.current_page == "Applicant Risk Analysis":
    render_html(
        """
        <div class="page-header">
            <h1>Applicant Risk Analysis</h1>
            <p>Understand the applicant's financial exposure and credit profile.</p>
        </div>
        """
    )

    col_inputs, col_visuals = st.columns([1, 1.4])

    with col_inputs:
        st.subheader("Financial Parameters")
        with st.container(border=True):
            user_income = st.number_input(
                "Annual Income (₹)",
                min_value=1.0,
                max_value=100000000.0,
                value=float(st.session_state.applicant_form_data.get("Annual_Income", 600000.0)),
                step=25000.0,
                key="risk_income",
            )
            user_loan = st.number_input(
                "Loan Amount (₹)",
                min_value=1.0,
                max_value=100000000.0,
                value=float(st.session_state.applicant_form_data.get("Loan_Amount", 700000.0)),
                step=25000.0,
                key="risk_loan",
            )
            user_debt = st.number_input(
                "Existing Debt (₹)",
                min_value=0.0,
                max_value=100000000.0,
                value=float(st.session_state.applicant_form_data.get("Existing_Debt", 150000.0)),
                step=10000.0,
                key="risk_debt",
            )
            user_credit = st.number_input(
                "Credit Score",
                min_value=300.0,
                max_value=900.0,
                value=float(st.session_state.applicant_form_data.get("Credit_Score", 720.0)),
                step=5.0,
                key="risk_credit",
            )

    dti_calc = (user_debt / user_income * 100) if user_income > 0 else 0.0
    loan_to_income = (user_loan / user_income * 100) if user_income > 0 else 0.0

    with col_visuals:
        st.subheader("Key Exposure Metrics")
        m1, m2 = st.columns(2)
        with m1:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Debt-to-Income (DTI)</div>
                    <div class="fin-metric-val">{dti_calc:.1f}%</div>
                    <div class="fin-metric-sub">{'Optimal (< 36%)' if dti_calc < 36 else 'Elevated (≥ 36%)'}</div>
                </div>
                """
            )
        with m2:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Loan-to-Income Ratio</div>
                    <div class="fin-metric-val">{loan_to_income:.1f}%</div>
                    <div class="fin-metric-sub">Relative leverage load</div>
                </div>
                """
            )

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        m3, m4 = st.columns(2)
        with m3:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Annual Income</div>
                    <div class="fin-metric-val">₹{user_income:,.0f}</div>
                    <div class="fin-metric-sub">Declared gross cashflow</div>
                </div>
                """
            )
        with m4:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Total Debt Burden</div>
                    <div class="fin-metric-val">₹{user_debt:,.0f}</div>
                    <div class="fin-metric-sub">Liabilities outstanding</div>
                </div>
                """
            )

    # LIGHT PLOTLY FINANCIAL COMPARISON CHART
    st.subheader("Financial Comparison")
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Annual Income", "Loan Amount", "Existing Debt"],
                y=[user_income, user_loan, user_debt],
                marker_color=["#2563EB", "#172554", "#64748B"],
                text=[f"₹{user_income:,.0f}", f"₹{user_loan:,.0f}", f"₹{user_debt:,.0f}"],
                textposition="outside",
                textfont=dict(family="Inter, sans-serif", size=12, color="#172554"),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=30, b=20),
        height=300,
        font=dict(family="Inter, sans-serif", size=13, color="#334155"),
        xaxis=dict(
            showgrid=False,
            color="#475569",
            tickfont=dict(size=12, weight=600, color="#172554"),
        ),
        yaxis=dict(
            gridcolor="#E2E8F0",
            color="#475569",
            zerolinecolor="#E2E8F0",
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    c_score_col, c_tier_col = st.columns([1.2, 1])

    with c_score_col:
        st.subheader("Credit Score Standing")
        score_ratio = min(max((user_credit - 300) / 600, 0.0), 1.0)
        st.progress(score_ratio)

        if user_credit >= 750:
            score_tier = "Excellent"
            tier_color = "#059669"
            tier_desc = "Strong credit history indicating very low risk of default."
        elif user_credit >= 670:
            score_tier = "Good"
            tier_color = "#2563EB"
            tier_desc = "Standard prime range with dependable repayment tendencies."
        elif user_credit >= 580:
            score_tier = "Fair"
            tier_color = "#D97706"
            tier_desc = "Moderate risk exposure; additional collateral or shorter terms recommended."
        else:
            score_tier = "Subprime / Poor"
            tier_color = "#DC2626"
            tier_desc = "Elevated default probability requiring tight credit restrictions."

        st.markdown(f"Standing: **{score_tier} ({user_credit:.0f} / 900)**")
        st.caption(tier_desc)

    with c_tier_col:
        st.subheader("Risk Interpretation")
        if user_credit >= 720 and dti_calc < 35:
            interp_badge = "LOW RISK"
            interp_class = "result-box-approved"
            interp_color = "#059669"
            interp_text = "Applicant demonstrates prime credit standing and sound debt serviceability."
        elif user_credit >= 620 and dti_calc <= 45:
            interp_badge = "MEDIUM RISK"
            interp_class = "result-box-medium"
            interp_color = "#D97706"
            interp_text = "Applicant exhibits moderate risk characteristics requiring standard verification."
        else:
            interp_badge = "HIGH RISK"
            interp_class = "result-box-rejected"
            interp_color = "#DC2626"
            interp_text = "Elevated leverage or subprime score profile indicates heightened default exposure."

        render_html(
            f"""
            <div class="{interp_class}">
                <div style="font-size: 0.7rem; font-weight: 700; color: #64748B;">CREDIT RISK RATING</div>
                <div style="font-size: 1.3rem; font-weight: 800; color: {interp_color}; margin: 0.2rem 0;">{interp_badge}</div>
                <div style="font-size: 0.82rem; color: #475569;">{interp_text}</div>
            </div>
            """
        )


# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================

elif st.session_state.current_page == "Model Performance":
    render_html(
        """
        <div class="page-header">
            <h1 style="font-size: 32px; font-weight: 800; color: #172554; margin: 0 0 0.25rem 0;">Model Performance</h1>
            <p style="font-size: 15px; color: #64748B; margin: 0;">Review the evaluation results of the trained Random Forest classification model.</p>
        </div>
        """
    )

    metrics = load_metrics()

    if not metrics:
        st.warning("No metrics.json file found in models/. Run training pipeline first.")
    else:
        m1, m2, m3, m4, m5 = st.columns(5, gap="medium")
        with m1:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Accuracy</div>
                    <div class="fin-metric-val">{metrics.get('accuracy', 0.0):.1%}</div>
                    <div class="fin-metric-sub">Overall correct rate</div>
                </div>
                """
            )
        with m2:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Precision</div>
                    <div class="fin-metric-val">{metrics.get('precision', 0.0):.1%}</div>
                    <div class="fin-metric-sub">Approval reliability</div>
                </div>
                """
            )
        with m3:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">Recall</div>
                    <div class="fin-metric-val">{metrics.get('recall', 0.0):.1%}</div>
                    <div class="fin-metric-sub">Qualified detection</div>
                </div>
                """
            )
        with m4:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">F1 Score</div>
                    <div class="fin-metric-val">{metrics.get('f1', 0.0):.3f}</div>
                    <div class="fin-metric-sub">Harmonic mean</div>
                </div>
                """
            )
        with m5:
            render_html(
                f"""
                <div class="fin-metric-card">
                    <div class="fin-metric-label">ROC-AUC</div>
                    <div class="fin-metric-val">{metrics.get('roc_auc', 0.0):.3f}</div>
                    <div class="fin-metric-sub">Area under curve</div>
                </div>
                """
            )

        st.markdown("<div style='margin-top: 1.75rem;'></div>", unsafe_allow_html=True)
        col_cm, col_meta = st.columns([2.1, 1], gap="large")

        with col_cm:
            render_html(
                """
                <h2 style="font-size: 26px; font-weight: 700; color: #172554; margin: 0 0 1rem 0; line-height: 1.2;">Confusion Matrix</h2>
                """
            )
            cm_data = metrics.get("confusion_matrix", [[113, 22], [38, 127]])
            tn, fp, fn, tp = cm_data[0][0], cm_data[0][1], cm_data[1][0], cm_data[1][1]

            render_html(
                f"""
                <div class="fin-card" style="padding: 1.5rem;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; text-align: center; margin: 0 auto;">
                        <thead>
                            <tr style="border-bottom: 2px solid #E2E8F0; color: #172554;">
                                <th style="padding: 0.85rem 1rem; text-align: left; font-size: 0.9rem; color: #172554;">Actual / Predicted</th>
                                <th style="padding: 0.85rem 1rem; background: #F8FAFC; font-size: 0.9rem; color: #172554; width: 35%;">Predicted Rejected</th>
                                <th style="padding: 0.85rem 1rem; background: #F8FAFC; font-size: 0.9rem; color: #172554; width: 35%;">Predicted Approved</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr style="border-bottom: 1px solid #F1F5F9;">
                                <td style="padding: 1rem; font-weight: 700; text-align: left; color: #172554; font-size: 0.9rem;">Actual Rejected</td>
                                <td style="padding: 1rem; background: #F0FDF4; font-weight: 700; color: #059669; font-size: 1.1rem;">
                                    {tn} <div style="font-size: 0.78rem; font-weight: 500; color: #64748B; margin-top: 0.2rem;">True Negative</div>
                                </td>
                                <td style="padding: 1rem; background: #FEF2F2; font-weight: 700; color: #DC2626; font-size: 1.1rem;">
                                    {fp} <div style="font-size: 0.78rem; font-weight: 500; color: #64748B; margin-top: 0.2rem;">False Positive</div>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding: 1rem; font-weight: 700; text-align: left; color: #172554; font-size: 0.9rem;">Actual Approved</td>
                                <td style="padding: 1rem; background: #FEF2F2; font-weight: 700; color: #DC2626; font-size: 1.1rem;">
                                    {fn} <div style="font-size: 0.78rem; font-weight: 500; color: #64748B; margin-top: 0.2rem;">False Negative</div>
                                </td>
                                <td style="padding: 1rem; background: #F0FDF4; font-weight: 700; color: #059669; font-size: 1.1rem;">
                                    {tp} <div style="font-size: 0.78rem; font-weight: 500; color: #64748B; margin-top: 0.2rem;">True Positive</div>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """
            )

        with col_meta:
            render_html(
                """
                <h2 style="font-size: 26px; font-weight: 700; color: #172554; margin: 0 0 1rem 0; line-height: 1.2;">Model Information</h2>
                """
            )
            render_html(
                f"""
                <div class="fin-card" style="padding: 1.5rem;">
                    <div style="margin-bottom: 1.15rem;">
                        <div style="font-size: 15px; font-weight: 500; color: #64748B; margin-bottom: 0.25rem;">Algorithm</div>
                        <div style="font-size: 19px; font-weight: 700; color: #172554; word-break: break-word; line-height: 1.3;">{metrics.get('model_name', 'RandomForestClassifier')}</div>
                    </div>
                    <div style="margin-bottom: 1.15rem;">
                        <div style="font-size: 15px; font-weight: 500; color: #64748B; margin-bottom: 0.25rem;">Model Version</div>
                        <div style="font-size: 19px; font-weight: 700; color: #172554;">v{metrics.get('model_version', '1.0.0')}</div>
                    </div>
                    <div style="margin-bottom: 1.15rem;">
                        <div style="font-size: 15px; font-weight: 500; color: #64748B; margin-bottom: 0.25rem;">Test Samples</div>
                        <div style="font-size: 19px; font-weight: 700; color: #172554;">{metrics.get('test_rows', 300):,}</div>
                    </div>
                    <div>
                        <div style="font-size: 15px; font-weight: 500; color: #64748B; margin-bottom: 0.25rem;">Training Samples</div>
                        <div style="font-size: 19px; font-weight: 700; color: #172554;">{metrics.get('training_rows', 1200):,}</div>
                    </div>
                </div>
                """
            )

        render_html(
            """
            <div class="fin-card" style="margin-top: 1.25rem;">
                <div style="font-size: 0.95rem; font-weight: 700; color: #172554; margin-bottom: 0.4rem;">Evaluation Insights</div>
                <div style="font-size: 0.88rem; color: #334155; line-height: 1.55;">
                    The ensemble classifier achieves high discriminant capacity (ROC-AUC 0.889) with balanced precision (85.2%) and recall (77.0%), mitigating false approvals while capturing qualified borrowers reliably.
                </div>
            </div>
            """
        )