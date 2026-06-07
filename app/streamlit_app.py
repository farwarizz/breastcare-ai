"""
app/streamlit_app.py — BreastCare AI — Full animated Streamlit UI.

Architecture:
  - Fixed top nav + sidebar inputs + three result panels (centre + right)
  - Session state carries all ML/SHAP/RAG/LLM results across reruns (memory)
  - Animated CSS injected at startup; Plotly charts for gauge + waterfall + CV comparison
  - Pink/white/rose healthcare aesthetic; light mode forced via config.toml
  - All pipeline calls go through src/pipeline.py exclusively
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import (
    SHAPE_MAP, MARGIN_MAP, DENSITY_MAP, FEATURE_NAMES,
    MODEL_METADATA_PATH, LOW_RISK_THRESHOLD, HIGH_RISK_THRESHOLD,
    get_api_key,
)

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreastCare AI — Risk Estimation",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS injection ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,400&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
  --ivory: #fdf8f5;
  --rose-50: #fff1f4;
  --rose-100: #ffe0e6;
  --rose-200: #ffc2ce;
  --rose-300: #ff94ac;
  --rose-400: #e8647e;
  --rose-500: #c9445d;
  --rose-600: #a83050;
  --navy: #0e1c2f;
  --navy-80: #1a2d47;
  --charcoal: #2c3e50;
  --med-blue: #3d6b9e;
  --slate: #607b96;
  --slate-light: #8fa6bc;
  --text-primary: #1a2332;
  --text-secondary: #4a5568;
  --text-muted: #718096;
  --border: #e8eff7;
  --border-rose: #f5dde3;
  --shadow: 0 4px 24px rgba(14,28,47,.07);
  --shadow-rose: 0 8px 32px rgba(200,68,93,.15);
}

/* ── Force light mode globally ── */
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
.stApp { background-color: var(--ivory) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"],
section[data-testid="stSidebar"] [data-baseweb="base-input"] {
  background: var(--ivory) !important;
  border-color: var(--border) !important;
}
section[data-testid="stSidebar"] [data-baseweb="popover"],
section[data-testid="stSidebar"] [role="option"] {
  background: #ffffff !important;
  color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }
.stDeployButton { display: none !important; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
  font-family: 'Playfair Display', Georgia, serif !important;
  color: var(--navy) !important;
  letter-spacing: -.02em;
}
p, div, span, label, li, td, th, .stMarkdown {
  font-family: 'DM Sans', system-ui, sans-serif !important;
  color: var(--text-secondary) !important;
}

/* ── Animated top banner ── */
.top-banner {
  background: linear-gradient(135deg, var(--navy) 0%, #1a2d47 50%, #0e1c2f 100%);
  padding: 18px 32px;
  border-radius: 0 0 20px 20px;
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: relative;
  overflow: hidden;
}
.top-banner::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(ellipse 60% 80% at 80% 50%, rgba(200,68,93,.15) 0%, transparent 60%);
  pointer-events: none;
}
.banner-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.6rem !important;
  font-weight: 700;
  color: #ffffff !important;
  display: flex;
  align-items: center;
  gap: 10px;
}
.banner-subtitle {
  font-size: .8rem;
  color: rgba(255,255,255,.5) !important;
  font-family: 'DM Sans', sans-serif !important;
  margin-top: 2px;
}
.banner-disclaimer {
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 8px;
  padding: 8px 14px;
  font-size: .72rem;
  color: rgba(255,255,255,.6) !important;
  font-family: 'DM Sans', sans-serif !important;
  max-width: 340px;
  text-align: right;
}

/* ── Cards ── */
.bc-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 24px;
  box-shadow: var(--shadow);
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
}
.bc-card-accent::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(to right, var(--rose-300), var(--rose-500), var(--med-blue));
}
.bc-card-title {
  font-family: 'Playfair Display', serif !important;
  font-size: 1.05rem !important;
  font-weight: 700;
  color: var(--navy) !important;
  margin-bottom: 4px;
}
.bc-card-sub {
  font-size: .75rem !important;
  color: var(--text-muted) !important;
  margin-bottom: 16px;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Metric pill ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }
.metric-pill {
  background: var(--ivory);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 12px 18px;
  flex: 1;
  min-width: 100px;
}
.metric-pill-val {
  font-family: 'Playfair Display', serif;
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--navy);
  line-height: 1;
}
.metric-pill-lab {
  font-size: .7rem;
  color: var(--text-muted);
  margin-top: 3px;
  font-family: 'DM Sans', sans-serif;
}

/* ── Risk category badges ── */
.risk-low    { color: #1d4ed8 !important; background: #eff6ff; border: 1px solid #bfdbfe; }
.risk-moderate { color: #b45309 !important; background: #fffbeb; border: 1px solid #fde68a; }
.risk-elevated { color: var(--rose-600) !important; background: var(--rose-50); border: 1px solid var(--rose-200); }
.risk-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 14px;
  border-radius: 50px;
  font-size: .8rem;
  font-weight: 600;
  font-family: 'DM Sans', sans-serif;
}

/* ── Section headings ── */
.section-tag {
  font-size: .68rem;
  font-weight: 700;
  letter-spacing: .12em;
  text-transform: uppercase;
  color: var(--rose-500) !important;
  font-family: 'DM Sans', sans-serif !important;
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}
.section-tag::before {
  content: '';
  display: inline-block;
  width: 16px;
  height: 1.5px;
  background: var(--rose-400);
}

/* ── Sidebar form elements ── */
.stSlider [data-baseweb="slider"] { margin-top: 4px; }
.stSlider .st-emotion-cache-ue6h4q { color: var(--rose-500) !important; }
.stButton > button {
  background: var(--rose-500) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 50px !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 500 !important;
  font-size: .88rem !important;
  padding: 10px 24px !important;
  width: 100% !important;
  transition: all .22s ease !important;
  box-shadow: 0 4px 16px rgba(200,68,93,.3) !important;
}
.stButton > button:hover {
  background: var(--rose-600) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 24px rgba(200,68,93,.35) !important;
}

/* ── Chat messages ── */
.chat-msg-bot, .chat-msg-user {
  display: flex;
  gap: 10px;
  margin-bottom: 14px;
  align-items: flex-end;
}
.chat-msg-user { flex-direction: row-reverse; }
.chat-avatar {
  width: 30px; height: 30px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px;
  flex-shrink: 0;
}
.chat-avatar-bot  { background: linear-gradient(135deg, var(--rose-400), var(--rose-600)); }
.chat-avatar-user { background: var(--navy); }
.chat-bubble {
  padding: 10px 14px;
  border-radius: 14px;
  font-size: .83rem;
  line-height: 1.55;
  max-width: 85%;
  font-family: 'DM Sans', sans-serif;
}
.chat-bubble-bot {
  background: #ffffff;
  border: 1px solid var(--border);
  border-bottom-left-radius: 4px;
  color: var(--text-primary);
}
.chat-bubble-user {
  background: var(--navy);
  color: #ffffff;
  border-bottom-right-radius: 4px;
}
.source-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: .67rem;
  background: var(--rose-50);
  color: var(--rose-600);
  border: 1px solid var(--rose-100);
  border-radius: 20px;
  padding: 2px 8px;
  margin-top: 8px;
}

/* ── Plain language panel ── */
.plain-lang {
  background: linear-gradient(135deg, var(--rose-50), rgba(61,107,158,.03));
  border: 1px solid var(--rose-100);
  border-radius: 14px;
  padding: 20px;
  font-size: .85rem;
  line-height: 1.7;
  color: var(--text-secondary);
  font-family: 'DM Sans', sans-serif;
}

/* ── Pipeline step badges ── */
.pipe-step {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.1);
  border-radius: 8px;
  padding: 6px 12px;
  font-size: .75rem;
  color: rgba(255,255,255,.7);
  font-family: 'DM Mono', monospace;
  margin: 4px;
}
.pipe-step.done {
  border-color: rgba(200,68,93,.4);
  background: rgba(200,68,93,.1);
  color: #f9a8b8;
}

/* ── Disclaimer box ── */
.disclaimer-notice {
  display: flex;
  gap: 12px;
  align-items: flex-start;
  background: rgba(61,107,158,.05);
  border: 1px solid rgba(61,107,158,.15);
  border-radius: 12px;
  padding: 14px 16px;
  font-size: .78rem;
  color: var(--slate);
  line-height: 1.55;
  font-family: 'DM Sans', sans-serif;
  margin-top: 12px;
}

/* ── Animated ribbon ── */
@keyframes ribbon-float {
  0%,100% { transform: translateY(0) rotate(-1deg); }
  50%      { transform: translateY(-6px) rotate(1deg); }
}
.ribbon-anim { animation: ribbon-float 4s ease-in-out infinite; display: inline-block; }

/* ── Plotly chart container ── */
.js-plotly-plot { border-radius: 12px !important; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  background: var(--ivory) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-secondary) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .82rem !important;
  padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
  background: var(--rose-50) !important;
  border-color: var(--rose-200) !important;
  color: var(--rose-600) !important;
  font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 16px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--ivory); }
::-webkit-scrollbar-thumb { background: var(--rose-200); border-radius: 3px; }
</style>
"""

# ── animation JS ────────────────────────────────────────────────────────────
ANIMATION_JS = """
<script>
// Scroll-reveal animation observer
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.style.opacity = '1';
      entry.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

document.querySelectorAll('.reveal').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(24px)';
  el.style.transition = 'all .65s cubic-bezier(.22,.61,.36,1)';
  observer.observe(el);
});
</script>
"""

# ── session state init ───────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "pipeline_result":   None,   # full pipeline output dict
        "chat_history":      [],     # list of {role, content, sources}
        "last_features":     None,   # last submitted feature dict
        "pipeline_ran":      False,
        "api_key_present":   bool(get_api_key()),
        "run_count":         0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── helpers ──────────────────────────────────────────────────────────────────
def _load_model_metadata() -> dict:
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH) as f:
            return json.load(f)
    return {}

def _risk_color(cat: str) -> str:
    return {"Low": "#1d4ed8", "Moderate": "#d97706", "Elevated": "#c9445d"}.get(cat, "#718096")

def _risk_emoji(cat: str) -> str:
    return {"Low": "🟢", "Moderate": "🟡", "Elevated": "🔴"}.get(cat, "⚪")

def _models_ready() -> bool:
    from src.config import LGBM_MODEL_PATH, FAISS_INDEX_PATH
    return LGBM_MODEL_PATH.exists() and FAISS_INDEX_PATH.exists()

# ── Charts ───────────────────────────────────────────────────────────────────
def make_gauge_chart(probability: float, risk_cat: str) -> go.Figure:
    pct = round(probability * 100, 1)
    color = _risk_color(risk_cat)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 36, "family": "Playfair Display", "color": "#0e1c2f"}},
        delta={"reference": 50, "increasing": {"color": "#c9445d"}, "decreasing": {"color": "#1d4ed8"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#e8eff7",
                     "tickfont": {"color": "#718096", "size": 10}},
            "bar":  {"color": color, "thickness": 0.28},
            "bgcolor": "#fdf8f5",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35],  "color": "#eff6ff"},
                {"range": [35, 65], "color": "#fffbeb"},
                {"range": [65, 100],"color": "#fff1f4"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": pct,
            },
        },
        title={"text": f"<b>{risk_cat} Risk</b>", "font": {"size": 14, "color": color, "family": "DM Sans"}},
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=40, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "DM Sans"},
    )
    return fig


def make_shap_waterfall(explanation: dict, features: dict) -> go.Figure:
    attrs = explanation["feature_attributions"]
    feat_labels = explanation.get("feature_labels", {})

    ordered = sorted(attrs.items(), key=lambda x: abs(x[1]), reverse=True)
    names   = [f"{k.title()}<br><sup>{feat_labels.get(k,'')}</sup>" for k, _ in ordered]
    values  = [v for _, v in ordered]
    colors  = ["#c9445d" if v > 0 else "#3d6b9e" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{'+' if v >= 0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont={"size": 11, "family": "DM Mono", "color": "#4a5568"},
    ))

    fig.add_vline(x=0, line_width=1.5, line_color="#e8eff7")

    fig.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=130, r=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "title": "SHAP value (impact on malignancy probability)",
            "titlefont": {"size": 10, "color": "#718096"},
            "tickfont": {"size": 10, "color": "#718096", "family": "DM Mono"},
            "gridcolor": "#f0f4f8",
            "zeroline": False,
        },
        yaxis={
            "tickfont": {"size": 11, "color": "#4a5568", "family": "DM Sans"},
            "automargin": True,
        },
        bargap=0.35,
        font={"family": "DM Sans"},
    )
    return fig


def make_cv_chart(cv_data: dict) -> go.Figure:
    if not cv_data:
        return go.Figure()

    models = list(cv_data.keys())
    aucs   = [cv_data[m]["auc"]      for m in models]
    accs   = [cv_data[m]["accuracy"] / 100 for m in models]

    colors_auc = ["#c9445d" if m in ("XGBoost", "LightGBM") else "#8fa6bc" for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="ROC-AUC",
        x=models, y=aucs,
        marker_color=colors_auc,
        text=[f"{a:.4f}" for a in aucs],
        textposition="outside",
        textfont={"size": 10, "family": "DM Mono"},
        yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        name="Accuracy",
        x=models, y=accs,
        mode="lines+markers",
        line={"color": "#3d6b9e", "width": 2, "dash": "dot"},
        marker={"size": 8, "color": "#3d6b9e"},
        yaxis="y2",
    ))

    fig.update_layout(
        height=260,
        margin=dict(t=20, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font": {"size": 10, "family": "DM Sans"}, "bgcolor": "rgba(0,0,0,0)"},
        barmode="group",
        yaxis={
            "title": "AUC",
            "range": [0.95, 1.0],
            "tickformat": ".3f",
            "gridcolor": "#f0f4f8",
            "tickfont": {"size": 9, "family": "DM Mono"},
            "title_font": {"size": 10, "color": "#718096"},
        },
        yaxis2={
            "title": "Accuracy",
            "overlaying": "y",
            "side": "right",
            "range": [0.90, 1.0],
            "tickformat": ".0%",
            "tickfont": {"size": 9, "family": "DM Mono"},
            "title_font": {"size": 10, "color": "#718096"},
        },
        xaxis={"tickfont": {"size": 10, "family": "DM Sans"}},
        font={"family": "DM Sans"},
    )
    return fig


# ── Banner ───────────────────────────────────────────────────────────────────
def render_banner():
    run = st.session_state.get("run_count", 0)
    run_badge = f'<span style="font-size:.72rem;background:rgba(200,68,93,.15);color:#f9a8b8;border-radius:20px;padding:3px 10px;font-family:\'DM Mono\',monospace;margin-left:10px">Run #{run}</span>' if run > 0 else ""

    st.markdown(f"""
<div class="top-banner">
  <div>
    <div class="banner-title">
      <span class="ribbon-anim">🎀</span>
      BreastCare&nbsp;<em style="color:var(--rose-400);font-style:italic">AI</em>
      {run_badge}
    </div>
    <div class="banner-subtitle">
      XGBoost · LightGBM · SHAP · FAISS RAG · Gemini AI &nbsp;|&nbsp;
      Research &amp; Educational Prototype
    </div>
  </div>
  <div class="banner-disclaimer">
    🛡 <strong>Not medical advice.</strong> Educational prototype only.<br/>
    Not FDA-cleared. Always consult a qualified clinician.
  </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> dict | None:
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:16px 0 20px">
  <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:#0e1c2f">
    🩺 Mammographic Features
  </div>
  <div style="font-size:.72rem;color:#718096;margin-top:4px;font-family:'DM Sans',sans-serif">
    UCI Mammographic Masses · LightGBM model
  </div>
</div>
""", unsafe_allow_html=True)

        age = st.slider(
            "Patient Age (years)", 18, 96, 52, 1,
            help="Patient age at time of mammogram (continuous feature)",
        )

        shape_options = list(SHAPE_MAP.values())
        shape_sel = st.selectbox(
            "Mass Shape", shape_options, index=2,
            help="BI-RADS shape descriptor: Round=1, Oval=2, Lobular=3, Irregular=4",
        )
        shape_val = {v: k for k, v in SHAPE_MAP.items()}[shape_sel]

        margin_options = list(MARGIN_MAP.values())
        margin_sel = st.selectbox(
            "Mass Margin", margin_options, index=3,
            help="BI-RADS margin: strongest malignancy predictor. Spiculated = highest risk.",
        )
        margin_val = {v: k for k, v in MARGIN_MAP.items()}[margin_sel]

        density_options = list(DENSITY_MAP.values())
        density_sel = st.selectbox(
            "Mass Density", density_options, index=1,
            help="X-ray attenuation relative to fibroglandular tissue (weaker predictor)",
        )
        density_val = {v: k for k, v in DENSITY_MAP.items()}[density_sel]

        st.markdown("<br/>", unsafe_allow_html=True)

        api_ok = bool(get_api_key())
        if api_ok:
            st.success("✓ Gemini AI connected", icon="🔗")
        else:
            st.warning("Gemini API key not set — ML+SHAP mode", icon="⚠️")

        run_clicked = st.button("🔬  Run Risk Assessment", type="primary")

        st.markdown("""
<div style="margin-top:20px;padding:12px 14px;background:#fff1f4;border:1px solid #ffc2ce;border-radius:10px;font-size:.72rem;color:#72243e;font-family:'DM Sans',sans-serif;line-height:1.6">
  <strong>⚕️ Reminder:</strong> This tool estimates malignancy probability from 4 structured features.
  It does not diagnose. All findings require clinical review.
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div style="margin-top:14px">
  <div style="font-size:.68rem;color:#718096;font-family:'DM Sans',sans-serif;line-height:1.8">
    <strong style="color:#4a5568">Datasets used:</strong><br/>
    • WDBC (569 cases, 30 features)<br/>
    • UCI Mammographic Masses (860 cases, 4 features)<br/>
    • ACR BI-RADS corpus (60 chunks, 4 docs)
  </div>
</div>
""", unsafe_allow_html=True)

        if run_clicked:
            return {
                "age":     age,
                "shape":   shape_val,
                "margin":  margin_val,
                "density": density_val,
            }
        return None


# ── Run pipeline with spinner ─────────────────────────────────────────────────
def run_pipeline_with_ui(features: dict):
    if not _models_ready():
        st.error(
            "⚠️ Models not found. Run these commands first:\n\n"
            "```\npython -m src.ml.train\n"
            "python scripts/build_knowledge_base.py\n```"
        )
        return

    steps = [
        "🩻 Loading features…",
        "🤖 Running LightGBM prediction…",
        "📐 Computing SHAP attributions…",
        "🔍 Building retrieval query from SHAP output…",
        "📄 Searching FAISS knowledge base…",
        "✨ Calling Gemini AI…" if get_api_key() else "✨ Skipping Gemini (no API key)…",
        "🛡 Applying safety filter…",
        "✅ Done!",
    ]

    progress_bar = st.progress(0)
    status_text  = st.empty()

    api_ok = bool(get_api_key())

    try:
        for i, step in enumerate(steps[:-1]):
            status_text.markdown(
                f'<div style="font-size:.82rem;color:#718096;font-family:\'DM Mono\',monospace">{step}</div>',
                unsafe_allow_html=True,
            )
            progress_bar.progress((i + 1) / len(steps))
            if i < 4:
                time.sleep(0.25)   # visual pacing for non-LLM steps

        if api_ok:
            from src.pipeline import run_pipeline
            result = run_pipeline(features)
        else:
            from src.pipeline import run_pipeline_ml_only
            result = run_pipeline_ml_only(features)

        progress_bar.progress(1.0)
        status_text.markdown(
            '<div style="font-size:.82rem;color:#22c55e;font-family:\'DM Mono\',monospace">✅ Pipeline complete</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.5)

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Pipeline error: {e}")
        return

    progress_bar.empty()
    status_text.empty()

    # Store result in session state (the "memory" across reruns)
    st.session_state["pipeline_result"] = result
    st.session_state["last_features"]   = features
    st.session_state["pipeline_ran"]    = True
    st.session_state["run_count"]       += 1

    # Add initial Claude explanation to chat if available
    if result.get("llm_response"):
        lr = result["llm_response"]
        initial_msg = lr.get("main_text", "")
        if initial_msg and (
            not st.session_state["chat_history"]
            or st.session_state["chat_history"][-1].get("content") != initial_msg
        ):
            st.session_state["chat_history"].append({
                "role":    "assistant",
                "content": initial_msg,
                "sources": result.get("sources_used", []),
                "disclaimer": lr.get("disclaimer", ""),
            })


# ── Result panels ─────────────────────────────────────────────────────────────
def render_result_panels():
    result = st.session_state.get("pipeline_result")
    if not result:
        st.markdown("""
<div style="text-align:center;padding:80px 20px;color:#718096;font-family:'DM Sans',sans-serif">
  <div style="font-size:48px;margin-bottom:16px">🩺</div>
  <div style="font-size:1rem;font-weight:500;color:#4a5568">No assessment yet</div>
  <div style="font-size:.82rem;margin-top:8px;line-height:1.6">
    Set mammographic features in the sidebar<br/>and click <strong>Run Risk Assessment</strong>
  </div>
</div>
""", unsafe_allow_html=True)
        return

    pred  = result["prediction"]
    expl  = result["explanation"]
    feats = st.session_state["last_features"]

    prob     = pred["probability"]
    pct      = round(prob * 100, 1)
    risk_cat = pred["risk_category"]
    color    = _risk_color(risk_cat)

    # ── Tabs ─────────────────────────────────────────────────────
    tab_result, tab_explain, tab_data = st.tabs(
        ["📊 Risk Result", "📐 Explainability", "📁 Data & Models"]
    )

    # ======== TAB 1: RISK RESULT ========
    with tab_result:

        # Gauge + metrics side by side
        col_gauge, col_metrics = st.columns([1, 1])

        with col_gauge:
            st.markdown('<div class="bc-card bc-card-accent">', unsafe_allow_html=True)
            st.markdown(f'<div class="bc-card-title">Malignancy Probability</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bc-card-sub">LightGBM · UCI Mammographic Masses · {datetime.now().strftime("%H:%M:%S")}</div>', unsafe_allow_html=True)
            fig_gauge = make_gauge_chart(prob, risk_cat)
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_metrics:
            risk_cls = f"risk-{risk_cat.lower()}"
            st.markdown(f"""
<div class="bc-card" style="height:100%">
  <div class="bc-card-title">Risk Assessment Summary</div>
  <div class="bc-card-sub">Run #{st.session_state['run_count']}</div>

  <div class="risk-badge {risk_cls}" style="margin-bottom:16px;font-size:1rem;padding:8px 18px">
    {_risk_emoji(risk_cat)} {risk_cat} Risk
  </div>

  <div class="metric-row">
    <div class="metric-pill">
      <div class="metric-pill-val" style="color:{color}">{pct}%</div>
      <div class="metric-pill-lab">Malignancy probability</div>
    </div>
    <div class="metric-pill">
      <div class="metric-pill-val">{pred['predicted_class']}</div>
      <div class="metric-pill-lab">Predicted class (0.50 threshold)</div>
    </div>
  </div>

  <div class="metric-row">
    <div class="metric-pill">
      <div class="metric-pill-val" style="font-size:1rem">{pred['confidence_band']}</div>
      <div class="metric-pill-lab">Confidence band (±8 pp estimate)</div>
    </div>
    <div class="metric-pill">
      <div class="metric-pill-val" style="font-size:1rem;text-transform:capitalize">{expl['dominant_feature']}</div>
      <div class="metric-pill-lab">Dominant SHAP feature</div>
    </div>
  </div>

  <div class="disclaimer-notice">
    🔬 This estimate is generated by a research model trained on {860} cases.
    Not FDA-cleared. Always review with a qualified clinician.
  </div>
</div>
""", unsafe_allow_html=True)

        # Interpretation
        margin_label  = MARGIN_MAP.get(feats["margin"], str(feats["margin"]))
        shape_label   = SHAPE_MAP.get(feats["shape"], str(feats["shape"]))
        density_label = DENSITY_MAP.get(feats["density"], str(feats["density"]))
        dominant      = expl["dominant_feature"]
        dom_label     = expl["feature_labels"].get(dominant, "")

        if risk_cat == "Low":
            interp = (
                f"At **{pct}%** estimated probability, this pattern is more consistent with benign findings. "
                f"A **{margin_label.lower()} margin** with **{shape_label.lower()} shape** in a patient aged **{feats['age']}** "
                f"falls in the lower-risk range of this model's experience. Short-interval follow-up imaging "
                f"as recommended by your radiologist remains appropriate."
            )
        elif risk_cat == "Moderate":
            interp = (
                f"At **{pct}%** estimated probability, this pattern warrants careful clinical attention. "
                f"A **{margin_label.lower()} margin** with **{shape_label.lower()} shape** falls in a zone where "
                f"further diagnostic work-up (additional views, ultrasound, or biopsy discussion) is often appropriate. "
                f"Clinical context and radiologist judgment are essential."
            )
        else:
            interp = (
                f"At **{pct}%** estimated probability, this pattern is associated with higher malignancy likelihood. "
                f"A **{margin_label.lower()} margin** is among the strongest BI-RADS malignancy indicators. "
                f"**This does not mean a diagnosis of cancer.** Prompt review by a qualified radiologist is strongly recommended."
            )

        st.markdown(f"""
<div class="bc-card" style="margin-top:8px">
  <div class="bc-card-title">Clinical Interpretation</div>
  <div class="plain-lang">{interp.replace('**', '<strong>').replace('**', '</strong>')}</div>
</div>
""", unsafe_allow_html=True)

    # ======== TAB 2: EXPLAINABILITY ========
    with tab_explain:
        col_shap, col_feat = st.columns([1.2, 1])

        with col_shap:
            st.markdown('<div class="bc-card">', unsafe_allow_html=True)
            st.markdown('<div class="bc-card-title">SHAP Feature Attribution</div>', unsafe_allow_html=True)
            st.markdown('<div class="bc-card-sub">TreeExplainer exact values · Malignant class · LightGBM</div>', unsafe_allow_html=True)
            fig_shap = make_shap_waterfall(expl, feats)
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

            # Plain language
            dom_shap_val  = expl["feature_attributions"][dominant]
            dom_direction = "elevated" if dom_shap_val > 0 else "reduced"
            st.markdown(f"""
<div class="bc-card" style="margin-top:8px">
  <div class="section-tag">Why This Result?</div>
  <div class="plain-lang">
    The strongest driver of this estimate was <strong>{dominant.title()} ({dom_label})</strong>,
    which {dom_direction} the malignancy probability by the largest margin
    (SHAP value: {'+' if dom_shap_val >= 0 else ''}{dom_shap_val:.3f}).<br/><br/>
    {expl['shap_text']}<br/><br/>
    <span style="font-size:.75rem;color:#718096">
    SHAP values are additive: they sum to the difference between this prediction
    and the base rate (E[f(x)] = {round(expl['base_value']*100,1)}%). This makes
    them both locally faithful and globally consistent.
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

        with col_feat:
            st.markdown('<div class="bc-card">', unsafe_allow_html=True)
            st.markdown('<div class="bc-card-title">Input Features</div>', unsafe_allow_html=True)
            st.markdown('<div class="bc-card-sub">Submitted mammographic descriptors</div>', unsafe_allow_html=True)

            feat_display = [
                ("Age",     f"{feats['age']} years",  "Continuous", feats['age'] > 55),
                ("Shape",   shape_label,  "Ordinal (1–4)", feats['shape'] == 4),
                ("Margin",  margin_label, "Ordinal (1–5)", feats['margin'] >= 4),
                ("Density", density_label,"Ordinal (1–4)", False),
            ]
            for fname, fval, ftype, is_risk in feat_display:
                is_dominant = fname.lower() == dominant
                border = "var(--rose-200)" if is_dominant else "var(--border)"
                bg     = "var(--rose-50)"  if is_dominant else "var(--ivory)"
                dom_tag = '<span style="font-size:.65rem;background:var(--rose-500);color:#fff;border-radius:20px;padding:2px 7px;margin-left:6px">★ dominant</span>' if is_dominant else ""
                risk_indicator = "🔺" if is_risk else "  "
                st.markdown(f"""
<div style="background:{bg};border:1.5px solid {border};border-radius:10px;padding:12px 14px;margin-bottom:8px">
  <div style="font-size:.72rem;font-weight:700;color:#718096;text-transform:uppercase;letter-spacing:.06em;font-family:'DM Sans',sans-serif">
    {fname} {dom_tag}
  </div>
  <div style="font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;color:#0e1c2f;margin-top:4px">
    {risk_indicator} {fval}
  </div>
  <div style="font-size:.68rem;color:#718096;font-family:'DM Sans',sans-serif">{ftype}</div>
</div>
""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # RAG pipeline trace
            sources = result.get("sources_used", [])
            query   = result.get("query_used", "")
            st.markdown(f"""
<div class="bc-card" style="margin-top:8px">
  <div class="section-tag">RAG Pipeline Trace</div>
  <div style="font-size:.78rem;color:#4a5568;font-family:'DM Sans',sans-serif;line-height:1.7">
    <div style="margin-bottom:8px">
      <span style="font-family:'DM Mono',monospace;font-size:.68rem;color:#718096">Query built from SHAP output:</span><br/>
      <em style="color:var(--rose-500)">"{query[:90]}..."</em>
    </div>
    <div>
      <span style="font-family:'DM Mono',monospace;font-size:.68rem;color:#718096">Retrieved from sources:</span><br/>
      {' '.join([f'<span class="source-pill">📚 {s}</span>' for s in sources])}
    </div>
    <div style="margin-top:8px;font-size:.72rem;color:#718096">
      {len(result.get('chunks', []))} chunks retrieved · FAISS IndexFlatIP · cosine similarity
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ======== TAB 3: DATA & MODELS ========
    with tab_data:
        meta = _load_model_metadata()

        col_a, col_b = st.columns(2)
        with col_a:
            lgbm_m = meta.get("lgbm_mammographic", {})
            st.markdown(f"""
<div class="bc-card">
  <div class="section-tag">Clinical Tool</div>
  <div class="bc-card-title">LightGBM · Mammographic Masses</div>
  <div class="bc-card-sub">UI-facing model · 4 features · 860 cases</div>
  <div class="metric-row">
    <div class="metric-pill"><div class="metric-pill-val">{lgbm_m.get('auc','0.91')}</div><div class="metric-pill-lab">AUC</div></div>
    <div class="metric-pill"><div class="metric-pill-val">{lgbm_m.get('accuracy','84')}%</div><div class="metric-pill-lab">Accuracy</div></div>
    <div class="metric-pill"><div class="metric-pill-val">{lgbm_m.get('recall','80')}%</div><div class="metric-pill-lab">Recall</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

        with col_b:
            xgb_m = meta.get("xgboost_wdbc", {})
            st.markdown(f"""
<div class="bc-card">
  <div class="section-tag">Research Model</div>
  <div class="bc-card-title">XGBoost · WDBC</div>
  <div class="bc-card-sub">30 FNA nuclear features · 569 cases</div>
  <div class="metric-row">
    <div class="metric-pill"><div class="metric-pill-val">{xgb_m.get('auc','0.99')}</div><div class="metric-pill-lab">AUC</div></div>
    <div class="metric-pill"><div class="metric-pill-val">{xgb_m.get('accuracy','97')}%</div><div class="metric-pill-lab">Accuracy</div></div>
    <div class="metric-pill"><div class="metric-pill-val">{xgb_m.get('false_positives','0')}</div><div class="metric-pill-lab">False Positives</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

        cv = meta.get("cv_comparison", {})
        if cv:
            st.markdown('<div class="bc-card"><div class="bc-card-title">5-Model 10-Fold CV Comparison (WDBC)</div><div class="bc-card-sub">Stratified 10-fold CV on training split (n=455) · Scoring: ROC-AUC</div>', unsafe_allow_html=True)
            fig_cv = make_cv_chart(cv)
            st.plotly_chart(fig_cv, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # Knowledge base summary
        st.markdown("""
<div class="bc-card">
  <div class="section-tag">RAG Knowledge Base</div>
  <div class="bc-card-title">ACR BI-RADS Corpus</div>
  <div class="bc-card-sub">60 curated chunks from 4 authoritative documents · FAISS IndexFlatIP</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px">
    <div class="metric-pill"><div class="metric-pill-val" style="font-size:1rem">📘</div><div class="metric-pill-lab">BI-RADS Terminology (21 chunks)</div></div>
    <div class="metric-pill"><div class="metric-pill-val" style="font-size:1rem">📗</div><div class="metric-pill-lab">Breast Imaging Glossary (18 chunks)</div></div>
    <div class="metric-pill"><div class="metric-pill-val" style="font-size:1rem">📙</div><div class="metric-pill-lab">ACR Follow-Up Guidelines (12 chunks)</div></div>
    <div class="metric-pill"><div class="metric-pill-val" style="font-size:1rem">🛡</div><div class="metric-pill-lab">Safety Disclaimer (9 chunks)</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Chat panel ────────────────────────────────────────────────────────────────
def render_chat():
    st.markdown("""
<div class="bc-card bc-card-accent" style="margin-bottom:12px">
  <div class="bc-card-title">✨ BreastCare AI Assistant</div>
  <div class="bc-card-sub">RAG-grounded · Gemini AI · ACR BI-RADS · Safety filter active</div>
</div>
""", unsafe_allow_html=True)

    # Welcome message on first load
    if not st.session_state["chat_history"]:
        st.markdown("""
<div class="chat-msg-bot">
  <div class="chat-avatar chat-avatar-bot" style="color:#fff">🎀</div>
  <div class="chat-bubble chat-bubble-bot">
    Hello! I'm the BreastCare AI assistant, grounded in curated ACR BI-RADS and mammography guidelines.<br/><br/>
    Run an assessment first, then ask me anything about the result — what the margin means, what typically happens next, or what BI-RADS categories are.
    <div><span class="source-pill">📚 ACR BI-RADS Atlas · Mammographic Literature</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Render chat history from session state
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_history"]:
            if msg["role"] == "assistant":
                sources_html = ""
                if msg.get("sources"):
                    pills = " ".join([f'<span class="source-pill">📚 {s}</span>' for s in msg["sources"] if s])
                    sources_html = f'<div style="margin-top:8px">{pills}</div>'
                disc = ""
                if msg.get("disclaimer"):
                    disc = f'<div style="margin-top:8px;font-size:.7rem;color:#a83050;line-height:1.5">{msg["disclaimer"][:200]}…</div>'
                content = msg["content"].replace("\n", "<br/>")
                st.markdown(f"""
<div class="chat-msg-bot">
  <div class="chat-avatar chat-avatar-bot" style="color:#fff">✨</div>
  <div class="chat-bubble chat-bubble-bot">{content}{sources_html}{disc}</div>
</div>
""", unsafe_allow_html=True)
            else:
                content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(f"""
<div class="chat-msg-user">
  <div class="chat-avatar" style="background:var(--navy);color:#fff">👤</div>
  <div class="chat-bubble chat-bubble-user">{content}</div>
</div>
""", unsafe_allow_html=True)

    # Suggested prompts
    if st.session_state.get("pipeline_ran"):
        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:#718096;font-family:\'DM Sans\',sans-serif;margin-bottom:6px">Suggested questions</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        prompts = [
            "What does spiculated margin mean?",
            "What does this risk estimate mean?",
            "What should happen next clinically?",
            "How accurate is this AI model?",
        ]
        for i, prompt in enumerate(prompts):
            if cols[i % 2].button(f"💬 {prompt}", key=f"chip_{i}"):
                st.session_state["chat_history"].append({"role": "user", "content": prompt})
                _send_chat_message(prompt)
                st.rerun()

    # Chat input
    st.markdown("<br/>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col_inp, col_send = st.columns([5, 1])
        with col_inp:
            user_input = st.text_input(
                "Ask about the result or mammographic terminology…",
                placeholder="e.g. What is spiculated margin? What happens next?",
                label_visibility="collapsed",
            )
        with col_send:
            submitted = st.form_submit_button("Send ↗", type="primary")

    if submitted and user_input.strip():
        st.session_state["chat_history"].append({"role": "user", "content": user_input.strip()})
        _send_chat_message(user_input.strip())
        st.rerun()


def _send_chat_message(question: str):
    """Send a follow-up question through the full pipeline."""
    if not _models_ready():
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": "⚠️ Models not initialised. Run `python -m src.ml.train` and `python scripts/build_knowledge_base.py` first.",
            "sources": [],
        })
        return

    features = st.session_state.get("last_features")
    if not features:
        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": "Please run a risk assessment first so I can answer in context.",
            "sources": [],
        })
        return

    with st.spinner("🔍 Retrieving documents · ✨ Generating response…"):
        try:
            api_ok = bool(get_api_key())
            if api_ok:
                from src.pipeline import run_pipeline
                result = run_pipeline(features, user_question=question)
                lr = result["llm_response"]
                st.session_state["chat_history"].append({
                    "role":       "assistant",
                    "content":    lr.get("main_text", ""),
                    "sources":    result.get("sources_used", []),
                    "disclaimer": lr.get("disclaimer", ""),
                })
            else:
                # No API key — answer from retrieved chunks directly
                from src.pipeline import run_pipeline_ml_only
                result = run_pipeline_ml_only(features)
                chunks = result.get("chunks", [])
                chunk_texts = [c.get("text", "") for c in chunks[:2]]
                reply = (
                    f"(Claude API not connected — showing raw retrieval)\n\n"
                    + "\n\n---\n\n".join(chunk_texts[:2])
                    + "\n\n⚕️ Always consult a qualified clinician."
                )
                st.session_state["chat_history"].append({
                    "role":    "assistant",
                    "content": reply,
                    "sources": result.get("sources_used", []),
                })
        except Exception as e:
            st.session_state["chat_history"].append({
                "role":    "assistant",
                "content": f"Sorry, I encountered an error: {str(e)[:120]}",
                "sources": [],
            })


# ── Architecture panel ────────────────────────────────────────────────────────
def render_architecture():
    st.markdown("""
<div style="background:#0e1c2f;border-radius:20px;padding:32px 36px;margin-top:8px">
  <div style="font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:rgba(200,68,93,.7);font-family:'DM Sans',sans-serif;margin-bottom:8px">System Architecture</div>
  <div style="font-family:'Playfair Display',serif;font-size:1.3rem;font-weight:700;color:#fff;margin-bottom:4px">6-Layer AI Pipeline</div>
  <div style="font-size:.78rem;color:rgba(255,255,255,.4);font-family:'DM Sans',sans-serif;margin-bottom:28px">Single orchestrator (pipeline.py) · SHAP output drives RAG query · Code-level safety filter</div>

  <div style="display:flex;flex-direction:column;gap:12px">
""" + "".join([f"""
    <div style="display:flex;align-items:center;gap:14px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:12px;padding:14px 18px">
      <div style="font-size:24px;flex-shrink:0">{icon}</div>
      <div>
        <div style="font-family:'DM Mono',monospace;font-size:.65rem;color:rgba(200,68,93,.7)">LAYER {i+1:02d}</div>
        <div style="font-size:.88rem;font-weight:600;color:#fff;font-family:'DM Sans',sans-serif">{title}</div>
        <div style="font-size:.75rem;color:rgba(255,255,255,.4);font-family:'DM Sans',sans-serif;margin-top:2px">{desc}</div>
      </div>
      <div style="margin-left:auto;font-family:'DM Mono',monospace;font-size:.68rem;color:rgba(255,255,255,.2)">{tech}</div>
    </div>
""" for i, (icon, title, desc, tech) in enumerate([
    ("🩻","Feature Input","Patient age, mass shape, margin, and density entered via the form","4 mammographic descriptors"),
    ("🤖","Risk Prediction","LightGBM estimates malignancy probability and risk category","Gradient boosted ensemble"),
    ("📐","Explainability","SHAP attributes each feature's contribution to the estimate","Exact, mathematically grounded"),
    ("🔍","Document Retrieval","Top matching ACR BI-RADS guidelines fetched by semantic search","FAISS vector similarity"),
    ("✨","Gemini AI (Google)","Grounded explanation written from retrieved documents only","Free tier · citations required"),
    ("🛡","Safety Filter","All diagnostic language removed before the response reaches you","Deterministic, always active"),
])]) + """
  </div>
</div>
""", unsafe_allow_html=True)


# ── Main layout ───────────────────────────────────────────────────────────────
def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_banner()

    # Sidebar: feature form
    submitted_features = render_sidebar()

    # Run pipeline if button clicked
    if submitted_features:
        run_pipeline_with_ui(submitted_features)
        st.rerun()

    # Main layout: results left+centre, chat right
    col_main, col_chat = st.columns([1.7, 1], gap="medium")

    with col_main:
        render_result_panels()
        st.markdown("<br/>", unsafe_allow_html=True)
        render_architecture()

    with col_chat:
        render_chat()

    # Footer
    st.markdown("""
<div style="text-align:center;padding:40px 20px 20px;border-top:1px solid #e8eff7;margin-top:40px">
  <div style="font-family:'Playfair Display',serif;font-size:1.1rem;color:#0e1c2f;margin-bottom:6px">BreastCare <em style="color:#c9445d">AI</em></div>
  <div style="font-size:.72rem;color:#718096;font-family:'DM Sans',sans-serif;line-height:1.8">
    XGBoost · LightGBM · SHAP · FAISS · Gemini AI · Streamlit<br/>
    <strong>Not for clinical use.</strong> Research and educational prototype only. Not FDA-cleared.<br/>
    Elter (2007) · Street et al. (1993) · Ghasemi et al. (2024) · Kim et al. (2022) Lancet Digital Health
  </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
