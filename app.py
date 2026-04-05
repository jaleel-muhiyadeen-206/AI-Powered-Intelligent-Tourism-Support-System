"""
=============================================================================
  Toura.lk — AI-Powered Intelligent Tourism Support System | Group 17
  Dark Emerald Theme | Auto-generates documentary on landmark detection
=============================================================================
  Layout (app.py sits at root, all other modules inside digital_story_teller/):

    project_root/
    ├── app.py                        ← this file
    ├── classifier.py                 ← landmark CNN
    ├── weather_app.py                ← LightGBM weather
    ├── recommendationapp.py          ← recommendation models
    └── digital_story_teller/
        ├── digital_storyteller_pipeline.py
        ├── sri_lanka_landmarks_final.csv
        ├── outputs/
        │   └── final_videos/
        └── reference_images/
=============================================================================
"""
import os, time, base64, datetime, importlib.util
from difflib import SequenceMatcher

import streamlit as st
import pandas as pd
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
DST_DIR     = os.path.join(ROOT_DIR, "digital_story_teller")   # sub-folder
FINAL_DIR   = os.path.join(DST_DIR, "outputs", "final_videos")

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE IMPORTS  — graceful fallback if any module is missing
# ─────────────────────────────────────────────────────────────────────────────
try:
    from classifier import init_classifier, get_prediction
except ImportError:
    init_classifier = get_prediction = None

try:
    import weather_app as wa
except Exception:
    wa = None

try:
    import recommendation_app as ra
except Exception:
    ra = None

def _load_pipeline():
    path = os.path.join(DST_DIR, "digital_storyteller_pipeline.py")
    if not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("pipeline", path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None

PIPELINE = _load_pipeline()

def safe_name(s: str) -> str:
    return s.replace(" ","_").replace("(","").replace(")","").replace("'","")

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Toura.lk — Explore Sri Lanka",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  DARK EMERALD THEME CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── ROOT TOKENS ── */
:root {
    --bg-base:        #060d0a;
    --bg-surface:     #0c1610;
    --bg-card:        #101d15;
    --bg-card-hover:  #152319;
    --bg-elevated:    #1a2e1f;
    --border-subtle:  rgba(16, 185, 129, 0.08);
    --border-medium:  rgba(16, 185, 129, 0.18);
    --border-strong:  rgba(16, 185, 129, 0.35);
    --text-primary:   #e8f5ee;
    --text-secondary: #7aad8a;
    --text-muted:     #4a7058;
    --accent:         #10b981;
    --accent-light:   #34d399;
    --accent-dark:    #059669;
    --accent-glow:    rgba(16, 185, 129, 0.25);
    --gold:           #f59e0b;
    --gold-glow:      rgba(245, 158, 11, 0.2);
    --red:            #f87171;
    --gradient-main:  linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
    --gradient-hero:  linear-gradient(135deg, #064e3b 0%, #065f46 40%, #047857 100%);
    --gradient-gold:  linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
    --radius-sm:  10px;
    --radius-md:  14px;
    --radius-lg:  20px;
    --radius-xl:  28px;
    --shadow-sm:  0 2px 12px rgba(0,0,0,0.4);
    --shadow-md:  0 6px 28px rgba(0,0,0,0.5);
    --shadow-lg:  0 16px 56px rgba(0,0,0,0.6);
    --shadow-glow: 0 0 40px var(--accent-glow);
    --transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"],
[data-testid="stApp"], .stApp {
    background: var(--bg-base) !important;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(5,150,105,0.07) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 90%, rgba(4,120,87,0.05) 0%, transparent 50%) !important;
    background-attachment: fixed !important;
}
[data-testid="stHeader"], header { visibility: hidden !important; }
footer, #MainMenu { visibility: hidden !important; }
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1280px !important;
}
section[data-testid="stSidebar"] { display: none !important; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-dark); }

/* ── ANIMATIONS ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes floatY {
    0%,100% { transform: translateY(0); }
    50%      { transform: translateY(-8px); }
}
@keyframes pulseGlow {
    0%,100% { box-shadow: 0 0 0 0 var(--accent-glow); }
    50%      { box-shadow: 0 0 20px 6px var(--accent-glow); }
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes gradShift {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
@keyframes blink {
    0%,100% { opacity: 1; } 50% { opacity: 0.3; }
}
@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position:  200% 0; }
}
.anim-up   { animation: fadeInUp 0.6s ease-out forwards; }
.anim-fade { animation: fadeIn 0.8s ease-out forwards; }

/* ── NAVBAR ── */
.navbar {
    background: rgba(6, 13, 10, 0.92);
    backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--border-subtle);
    padding: 0.9rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    margin: 0 -1rem 0 -1rem;
    position: sticky; top: 0; z-index: 999;
}
.nav-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem; font-weight: 800;
    display: flex; align-items: center; gap: 10px;
    color: var(--text-primary);
}
.nav-logo-icon {
    width: 36px; height: 36px;
    background: var(--gradient-main);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; font-weight: 900; color: white;
    box-shadow: 0 4px 14px rgba(16,185,129,0.35);
}
.nav-dot { color: var(--accent-light); }
.nav-links { display: flex; gap: 2.2rem; }
.nav-links a {
    color: var(--text-secondary); text-decoration: none;
    font-size: 0.88rem; font-weight: 500; transition: var(--transition);
    position: relative;
}
.nav-links a::after {
    content: ''; position: absolute; bottom: -4px; left: 0;
    width: 0; height: 2px;
    background: var(--gradient-main); border-radius: 1px;
    transition: width 0.3s ease;
}
.nav-links a:hover { color: var(--accent-light); }
.nav-links a:hover::after { width: 100%; }
.nav-badge {
    padding: 4px 12px; background: rgba(16,185,129,0.12);
    border: 1px solid var(--border-medium); border-radius: 100px;
    font-size: 0.75rem; font-weight: 600; color: var(--accent-light);
}

/* ── HERO ── */
.hero-wrap {
    text-align: center;
    padding: 5.5rem 2rem 4rem;
    animation: fadeInUp 0.8s ease-out;
    position: relative;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: 0; left: 50%; transform: translateX(-50%);
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(16,185,129,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 0.5rem 1.4rem;
    background: rgba(16,185,129,0.08);
    border: 1px solid var(--border-medium); border-radius: 100px;
    font-size: 0.78rem; font-weight: 700; color: var(--accent-light);
    letter-spacing: 0.06em; text-transform: uppercase;
    margin-bottom: 1.6rem;
    animation: pulseGlow 4s ease-in-out infinite;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 3.8rem; font-weight: 800;
    line-height: 1.1; letter-spacing: -1.5px;
    color: var(--text-primary);
    margin-bottom: 1.2rem;
}
.gradient-text {
    background: var(--gradient-main);
    background-size: 200% 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradShift 5s ease-in-out infinite;
}
.hero-sub {
    display: inline-flex; align-items: center;
    font-size: 1.1rem; color: var(--text-secondary);
    max-width: 560px; margin: 0 auto 2.5rem; line-height: 1.75;
}

/* ── GLASS CARD ── */
.glass-card {
    background: rgba(16, 29, 21, 0.7);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 2rem;
    box-shadow: var(--shadow-md);
    transition: var(--transition);
}
.glass-card:hover {
    border-color: var(--border-medium);
    box-shadow: var(--shadow-glow);
    transform: translateY(-3px);
}

/* ── UPLOAD ZONE ── */
div[data-testid="stFileUploader"] {
    background: rgba(16,29,21,0.6) !important;
    border: 2px dashed var(--border-medium) !important;
    border-radius: var(--radius-lg) !important;
    transition: var(--transition) !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
    background: rgba(16,185,129,0.04) !important;
    box-shadow: 0 0 0 4px rgba(16,185,129,0.06) !important;
}
div[data-testid="stFileUploader"] * { color: var(--text-secondary) !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: var(--gradient-main) !important;
    color: white !important; border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important; font-size: 0.92rem !important;
    letter-spacing: 0.2px !important;
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
    box-shadow: 0 4px 18px rgba(16,185,129,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 32px rgba(16,185,129,0.45) !important;
}
.stButton > button:active { transform: translateY(-1px) scale(0.99) !important; }
.stButton > button:focus { outline: none !important; box-shadow: 0 0 0 3px rgba(16,185,129,0.25) !important; }

/* ── SELECTBOX ── */
div[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    transition: var(--transition) !important;
}
div[data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.1) !important;
}
div[data-testid="stSelectbox"] label { color: var(--text-secondary) !important; font-size: 0.85rem !important; }

/* ── DATE INPUT ── */
[data-testid="stDateInput"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
[data-testid="stDateInput"] > div > div:hover {
    border-color: var(--accent) !important;
}
[data-testid="stDateInput"] input {
    background: transparent !important;
    color: var(--text-primary) !important;
}
[data-testid="stDateInput"] input::-webkit-calendar-picker-indicator {
    filter: invert(0.6) sepia(1) hue-rotate(120deg) saturate(3) !important;
    cursor: pointer;
}
[data-testid="stDateInput"] label { color: var(--text-secondary) !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem; background: transparent;
    border-bottom: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: 1px solid transparent; border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    padding: 10px 20px; color: var(--text-muted);
    font-weight: 500; transition: var(--transition);
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(16,185,129,0.06); color: var(--accent-light);
}
.stTabs [aria-selected="true"] {
    background: rgba(16,185,129,0.1) !important;
    border-color: var(--border-medium) !important;
    border-bottom-color: transparent !important;
    color: var(--accent-light) !important;
    font-weight: 700 !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── VIDEO ── */
[data-testid="stVideo"] {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-lg) !important;
    border: 1px solid var(--border-subtle) !important;
}
[data-testid="stVideo"] video {
    border-radius: var(--radius-lg) !important;
}

/* ── SECTION HEADER ── */
.sec-head { margin-bottom: 1.5rem; animation: fadeInUp 0.5s ease-out; }
.sec-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.75rem; font-weight: 700; color: var(--text-primary);
    display: flex; align-items: center; gap: 12px; margin-bottom: 0.4rem;
}
.sec-icon {
    width: 44px; height: 44px;
    background: rgba(16,185,129,0.1);
    border: 1px solid var(--border-medium);
    border-radius: 13px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 1.25rem;
    box-shadow: inset 0 0 12px rgba(16,185,129,0.06);
}
.sec-sub { color: var(--text-muted); font-size: 0.9rem; line-height: 1.6; }

/* ── PLACE HEADER ── */
.place-header {
    position: relative; border-radius: var(--radius-xl);
    overflow: hidden; margin-bottom: 2.5rem;
    min-height: 300px;
    background: var(--gradient-hero);
    animation: fadeIn 0.8s ease-out;
    border: 1px solid var(--border-subtle);
}
.place-header-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(180deg,
        rgba(6,13,10,0.2) 0%,
        rgba(6,13,10,0.75) 100%);
}
.place-header-content {
    position: relative; z-index: 2;
    padding: 3rem 2.5rem;
    color: white;
    display: flex; flex-direction: column;
    justify-content: flex-end; min-height: 300px;
}
.place-name {
    font-family: 'Outfit', sans-serif;
    font-size: 3rem; font-weight: 800;
    line-height: 1.1; margin-bottom: 0.5rem;
    text-shadow: 0 2px 16px rgba(0,0,0,0.5);
}
.place-location {
    font-size: 0.95rem; opacity: 0.8;
    display: flex; align-items: center; gap: 6px;
    margin-bottom: 0.8rem;
}
.place-desc {
    font-size: 0.9rem; opacity: 0.75;
    line-height: 1.7; max-width: 700px;
}

/* ── VIDEO PLACEHOLDER ── */
.vid-placeholder {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-elevated));
    border: 1px dashed var(--border-medium);
    border-radius: var(--radius-lg);
    height: 320px;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; gap: 12px;
    transition: var(--transition);
}
.vid-placeholder:hover { border-color: var(--border-strong); }
.vid-placeholder-icon {
    font-size: 3rem;
    animation: floatY 3s ease-in-out infinite;
}

/* ── VIDEO INFO SIDEBAR ── */
.vid-info-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem; height: 100%;
}
.vid-info-card h4 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700; color: var(--text-primary);
    margin: 0 0 0.8rem;
}
.vid-info-card p {
    color: var(--text-muted); font-size: 0.83rem; line-height: 1.7;
}
.vid-feature {
    font-size: 0.78rem; color: var(--accent-light);
    font-weight: 600; line-height: 2;
}
.vid-status {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 14px;
    border-radius: var(--radius-sm);
    font-size: 0.82rem; font-weight: 600;
    margin-top: 1rem;
}
.vid-status-ready {
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.2);
    color: var(--accent-light);
}
.vid-status-pending {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    color: #f59e0b;
}

/* ── WEATHER ── */
.wx-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 2rem;
}
.wx-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin: 1.5rem 0;
}
.wx-stat {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 1.3rem; text-align: center;
    transition: var(--transition);
    animation: fadeInUp 0.5s ease-out forwards;
}
.wx-stat:hover {
    border-color: var(--border-medium);
    transform: translateY(-4px);
    box-shadow: var(--shadow-glow);
}
.wx-icon { font-size: 1.9rem; margin-bottom: 0.5rem; }
.wx-val {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem; font-weight: 700;
    color: var(--text-primary); line-height: 1.2;
}
.wx-label {
    font-size: 0.75rem; color: var(--text-muted);
    font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; margin-top: 0.3rem;
}
.score-max { font-size: 0.9rem; color: var(--text-muted); font-weight: 400; }
.wx-verdict {
    display: flex; align-items: center; gap: 14px;
    padding: 1rem 1.4rem; border-radius: var(--radius-md);
    font-size: 0.9rem; font-weight: 500;
    animation: fadeIn 0.6s ease-out;
}
.verdict-good {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.2); color: var(--accent-light);
}
.verdict-warn {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2); color: #f59e0b;
}
.verdict-bad {
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.2); color: #f87171;
}

/* ── UBER CARD ── */
.uber-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 1.5rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 1.5rem; transition: var(--transition);
}
.uber-card:hover { border-color: var(--border-medium); transform: translateY(-2px); }
.uber-inner { display: flex; align-items: center; gap: 16px; }
.uber-ico {
    width: 52px; height: 52px;
    background: var(--gradient-main); border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 4px 14px rgba(16,185,129,0.3);
}
.uber-txt h4 {
    margin: 0 0 3px; font-family: 'Outfit', sans-serif;
    font-weight: 700; color: var(--text-primary);
}
.uber-txt p { margin: 0; font-size: 0.83rem; color: var(--text-muted); }
.uber-btn-link {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--text-primary); color: var(--bg-base) !important;
    text-decoration: none !important;
    padding: 0.65rem 1.4rem; border-radius: var(--radius-sm);
    font-weight: 700; font-size: 0.88rem; transition: var(--transition);
}
.uber-btn-link:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(232,245,238,0.2);
}

/* ── REC CARDS ── */
.rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden; transition: var(--transition);
    height: 100%; display: flex; flex-direction: column;
}
.rec-card:hover {
    transform: translateY(-6px);
    border-color: var(--border-strong);
    box-shadow: var(--shadow-glow);
}
.rec-img-wrap { position: relative; height: 165px; overflow: hidden; background: var(--bg-elevated); }
.rec-img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.5s ease; }
.rec-card:hover .rec-img { transform: scale(1.08); }
.rec-badge {
    position: absolute; top: 10px; left: 10px;
    background: var(--gradient-main); color: white;
    font-size: 0.68rem; font-weight: 800;
    padding: 4px 10px; border-radius: 20px;
}
.rec-dist {
    position: absolute; top: 10px; right: 10px;
    background: rgba(6,13,10,0.7); backdrop-filter: blur(8px);
    color: var(--accent-light); font-size: 0.68rem; font-weight: 600;
    padding: 4px 10px; border-radius: 20px;
}
.rec-body { padding: 1rem; flex: 1; display: flex; flex-direction: column; }
.rec-name {
    font-family: 'Outfit', sans-serif; font-size: 0.95rem; font-weight: 700;
    color: var(--text-primary); margin-bottom: 0.3rem; line-height: 1.3;
}
.rec-meta {
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.78rem; margin-bottom: 0.5rem;
}
.rec-rating { color: var(--gold); font-weight: 700; }
.rec-type { color: var(--accent); font-weight: 500; font-size: 0.73rem; }
.rec-review {
    font-size: 0.78rem; color: var(--text-muted); line-height: 1.5;
    font-style: italic; margin-top: auto;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}

/* ── SERVICE CARDS ── */
.svc-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    overflow: hidden; display: flex;
    transition: var(--transition); margin-bottom: 0.75rem;
}
.svc-card:hover {
    transform: translateX(6px);
    border-color: var(--border-strong);
    box-shadow: var(--shadow-md);
}
.svc-img {
    width: 115px; min-height: 105px;
    object-fit: cover; flex-shrink: 0;
}
.svc-img-ph {
    width: 115px; min-height: 105px;
    background: var(--bg-elevated);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
}
.svc-body { padding: 0.85rem 1rem; flex: 1; }
.svc-name {
    font-family: 'Outfit', sans-serif; font-weight: 600;
    color: var(--text-primary); margin-bottom: 0.3rem;
}
.svc-meta { display: flex; gap: 1rem; font-size: 0.78rem; margin-bottom: 0.2rem; }
.svc-rating { color: var(--gold); font-weight: 700; }
.svc-dist { color: var(--text-muted); }
.svc-review {
    font-size: 0.76rem; color: var(--text-muted); font-style: italic;
    line-height: 1.45;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}

/* ── STATUS BADGES ── */
.badge-detected {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px;
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.22);
    border-radius: var(--radius-sm); margin: 8px 0;
}
.badge-warn {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: var(--radius-sm); margin: 8px 0;
}
.badge-err {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px;
    background: rgba(248,113,113,0.08);
    border: 1px solid rgba(248,113,113,0.2);
    border-radius: var(--radius-sm); margin: 8px 0;
}

/* ── ANALYZING OVERLAY ── */
.overlay {
    position: fixed; inset: 0;
    background: rgba(6,13,10,0.97);
    backdrop-filter: blur(28px);
    z-index: 9999;
    display: flex; align-items: center; justify-content: center;
}
.overlay-inner { text-align: center; }
.spinner-wrap {
    position: relative; width: 84px; height: 84px;
    margin: 0 auto 2rem;
}
.spin-ring {
    position: absolute; inset: 0; border-radius: 50%;
    border: 3px solid transparent;
}
.spin-ring:nth-child(1) { border-top-color: var(--accent); animation: spin 1.2s linear infinite; }
.spin-ring:nth-child(2) { inset: 8px; border-right-color: var(--accent-light); animation: spin 1.7s linear infinite reverse; }
.spin-ring:nth-child(3) { inset: 16px; border-bottom-color: var(--gold); animation: spin 2.2s linear infinite; }
.overlay-title {
    font-family: 'Outfit', sans-serif; font-size: 1.65rem; font-weight: 700;
    color: var(--text-primary); margin-bottom: 0.5rem;
}
.overlay-step { color: var(--accent-light); font-size: 0.9rem; margin-bottom: 1.5rem; }
.progress-outer {
    width: 260px; height: 3px;
    background: var(--bg-elevated); border-radius: 2px;
    overflow: hidden; margin: 0 auto;
}
.progress-bar {
    height: 100%;
    background: var(--gradient-main);
    border-radius: 2px; transition: width 0.4s ease;
}

/* ── FOOTER ── */
.site-footer {
    text-align: center; padding: 3rem 2rem;
    margin-top: 4rem; border-top: 1px solid var(--border-subtle);
    color: var(--text-muted); font-size: 0.83rem;
}
.footer-logo {
    font-family: 'Outfit', sans-serif; font-size: 1.4rem; font-weight: 800;
    color: var(--text-primary); margin-bottom: 0.5rem;
}
.footer-logo span { color: var(--accent-light); }
.footer-links { display: flex; gap: 2rem; justify-content: center; margin-top: 0.8rem; }
.footer-links a { color: var(--text-muted); text-decoration: none; transition: color 0.2s; }
.footer-links a:hover { color: var(--accent-light); }

/* ── MISC ── */
[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
div[data-testid="column"] { min-height: 1px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CACHED MODEL INIT
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🌿 Loading AI models…")
def initialize_models():
    if init_classifier:
        init_classifier()
    wx_df = wx_models = wx_feats = rec_models = None
    if wa:
        wx_df = wa.load_data()
        wx_models, wx_feats = wa.load_models()
    if ra:
        rec_models = ra.load_models()
    return wx_df, wx_models, wx_feats, rec_models

wx_df, wx_models, wx_feats, rec_models = initialize_models()

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def fuzzy_match(name, candidates):
    best_r, best = 0, name
    for c in candidates:
        r = SequenceMatcher(None, name.lower(), str(c).lower()).ratio()
        if r > best_r:
            best_r, best = r, c
    return best if best_r > 0.45 else None

def video_exists(landmark: str) -> str | None:
    p = os.path.join(FINAL_DIR, f"{safe_name(landmark)}_Documentary.mp4")
    return p if os.path.exists(p) else None

def generate_video(landmark: str) -> str | None:
    """Run the pipeline. Returns path or None."""
    if not PIPELINE:
        return None
    try:
        result = PIPELINE.run_pipeline(landmark)
        if result and os.path.exists(result):
            return result
    except Exception as e:
        st.session_state["pipeline_error"] = str(e)
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_navbar():
    st.markdown("""
<div class="navbar">
  <div class="nav-logo">
    <div class="nav-logo-icon">◈</div>
    Toura<span class="nav-dot">.lk</span>
  </div>
  <div class="nav-links">
    <a href="#">Home</a>
    <a href="#">Explore</a>
    <a href="#">About</a>
    <div class="nav-badge">🌿 AI Powered</div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYZING OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def render_analyzing(landmark: str):
    """Show animated overlay, run pipeline, then redirect to results."""
    steps = [
        ("Running CNN landmark recognition",      0.5),
        ("Loading landmark dataset & facts",      0.5),
        ("Generating cinematic scene scripts",     0.6),
        ("Generating video clips via Veo 3.1",    0.5),
        ("Embedding native voiceover audio",       0.5),
        ("Assembling final documentary",           0.4),
    ]
    ph = st.empty()
    for idx, (txt, delay) in enumerate(steps):
        pct = int((idx + 1) / len(steps) * 100)
        with ph.container():
            st.markdown(f"""
<div class="overlay">
  <div class="overlay-inner">
    <div class="spinner-wrap">
      <div class="spin-ring"></div>
      <div class="spin-ring"></div>
      <div class="spin-ring"></div>
    </div>
    <div class="overlay-title">Creating your documentary…</div>
    <div class="overlay-step">{txt}</div>
    <div class="progress-outer">
      <div class="progress-bar" style="width:{pct}%;"></div>
    </div>
    <p style="color:var(--text-muted);font-size:0.78rem;margin-top:1rem;">
      {landmark}
    </p>
  </div>
</div>""", unsafe_allow_html=True)
        time.sleep(delay)

    # ── ACTUAL PIPELINE CALL ──────────────────────────────────────────────
    vid = generate_video(landmark)
    st.session_state.video_path = vid
    st.session_state.page = "results"
    ph.empty()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — HOME / UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
def render_home():
    st.markdown("""
<div class="hero-wrap">
  <div class="hero-badge">🌿 AI-Powered Tourism Intelligence</div>
  <h1 class="hero-title">
    Explore Sri Lanka<br>
    <span class="gradient-text">Through Your Lens</span>
  </h1>
  <p class="hero-sub">
    Upload a landmark photo — our CNN identifies it, then our AI pipeline
    generates a cinematic documentary, predicts weather, and curates
    personalised recommendations, all in one seamless experience.
  </p>
</div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        # Upload card
        st.markdown("""
<div class="glass-card" style="text-align:center;margin-bottom:1.5rem;">
  <div style="font-size:2.8rem;animation:floatY 3s ease-in-out infinite;margin-bottom:0.5rem;">📸</div>
  <h3 style="font-family:'Outfit';font-weight:700;color:var(--text-primary);margin-bottom:0.3rem;">
    Upload a Landmark Photo
  </h3>
  <p style="color:var(--text-muted);font-size:0.85rem;">
    Supports JPG · PNG · WebP — our CNN classifier will identify it instantly
  </p>
</div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload landmark image",
            type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed",
        )

        ai_detected = None
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)

            if get_prediction:
                with st.spinner("🔍 Identifying landmark with CNN…"):
                    result = get_prediction(image)

                if result and result.get("name"):
                    raw_name = result["name"].strip()
                    # Try to match against recommendation dataset first
                    matched = None
                    if rec_models:
                        matched = fuzzy_match(raw_name, rec_models[0]["place_name"].tolist())
                    if not matched:
                        matched = raw_name
                    ai_detected = matched
                    st.markdown(f"""
<div class="badge-detected">
  <span style="font-size:1.2rem;">✅</span>
  <div>
    <span style="color:var(--accent-light);font-weight:700;font-size:0.82rem;">
      Landmark identified
    </span><br>
    <span style="color:var(--text-primary);font-size:0.97rem;font-weight:700;">
      {matched}
    </span>
    {f'<span style="color:var(--text-muted);font-size:0.78rem;"> · {result.get("place","")}</span>' if result.get("place") else ""}
  </div>
</div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
<div class="badge-err">
  <span style="font-size:1.2rem;">❌</span>
  <div>
    <span style="color:#f87171;font-weight:700;font-size:0.82rem;">Could not identify landmark</span><br>
    <span style="color:var(--text-muted);font-size:0.8rem;">Please select manually below</span>
  </div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div class="badge-warn">
  <span>⚠️</span>
  <span style="color:#f59e0b;font-size:0.82rem;font-weight:600;">
    classifier.py not found — select landmark manually
  </span>
</div>""", unsafe_allow_html=True)

        # Separator
        st.markdown("""
<div style="text-align:center;color:var(--text-muted);margin:1.2rem 0 0.6rem;font-size:0.87rem;">
  — or select a destination manually —
</div>""", unsafe_allow_html=True)

        # ── Selectbox populated from the PIPELINE CSV (these are the valid names) ──
        # Using CSV names ensures what the user picks always matches the pipeline.
        csv_path = os.path.join(DST_DIR, "sri_lanka_landmarks_final.csv")
        all_places = ["Sigiriya"]
        if os.path.exists(csv_path):
            try:
                _csv_df = pd.read_csv(csv_path)
                all_places = sorted(_csv_df["Landmark"].tolist())
            except Exception:
                if rec_models:
                    all_places = sorted(rec_models[0]["place_name"].tolist())
        elif rec_models:
            all_places = sorted(rec_models[0]["place_name"].tolist())

        # ── Fuzzy-match classifier output against the CSV names ──────────────
        # This maps e.g. "Sripada (Adam's Peak)" -> "Adam's Peak (Sri Pada)"
        default_idx = 0
        if ai_detected:
            best_csv_match = fuzzy_match(ai_detected, all_places)
            if best_csv_match and best_csv_match in all_places:
                default_idx = all_places.index(best_csv_match)
            elif "Sigiriya" in all_places:
                default_idx = all_places.index("Sigiriya")
        elif "Sigiriya" in all_places:
            default_idx = all_places.index("Sigiriya")

        manual = st.selectbox(
            "Select landmark",
            all_places,
            index=default_idx,
            label_visibility="collapsed",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🎬  Generate Documentary", use_container_width=True):
                # Always use `manual` (the selectbox) — it is a CSV-valid name.
                # ai_detected may differ from both CSV and rec_models names.
                st.session_state.landmark   = manual
                st.session_state.video_path = None
                st.session_state.page       = "analyzing"
                st.rerun()
        with c2:
            if st.button("✨  Sigiriya Demo", use_container_width=True):
                st.session_state.landmark   = "Sigiriya"
                st.session_state.video_path = None
                st.session_state.page       = "analyzing"
                st.rerun()

        st.markdown("""
<p style="text-align:center;color:var(--text-muted);font-size:0.8rem;margin-top:1rem;">
  ✨ Click <strong style="color:var(--accent-light);">Sigiriya Demo</strong> for a quick preview
</p>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — RESULTS DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def render_results():
    landmark   = st.session_state.get("landmark", "Sigiriya")
    video_path = st.session_state.get("video_path")

    # Show pipeline error if any
    if "pipeline_error" in st.session_state:
        st.warning(f"Pipeline note: {st.session_state.pop('pipeline_error')}")

    # Back button
    if st.button("← Back to Home", key="back_btn"):
        st.session_state.page = "home"
        st.session_state.video_path = None
        st.rerun()

    # ── PLACE HEADER ─────────────────────────────────────────────────────
    desc = f"Discover the captivating heritage and natural beauty of {landmark}, one of Sri Lanka's most iconic destinations."
    # Fuzzy-match landmark (CSV name) to rec_models place names for description
    rec_landmark = landmark
    if rec_models:
        _rec_match = fuzzy_match(landmark, rec_models[0]["place_name"].tolist())
        if _rec_match:
            rec_landmark = _rec_match
        match = rec_models[0][rec_models[0]["place_name"] == rec_landmark]
        if not match.empty:
            row = match.iloc[0]
            if "description" in row and pd.notna(row.get("description", "")):
                desc = str(row["description"])

    st.markdown(f"""
<div class="place-header">
  <div class="place-header-overlay"></div>
  <div class="place-header-content">
    <div style="display:inline-flex;align-items:center;gap:8px;padding:5px 14px;
         background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);
         border-radius:100px;font-size:0.72rem;font-weight:700;
         color:var(--accent-light);letter-spacing:0.06em;
         text-transform:uppercase;margin-bottom:1rem;">
      ✅ Landmark Identified
    </div>
    <div class="place-name">{landmark}</div>
    <div class="place-location">📍 Sri Lanka</div>
    <div class="place-desc">{desc}</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── SECTION 1: DOCUMENTARY VIDEO ─────────────────────────────────────
    st.markdown("""
<div class="sec-head">
  <div class="sec-title">
    <span class="sec-icon">🎬</span>
    Immersive Documentary
  </div>
  <div class="sec-sub">
    AI-generated cinematic experience powered by our Digital Storyteller Engine —
    Llama 3.1 scripting · Veo 3.1 visuals · native narration audio
  </div>
</div>""", unsafe_allow_html=True)

    vc1, vc2 = st.columns([3, 1])
    with vc1:
        # Check if video already exists on disk (cache from previous run)
        if not video_path:
            video_path = video_exists(landmark)
            st.session_state.video_path = video_path

        if video_path and os.path.exists(video_path):
            st.video(video_path)
        else:
            st.markdown(f"""
<div class="vid-placeholder">
  <div class="vid-placeholder-icon">🎥</div>
  <h3 style="color:var(--text-primary);margin:0 0 0.4rem;">Documentary not generated yet</h3>
  <p style="color:var(--text-muted);font-size:0.87rem;margin:0;">
    Click <strong style="color:var(--accent-light);">Generate Documentary</strong>
    to create one using our AI pipeline.
  </p>
</div>""", unsafe_allow_html=True)

    with vc2:
        st.markdown(f"""
<div class="vid-info-card">
  <h4>Digital Storyteller Engine</h4>
  <p>
    Our pipeline uses Llama 3.1 to craft a 7-scene script, then
    Google Veo 3.1 to render each 8-second cinematic clip with
    native voiceover audio embedded directly.
  </p>
  <hr style="border-color:var(--border-subtle);margin:1rem 0;">
  <div class="vid-feature">
    ✓ Context &amp; scene extraction<br>
    ✓ Cinematic Veo 3.1 generation<br>
    ✓ Native narration audio<br>
    ✓ Final documentary assembly
  </div>
  <div class="{'vid-status vid-status-ready' if (video_path and os.path.exists(str(video_path))) else 'vid-status vid-status-pending'}">
    {'✅ &nbsp; Documentary ready' if (video_path and os.path.exists(str(video_path))) else '⏳ &nbsp; Not generated yet'}
  </div>
</div>""", unsafe_allow_html=True)

        if st.button("🎬 Generate / Re-generate", use_container_width=True, key="gen_btn"):
            st.session_state.video_path = None
            st.session_state.page = "analyzing"
            st.rerun()

    # ── SECTION 2: WEATHER ────────────────────────────────────────────────
    st.markdown("""
<div style="margin-top:3rem;" class="sec-head">
  <div class="sec-title">
    <span class="sec-icon">🌤️</span>
    Smart Weather Forecast
  </div>
  <div class="sec-sub">
    Plan your visit with our LightGBM weather prediction model,
    trained on years of local Sri Lankan meteorological data.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="wx-card">', unsafe_allow_html=True)
    wc1, wc2, wc3 = st.columns([2, 1, 1])
    with wc1:
        visit_date = st.date_input(
            "Select visit date",
            value=datetime.date.today() + datetime.timedelta(days=1),
            min_value=datetime.date.today(), key="wx_date",
        )
    with wc3:
        st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Weather", use_container_width=True, key="wx_btn")

    if predict_btn:
        if wa and wx_df is not None:
            station = None
            loc_info = wa.get_location_info(wx_df)
            station  = fuzzy_match(landmark, loc_info["name"].tolist())
            if station:
                with st.spinner("Predicting weather…"):
                    pred = wa.predict_weather(station, visit_date, wx_df, wx_models, wx_feats)
                if pred:
                    temp  = pred["temperature_2m_mean"]
                    rain  = pred["precipitation_sum"]
                    wind  = pred["windspeed_10m_max"]
                    sc    = wa.score_weather(temp, rain, wind)
                    stype, sicon, stext = wa.get_suggestion(sc)
                    rain_pct  = min(99, round(rain * 3))
                    sc_color  = "#10b981" if sc >= 70 else ("#f59e0b" if sc >= 50 else "#f87171")
                    v_cls     = "verdict-good" if stype=="good" else ("verdict-warn" if stype=="warn" else "verdict-bad")
                    v_icon    = "✅" if stype=="good" else ("⚠️" if stype=="warn" else "🌧️")
                    st.markdown(f"""
<div class="wx-grid">
  <div class="wx-stat stagger-1"><div class="wx-icon">🌡️</div>
    <div class="wx-val">{round(temp)}°C</div><div class="wx-label">Temperature</div></div>
  <div class="wx-stat stagger-2"><div class="wx-icon">💧</div>
    <div class="wx-val">{rain_pct}%</div><div class="wx-label">Precipitation</div></div>
  <div class="wx-stat stagger-3"><div class="wx-icon">🌬️</div>
    <div class="wx-val">{round(wind)} km/h</div><div class="wx-label">Wind Speed</div></div>
  <div class="wx-stat stagger-4"><div class="wx-icon">⭐</div>
    <div class="wx-val" style="color:{sc_color};">{sc}<span class="score-max">/100</span></div>
    <div class="wx-label">Travel Score</div></div>
</div>
<div class="wx-verdict {v_cls}">
  <span style="font-size:1.3rem;">{v_icon}</span>
  <span><strong>Smart Travel Suggestion:</strong> {stext}</span>
</div>""", unsafe_allow_html=True)
                    # Uber card with real coords
                    loc_match = loc_info[loc_info["name"] == station]
                    if not loc_match.empty:
                        lr = loc_match.iloc[0]
                        uber_url = (f"https://m.uber.com/ul/?action=setPickup"
                                    f"&pickup=my_location"
                                    f"&dropoff[latitude]={lr['location_lat']}"
                                    f"&dropoff[longitude]={lr['location_lng']}"
                                    f"&dropoff[nickname]={station.replace(' ','%20')}")
                        st.markdown(f"""
<div class="uber-card">
  <div class="uber-inner">
    <div class="uber-ico">🚗</div>
    <div class="uber-txt">
      <h4>Ready to visit?</h4>
      <p>Book a direct ride to {landmark}</p>
    </div>
  </div>
  <a href="{uber_url}" target="_blank" class="uber-btn-link">🚗 Connect with Uber</a>
</div>""", unsafe_allow_html=True)
                else:
                    st.warning("Prediction unavailable for this date/location.")
            else:
                st.warning("Could not match this landmark to a weather station.")
        else:
            st.info("Weather module not connected. Ensure weather_app.py is in the same folder.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── SECTION 3: RECOMMENDATIONS ───────────────────────────────────────
    st.markdown("""
<div style="margin-top:3rem;" class="sec-head">
  <div class="sec-title">
    <span class="sec-icon">💎</span>
    Personalised Recommendations
  </div>
  <div class="sec-sub">
    Curated by our hybrid recommendation engine using semantic analysis
    and graph neural networks. Discover gems near your destination.
  </div>
</div>""", unsafe_allow_html=True)

    if rec_models:
        sub1, model1, model2, model3 = rec_models
        # Fuzzy-match the CSV landmark name to rec_models place names
        # e.g. "Adam's Peak (Sri Pada)" -> "Adam's Peak" in rec_models
        rec_lm = fuzzy_match(landmark, sub1["place_name"].tolist()) or landmark
        match  = sub1[sub1["place_name"] == rec_lm]

        if match.empty:
            st.warning(f"'{landmark}' is not mapped in our recommendation database.")
        else:
            pid = match.iloc[0]["place_id"]

            def render_rec_cards(df, show_type=False):
                if df.empty:
                    st.info("No recommendations found.")
                    return
                cols = st.columns(min(len(df), 4))
                for i, (_, rec) in enumerate(df.head(4).iterrows()):
                    with cols[i]:
                        img_src = rec.get("image", "")
                        if img_src and str(img_src).startswith("http"):
                            img_html = f'<img src="{img_src}" class="rec-img" onerror="this.parentElement.innerHTML=\'📍\'">'
                        else:
                            img_html = '<div style="width:100%;height:165px;background:var(--bg-elevated);display:flex;align-items:center;justify-content:center;font-size:2.5rem;">📍</div>'
                        dist     = f'{rec.get("distance_km","")} km' if "distance_km" in rec else ""
                        review   = str(rec.get("review","A wonderful destination."))[:120]
                        t_html   = f'<span class="rec-type">{rec.get("type","")}</span>' if show_type and rec.get("type") else ""
                        st.markdown(f"""
<div class="rec-card">
  <div class="rec-img-wrap">
    {img_html}
    <div class="rec-badge">#{i+1}</div>
    <div class="rec-dist">{dist}</div>
  </div>
  <div class="rec-body">
    <div class="rec-name">{rec['name']}</div>
    <div class="rec-meta">
      <span class="rec-rating">★ {rec['rating']}/5</span>
      {t_html}
    </div>
    <div class="rec-review">"{review}"</div>
  </div>
</div>""", unsafe_allow_html=True)

            def render_svc_cards(df):
                if df.empty:
                    st.info("No services found nearby.")
                    return
                for i, (_, rec) in enumerate(df.head(5).iterrows()):
                    img_src = rec.get("image", "")
                    if img_src and str(img_src).startswith("http"):
                        img_html = f'<img src="{img_src}" class="svc-img" onerror="this.style.display=\'none\'">'
                    else:
                        img_html = '<div class="svc-img-ph">🏨</div>'
                    review = str(rec.get("review","Excellent service."))[:120]
                    st.markdown(f"""
<div class="svc-card">
  {img_html}
  <div class="svc-body">
    <div class="svc-name">
      <span style="background:var(--gradient-main);color:white;
            font-size:0.68rem;font-weight:800;padding:2px 8px;
            border-radius:6px;margin-right:8px;">#{i+1}</span>
      {rec['name']}
    </div>
    <div class="svc-meta">
      <span class="svc-rating">★ {rec['rating']}/5</span>
      <span class="svc-dist">{rec.get('distance_km','')} km</span>
    </div>
    <div class="svc-review">"{review}"</div>
  </div>
</div>""", unsafe_allow_html=True)

            t1, t2, t3 = st.tabs(["✨ You May Also Like", "🔥 Popular Nearby", "🏨 Nearby Services"])
            with t1:
                st.markdown('<p style="color:var(--text-muted);font-size:0.83rem;margin-bottom:1rem;">Similar places via semantic matching &amp; graph neural networks</p>', unsafe_allow_html=True)
                try:    render_rec_cards(model1.recommend(pid, top_n=4), show_type=True)
                except Exception as e: st.warning(f"Unavailable: {e}")
            with t2:
                st.markdown('<p style="color:var(--text-muted);font-size:0.83rem;margin-bottom:1rem;">Trending destinations ranked by visitor sentiment analysis</p>', unsafe_allow_html=True)
                try:    render_rec_cards(model2.recommend(pid, top_n=4))
                except Exception as e: st.warning(f"Unavailable: {e}")
            with t3:
                st.markdown('<p style="color:var(--text-muted);font-size:0.83rem;margin-bottom:1rem;">Hotels, dining &amp; activities curated for your convenience</p>', unsafe_allow_html=True)
                s1, s2, s3 = st.tabs(["🏨 Hotels", "🍽️ Dining", "🧭 Activities"])
                for stab, stype in [(s1,"Hotels"),(s2,"Dining"),(s3,"Activities")]:
                    with stab:
                        try:    render_svc_cards(model3.recommend(pid, service_type=stype, top_n=5))
                        except: st.info(f"No {stype.lower()} found nearby.")
    else:
        st.info("Recommendation module not connected. Ensure recommendationapp.py is in the same folder.")

    # ── FOOTER ────────────────────────────────────────────────────────────
    st.markdown("""
<div class="site-footer">
  <div class="footer-logo">◈ Toura<span>.lk</span></div>
  <p>AI-Powered Intelligent Tourism Support System &nbsp;·&nbsp; Sri Lanka</p>
  <p style="font-size:0.73rem;margin-top:0.3rem;">© 2026 Group 17 · All rights reserved</p>
  <div class="footer-links">
    <a href="#">Privacy</a><a href="#">Terms</a><a href="#">Support</a>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for k, v in [("page","home"),("landmark",None),("video_path",None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    render_navbar()

    page = st.session_state.page

    if page == "home":
        render_home()
    elif page == "analyzing":
        render_analyzing(st.session_state.get("landmark","Sigiriya"))
    elif page == "results":
        render_results()

if __name__ == "__main__":
    main()