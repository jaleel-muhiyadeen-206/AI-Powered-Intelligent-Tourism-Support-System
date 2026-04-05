"""
Streamlit Application: Toura.lk
=================================
AI-driven smart tourism recommendation system for Sri Lanka.

Models:
    Model1: You May Also Like (SBERT + GNN + LambdaMART)
    Model2: Popular Places Nearby (BERT + Kaggle + LambdaMART)
    Model3: Nearby Essentials (BERT + per-type LambdaMART)

Input Modes:
    1. Search & Select: User types a place name and selects from dataset
    2. External Input: Another model passes a place name via URL query parameter

User Interactions:
    - Likes: Liked places get priority boost in recommendations
    - Reviews: Replace display_review in Model 1, replace review columns in Model 2
    - Ratings: Stored in CSV, affect displayed rating
"""

import streamlit as st
import pandas as pd
import joblib
import os
import html as html_module
from datetime import datetime
from difflib import SequenceMatcher

from models.model_1_you_may_also_like import Model1_YouMayAlsoLike
from models.model_2_popular_nearby import Model2_PopularNearby
from models.model_3_nearby_essentials import Model3_NearbyEssentials

st.set_page_config(
    page_title="Toura.lk — AI-Driven Smart Tourism",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

USER_LIKES_PATH = 'data/user_likes.csv'
USER_REVIEWS_PATH = 'data/user_reviews.csv'
USER_RATINGS_PATH = 'data/user_ratings.csv'


def _ensure_user_data_files():
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(USER_LIKES_PATH):
        pd.DataFrame(columns=['place_id', 'timestamp']).to_csv(USER_LIKES_PATH, index=False)
    if not os.path.exists(USER_REVIEWS_PATH):
        pd.DataFrame(columns=['place_id', 'review', 'timestamp']).to_csv(USER_REVIEWS_PATH, index=False)
    if not os.path.exists(USER_RATINGS_PATH):
        pd.DataFrame(columns=['place_id', 'rating', 'timestamp']).to_csv(USER_RATINGS_PATH, index=False)


def add_like(place_id):
    _ensure_user_data_files()
    df = pd.read_csv(USER_LIKES_PATH)
    if place_id not in df['place_id'].values:
        new_row = pd.DataFrame([{'place_id': place_id, 'timestamp': datetime.now().isoformat()}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(USER_LIKES_PATH, index=False)


def remove_like(place_id):
    _ensure_user_data_files()
    df = pd.read_csv(USER_LIKES_PATH)
    df = df[df['place_id'] != place_id]
    df.to_csv(USER_LIKES_PATH, index=False)


def is_liked(place_id):
    _ensure_user_data_files()
    df = pd.read_csv(USER_LIKES_PATH)
    return place_id in df['place_id'].values


def get_liked_place_ids():
    _ensure_user_data_files()
    df = pd.read_csv(USER_LIKES_PATH)
    return set(df['place_id'].tolist())


def add_review_and_rating(place_id, review_text, rating):
    _ensure_user_data_files()
    if review_text and str(review_text).strip():
        rdf = pd.read_csv(USER_REVIEWS_PATH)
        new_row = pd.DataFrame([{
            'place_id': place_id, 'review': review_text,
            'timestamp': datetime.now().isoformat()
        }])
        rdf = pd.concat([rdf, new_row], ignore_index=True)
        rdf.to_csv(USER_REVIEWS_PATH, index=False)
    if rating and rating > 0:
        rtdf = pd.read_csv(USER_RATINGS_PATH)
        new_row = pd.DataFrame([{
            'place_id': place_id, 'rating': rating,
            'timestamp': datetime.now().isoformat()
        }])
        rtdf = pd.concat([rtdf, new_row], ignore_index=True)
        rtdf.to_csv(USER_RATINGS_PATH, index=False)


def get_reviews_for_place(place_id):
    _ensure_user_data_files()
    df = pd.read_csv(USER_REVIEWS_PATH)
    return df[df['place_id'] == place_id]['review'].tolist()


def get_average_rating(place_id, original_rating):
    _ensure_user_data_files()
    df = pd.read_csv(USER_RATINGS_PATH)
    user_ratings = df[df['place_id'] == place_id]['rating'].tolist()
    if not user_ratings:
        return original_rating
    all_ratings = [original_rating] + user_ratings
    return round(sum(all_ratings) / len(all_ratings), 1)


def get_rating_count(place_id):
    _ensure_user_data_files()
    df = pd.read_csv(USER_RATINGS_PATH)
    return len(df[df['place_id'] == place_id])


def update_model1_display_review(sub1_df, place_id, user_reviews):
    if user_reviews:
        mask = sub1_df['place_id'] == place_id
        if mask.any():
            sub1_df.loc[mask, 'display_review'] = user_reviews[-1]
    return sub1_df


def update_model2_reviews(model2, place_id, user_reviews):
    if not user_reviews or not hasattr(model2, 'df'):
        return
    mask = model2.df['place_id'] == place_id
    if not mask.any():
        return
    review_cols = ['review_1', 'review_2', 'review_3', 'review_4', 'review_5']
    for i, review in enumerate(user_reviews[:5]):
        if i < len(review_cols):
            model2.df.loc[mask, review_cols[i]] = review
    if 'display_review' in model2.df.columns and user_reviews:
        model2.df.loc[mask, 'display_review'] = user_reviews[-1]


def boost_liked_recommendations(recs_df, liked_ids):
    if recs_df.empty or not liked_ids:
        return recs_df
    recs = recs_df.copy()
    for idx, row in recs.iterrows():
        name = row.get('name', '')
        if any(pid in str(name) for pid in liked_ids):
            recs.at[idx, 'final_score'] = row['final_score'] * 1.15
    recs = recs.sort_values('final_score', ascending=False).reset_index(drop=True)
    recs['rank'] = range(1, len(recs) + 1)
    return recs


def fuzzy_match_place(place_name, sub1_df, threshold=0.60):
    best_ratio, best_match = 0, None
    name_lower = place_name.lower().strip()
    for _, row in sub1_df.iterrows():
        ratio = SequenceMatcher(None, name_lower, row['place_name'].lower().strip()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = row
    if best_ratio >= threshold and best_match is not None:
        return best_match['place_id'], best_match['place_name'], best_ratio
    return None, None, 0.0


@st.cache_resource
def load_models():
    sub1 = joblib.load('output/preprocessed/submodel_1.joblib')
    model1 = Model1_YouMayAlsoLike(sub1)
    model1.load('output/trained_models/model_1.joblib')
    model2 = Model2_PopularNearby(pd.DataFrame())
    model2.load('output/trained_models/model_2.joblib')
    model3 = Model3_NearbyEssentials(pd.DataFrame(), pd.DataFrame())
    model3.load('output/trained_models/model_3.joblib')
    return sub1, model1, model2, model3


def inject_css():
    st.markdown('<meta name="referrer" content="no-referrer">', unsafe_allow_html=True)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    :root {
        --bg-primary: #0a0a1a;
        --bg-secondary: #12122a;
        --bg-card: #1a1a3e;
        --bg-card-hover: #222260;
        --text-primary: #ffffff;
        --text-secondary: #a0a0c0;
        --accent-gradient: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%);
        --accent-purple: #a855f7;
        --accent-violet: #7c3aed;
        --gold: #f59e0b;
        --green: #10b981;
        --border-color: rgba(168, 85, 247, 0.15);
    }
    .stApp {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] { display: none !important; }
    header[data-testid="stHeader"] { background: transparent !important; }
    .block-container { max-width: 1200px !important; padding-top: 1rem !important; }

    .nav-bar {
        background: rgba(10, 10, 26, 0.95);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--border-color);
        padding: 0.8rem 2rem;
        display: flex; align-items: center; gap: 1rem;
        position: sticky; top: 0; z-index: 1000;
        margin: -1rem -1rem 2rem -1rem;
    }
    .nav-logo {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem; font-weight: 800; color: white;
    }
    .nav-dot { color: var(--accent-purple); }

    .hero-section {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1040 50%, #0a0a1a 100%);
        border-radius: 24px; padding: 4rem 2rem; text-align: center;
        margin-bottom: 2rem; border: 1px solid var(--border-color);
        position: relative; overflow: hidden;
    }
    .hero-section::before {
        content: ''; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 30% 50%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
                    radial-gradient(circle at 70% 50%, rgba(168, 85, 247, 0.06) 0%, transparent 50%);
        animation: pulse-bg 8s ease-in-out infinite;
    }
    @keyframes pulse-bg {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }
    .hero-badge {
        display: inline-block; padding: 0.4rem 1.2rem;
        background: rgba(168, 85, 247, 0.15);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 100px; font-size: 0.75rem; font-weight: 600;
        color: var(--accent-purple); letter-spacing: 0.1em;
        text-transform: uppercase; margin-bottom: 1.5rem; position: relative;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 800;
        line-height: 1.1; margin-bottom: 1rem; color: white; position: relative;
    }
    .gradient-text {
        background: var(--accent-gradient);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 1.1rem; color: var(--text-secondary);
        max-width: 600px; margin: 0 auto 2rem; line-height: 1.6; position: relative;
    }

    .place-header {
        position: relative; border-radius: 20px; overflow: hidden;
        margin-bottom: 2rem; min-height: 320px;
    }
    .place-header-bg {
        width: 100%; height: 320px; object-fit: cover;
        filter: brightness(0.4); border-radius: 20px;
    }
    .place-header-overlay {
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(180deg, transparent 30%, rgba(10, 10, 26, 0.95) 100%);
        border-radius: 20px;
    }
    .place-header-content {
        position: absolute; bottom: 0; left: 0; right: 0; padding: 2rem;
    }
    .place-name-text {
        font-family: 'Outfit', sans-serif; font-size: 2.5rem;
        font-weight: 800; color: white; margin-bottom: 0.5rem;
    }
    .place-location-text {
        color: var(--text-secondary); font-size: 0.95rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .place-description-text {
        color: var(--text-secondary); font-size: 0.95rem;
        line-height: 1.7; margin-top: 1rem;
    }

    .section-title {
        font-family: 'Outfit', sans-serif; font-size: 1.5rem;
        font-weight: 700; color: white; margin-bottom: 0.3rem;
    }
    .section-subtitle {
        color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1.5rem;
    }

    /* Force all st.columns children to stretch equally */
    [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        display: flex !important;
        flex-direction: column !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div {
        flex: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }

    .rec-card {
        background: var(--bg-card); border-radius: 16px; overflow: hidden;
        border: 1px solid var(--border-color); transition: all 0.3s ease;
        display: flex; flex-direction: column; min-height: 380px; height: 100%;
    }
    .rec-card:hover {
        transform: translateY(-4px);
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 12px 30px rgba(124, 58, 237, 0.15);
    }
    .rec-card-img-wrap { position: relative; width: 100%; flex-shrink: 0; }
    .rec-card-img { width: 100%; height: 160px; object-fit: cover; display: block; }
    .rec-card-placeholder {
        width: 100%; height: 160px; background: var(--bg-card-hover);
        display: flex; align-items: center; justify-content: center;
        color: var(--text-secondary); font-size: 2rem;
    }
    .rec-card-badge {
        position: absolute; top: 10px; left: 10px;
        background: var(--accent-gradient); color: white;
        font-weight: 700; font-size: 0.7rem; padding: 3px 8px; border-radius: 6px;
    }
    .rec-card-distance {
        position: absolute; top: 10px; right: 10px;
        background: rgba(0, 0, 0, 0.65); backdrop-filter: blur(8px);
        color: white; font-size: 0.65rem; padding: 3px 7px; border-radius: 5px;
    }
    .rec-card-body {
        padding: 0.8rem; flex: 1; display: flex; flex-direction: column;
    }
    .rec-card-name {
        font-family: 'Outfit', sans-serif; font-weight: 600;
        font-size: 0.9rem; color: white; margin-bottom: 0.3rem;
        line-height: 1.3; min-height: 2.4em;
        overflow: hidden; text-overflow: ellipsis;
        display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
    }
    .rec-card-meta {
        display: flex; align-items: center; gap: 0.5rem;
        margin-bottom: 0.3rem; font-size: 0.75rem;
    }
    .rec-card-rating { color: var(--gold); font-weight: 600; }
    .rec-card-type { color: var(--accent-purple); font-weight: 500; font-size: 0.7rem; }
    .rec-card-district { color: var(--text-secondary); font-size: 0.7rem; margin-bottom: 0.4rem; }
    .rec-card-review {
        color: var(--text-secondary); font-size: 0.72rem; line-height: 1.4;
        font-style: italic; display: -webkit-box;
        -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
        margin-top: auto;
    }

    .service-card {
        background: var(--bg-card); border-radius: 14px; overflow: hidden;
        border: 1px solid var(--border-color); display: flex;
        transition: all 0.3s ease; margin-bottom: 0.8rem;
    }
    .service-card:hover {
        border-color: rgba(168, 85, 247, 0.4); transform: translateX(4px);
    }
    .service-card-img { width: 120px; min-height: 120px; object-fit: cover; flex-shrink: 0; }
    .service-card-body { padding: 0.8rem 1rem; flex: 1; }
    .service-card-name {
        font-family: 'Outfit', sans-serif; font-weight: 600;
        color: white; margin-bottom: 0.3rem;
    }
    .service-card-meta { display: flex; gap: 1rem; font-size: 0.8rem; margin-bottom: 0.4rem; }
    .service-card-rating { color: var(--gold); }
    .service-card-distance { color: var(--text-secondary); }
    .service-card-budget { color: var(--green); font-size: 0.75rem; margin-bottom: 0.3rem; }
    .service-card-review {
        color: var(--text-secondary); font-size: 0.75rem; font-style: italic;
        display: -webkit-box; -webkit-line-clamp: 2;
        -webkit-box-orient: vertical; overflow: hidden;
    }
    .rank-badge {
        display: inline-block; background: var(--accent-gradient); color: white;
        font-weight: 700; font-size: 0.7rem; padding: 2px 8px;
        border-radius: 6px; margin-right: 0.5rem;
    }
    .footer {
        text-align: center; padding: 2rem; color: var(--text-secondary);
        font-size: 0.8rem; border-top: 1px solid var(--border-color); margin-top: 3rem;
    }

    div[data-testid="stSelectbox"] label { color: var(--text-secondary) !important; }
    div[data-testid="stSelectbox"] > div > div {
        background: var(--bg-card) !important;
        border-color: var(--border-color) !important;
        color: white !important;
    }
    .stTextInput input, .stTextArea textarea {
        background: var(--bg-card) !important;
        border-color: var(--border-color) !important;
        color: white !important;
    }
    div[data-testid="stTabs"] button {
        color: var(--text-secondary) !important; background: transparent !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 10px !important; padding: 0.5rem 1.2rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: var(--accent-gradient) !important; color: white !important;
        border-color: var(--accent-purple) !important;
    }
    .stButton button {
        background: var(--accent-gradient) !important; color: white !important;
        border: none !important; border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important; font-weight: 600 !important;
    }
    .stButton button:hover {
        box-shadow: 0 8px 25px rgba(124, 58, 237, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_navbar():
    st.markdown("""
    <div class="nav-bar">
        <div class="nav-logo">◈ Toura<span class="nav-dot">.lk</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_hero():
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">AI-Powered Tourism Intelligence</div>
        <div class="hero-title">Discover Sri Lanka<br><span class="gradient-text">Through AI Recommendations</span></div>
        <p class="hero-sub">Search for any Sri Lankan landmark and get personalised recommendations powered by SBERT, GNN, and LambdaMART.</p>
    </div>
    """, unsafe_allow_html=True)


def _is_valid_image_url(url):
    """Check if the URL is a real image URL (not a placeholder)."""
    if not url or not str(url).startswith("http"):
        return False
    url_str = str(url).lower()
    if 'placeholder.com' in url_str or 'placeholder' in url_str.split('?')[0]:
        return False
    return True


def render_rec_card_html(rank, name, rating, distance, review, place_type="",
                         district="", image_url=""):
    """Generate a single recommendation card as HTML."""
    safe_name = html_module.escape(str(name))
    safe_review = html_module.escape(str(review)[:120])
    if len(str(review)) > 120:
        safe_review += "..."
    safe_type = html_module.escape(str(place_type)) if place_type else ''
    safe_district = html_module.escape(str(district)) if district else ''
    # Generate initials for fallback
    initials = ''.join(w[0] for w in str(name).split()[:2] if w).upper() or '📍'
    fallback_div = (f'<div class="rec-card-placeholder" style="background:linear-gradient(135deg,#7c3aed 0%,#a855f7 100%);'
                    f'display:flex;align-items:center;justify-content:center;font-size:1.5rem;font-weight:700;'
                    f'color:rgba(255,255,255,0.85);">{initials}</div>')
    if _is_valid_image_url(image_url):
        onerror = ("this.style.display='none';" 
                   "this.nextElementSibling.style.display='flex';")
        img_html = f'<img src="{image_url}" class="rec-card-img" referrerpolicy="no-referrer" onerror="{onerror}">'
        placeholder = f'<div class="rec-card-placeholder" style="display:none;background:linear-gradient(135deg,#7c3aed 0%,#a855f7 100%);color:rgba(255,255,255,0.85);font-size:1.5rem;font-weight:700;">{initials}</div>'
    else:
        img_html = ''
        placeholder = fallback_div
    type_html = f'<span class="rec-card-type">{safe_type}</span>' if safe_type else ''
    district_html = f'<div class="rec-card-district">📍 {safe_district}</div>' if safe_district else ''
    return f"""
    <div class="rec-card">
        <div class="rec-card-img-wrap">
            {img_html}{placeholder}
            <div class="rec-card-badge">#{rank}</div>
            <div class="rec-card-distance">{distance} km</div>
        </div>
        <div class="rec-card-body">
            <div class="rec-card-name">{safe_name}</div>
            <div class="rec-card-meta">
                <span class="rec-card-rating">★ {rating}/5</span>
                {type_html}
            </div>
            {district_html}
            <div class="rec-card-review">"{safe_review}"</div>
        </div>
    </div>
    """


def render_cards_row(recs_df, show_type=False):
    """Render 5 cards using st.columns — each card is forced to same height via CSS."""
    if recs_df.empty:
        st.info("No recommendations found within the distance limit.")
        return
    n = len(recs_df)
    cols = st.columns(max(n, 5))
    for i, (_, rec) in enumerate(recs_df.iterrows()):
        if i >= 5:
            break
        with cols[i]:
            html = render_rec_card_html(
                rec['rank'], rec['name'], rec['rating'],
                rec['distance_km'],
                rec.get('review', 'A remarkable destination.'),
                rec.get('type', '') if show_type else '',
                rec.get('district', ''),
                rec.get('image', '')
            )
            st.markdown(html, unsafe_allow_html=True)


def render_service_card(rank, name, rating, distance, review, budget="", image_url=""):
    safe_name = html_module.escape(str(name))
    safe_review = html_module.escape(str(review)[:120])
    if len(str(review)) > 120:
        safe_review += "..."
    safe_budget = html_module.escape(str(budget)) if budget else ''
    fallback_div = ('<div style="width:120px;min-height:120px;background:linear-gradient(135deg,#7c3aed 0%,#a855f7 100%);'
                    'display:flex;align-items:center;justify-content:center;'
                    'color:rgba(255,255,255,0.85);font-size:1.2rem;font-weight:700;flex-shrink:0;">🏨</div>')
    if _is_valid_image_url(image_url):
        onerror = ("this.style.display='none';" 
                   "this.nextElementSibling.style.display='flex';")
        img_html = f'<img src="{image_url}" class="service-card-img" referrerpolicy="no-referrer" onerror="{onerror}">'
        img_html += ('<div style="display:none;width:120px;min-height:120px;background:linear-gradient(135deg,#7c3aed 0%,#a855f7 100%);'
                     'align-items:center;justify-content:center;'
                     'color:rgba(255,255,255,0.85);font-size:1.2rem;font-weight:700;flex-shrink:0;">🏨</div>')
    else:
        img_html = fallback_div
    budget_html = f'<div class="service-card-budget">💰 {safe_budget}</div>' if safe_budget else ''
    return f"""
    <div class="service-card">
        {img_html}
        <div class="service-card-body">
            <div class="service-card-name"><span class="rank-badge">#{rank}</span>{safe_name}</div>
            <div class="service-card-meta">
                <span class="service-card-rating">★ {rating}/5</span>
                <span class="service-card-distance">{distance} km</span>
            </div>
            {budget_html}
            <div class="service-card-review">"{safe_review}"</div>
        </div>
    </div>
    """


def render_star_rating(pid):
    """Render Google Maps-style clickable star rating using 5 star buttons."""
    if f'user_star_{pid}' not in st.session_state:
        st.session_state[f'user_star_{pid}'] = 0

    current = st.session_state[f'user_star_{pid}']

    star_cols = st.columns([1, 1, 1, 1, 1, 4])
    for i in range(5):
        star_num = i + 1
        with star_cols[i]:
            label = "★" if star_num <= current else "☆"
            if st.button(label, key=f"star_{pid}_{star_num}",
                         help=f"{star_num} star{'s' if star_num > 1 else ''}"):
                st.session_state[f'user_star_{pid}'] = star_num
                st.rerun()

    return current


def render_place_results(place, sub1, model1, model2, model3):
    pid = place['place_id']
    pname = place['place_name']
    district = place['district']
    desc = place.get('description', '')
    original_rating = place.get('avg_rating', 0)
    image_url = place.get('image_url', '')

    avg_rating = get_average_rating(pid, original_rating)
    user_rating_count = get_rating_count(pid)
    user_reviews = get_reviews_for_place(pid)
    liked_ids = get_liked_place_ids()

    if user_reviews:
        update_model1_display_review(sub1, pid, user_reviews)
        update_model2_reviews(model2, pid, user_reviews)

    # ── Place Header ───────────────────────────────────────────────────
    if _is_valid_image_url(image_url):
        st.markdown(f"""
        <div class="place-header">
            <img src="{image_url}" class="place-header-bg" referrerpolicy="no-referrer" onerror="this.src='';this.style.background='linear-gradient(135deg,#7c3aed 0%,#1a1040 100%)';this.style.display='block';">
            <div class="place-header-overlay"></div>
            <div class="place-header-content">
                <div class="place-name-text">{pname}</div>
                <div class="place-location-text">📍 {district}, Sri Lanka • ★ {avg_rating}/5{f' ({user_rating_count + 1} ratings)' if user_rating_count > 0 else ''}</div>
                <div class="place-description-text">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: var(--bg-card); border-radius: 20px; padding: 2rem; margin-bottom: 2rem; border: 1px solid var(--border-color);">
            <div class="place-name-text">{pname}</div>
            <div class="place-location-text">📍 {district}, Sri Lanka • ★ {avg_rating}/5{f' ({user_rating_count + 1} ratings)' if user_rating_count > 0 else ''}</div>
            <div class="place-description-text">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Like button ────────────────────────────────────────────────────
    liked = is_liked(pid)
    if st.button("❤️ Liked" if liked else "🤍 Like this place", key=f"like_btn_{pid}"):
        if liked:
            remove_like(pid)
            st.toast("Removed from favorites")
        else:
            add_like(pid)
            st.toast("❤️ Added to favorites! Liked places get priority in recommendations.")
        st.rerun()

    # ── Combined Rate & Review ─────────────────────────────────────────
    with st.expander("⭐ Rate & Review", expanded=False):
        st.markdown(f"""
        <div style="color:white;font-family:'Outfit',sans-serif;font-weight:700;font-size:1.1rem;margin-bottom:0.3rem;">
            Rate your experience at {pname}
        </div>
        <div style="color:var(--text-secondary);font-size:0.85rem;margin-bottom:0.8rem;">
            Click a star to set your rating, then write your review below
        </div>
        """, unsafe_allow_html=True)

        # Clickable star rating
        star_rating = render_star_rating(pid)

        current_stars = st.session_state.get(f'user_star_{pid}', 0)
        if current_stars > 0:
            st.markdown(
                f'<div style="color:var(--gold);font-weight:600;margin-bottom:0.8rem;">'
                f'Your rating: {"★" * current_stars}{"☆" * (5 - current_stars)} ({current_stars}/5)</div>',
                unsafe_allow_html=True
            )

        review_text = st.text_area(
            "Write your review",
            placeholder="Tell others about your experience at this place...",
            key=f"review_text_{pid}",
            height=100,
            label_visibility="collapsed"
        )

        if st.button("Submit", key=f"submit_rr_{pid}"):
            final_rating = st.session_state.get(f'user_star_{pid}', 0)
            if final_rating == 0 and not review_text:
                st.warning("Please select a star rating or write a review.")
            else:
                add_review_and_rating(pid, review_text, final_rating)
                st.session_state[f'user_star_{pid}'] = 0
                parts = []
                if review_text:
                    parts.append("review")
                if final_rating > 0:
                    parts.append(f"{final_rating}-star rating")
                st.toast(f"✅ Submitted {' and '.join(parts)}!")
                st.rerun()

    # ── Show user's past reviews ───────────────────────────────────────
    if user_reviews:
        st.markdown(f"""
        <div style="background:var(--bg-card);border-radius:12px;padding:1rem;margin-bottom:1rem;border:1px solid var(--border-color);">
            <div style="color:var(--accent-purple);font-weight:600;margin-bottom:0.5rem;">Your Reviews ({len(user_reviews)})</div>
            {''.join(f'<div style="color:var(--text-secondary);font-style:italic;margin-bottom:0.3rem;">"' + str(r) + '"</div>' for r in user_reviews[-3:])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model 1: You May Also Like ─────────────────────────────────────
    st.markdown("""
    <div class="section-title">💎 You May Also Like</div>
    <div class="section-subtitle">Similar places based on semantic similarity, graph relations, and learned ranking</div>
    """, unsafe_allow_html=True)

    try:
        recs1 = model1.recommend(pid, top_n=5)
        recs1 = boost_liked_recommendations(recs1, liked_ids)
        render_cards_row(recs1, show_type=True)
    except Exception as e:
        st.warning(f"Model 1 recommendations unavailable: {e}")

    st.markdown("---")

    # ── Model 2: Popular Places Nearby ─────────────────────────────────
    st.markdown("""
    <div class="section-title">🔥 Popular Places Nearby</div>
    <div class="section-subtitle">Trending destinations ranked by BERT sentiment and Kaggle review enrichment</div>
    """, unsafe_allow_html=True)

    try:
        recs2 = model2.recommend(pid, top_n=5)
        recs2 = boost_liked_recommendations(recs2, liked_ids)
        render_cards_row(recs2, show_type=False)
    except Exception as e:
        st.warning(f"Model 2 recommendations unavailable: {e}")

    st.markdown("---")

    # ── Model 3: Nearby Services ───────────────────────────────────────
    st.markdown("""
    <div class="section-title">🏨 Nearby Services</div>
    <div class="section-subtitle">Hotels, dining, and activities ranked by BERT sentiment with per-type LambdaMART</div>
    """, unsafe_allow_html=True)

    tab_hotels, tab_dining, tab_activities = st.tabs(["🏨 Hotels", "🍽️ Dining", "🧭 Activities"])

    for tab, stype in [(tab_hotels, 'Hotels'), (tab_dining, 'Dining'),
                       (tab_activities, 'Activities')]:
        with tab:
            try:
                recs3 = model3.recommend(pid, service_type=stype, top_n=5)
                if not recs3.empty:
                    for _, rec in recs3.iterrows():
                        st.markdown(render_service_card(
                            rec['rank'], rec['name'], rec['rating'],
                            rec['distance_km'],
                            rec.get('review', 'Service awaiting review.'),
                            rec.get('budget', ''), rec.get('image', '')
                        ), unsafe_allow_html=True)
                else:
                    st.info(f"No {stype.lower()} found within the distance limit.")
            except Exception as e:
                st.warning(f"{stype} recommendations unavailable: {e}")

    st.markdown("""
    <div class="footer">
        <div style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;margin-bottom:0.5rem;">
            ◈ Toura<span style="color:var(--accent-purple)">.lk</span>
        </div>
        <p>AI-Driven Smart Tourism • Sri Lanka</p>
    </div>
    """, unsafe_allow_html=True)


def recommend_by_name(place_name, sub1):
    pid, matched_name, score = fuzzy_match_place(place_name, sub1)
    return pid, matched_name, score


def main():
    _ensure_user_data_files()
    inject_css()
    render_navbar()

    sub1, model1, model2, model3 = load_models()

    if 'selected_place_id' not in st.session_state:
        st.session_state['selected_place_id'] = None

    query_params = st.query_params
    ext_place_name = query_params.get("place_name", None)
    if ext_place_name and st.session_state['selected_place_id'] is None:
        pid, matched_name, score = recommend_by_name(ext_place_name, sub1)
        if pid:
            st.session_state['selected_place_id'] = pid
            st.toast(f"Matched '{ext_place_name}' → {matched_name} ({score:.0%})")
            st.rerun()
        else:
            st.warning(f"Could not match '{ext_place_name}' to any place in the dataset.")

    if st.session_state['selected_place_id'] is None:
        render_hero()

        place_names = sorted(sub1['place_name'].tolist())
        selected = st.selectbox(
            "Search for a Sri Lankan landmark",
            options=[""] + place_names, index=0,
            key="place_search", placeholder="Type a place name..."
        )

        if selected:
            match = sub1[sub1['place_name'] == selected]
            if not match.empty:
                st.session_state['selected_place_id'] = match.iloc[0]['place_id']
                st.rerun()
    else:
        pid = st.session_state['selected_place_id']
        place = sub1[sub1['place_id'] == pid]

        if place.empty:
            st.error("Place not found in dataset.")
            if st.button("⬅ Back to search"):
                st.session_state['selected_place_id'] = None
                st.rerun()
            return

        place = place.iloc[0]

        if st.button("⬅ Back to search", key="back_btn"):
            st.session_state['selected_place_id'] = None
            st.rerun()

        render_place_results(place, sub1, model1, model2, model3)


if __name__ == "__main__":
    main()
