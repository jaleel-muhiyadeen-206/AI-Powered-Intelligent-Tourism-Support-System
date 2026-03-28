
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta

st.set_page_config(
    page_title="Smart Weather Prediction",
    page_icon="⛅",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d0d0d; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 1.5rem; max-width: 760px; }

.page-title {
    font-size: 1.3rem; font-weight: 700; color: #ffffff;
    margin-bottom: 0.2rem; display: flex; align-items: center; gap: 8px;
}
.page-subtitle { font-size: 0.8rem; color: rgba(255,255,255,0.35); margin-bottom: 1.8rem; }

.main-card {
    background: #1a1a1a; border-radius: 18px;
    padding: 1.5rem; margin-bottom: 1.2rem;
}

.metric-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-bottom: 1.2rem;
}

.metric-tile {
    background: #242424; border-radius: 14px;
    padding: 1.1rem 0.8rem; text-align: center;
}
.metric-tile .icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
.metric-tile .val {
    font-size: 1.4rem; font-weight: 700; color: #ffffff;
    line-height: 1; margin-bottom: 0.3rem;
}
.metric-tile .val .unit { font-size: 0.75rem; font-weight: 400; color: rgba(255,255,255,0.4); }
.metric-tile .lbl {
    font-size: 0.7rem; color: rgba(255,255,255,0.35);
    text-transform: uppercase; letter-spacing: 0.08em;
}
.range-text { font-size: 0.68rem; color: rgba(255,255,255,0.25); margin-top: 3px; }
.score-denom { font-size: 0.7rem; color: rgba(255,255,255,0.35); }

.suggestion-banner {
    display: flex; align-items: center; gap: 10px;
    background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.25);
    border-radius: 12px; padding: 0.9rem 1.1rem;
    font-size: 0.83rem; color: rgba(255,255,255,0.75);
}
.check {
    width: 22px; height: 22px; background: #22c55e;
    border-radius: 6px; display: flex; align-items: center;
    justify-content: center; flex-shrink: 0; font-size: 0.75rem; color: white; font-weight: 700;
}
.suggestion-bad  { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.25); }
.suggestion-bad .check  { background: #ef4444; }
.suggestion-warn { background: rgba(234,179,8,0.1); border-color: rgba(234,179,8,0.25); }
.suggestion-warn .check { background: #eab308; }

.season-row {
    background: #1a1a1a; border-radius: 14px; padding: 1rem 1.3rem;
    margin-bottom: 1.2rem; display: flex; align-items: flex-start; gap: 12px;
}
.season-icon { font-size: 1.4rem; margin-top: 2px; }
.season-name { font-size: 0.85rem; font-weight: 600; color: #ffffff; margin-bottom: 3px; }
.season-desc { font-size: 0.78rem; color: rgba(255,255,255,0.4); line-height: 1.5; }

.uber-card {
    background: #1a1a1a; border-radius: 18px; padding: 1.2rem 1.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.uber-left { display: flex; align-items: center; gap: 14px; }
.uber-globe {
    width: 44px; height: 44px; background: #242424; border-radius: 50%;
    display: flex; align-items: center; justify-content: center; font-size: 1.3rem;
}
.uber-title { font-size: 0.92rem; font-weight: 600; color: #ffffff; margin-bottom: 3px; }
.uber-sub   { font-size: 0.75rem; color: rgba(255,255,255,0.35); }
.uber-btn-link {
    display: inline-flex; align-items: center; gap: 8px;
    background: #ffffff; color: #0d0d0d !important;
    text-decoration: none !important; font-size: 0.82rem; font-weight: 600;
    padding: 0.65rem 1.2rem; border-radius: 10px; white-space: nowrap; transition: opacity 0.2s;
}
.uber-btn-link:hover { opacity: 0.85; }

/* Text input styles */
.stTextInput > div > div > input {
    background: #242424 !important; 
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important; 
    color: #ffffff !important; 
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    padding: 0.65rem 0.85rem !important;
}
.stTextInput > div > div > input:hover {
    border-color: rgba(168,85,247,0.6) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.15) !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(168,85,247,0.8) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.2) !important;
}

/* Date input hover effect */
.stDateInput > div > div {
    transition: all 0.2s ease !important;
}
.stDateInput > div > div:hover {
    border-color: rgba(168,85,247,0.6) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.15) !important;
    transition: all 0.2s ease !important;
}
/* Focus state too */
.stDateInput > div > div:focus-within {
    border-color: rgba(168,85,247,0.8) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.2) !important;
}
/* Streamlit overrides */
.stSelectbox > div > div {
    background: #242424 !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important; color: #ffffff !important;
}
.stSelectbox > div > div:hover {
    border-color: rgba(168,85,247,0.6) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,0.15) !important;
    transition: all 0.2s ease !important;
}
.stDateInput > div > div > input {
    background: #242424 !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important; color: #ffffff !important; font-size: 0.9rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #a855f7, #7c3aed) !important;
    color: #ffffff !important; font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    border: none !important; border-radius: 10px !important;
    padding: 0.65rem 1.4rem !important; transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(124,58,237,0.5) !important;
}
.input-label {
    font-size: 0.75rem; color: rgba(255,255,255,0.4);
    margin-bottom: 5px; font-weight: 500; letter-spacing: 0.03em;
}
.stSpinner > div { border-top-color: #a855f7 !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load raw weather data - FAST because no processing"""
    df = pd.read_csv('C:\\Users\\User\PycharmProjects\DSGP\weather_prediction2\weather_data_lightweight_smart.csv',
                     usecols=['time', 'name', 'station_lat', 'station_lng',
                              'temperature_2m_max', 'temperature_2m_min',
                              'temperature_2m_mean', 'apparent_temperature_mean',
                              'weathercode', 'precipitation_sum', 'rain_sum',
                              'precipitation_hours', 'windspeed_10m_max',
                              'sunrise', 'sunset', 'address', 'district',
                              'keyword', 'location_lat', 'location_lng',
                              'city_x', 'elevation_x', 'distance_km'])
    df['time'] = pd.to_datetime(df['time'])
    return df.sort_values('time').reset_index(drop=True)


@st.cache_data
def get_location_info(df):
    """Extract unique locations with their details"""
    location_info = df.groupby('name').agg({
        'address': 'first',
        'district': 'first',
        'city_x': 'first',
        'location_lat': 'first',
        'location_lng': 'first'
    }).reset_index()

    # Create display name with address
    location_info['display_name'] = location_info.apply(
        lambda x: f"{x['name']} - {x['district']}, {x['city_x']}" if pd.notna(x['district']) else x['name'],
        axis=1
    )

    return location_info


@st.cache_resource
def load_models():
    """Load pre-trained models - cached so only loads once"""
    m = {
        'temperature_2m_mean': lgb.Booster(model_file=r'weather_prediction2\lightgbm_model_temperature_2m_mean.txt'),
        'precipitation_sum': lgb.Booster(model_file=r'weather_prediction2\lightgbm_model_precipitation_sum.txt'),
        'windspeed_10m_max': lgb.Booster(model_file=r'weather_prediction2\lightgbm_model_windspeed_10m_max.txt')
    }
    return m, joblib.load('weather_prediction2/feature_columns.pkl')


def create_features_for_location(df, location_name):
    """
    OPTIMIZED: Create features only for selected location (~220 rows instead of 208K)
    This is called on-demand when user clicks predict, not on page load
    """
    # Filter FIRST - reduces processing from 208K to ~220 rows
    loc_df = df[df['name'] == location_name].copy()

    if len(loc_df) == 0:
        return None

    d = loc_df.copy()

    # Time-based features
    d['hour'] = 12
    d['day'] = d['time'].dt.day
    d['month'] = d['time'].dt.month
    d['year'] = d['time'].dt.year
    d['day_of_year'] = d['time'].dt.dayofyear
    d['day_of_week'] = d['time'].dt.dayofweek
    d['is_weekend'] = d['day_of_week'].isin([5, 6]).astype(int)

    # Cyclical encoding
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['day_sin'] = np.sin(2 * np.pi * d['day_of_year'] / 365)
    d['day_cos'] = np.cos(2 * np.pi * d['day_of_year'] / 365)

    # Label encoding (only for this location's unique values)
    le_d, le_k, le_c = LabelEncoder(), LabelEncoder(), LabelEncoder()
    d['district_encoded'] = le_d.fit_transform(d['district'].fillna('Unknown'))
    d['keyword_encoded'] = le_k.fit_transform(d['keyword'].fillna('Unknown'))
    d['city_encoded'] = le_c.fit_transform(d['city_x'].fillna('Unknown'))

    # Weather-based features
    d['temp_range'] = d['temperature_2m_max'] - d['temperature_2m_min']
    d['apparent_diff'] = d['apparent_temperature_mean'] - d['temperature_2m_mean']
    d['rain_intensity'] = d['rain_sum'] / (d['precipitation_hours'] + 1)

    # Sort by time for lag/rolling features (only for this location)
    d = d.sort_values('time')

    # Lag features
    d['temp_lag_1'] = d['temperature_2m_mean'].shift(1)
    d['temp_lag_7'] = d['temperature_2m_mean'].shift(7)
    d['rain_lag_1'] = d['rain_sum'].shift(1)
    d['wind_lag_1'] = d['windspeed_10m_max'].shift(1)
    d['wind_lag_7'] = d['windspeed_10m_max'].shift(7)

    # Rolling features
    d['temp_rolling_3'] = d['temperature_2m_mean'].rolling(3, min_periods=1).mean()
    d['temp_rolling_7'] = d['temperature_2m_mean'].rolling(7, min_periods=1).mean()
    d['wind_rolling_3'] = d['windspeed_10m_max'].rolling(3, min_periods=1).mean()
    d['wind_rolling_7'] = d['windspeed_10m_max'].rolling(7, min_periods=1).mean()

    # Wind log transform
    d['wind_log'] = np.log1p(d['windspeed_10m_max'])

    # Fill missing values
    nc = d.select_dtypes(include='number').columns
    d[nc] = d[nc].bfill().ffill()

    return d


def predict_weather(location_name, target_date, df_raw, models, feature_cols):
    """
    Creates features on-demand for selected location only
    """
    # Create features only for this specific location
    df_enhanced = create_features_for_location(df_raw, location_name)

    if df_enhanced is None or len(df_enhanced) == 0:
        return None

    td = pd.to_datetime(target_date)

    # Get most recent data for this location
    lat = df_enhanced.sort_values('time').iloc[-1:].copy()

    # Build feature dictionary
    feat = {col: (lat[col].values[0] if col in lat.columns else 0) for col in feature_cols}

    # Update time-based features for target date
    feat.update({
        'day': td.day,
        'month': td.month,
        'year': td.year,
        'day_of_year': td.dayofyear,
        'day_of_week': td.dayofweek,
        'is_weekend': 1 if td.dayofweek >= 5 else 0,
        'month_sin': np.sin(2 * np.pi * td.month / 12),
        'month_cos': np.cos(2 * np.pi * td.month / 12),
        'day_sin': np.sin(2 * np.pi * td.dayofyear / 365),
        'day_cos': np.cos(2 * np.pi * td.dayofyear / 365),
        'hour': 12
    })

    pdf = pd.DataFrame([feat])

    # Make predictions
    out = {}
    for t, m in models.items():
        p = m.predict(pdf[feature_cols], num_iteration=m.best_iteration)[0]
        # FIXED: NO expm1 transformation - model outputs correct values
        out[t] = round(p, 2)

    return out


def score_weather(temp, rain, wind):
    """Calculate 0-100 travel score based on weather conditions"""
    s = 0

    # Temperature scoring (max 40 points)
    if 24 <= temp <= 30:
        s += 40
    elif 22 <= temp < 24 or 30 < temp <= 33:
        s += 33
    elif 20 <= temp < 22 or 33 < temp <= 36:
        s += 22
    else:
        s += 10

    # Precipitation scoring (max 35 points)
    if rain == 0:
        s += 35
    elif rain < 5:
        s += 30
    elif rain < 15:
        s += 22
    elif rain < 30:
        s += 14
    elif rain < 50:
        s += 7

    # Wind speed scoring (max 25 points)
    if wind < 10:
        s += 25
    elif wind < 20:
        s += 20
    elif wind < 30:
        s += 13
    elif wind < 45:
        s += 6

    return s


def get_suggestion(score):
    """Get travel suggestion based on weather score"""
    if score >= 70:
        return "good", "✓", "Ideal weather for exploring heritage sites!"
    elif score >= 50:
        return "warn", "!", "Fair conditions. Carry an umbrella just in case."
    else:
        return "bad", "✕", "Poor weather. Consider rescheduling your visit."


def get_season(month):
    """Get seasonal information for Sri Lanka"""
    if month in [12, 1, 2]:
        return "❄️", "Northeast Monsoon", "Cool and dry in most areas. Northeast coast may experience rain."
    elif month in [3, 4]:
        return "🌤", "First Inter-Monsoon", "Hot and humid island-wide. Afternoon thundershowers are common."
    elif month in [5, 6, 7, 8, 9]:
        return "🌧", "Southwest Monsoon", "Heavy rain on west & south coasts. Dry in the north and east."
    else:
        return "🌦", "Second Inter-Monsoon", "Rainfall island-wide. Both coasts can experience showers."


# =============================================================================
# MAIN APP - Load data (NO feature engineering on startup)
# =============================================================================

with st.spinner("Loading..."):
    df = load_data()  # Just loads raw CSV - FAST
    location_info = get_location_info(df)
    models, feature_cols = load_models()

st.markdown("""
<div class="page-title">⛅ Smart Weather Prediction</div>
<div class="page-subtitle">Plan your perfect heritage site visit</div>
""", unsafe_allow_html=True)

# Show total locations count
st.markdown(
    f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.25);margin-bottom:10px;">📊 {len(location_info)} locations available</div>',
    unsafe_allow_html=True)

# SEARCHABLE LOCATION INPUT
st.markdown('<div class="input-label">🔍 Search locations</div>', unsafe_allow_html=True)

# User can type to search
user_input = st.text_input("Search",
                           placeholder="Type to search (e.g., Sigiriya, Colombo, Kandy)...",
                           label_visibility="collapsed",
                           key="location_search")

# Filter locations based on search
if user_input:
    # Search in both name and address
    filtered_locations = location_info[
        location_info['name'].str.lower().str.contains(user_input.lower(), na=False) |
        location_info['display_name'].str.lower().str.contains(user_input.lower(), na=False)
        ]

    if len(filtered_locations) > 0:
        st.markdown(
            f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.35);margin:8px 0 4px;">💡 Found {len(filtered_locations)} matching location(s):</div>',
            unsafe_allow_html=True)

        # Show dropdown with filtered results
        selected_location = st.selectbox(
            "Select location",
            options=filtered_locations['name'].tolist(),
            format_func=lambda x: location_info[location_info['name'] == x]['display_name'].values[0],
            label_visibility="collapsed",
            key="filtered_dropdown"
        )
        landmark = selected_location

        # Show location details
        loc_details = location_info[location_info['name'] == landmark].iloc[0]
        st.markdown(
            f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.28);margin:4px 0 14px;">📍 {loc_details["address"]}</div>',
            unsafe_allow_html=True)
    else:
        st.warning(f"No locations found matching '{user_input}'. Try a different search term.")
        landmark = None
else:
    # Default: show all locations
    st.markdown(
        '<div style="font-size:0.7rem;color:rgba(255,255,255,0.35);margin:8px 0 4px;">💡 All available locations:</div>',
        unsafe_allow_html=True)

    landmark = st.selectbox(
        "Select location",
        options=location_info['name'].tolist(),
        format_func=lambda x: location_info[location_info['name'] == x]['display_name'].values[0],
        label_visibility="collapsed",
        key="all_dropdown"
    )

    # Show location details
    loc_details = location_info[location_info['name'] == landmark].iloc[0]
    st.markdown(
        f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.28);margin:4px 0 14px;">📍 {loc_details["address"]}</div>',
        unsafe_allow_html=True)

col_date, col_btn = st.columns([3, 1])
with col_date:
    st.markdown('<div class="input-label">Select your visit date</div>', unsafe_allow_html=True)
    selected_date = st.date_input("Date",
                                  value=date.today() + timedelta(days=1),
                                  min_value=date.today(),
                                  max_value=date.today() + timedelta(days=365 * 4),
                                  label_visibility="collapsed",
                                  format="DD/MM/YYYY")
with col_btn:
    st.markdown('<div class="input-label">&nbsp;</div>', unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Weather")

if predict_btn and landmark:
    with st.spinner("Predicting..."):
        # Features created on-demand for selected location only - FAST
        pred = predict_weather(landmark, selected_date, df, models, feature_cols)

    if pred:
        temp = pred['temperature_2m_mean']
        rain = pred['precipitation_sum']
        wind = pred['windspeed_10m_max']
        sc = score_weather(temp, rain, wind)
        stype, sicon, stext = get_suggestion(sc)
        s_ico, s_nm, s_desc = get_season(selected_date.month)

        rain_pct = min(99, round(rain * 3))
        temp_range = f"{round(temp - 1.5, 1)}–{round(temp + 1.5, 1)}"
        wind_range = f"{max(0, round(wind - 3))}–{round(wind + 3)}"

        # Metric tiles
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="icon">🌡️</div>
                <div class="val">{round(temp)}<span class="unit">°C</span></div>
                <div class="lbl">Temperature</div>
                <div class="range-text">{temp_range}°C</div>
            </div>
            <div class="metric-tile">
                <div class="icon">💧</div>
                <div class="val">{rain_pct}<span class="unit">%</span></div>
                <div class="lbl">Precipitation</div>
                <div class="range-text">{round(rain, 1)} mm</div>
            </div>
            <div class="metric-tile">
                <div class="icon">🌬️</div>
                <div class="val">{round(wind)}<span class="unit"> km/h</span></div>
                <div class="lbl">Wind Speed</div>
                <div class="range-text">{wind_range} km/h</div>
            </div>
            <div class="metric-tile">
                <div class="icon">⭐</div>
                <div class="val">{sc}<span class="score-denom">/100</span></div>
                <div class="lbl">Weather Score</div>
            </div>
        </div>
        <div class="suggestion-banner{'  suggestion-bad' if stype == 'bad' else ' suggestion-warn' if stype == 'warn' else ''}">
            <div class="check">{sicon}</div>
            <div><strong>Smart Travel Suggestion:</strong> {stext}</div>
        </div>
        """, unsafe_allow_html=True)

        # Season
        st.markdown(f"""
        <div class="season-row">
            <div class="season-icon">{s_ico}</div>
            <div>
                <div class="season-name">{s_nm}</div>
                <div class="season-desc">{s_desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Uber
        loc_details = location_info[location_info['name'] == landmark].iloc[0]

        uber_link = f"https://m.uber.com/ul/?action=setPickup&pickup=my_location&dropoff[latitude]={loc_details['location_lat']}&dropoff[longitude]={loc_details['location_lng']}&dropoff[nickname]={landmark.replace(' ', '%20')}"

        st.markdown(f"""
                <div class="uber-card">
                    <div class="uber-left">
                        <div class="uber-globe">🌐</div>
                        <div>
                            <div class="uber-title">Ready to visit?</div>
                            <div class="uber-sub">Book a ride directly to {landmark}</div>
                        </div>
                    </div>
                    <a href="{uber_link}"
                       target="_blank" class="uber-btn-link">🚗 Connect with Uber</a>
                </div>
                """, unsafe_allow_html=True)

    else:
        st.error(f"⚠️ No weather data found for **{landmark}**.")