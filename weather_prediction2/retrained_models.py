# retrain_models.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('weather_data_lightweight_smart_with_landmarks.csv',
                 usecols=['time', 'name', 'station_lat', 'station_lng',
                          'temperature_2m_max', 'temperature_2m_min',
                          'temperature_2m_mean', 'apparent_temperature_mean',
                          'weathercode', 'precipitation_sum', 'rain_sum',
                          'precipitation_hours', 'windspeed_10m_max',
                          'sunrise', 'sunset', 'address', 'district',
                          'keyword', 'location_lat', 'location_lng',
                          'city_x', 'elevation_x', 'distance_km'])
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# Feature engineering (mirrors weather_app.py)
df['hour'] = 12
df['day'] = df['time'].dt.day
df['month'] = df['time'].dt.month
df['year'] = df['time'].dt.year
df['day_of_year'] = df['time'].dt.dayofyear
df['day_of_week'] = df['time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

le_d, le_k, le_c = LabelEncoder(), LabelEncoder(), LabelEncoder()
df['district_encoded'] = le_d.fit_transform(df['district'].fillna('Unknown'))
df['keyword_encoded']  = le_k.fit_transform(df['keyword'].fillna('Unknown'))
df['city_encoded']     = le_c.fit_transform(df['city_x'].fillna('Unknown'))

df['temp_range']     = df['temperature_2m_max'] - df['temperature_2m_min']
df['apparent_diff']  = df['apparent_temperature_mean'] - df['temperature_2m_mean']
df['rain_intensity'] = df['rain_sum'] / (df['precipitation_hours'] + 1)

df['temp_lag_1']  = df['temperature_2m_mean'].shift(1)
df['temp_lag_7']  = df['temperature_2m_mean'].shift(7)
df['rain_lag_1']  = df['rain_sum'].shift(1)
df['wind_lag_1']  = df['windspeed_10m_max'].shift(1)
df['wind_lag_7']  = df['windspeed_10m_max'].shift(7)

df['temp_rolling_3'] = df['temperature_2m_mean'].rolling(3, min_periods=1).mean()
df['temp_rolling_7'] = df['temperature_2m_mean'].rolling(7, min_periods=1).mean()
df['wind_rolling_3'] = df['windspeed_10m_max'].rolling(3, min_periods=1).mean()
df['wind_rolling_7'] = df['windspeed_10m_max'].rolling(7, min_periods=1).mean()
df['wind_log']       = np.log1p(df['windspeed_10m_max'])

nc = df.select_dtypes(include='number').columns
df[nc] = df[nc].bfill().ffill()

# Define targets and features
targets = ['temperature_2m_mean', 'precipitation_sum', 'windspeed_10m_max']

exclude = targets + ['time', 'name', 'address', 'district', 'keyword',
                     'city_x', 'sunrise', 'sunset']
feature_cols = [c for c in df.columns if c not in exclude]

print(f"Training with {len(feature_cols)} features on {len(df)} rows...")

# Train and save one model per target
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'verbose': -1
}

for target in targets:
    print(f"Training model for: {target}")
    X = df[feature_cols]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, shuffle=False)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    save_path = f'lightgbm_model_{target}.txt'
    model.save_model(save_path)
    print(f"  Saved → {save_path}")

# Save feature columns list
joblib.dump(feature_cols, 'feature_columns.pkl')
print(f"Saved feature_columns.pkl")
print("Done! All models retrained and saved.")