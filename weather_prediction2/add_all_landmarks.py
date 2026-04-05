"""
Add 49 Sri Lankan landmark locations to weather_data_lightweight_smart.csv
by borrowing weather data from the nearest weather station for each landmark.

UPDATED: Now includes original 24 + 25 new landmarks = 49 total

Usage:
    python add_all_landmarks_to_weather.py

Input:  weather_data_lightweight_smart.csv
Output: weather_data_lightweight_smart_with_landmarks.csv
"""

import numpy as np
import pandas as pd

INPUT_CSV  = "weather_prediction2/weather_data_lightweight_smart.csv"
OUTPUT_CSV = "weather_data_lightweight_smart_with_landmarks.csv"

# ============================================================================
# LANDMARK COORDINATES - 49 TOTAL (24 ORIGINAL + 25 NEW)
# ============================================================================

LANDMARK_COORDS = {
    # ──────────────────────────────────────────────────────────────────────
    # ORIGINAL 24 LANDMARKS
    # ──────────────────────────────────────────────────────────────────────
    "Adams Peak":                    {"lat": 6.8096, "lng": 80.4994},
    "Ancient City of Polonnaruwa":   {"lat": 7.9403, "lng": 81.0188},
    "Beruwala Light House":          {"lat": 6.4785, "lng": 79.9836},
    "British War Cemetery":          {"lat": 7.2906, "lng": 80.6337},
    "Bundala National Park":         {"lat": 6.1833, "lng": 81.2167},
    "Delft Island":                  {"lat": 9.6500, "lng": 79.6833},
    "Dowa Rock Temple":              {"lat": 6.9667, "lng": 81.0167},
    "Ganagaramaya Temple":           {"lat": 6.9167, "lng": 79.8567},
    "Henarathgoda Botanical Garden": {"lat": 7.0833, "lng": 80.0167},
    "Hortains Plain":                {"lat": 6.8004, "lng": 80.8005},
    "Independance Square":           {"lat": 6.9167, "lng": 79.8667},
    "Jaya Sri Maha Bodhi":           {"lat": 8.3456, "lng": 80.3956},
    "Lotus Tower":                   {"lat": 6.9271, "lng": 79.8612},
    "Maligawa Buddha Statue":        {"lat": 7.2931, "lng": 80.6386},
    "Nine Arches Bridge":            {"lat": 6.8750, "lng": 81.0597},
    "Pinnawala Elephant Orphanage":  {"lat": 7.3000, "lng": 80.3833},
    "Sigiriya":                      {"lat": 7.9572, "lng": 80.7600},
    "Sinharaja Forest":              {"lat": 6.4167, "lng": 80.5000},
    "Sri Dalada Maligawa":           {"lat": 7.2936, "lng": 80.6413},
    "Star Fort":                     {"lat": 5.9449, "lng": 80.5361},
    "Turtle Hatchery":               {"lat": 6.2167, "lng": 80.0500},
    "Vavuniya Archaeological Museum":{"lat": 8.7514, "lng": 80.4972},
    "Wilpattu National Park":        {"lat": 8.4500, "lng": 80.0167},
    "Yapahuwa Rock Fortress":        {"lat": 8.0500, "lng": 80.3333},
    
    # ──────────────────────────────────────────────────────────────────────
    # NEW 25 LANDMARKS
    # ──────────────────────────────────────────────────────────────────────
    "Demodara Nine Arch Bridge":     {"lat": 6.8750, "lng": 81.0597},  # Same as original Nine Arches
    "British Garrison Cemetery":     {"lat": 7.2906, "lng": 80.6337},  # Same as British War Cemetery
    "Bundala Ancient Shell Middens": {"lat": 6.1833, "lng": 81.2167},  # Near Bundala NP
    "Gangaramaya Temple":            {"lat": 6.9167, "lng": 79.8567},  # Same as Ganagaramaya
    "Embekke Devalaya":              {"lat": 7.2667, "lng": 80.5833},
    "Independence Memorial Hall":    {"lat": 6.9167, "lng": 79.8667},  # Same as Independence Square
    "Henaratgoda Botanical Garden":  {"lat": 7.0833, "lng": 80.0167},  # Same spelling variant
    "Sripada (Adam's Peak)":         {"lat": 6.8096, "lng": 80.4994},  # Same as Adams Peak
    "Jaffna Fort":                   {"lat": 9.6620, "lng": 80.0070},
    "Seetha Amman Kovil":            {"lat": 6.9500, "lng": 80.7667},
    "Sri Bhakta Hanuman Temple":     {"lat": 7.1167, "lng": 80.7333},
    "Old Dutch Hospital":            {"lat": 6.9354, "lng": 79.8474},
    "Iranamadu Tank":                {"lat": 9.5167, "lng": 80.4167},
    "Pallekele Archaeological Site": {"lat": 7.3167, "lng": 80.6167},
    "Kataragama Devalaya":           {"lat": 6.4133, "lng": 81.3342},
    "Tissamaharama Raja Maha Vihara":{"lat": 6.2833, "lng": 81.2833},
    "Martin Wickramasinghe Museum":  {"lat": 5.9833, "lng": 80.5500},
    "Dambulla Cave Temple":          {"lat": 7.8567, "lng": 80.6489},
    "Jaffna Public Library":         {"lat": 9.6615, "lng": 80.0255},
}

# ============================================================================
# LOCATION MAPPING - 49 TOTAL
# ============================================================================

LOCATION_MAP = {
    # ──────────────────────────────────────────────────────────────────────
    # ORIGINAL 24 LANDMARKS
    # ──────────────────────────────────────────────────────────────────────
    "Adams Peak":                    "Rathnapura, Sabaragamuwa Province, Sri Lanka",
    "Ancient City of Polonnaruwa":   "Polonnaruwa, North Central Province, Sri Lanka",
    "Beruwala Light House":          "Beruwala, Western Province, Sri Lanka",
    "British War Cemetery":          "Kandy, Central Province, Sri Lanka",
    "Bundala National Park":         "Hambantota, Southern Province, Sri Lanka",
    "Delft Island":                  "Jaffna, Northern Province, Sri Lanka",
    "Dowa Rock Temple":              "Bandarawela, Uva Province, Sri Lanka",
    "Ganagaramaya Temple":           "Colombo, Western Province, Sri Lanka",
    "Henarathgoda Botanical Garden": "Gampaha, Western Province, Sri Lanka",
    "Hortains Plain":                "Nuwara Eliya, Central Province, Sri Lanka",
    "Independance Square":           "Colombo, Western Province, Sri Lanka",
    "Jaya Sri Maha Bodhi":           "Anuradhapura, North Central Province, Sri Lanka",
    "Lotus Tower":                   "Colombo, Western Province, Sri Lanka",
    "Maligawa Buddha Statue":        "Kandy, Central Province, Sri Lanka",
    "Nine Arches Bridge":            "Ella, Uva Province, Sri Lanka",
    "Pinnawala Elephant Orphanage":  "Kegalle, Sabaragamuwa Province, Sri Lanka",
    "Sigiriya":                      "Matale, Central Province, Sri Lanka",
    "Sinharaja Forest":              "Ratnapura, Sabaragamuwa Province, Sri Lanka",
    "Sri Dalada Maligawa":           "Kandy, Central Province, Sri Lanka",
    "Star Fort":                     "Matara, Southern Province, Sri Lanka",
    "Turtle Hatchery":               "Kosgoda, Southern Province, Sri Lanka",
    "Vavuniya Archaeological Museum":"Vavuniya, Northern Province, Sri Lanka",
    "Wilpattu National Park":        "Puttalam, North Western Province, Sri Lanka",
    "Yapahuwa Rock Fortress":        "Yapahuwa, North Western Province, Sri Lanka",
    
    # ──────────────────────────────────────────────────────────────────────
    # NEW 25 LANDMARKS
    # ──────────────────────────────────────────────────────────────────────
    "Demodara Nine Arch Bridge":     "Ella, Uva Province, Sri Lanka",
    "British Garrison Cemetery":     "Kandy, Central Province, Sri Lanka",
    "Bundala Ancient Shell Middens": "Hambantota, Southern Province, Sri Lanka",
    "Gangaramaya Temple":            "Colombo, Western Province, Sri Lanka",
    "Embekke Devalaya":              "Kandy, Central Province, Sri Lanka",
    "Independence Memorial Hall":    "Colombo, Western Province, Sri Lanka",
    "Henaratgoda Botanical Garden":  "Gampaha, Western Province, Sri Lanka",
    "Sripada (Adam's Peak)":         "Rathnapura, Sabaragamuwa Province, Sri Lanka",
    "Jaffna Fort":                   "Jaffna, Northern Province, Sri Lanka",
    "Seetha Amman Kovil":            "Nuwara Eliya, Central Province, Sri Lanka",
    "Sri Bhakta Hanuman Temple":     "Nuwara Eliya, Central Province, Sri Lanka",
    "Old Dutch Hospital":            "Colombo, Western Province, Sri Lanka",
    "Iranamadu Tank":                "Kilinochchi, Northern Province, Sri Lanka",
    "Pallekele Archaeological Site": "Kandy, Central Province, Sri Lanka",
    "Kataragama Devalaya":           "Kataragama, Uva Province, Sri Lanka",
    "Tissamaharama Raja Maha Vihara":"Hambantota, Southern Province, Sri Lanka",
    "Martin Wickramasinghe Museum":  "Matara, Southern Province, Sri Lanka",
    "Dambulla Cave Temple":          "Matale, Central Province, Sri Lanka",
    "Jaffna Public Library":         "Jaffna, Northern Province, Sri Lanka",
}


def haversine(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def find_nearest_station(landmark_lat, landmark_lng, stations_df):
    """Return the row in stations_df closest to the given coordinates."""
    dists = haversine(
        landmark_lat, landmark_lng,
        stations_df["station_lat"].values,
        stations_df["station_lng"].values,
    )
    idx = dists.argmin()
    return stations_df.iloc[idx], dists[idx]


# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 70)
print("ADDING 49 LANDMARKS TO LIGHTWEIGHT SMART CSV")
print("=" * 70)

print(f"\n1. Loading {INPUT_CSV} ...")
df = pd.read_csv(INPUT_CSV)
df["time"] = pd.to_datetime(df["time"])
print(f"   Loaded {len(df):,} rows | {df['name'].nunique()} unique locations")


# One row per unique station (station_lat / station_lng pair)
stations = (
    df[["station_lat", "station_lng", "city_x"]]
    .drop_duplicates()
    .reset_index(drop=True)
)
print(f"   Found {len(stations)} unique weather stations")


existing_names = set(df["name"].unique())
landmarks_to_add = {k: v for k, v in LANDMARK_COORDS.items() if k not in existing_names}
already_present  = [k for k in LANDMARK_COORDS if k in existing_names]

if already_present:
    print(f"\n   Already in dataset (skipping {len(already_present)}):")
    for name in sorted(already_present)[:10]:  # Show first 10
        print(f"     - {name}")
    if len(already_present) > 10:
        print(f"     ... and {len(already_present) - 10} more")

print(f"\n2. Landmarks to inject: {len(landmarks_to_add)}/{len(LANDMARK_COORDS)}")
print(f"   Total landmarks in mapping: {len(LANDMARK_COORDS)}")


new_chunks = []

for i, (landmark, coords) in enumerate(landmarks_to_add.items(), 1):
    nearest_station, dist_km = find_nearest_station(
        coords["lat"], coords["lng"], stations
    )

    # Pull all rows for that station from the smart CSV
    mask = (
        (df["station_lat"] == nearest_station["station_lat"]) &
        (df["station_lng"] == nearest_station["station_lng"])
    )
    station_rows = df[mask].copy()

    if len(station_rows) == 0:
        print(f"   ⚠  [{i:2d}] No rows for station {nearest_station['city_x']} — skipping {landmark}")
        continue

    # Overwrite location-specific columns with landmark identity
    station_rows["name"]         = landmark
    station_rows["address"]      = LOCATION_MAP[landmark]
    station_rows["district"]     = LOCATION_MAP[landmark].split(",")[0].strip()
    station_rows["keyword"]      = "heritage site"
    station_rows["location_lat"] = coords["lat"]
    station_rows["location_lng"] = coords["lng"]
    station_rows["distance_km"]  = dist_km

    new_chunks.append(station_rows)
    print(f"   ✓  [{i:2d}] {landmark:<42} → {nearest_station['city_x']:<15} ({dist_km:5.1f} km) | {len(station_rows):>4} rows")


if new_chunks:
    new_df      = pd.concat(new_chunks, ignore_index=True)
    df_combined = pd.concat([df, new_df], ignore_index=True)
else:
    print("\n   Nothing to add — all landmarks were already present.")
    df_combined = df

print(f"\n3. Row counts")
print(f"   Original    : {len(df):>10,}")
print(f"   New added   : {len(df_combined) - len(df):>10,}")
print(f"   Combined    : {len(df_combined):>10,}")
print(f"   Locations   : {df_combined['name'].nunique():>10,}")

# Show landmark breakdown
original_locs = len([x for x in df_combined['name'].unique() if x not in LANDMARK_COORDS])
landmark_locs = len([x for x in df_combined['name'].unique() if x in LANDMARK_COORDS])
print(f"   - Original locations : {original_locs:>5,}")
print(f"   - Landmark locations : {landmark_locs:>5,}")

print(f"\n4. Saving → {OUTPUT_CSV} ...")
df_combined.to_csv(OUTPUT_CSV, index=False)
print(f"   ✓ Done — {OUTPUT_CSV}")
print("=" * 70)
print("\n📊 SUMMARY:")
print(f"   Total landmarks added to dataset: {landmark_locs}")
print(f"   Total locations in final dataset: {df_combined['name'].nunique()}")
print("=" * 70)