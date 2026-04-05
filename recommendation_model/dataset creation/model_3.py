"""
Dataset Creation: Model 3 (nearby_accommodation.csv)
=====================================================
Fetches nearby services (Hotels, Dining, Activities) for each place.
Input:  data/submodel_1.csv (608 places)
Output: data/nearby_accommodation.csv (up to 15 services per place)

Columns produced:
    place_id, place_name, service_id, service_name, service_type,
    service_latitude, service_longitude, service_avg_rating,
    service_budget_lkr, service_image_url, service_display_review

Search logic:
    1. Nearby Search within 5km radius
    2. If < 5 results, fallback to Text Search in district
    3. Place Details for rating, reviews, price_level
    4. Up to 5 services per category (Hotels, Dining, Activities)

Resume support:
    If nearby_accommodation.csv already exists, skips completed places.
"""

import requests
import pandas as pd
import time
import os

# Configuration
API_KEY = "AIzaSyDPgpt69x2gPCN2K4Lza3PdMPSAHlo3gJ4"

# Paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "submodel_1.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "nearby_accommodation.csv")


def format_lkr_budget(price_level):
    """
    Convert Google price_level (0-4) to LKR budget string.

    0 -> Free
    1 -> Rs. 500 - 2,500
    2 -> Rs. 2,500 - 7,500
    3 -> Rs. 7,500 - 15,000
    4 -> Rs. 15,000+
    None -> Rs. 1,500 - 5,000 (Estimated)
    """
    mapping = {
        0: "Free",
        1: "Rs. 500 - 2,500",
        2: "Rs. 2,500 - 7,500",
        3: "Rs. 7,500 - 15,000",
        4: "Rs. 15,000+"
    }
    return mapping.get(price_level, "Rs. 1,500 - 5,000 (Estimated)")


def safe_request(url, params, retries=3):
    """Handles connection resets by retrying after a short wait."""
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.json()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(f"\n Connection lost. Retrying ({i + 1}/{retries}) in 5 seconds...")
            time.sleep(5)
    return {}


def get_services_for_landmark(row):
    """
    Fetch up to 5 services per category (Hotels, Dining, Activities)
    for a given landmark.

    Search strategy:
        1. Nearby Search (5km radius) for each category
        2. Text Search fallback for categories with < 5 results
        3. Place Details for reviews, rating, price_level

    Returns:
        List of service dictionaries
    """
    lat, lng = row['latitude'], row['longitude']
    categories = {
        "Hotels": {"type": "lodging", "keyword": "hotel"},
        "Dining": {"type": "restaurant", "keyword": "restaurant"},
        "Activities": {"keyword": "rock climbing|hiking|diving|cycling|swimming|surfing|safari"}
    }

    landmark_services = []
    for cat_name, query_data in categories.items():
        params = {
            "location": f"{lat},{lng}",
            "radius": 5000,
            "key": API_KEY,
            **query_data
        }

        # Step 1: Nearby Search
        data = safe_request(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params
        )
        results = data.get("results", [])[:5]

        # Step 2: Fallback to Text Search if < 5 results
        if len(results) < 5:
            text_params = {
                "query": f"Best {cat_name} in {row['district']} district, Sri Lanka",
                "key": API_KEY
            }
            text_data = safe_request(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                text_params
            )
            for p in text_data.get("results", []):
                if p['place_id'] not in [s['place_id'] for s in results]:
                    results.append(p)
                    if len(results) >= 5:
                        break

        # Step 3: Get details for each service
        for r in results[:5]:
            det_params = {
                "place_id": r["place_id"],
                "fields": "reviews,price_level,rating",
                "key": API_KEY
            }
            d = safe_request(
                "https://maps.googleapis.com/maps/api/place/details/json",
                det_params
            ).get("result", {})

            # Review with fallback
            reviews = d.get("reviews", [])
            review_text = (
                reviews[0].get("text", "no reviews yet, be the first to share your experience")
                if reviews
                else "no reviews yet, be the first to share your experience"
            )

            # Image URL
            image_url = ""
            if "photos" in r:
                photo_ref = r['photos'][0]['photo_reference']
                image_url = (
                    f"https://maps.googleapis.com/maps/api/place/photo?"
                    f"maxwidth=400&photoreference={photo_ref}&key={API_KEY}"
                )

            landmark_services.append({
                "place_id": row['place_id'],
                "place_name": row['place_name'],
                "service_id": r["place_id"],
                "service_name": r["name"],
                "service_type": cat_name,
                "service_latitude": r["geometry"]["location"]["lat"],
                "service_longitude": r["geometry"]["location"]["lng"],
                "service_avg_rating": d.get("rating", r.get("rating", 4.0)),
                "service_budget_lkr": format_lkr_budget(d.get("price_level")),
                "service_image_url": image_url,
                "service_display_review": review_text
            })

    return landmark_services


# --- MAIN LOGIC WITH RESUME SUPPORT ---
df = pd.read_csv(INPUT_FILE)
processed_ids = []

# Check if output already exists (for resume)
file_exists = os.path.exists(OUTPUT_FILE)
if file_exists:
    existing_df = pd.read_csv(OUTPUT_FILE)
    processed_ids = existing_df['place_id'].unique().tolist()
    print(f" Resuming: Found {len(processed_ids)} landmarks already completed. Skipping...")

remaining = len(df) - len(processed_ids)
print(f" Processing {remaining} remaining landmarks...")
print(f" Input:  {INPUT_FILE}")
print(f" Output: {OUTPUT_FILE}")

for index, row in df.iterrows():
    if row['place_id'] in processed_ids:
        continue  # Skip already completed landmarks

    print(f"[{index + 1}/{len(df)}] Processing: {row['place_name']}...")
    services = get_services_for_landmark(row)

    if services:
        services_df = pd.DataFrame(services)
        # Append mode: write header only if file doesn't exist yet
        write_header = not os.path.exists(OUTPUT_FILE)
        services_df.to_csv(OUTPUT_FILE, mode='a', index=False, header=write_header)

    time.sleep(0.2)  # Prevent rate limiting

print(f"\n All landmarks completed! Final dataset: {OUTPUT_FILE}")

# Print summary
if os.path.exists(OUTPUT_FILE):
    final_df = pd.read_csv(OUTPUT_FILE)
    print(f"   {len(final_df)} total services")
    print(f"   {final_df['place_id'].nunique()} unique places")
    print(f"   Service types: {final_df['service_type'].value_counts().to_dict()}")
    print(f"   Columns: {list(final_df.columns)}")