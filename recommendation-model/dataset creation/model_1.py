"""
Dataset Creation: Model 1 (submodel_1.csv)
==========================================
Creates the base places dataset from Google Places API.
Input:  places_input.csv (608 places with place_name, district)
Output: data/submodel_1.csv (608 rows, 10 columns)

Columns produced:
    place_id, place_name, district, latitude, longitude,
    place_type, description, display_review, image_url, avg_rating
"""

import requests
import pandas as pd
import uuid
import os

API_KEY = "AIzaSyDPgpt69x2gPCN2K4Lza3PdMPSAHlo3gJ4"

# Paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "places_input.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "submodel_1.csv")


def geocode_place(place_name):
    """
    Fetch place details from Google Places API.

    Returns:
        lat, lng, place_type, description, display_review, image_url, avg_rating
    """
    # Step 1: Text Search to get place_id and basic info
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": place_name + ", Sri Lanka", "key": API_KEY}
    r = requests.get(url, params=params).json()["results"][0]

    lat = r["geometry"]["location"]["lat"]
    lng = r["geometry"]["location"]["lng"]
    place_id = r["place_id"]
    place_type = ", ".join(r.get("types", []))
    avg_rating = r.get("rating", 0)  # Get avg_rating from search results
    image_url = None

    if "photos" in r:
        photo_ref = r["photos"][0]["photo_reference"]
        image_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=800&photoreference={photo_ref}&key={API_KEY}"

    # Step 2: Place Details to get description and review
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    details_params = {
        "place_id": place_id,
        "fields": "editorial_summary,reviews,rating",
        "key": API_KEY
    }
    d = requests.get(details_url, params=details_params).json()["result"]

    description = d.get("editorial_summary", {}).get("overview", "")
    display_review = d.get("reviews", [{}])[0].get("text", "")

    # Use the more accurate rating from details if available
    if "rating" in d:
        avg_rating = d["rating"]

    return lat, lng, place_type, description, display_review, image_url, avg_rating


# Load input places
df = pd.read_csv(INPUT_FILE)
print(f"Processing {len(df)} places from {INPUT_FILE}...")

rows = []

for index, p in df.iterrows():
    print(f"[{index + 1}/{len(df)}] Fetching: {p.place_name}...")
    lat, lng, ptype, desc, review, img, rating = geocode_place(p.place_name)

    rows.append({
        "place_id": str(uuid.uuid4()),
        "place_name": p.place_name,
        "district": p.district,
        "latitude": lat,
        "longitude": lng,
        "place_type": ptype,
        "description": desc,
        "display_review": review,
        "image_url": img,
        "avg_rating": rating
    })

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

result_df = pd.DataFrame(rows)
result_df.to_csv(OUTPUT_FILE, index=False)
print(f"\n Model 1 dataset created: {OUTPUT_FILE}")
print(f"   {len(result_df)} places, {len(result_df.columns)} columns")
print(f"   Columns: {list(result_df.columns)}")
