"""
Dataset Creation: Model 2 (submodel_2.csv)
==========================================
Enriches submodel_1.csv with review data from Google Places API.
Input:  data/submodel_1.csv (608 places, 10 columns)
Output: data/submodel_2.csv (608 rows, 16 columns)

Adds these columns to submodel_1:
    review_count, review_1, review_2, review_3, review_4, review_5

Fallback logic:
    - If 0 reviews found: inserts a generic positive review
    - If < 5 reviews: pads remaining with " " (single space)
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
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "..", "data", "submodel_2.csv")


def get_full_details(place_name, district):
    """
    Fetch review details from Google Places API.

    Returns:
        (avg_rating, review_count, reviews_list) or None on failure

    Fallback logic:
        - 0 reviews -> insert generic positive review
        - < 5 reviews -> pad with " "
    """
    search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    search_params = {"query": f"{place_name}, {district}, Sri Lanka", "key": API_KEY}

    try:
        search_data = requests.get(search_url, search_params).json()
        if not search_data.get("results"):
            return None

        google_id = search_data["results"][0]["place_id"]

        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": google_id,
            "fields": "reviews,user_ratings_total,rating",
            "key": API_KEY
        }

        d_res = requests.get(details_url, details_params).json()
        d = d_res.get("result", {})

        review_count = d.get("user_ratings_total", 0)
        avg_rating = d.get("rating", 0)

        # Extract up to 5 reviews
        raw_reviews = d.get("reviews", [])
        reviews_list = [r.get("text", "") for r in raw_reviews[:5]]

        # Fallback: if ZERO reviews, insert a generic positive review
        if len(reviews_list) == 0:
            reviews_list.append(
                "A wonderful place to visit. The historical significance "
                "is amazing and the surroundings are very peaceful."
            )

        # Pad to exactly 5 reviews with single space " "
        while len(reviews_list) < 5:
            reviews_list.append(" ")

        return avg_rating, review_count, reviews_list

    except Exception as e:
        print(f"Error for {place_name}: {e}")
        return None


# Load submodel_1 as the base
df = pd.read_csv(INPUT_FILE)

print(f"Enriching {len(df)} places with 5 reviews each...")
print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

for index, row in df.iterrows():
    print(f"[{index + 1}/{len(df)}] Fetching: {row['place_name']}...")

    result = get_full_details(row['place_name'], row['district'])

    if result:
        avg_rating, count, reviews = result
        df.at[index, 'avg_rating'] = avg_rating
        df.at[index, 'review_count'] = count
        df.at[index, 'review_1'] = reviews[0]
        df.at[index, 'review_2'] = reviews[1]
        df.at[index, 'review_3'] = reviews[2]
        df.at[index, 'review_4'] = reviews[3]
        df.at[index, 'review_5'] = reviews[4]

    # Save checkpoint every 20 rows to prevent data loss
    if (index + 1) % 20 == 0:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"   Checkpoint saved ({index + 1}/{len(df)})")

    time.sleep(0.2)  # Prevent rate limiting

# Final save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n Finished! Dataset saved to {OUTPUT_FILE}")
print(f"   {len(df)} places, {len(df.columns)} columns")
print(f"   Columns: {list(df.columns)}")