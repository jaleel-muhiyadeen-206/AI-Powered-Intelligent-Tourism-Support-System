"""
Test script for all recommendation models.
Loads pre-trained models and generates sample recommendations for Sigiriya.
"""

import os
import pandas as pd
import joblib
from models.model_1_you_may_also_like import Model1_YouMayAlsoLike
from models.model_2_popular_nearby import Model2_PopularNearby
from models.model_3_nearby_essentials import Model3_NearbyEssentials


def print_recs(title, df):
    print(f"\n  {title}")
    print("  " + "-" * 70)
    if df.empty:
        print("  No recommendations found within distance limit.")
        return
    for _, row in df.iterrows():
        print(f"\n  RANK {row['rank']}")
        print(f"    Name:     {row['name']}")
        if 'district' in row:
            print(f"    District: {row['district']}")
        if 'type' in row:
            print(f"    Type:     {row['type']}")
        print(f"    Rating:   {row.get('rating', 'N/A')}/5")
        print(f"    Distance: {row['distance_km']} km")
        if 'review' in row and str(row['review']).strip():
            print(f"    Review:   {str(row['review'])[:150]}...")
        print(f"    Score:    {row['final_score']}")


def main():
    required = [
        'output/preprocessed/submodel_1.joblib',
        'output/trained_models/model_1.joblib',
        'output/trained_models/model_2.joblib',
        'output/trained_models/model_3.joblib',
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("Missing required files:")
        for f in missing:
            print(f"  {f}")
        print("\nRun preprocessing.py and model training scripts first.")
        return

    print("=" * 70)
    print("LOADING SAVED MODELS")
    print("=" * 70)

    sub1 = joblib.load('output/preprocessed/submodel_1.joblib')

    model1 = Model1_YouMayAlsoLike(sub1)
    model1.load('output/trained_models/model_1.joblib')

    model2 = Model2_PopularNearby(pd.DataFrame())
    model2.load('output/trained_models/model_2.joblib')

    model3 = Model3_NearbyEssentials(pd.DataFrame(), pd.DataFrame())
    model3.load('output/trained_models/model_3.joblib')

    search_term = "Sigiriya"
    matches = sub1[sub1['place_name'].str.contains(search_term, case=False, na=False)]

    if matches.empty:
        print(f"\nNo place matching '{search_term}' found.")
        return

    place = matches.iloc[0]
    pid = place['place_id']
    name = place['place_name']
    dist = place['district']

    print(f"\n" + "=" * 70)
    print(f"TESTING RECOMMENDATIONS")
    print(f"  Query: {name} ({dist})")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("MODEL 1 — You May Also Like")
    print("=" * 70)
    recs1 = model1.recommend(pid, top_n=5)
    print_recs("Similar places nearby", recs1)

    print("\n" + "=" * 70)
    print("MODEL 2 — Popular Places Nearby")
    print("=" * 70)
    recs2 = model2.recommend(pid, top_n=5)
    print_recs("Popular places nearby", recs2)

    for stype in ['Hotels', 'Dining', 'Activities']:
        print(f"\n" + "=" * 70)
        print(f"MODEL 3 — Nearby {stype}")
        print("=" * 70)
        recs3 = model3.recommend(pid, service_type=stype, top_n=5)
        print_recs(f"Nearby {stype}", recs3)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()