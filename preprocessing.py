"""
Data Preprocessing Module
==========================
Processes datasets for the Sri Lanka tourism recommendation system.

Components:
    1. Submodel 1: 608 places with descriptions, coordinates, districts, types
    2. Submodel 2: 608 places with 5 reviews each, ratings, review counts
    3. Nearby Accommodation: 3,870 services with ratings, budget, reviews
    4. Kaggle Review Enrichment: 35,434 reviews for 236 destinations

Processing Pipeline:
    - District coordinate validation and correction
    - Missing value imputation
    - BERT sentiment analysis (nlptown/bert-base-multilingual-uncased-sentiment)
    - Kaggle review matching via fuzzy string matching
    - BERT review selection for best positive display review
    - Numeric budget score extraction
    - Outlier clipping and popularity scoring
"""

import pandas as pd  # Data structures
import numpy as np  # Mathematical routines
import torch  # GPU tensor computation
import joblib  # Model artifact persistence
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from difflib import SequenceMatcher  # Fuzzy string heuristics
import warnings  # Diagnostic noise reduction
import os  # Underlying OS integration paths

warnings.filterwarnings('ignore')  # Ensure sterile output logs

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # Anchor origin
# BERT offline persistent store
LOCAL_BERT_DIR = os.path.join(PROJECT_DIR, 'models', 'bert_sentiment')
BERT_CACHE_DIR = os.path.join(PROJECT_DIR, 'models', 'bert_cache')
BERT_MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"

DISTRICT_CENTROIDS = {
    'Colombo':       (6.9271, 79.8612),
    'Gampaha':       (7.0840, 80.0144),
    'Kalutara':      (6.5854, 80.1117),
    'Kandy':         (7.2906, 80.6337),
    'Matale':        (7.4675, 80.6234),
    'Nuwara Eliya':  (6.9497, 80.7891),
    'Galle':         (6.0535, 80.2210),
    'Matara':        (5.9549, 80.5550),
    'Hambantota':    (6.1241, 81.1185),
    'Jaffna':        (9.6615, 80.0255),
    'Kilinochchi':   (9.3803, 80.3770),
    'Mannar':        (8.9810, 79.9044),
    'Vavuniya':      (8.7514, 80.4971),
    'Mullaitivu':    (9.2671, 80.8142),
    'Batticaloa':    (7.7310, 81.6747),
    'Ampara':        (7.2916, 81.6724),
    'Trincomalee':   (8.5874, 81.2152),
    'Kurunegala':    (7.4863, 80.3623),
    'Puttalam':      (8.0362, 79.8283),
    'Anuradhapura':  (8.3114, 80.4037),
    'Polonnaruwa':   (7.9403, 81.0188),
    'Badulla':       (6.9934, 81.0550),
    'Monaragala':    (6.8728, 81.3507),
    'Ratnapura':     (6.6828, 80.3992),
    'Kegalle':       (7.2513, 80.3464),
}

DISTRICT_ADJACENCY = {
    'Colombo':       {'Gampaha', 'Kalutara', 'Kegalle', 'Ratnapura'},
    'Gampaha':       {'Colombo', 'Kalutara', 'Kegalle', 'Kurunegala', 'Puttalam'},
    'Kalutara':      {'Colombo', 'Gampaha', 'Ratnapura', 'Galle'},
    'Kandy':         {'Matale', 'Nuwara Eliya', 'Kegalle', 'Badulla', 'Kurunegala'},
    'Matale':        {'Kandy', 'Kurunegala', 'Anuradhapura', 'Polonnaruwa', 'Nuwara Eliya', 'Badulla'},
    'Nuwara Eliya':  {'Kandy', 'Matale', 'Badulla', 'Ratnapura', 'Kegalle', 'Monaragala'},
    'Galle':         {'Kalutara', 'Matara', 'Ratnapura', 'Hambantota'},
    'Matara':        {'Galle', 'Hambantota'},
    'Hambantota':    {'Matara', 'Galle', 'Ratnapura', 'Monaragala', 'Badulla', 'Ampara'},
    'Jaffna':        {'Kilinochchi'},
    'Kilinochchi':   {'Jaffna', 'Mullaitivu', 'Mannar', 'Vavuniya'},
    'Mannar':        {'Kilinochchi', 'Vavuniya', 'Puttalam', 'Anuradhapura'},
    'Vavuniya':      {'Kilinochchi', 'Mullaitivu', 'Mannar', 'Anuradhapura'},
    'Mullaitivu':    {'Kilinochchi', 'Vavuniya', 'Trincomalee', 'Anuradhapura'},
    'Batticaloa':    {'Trincomalee', 'Ampara', 'Polonnaruwa'},
    'Ampara':        {'Batticaloa', 'Monaragala', 'Badulla', 'Hambantota', 'Polonnaruwa', 'Trincomalee'},
    'Trincomalee':   {'Mullaitivu', 'Batticaloa', 'Polonnaruwa', 'Anuradhapura', 'Ampara'},
    'Kurunegala':    {'Gampaha', 'Kandy', 'Matale', 'Anuradhapura', 'Puttalam', 'Kegalle'},
    'Puttalam':      {'Gampaha', 'Kurunegala', 'Anuradhapura', 'Mannar'},
    'Anuradhapura':  {'Kurunegala', 'Matale', 'Polonnaruwa', 'Puttalam', 'Vavuniya', 'Mullaitivu', 'Trincomalee', 'Mannar'},
    'Polonnaruwa':   {'Matale', 'Anuradhapura', 'Trincomalee', 'Batticaloa', 'Badulla', 'Ampara', 'Monaragala'},
    'Badulla':       {'Kandy', 'Matale', 'Nuwara Eliya', 'Monaragala', 'Ampara', 'Polonnaruwa', 'Hambantota'},
    'Monaragala':    {'Badulla', 'Ampara', 'Hambantota', 'Ratnapura', 'Nuwara Eliya', 'Polonnaruwa'},
    'Ratnapura':     {'Colombo', 'Kalutara', 'Kegalle', 'Nuwara Eliya', 'Galle', 'Hambantota', 'Monaragala'},
    'Kegalle':       {'Colombo', 'Gampaha', 'Kandy', 'Kurunegala', 'Ratnapura', 'Nuwara Eliya'},
}


def are_districts_adjacent(d1: str, d2: str) -> bool:
    if d1 == d2:
        return True
    return d2 in DISTRICT_ADJACENCY.get(d1, set())


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _is_local_model_ready(local_dir: str) -> bool:
    if not os.path.isdir(local_dir):
        return False
    required = ['config.json', 'tokenizer_config.json']
    for f in required:
        if not os.path.exists(os.path.join(local_dir, f)):
            return False
    return any(f.endswith(('.bin', '.safetensors')) for f in os.listdir(local_dir))


def correct_coordinates(df, lat_col: str='latitude', lon_col: str='longitude',
                        district_col='district', threshold_km=80):
    corrected = 0
    df = df.copy()
    for idx, row in df.iterrows():
        district = row[district_col]
        if district not in DISTRICT_CENTROIDS:
            continue
        centroid_lat, centroid_lon = DISTRICT_CENTROIDS[district]
        dist_km = _haversine(row[lat_col], row[lon_col], centroid_lat, centroid_lon)
        if dist_km > threshold_km:
            df.at[idx, lat_col] = centroid_lat
            df.at[idx, lon_col] = centroid_lon
            corrected += 1
    if corrected > 0:
        print(f"  Corrected {corrected} coordinate mismatches (>{threshold_km} km from centroid)")
    return df


class BERTSentimentAnalyser:
    """
    Sentiment analysis using nlptown/bert-base-multilingual-uncased-sentiment.
    Outputs star ratings (1-5) normalised to [-1, 1].
    Loads from local saved model if available, otherwise downloads once and saves locally.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if _is_local_model_ready(LOCAL_BERT_DIR):
            print("  Loading BERT sentiment model from local files...")
            self.tokenizer = AutoTokenizer.from_pretrained(LOCAL_BERT_DIR, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(LOCAL_BERT_DIR, local_files_only=True)
        else:
            hf_token = os.environ.get("HF_TOKEN", "")
            if not hf_token:
                raise RuntimeError(
                    "BERT model not found locally and HF_TOKEN environment variable is not set.\n"
                    "Either run: python download_bert_model.py\n"
                    "Or set: export HF_TOKEN=your_huggingface_token"
                )
            print("  Downloading BERT sentiment model (one-time)...")
            os.makedirs(BERT_CACHE_DIR, exist_ok=True)
            os.makedirs(LOCAL_BERT_DIR, exist_ok=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                BERT_MODEL_NAME, cache_dir=BERT_CACHE_DIR, token=hf_token
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                BERT_MODEL_NAME, cache_dir=BERT_CACHE_DIR, token=hf_token
            )
            self.tokenizer.save_pretrained(LOCAL_BERT_DIR)
            self.model.save_pretrained(LOCAL_BERT_DIR)
            print(f"  Model saved to {LOCAL_BERT_DIR} for offline use")

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        print("  BERT sentiment model ready")

    def analyse(self, text):
        if not self._loaded:
            self.load()
        if pd.isna(text) or str(text).strip() == "":
            return 0.0
        text = str(text)[:512]
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            star_weights = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).to(self.device)
            weighted_score = (probs * star_weights).sum().item()
            return round((weighted_score - 3) / 2, 4)
        except Exception:
            return 0.0

    def analyse_batch(self, texts, batch_size=32):
        if not self._loaded:
            self.load()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            cleaned = []
            for t in batch:
                if pd.isna(t) or str(t).strip() == "":
                    cleaned.append("")
                else:
                    cleaned.append(str(t)[:512])
            non_empty_indices = [j for j, t in enumerate(cleaned) if t]
            non_empty_texts = [cleaned[j] for j in non_empty_indices]
            batch_scores = [0.0] * len(cleaned)
            if non_empty_texts:
                try:
                    inputs = self.tokenizer(
                        non_empty_texts, return_tensors="pt", truncation=True,
                        max_length=512, padding=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    star_weights = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).to(self.device)
                    for idx_pos, j in enumerate(non_empty_indices):
                        weighted = (probs[idx_pos] * star_weights).sum().item()
                        batch_scores[j] = round((weighted - 3) / 2, 4)
                except Exception:
                    pass
            results.extend(batch_scores)
        return results


class KaggleReviewEnricher:
    """Coordinates integration heuristics for global datasets."""
    """Enriches the dataset with 35,434 real tourist reviews from Kaggle via fuzzy matching."""

    def __init__(self, kaggle_csv_path):
        self.kaggle_path = kaggle_csv_path
        self.kaggle_df = None
        self.match_map = {}
        self.loaded = False

    def load(self):
        if not os.path.exists(self.kaggle_path):
            print(f"  Kaggle CSV not found at {self.kaggle_path}")
            return
        self.kaggle_df = pd.read_csv(self.kaggle_path)
        self.loaded = True
        print(f"  Loaded {len(self.kaggle_df)} Kaggle reviews for "
              f"{self.kaggle_df['Destination'].nunique()} destinations")

    def match_places(self, place_names: list, threshold: float=0.70):
        if not self.loaded:
            return {}
        kaggle_names = list(self.kaggle_df['Destination'].unique())
        self.match_map = {}
        for pn in place_names:
            best_ratio, best_match = 0, None
            pn_lower = pn.lower().strip()
            for kn in kaggle_names:
                ratio = SequenceMatcher(None, pn_lower, kn.lower().strip()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = kn
            if best_ratio >= threshold:
                self.match_map[pn] = best_match
        print(f"  Matched {len(self.match_map)} places to Kaggle destinations (threshold={threshold})")
        return self.match_map

    def get_reviews_for_place(self, kaggle_destination_name):
        if not self.loaded or self.kaggle_df is None:
            return []
        return self.kaggle_df[
            self.kaggle_df['Destination'] == kaggle_destination_name
        ]['Review'].dropna().tolist()

    def get_review_count_for_place(self, place_name):
        if place_name not in self.match_map:
            return 0
        return len(self.get_reviews_for_place(self.match_map[place_name]))


class BERTReviewSelector:
    """Calculates peak narrative representations utilizing sequence tensors."""
    """Selects the most positive and informative review using BERT sentiment scoring."""

    def __init__(self, sentiment_analyser):
        self.analyser = sentiment_analyser

    def select_best_review(self, reviews: list) -> str:
        default = "A hidden gem waiting to be explored."
        if not reviews:
            return default
        valid = [r for r in reviews if r and str(r).strip() and len(str(r).strip()) > 10]
        if not valid:
            return reviews[0] if reviews else default
        scored = []
        for review in valid:
            sentiment = self.analyser.analyse(review)
            length_bonus = min(len(str(review)) / 500.0, 0.3)
            scored.append((review, sentiment + length_bonus))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]


def extract_primary_type(place_type_str: str) -> str:
    if pd.isna(place_type_str):
        return 'General Attraction'
    types = [t.strip() for t in str(place_type_str).split(',')]
    priority_keywords = {
        'temple': 'Temple', 'church': 'Church', 'mosque': 'Mosque',
        'place_of_worship': 'Religious Site', 'museum': 'Museum',
        'park': 'Park', 'beach': 'Beach', 'zoo': 'Zoo',
        'landmark': 'Landmark', 'natural_feature': 'Natural Site'
    }
    for ptype in types:
        for keyword, category in priority_keywords.items():
            if keyword in ptype.lower():
                return category
    return types[0].replace('_', ' ').title() if types else 'General Attraction'


DEFAULT_REVIEW = "A hidden gem waiting to be explored."  # Global fallback text
DEFAULT_SERVICE_REVIEW = "This service is awaiting its first detailed review."


def preprocess_submodel_1(df_raw, sentiment_analyser, kaggle_enricher=None,
                          review_selector=None):
    print("\n[INFO] [1/3] Preprocessing Submodel 1 pipeline...")
    df = df_raw.copy()
    df = correct_coordinates(df)

    df['display_review'] = df['display_review'].fillna(DEFAULT_REVIEW)
    df['description'] = df['description'].fillna('')
    df['avg_rating'] = df['avg_rating'].fillna(0.0)
    df['place_category'] = df['place_type'].apply(extract_primary_type)
    df['combined_features'] = df['place_category'] + ' ' + df['description'] + ' ' + df['district']

    print("  Computing BERT sentiment scores...")
    df['sentiment_score'] = sentiment_analyser.analyse_batch(df['display_review'].tolist(), batch_size=16)

    if kaggle_enricher is not None and kaggle_enricher.loaded:
        print("  Enriching reviews with Kaggle data...")
        kaggle_enricher.match_places(df['place_name'].tolist())
        kaggle_review_counts, enriched_reviews = [], []
        for _, row in df.iterrows():
            pname = row['place_name']
            if pname in kaggle_enricher.match_map:
                kaggle_revs = kaggle_enricher.get_reviews_for_place(kaggle_enricher.match_map[pname])
                kaggle_review_counts.append(len(kaggle_revs))
                all_reviews = [row['display_review']] + kaggle_revs
                if review_selector is not None:
                    enriched_reviews.append(review_selector.select_best_review(all_reviews[:20]))
                else:
                    enriched_reviews.append(row['display_review'])
            else:
                kaggle_review_counts.append(0)
                enriched_reviews.append(row['display_review'])
        df['kaggle_review_count'] = kaggle_review_counts
        df['display_review'] = enriched_reviews
        enriched = sum(1 for c in kaggle_review_counts if c > 0)
        print(f"  Enriched {enriched} places with {sum(kaggle_review_counts)} Kaggle reviews")
    else:
        df['kaggle_review_count'] = 0

    coord_counts = df.groupby(['latitude', 'longitude']).agg(
        n_districts=('district', 'nunique'),
        count=('place_id', 'count'),
        districts=('district', lambda x: list(x.unique()))
    ).reset_index()
    bad_coords = coord_counts[(coord_counts['count'] > 1) & (coord_counts['n_districts'] > 1)]
    df['has_shared_coords'] = False
    for _, bc in bad_coords.iterrows():
        mask = (df['latitude'] == bc['latitude']) & (df['longitude'] == bc['longitude'])
        df.loc[mask, 'has_shared_coords'] = True

    print(f"  Processed {len(df)} places")
    return df


def preprocess_submodel_2(df_raw, sentiment_analyser, kaggle_enricher=None,
                          review_selector=None):
    print("\n[2/3] Preprocessing Submodel 2...")
    df = df_raw.copy()
    df = correct_coordinates(df)

    zero_count = len(df[df['avg_rating'] == 0])
    if zero_count > 0:
        df.loc[df['avg_rating'] == 0, 'avg_rating'] = 0.0
        print(f"  Marked {zero_count} zero ratings")

    df['display_review'] = df['display_review'].fillna(DEFAULT_REVIEW)
    df['review_count'] = df['review_count'].fillna(0)
    df['description'] = df['description'].fillna('')

    review_cols = ['review_1', 'review_2', 'review_3', 'review_4', 'review_5']
    for col in review_cols:
        df[col] = df[col].fillna("")

    df['place_category'] = df['place_type'].apply(extract_primary_type)

    print("  Computing BERT sentiment for all review columns...")
    df['sentiment_score'] = sentiment_analyser.analyse_batch(df['display_review'].tolist(), batch_size=16)

    for col in review_cols:
        print(f"    Processing {col}...")
        df[f'{col}_sentiment'] = sentiment_analyser.analyse_batch(df[col].tolist(), batch_size=16)

    sentiment_cols = [f'{col}_sentiment' for col in review_cols]
    df['avg_review_sentiment'] = df[sentiment_cols].mean(axis=1)
    df['non_empty_reviews'] = df[review_cols].apply(
        lambda row: sum(1 for x in row if str(x).strip() != ""), axis=1
    )

    if kaggle_enricher is not None and kaggle_enricher.loaded:
        print("  Enriching with Kaggle reviews...")
        if not kaggle_enricher.match_map:
            kaggle_enricher.match_places(df['place_name'].tolist())
        kaggle_counts = []
        for idx, row in df.iterrows():
            pname = row['place_name']
            if pname in kaggle_enricher.match_map:
                kaggle_revs = kaggle_enricher.get_reviews_for_place(kaggle_enricher.match_map[pname])
                kaggle_counts.append(len(kaggle_revs))
                if kaggle_revs:
                    kaggle_sents = sentiment_analyser.analyse_batch(kaggle_revs[:50], batch_size=16)
                    avg_kaggle_sent = np.mean(kaggle_sents)
                    original_sent = row['avg_review_sentiment']
                    df.at[idx, 'avg_review_sentiment'] = original_sent * 0.4 + avg_kaggle_sent * 0.6
                    all_revs = [row['display_review']]
                    for rc in review_cols:
                        if str(row[rc]).strip():
                            all_revs.append(str(row[rc]))
                    all_revs.extend(kaggle_revs[:20])
                    if review_selector is not None:
                        df.at[idx, 'display_review'] = review_selector.select_best_review(all_revs)
            else:
                kaggle_counts.append(0)
        df['kaggle_review_count'] = kaggle_counts
        print(f"  Enriched {sum(1 for c in kaggle_counts if c > 0)} places")
    else:
        df['kaggle_review_count'] = 0

    p95 = df['review_count'].quantile(0.95)
    df['review_count_clipped'] = df['review_count'].clip(upper=p95)
    df['review_count_log'] = np.log1p(df['review_count'])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    review_normalized = scaler.fit_transform(df[['review_count_clipped']])
    rating_normalized = scaler.fit_transform(df[['avg_rating']])
    sentiment_normalized = (df[['avg_review_sentiment']].values + 1) / 2
    df['popularity_score'] = (
        review_normalized * 0.3 + rating_normalized * 0.3 + sentiment_normalized * 0.4
    ).flatten()

    print(f"  Processed {len(df)} places")
    print(f"  Avg BERT sentiment: {df['avg_review_sentiment'].mean():.3f}")
    return df


def preprocess_nearby_accommodation(df_raw, sentiment_analyser):
    print("\n[3/3] Preprocessing Nearby Accommodation...")
    df = df_raw.copy()

    df['service_display_review'] = df['service_display_review'].fillna(DEFAULT_SERVICE_REVIEW)
    df['service_image_url'] = df['service_image_url'].fillna('https://via.placeholder.com/300x200?text=No+Image')
    df['service_avg_rating'] = df['service_avg_rating'].fillna(0.0)

    print("  Computing BERT sentiment for service reviews...")
    df['service_sentiment'] = sentiment_analyser.analyse_batch(df['service_display_review'].tolist(), batch_size=16)

    budget_mapping = {
        'Rs. 500 - 2,500': 1,
        'Rs. 1,500 - 5,000 (Estimated)': 2,
        'Rs. 2,500 - 7,500': 3,
        'Rs. 7,500 - 15,000': 4,
        'Rs. 15,000+': 5
    }
    df['budget_score'] = df['service_budget_lkr'].map(budget_mapping).fillna(2)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    rating_normalized = scaler.fit_transform(df[['service_avg_rating']])
    sentiment_normalized = (df[['service_sentiment']].values + 1) / 2
    budget_affordability = 1 - (df['budget_score'] / 5.0)
    df['service_quality_score'] = (
        rating_normalized * 0.5 + sentiment_normalized * 0.3 +
        budget_affordability.values.reshape(-1, 1) * 0.2
    ).flatten()

    print(f"  Processed {len(df)} services")
    print(f"  Avg BERT sentiment: {df['service_sentiment'].mean():.3f}")
    return df


def load_and_preprocess_all(data_dir='data', kaggle_csv='Destination Reviews (final).csv'):
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATASETS")
    print("=" * 80)

    sub1_raw = pd.read_csv(os.path.join(data_dir, 'submodel_1.csv'))
    sub2_raw = pd.read_csv(os.path.join(data_dir, 'submodel_2.csv'))
    acc_raw = pd.read_csv(os.path.join(data_dir, 'nearby_accommodation.csv'))

    print(f"\nLoaded raw data:")
    print(f"  Submodel 1: {sub1_raw.shape}")
    print(f"  Submodel 2: {sub2_raw.shape}")
    print(f"  Accommodation: {acc_raw.shape}")

    sentiment_analyser = BERTSentimentAnalyser()
    sentiment_analyser.load()

    kaggle_enricher = KaggleReviewEnricher(kaggle_csv)
    kaggle_enricher.load()

    review_selector = BERTReviewSelector(sentiment_analyser)

    sub1 = preprocess_submodel_1(sub1_raw, sentiment_analyser, kaggle_enricher, review_selector)
    sub2 = preprocess_submodel_2(sub2_raw, sentiment_analyser, kaggle_enricher, review_selector)
    acc = preprocess_nearby_accommodation(acc_raw, sentiment_analyser)

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)

    return sub1, sub2, acc


if __name__ == "__main__":
    sub1, sub2, acc = load_and_preprocess_all('data')

    out_dir = os.path.join('output', 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(sub1, os.path.join(out_dir, 'submodel_1.joblib'), compress=3)
    joblib.dump(sub2, os.path.join(out_dir, 'submodel_2.joblib'), compress=3)
    joblib.dump(acc, os.path.join(out_dir, 'accommodation.joblib'), compress=3)

    print(f"\nSaved to {out_dir}/")
    print(f"  submodel_1.joblib  ({len(sub1)} places)")
    print(f"  submodel_2.joblib  ({len(sub2)} places)")
    print(f"  accommodation.joblib ({len(acc)} services)")