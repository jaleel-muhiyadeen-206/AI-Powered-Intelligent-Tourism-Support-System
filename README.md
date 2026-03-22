# Sri Lanka Tourism Recommendation System

A machine learning–based recommendation system for Sri Lankan tourist destinations, built on **608 real places** across **25 districts**, with **3,870 nearby services** and **35,434 Kaggle tourist reviews**.

The system provides three types of recommendations:

| Model | Purpose | Core Technique |
|-------|---------|----------------|
| **Model 1** — You May Also Like | Similar places to what you're viewing | SBERT + GNN + LambdaMART |
| **Model 2** — Popular Places Nearby | Highest-rated popular places near you | BERT Sentiment + Kaggle Enrichment + LambdaMART |
| **Model 3** — Nearby Essentials | Top 5 Hotels, Dining, Activities near you | BERT Sentiment + Per-Type LambdaMART |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Model 1: You May Also Like](#model-1-you-may-also-like)
- [Model 2: Popular Places Nearby](#model-2-popular-places-nearby)
- [Model 3: Nearby Essentials](#model-3-nearby-essentials)
- [Shared Components](#shared-components)
- [Model Comparison](#model-comparison)
- [Evaluation Methodology](#evaluation-methodology)
- [Integration](#integration)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)

---

## Project Overview

### Problem Statement

Tourists visiting Sri Lanka need personalised place recommendations that are:
1. **Contextually relevant** — similar to what they are currently viewing
2. **Geographically proximate** — within a practical travel distance
3. **Quality-driven** — backed by real sentiments from actual tourist reviews

### Why These Models?

Traditional recommendation approaches (TF-IDF + manual weights) suffer from three limitations:

1. **No semantic understanding** — TF-IDF treats words as independent tokens. "Beautiful beach with clear water" and "Pristine shoreline with crystal sea" are treated as dissimilar despite meaning the same thing.

2. **No learned ranking** — Manual weight formulas (e.g., `0.4 × rating + 0.3 × sentiment + 0.3 × proximity`) are arbitrary. Different contexts require different weight distributions, which manual tuning cannot capture.

3. **Limited review data** — Only 5 reviews per place is insufficient to reliably assess sentiment. The Kaggle enrichment adds 100+ real reviews per matched place.

### Solution

All three models use a **three-stage pipeline**:

```
Stage 1: Feature Extraction     → SBERT / BERT (semantic understanding)
Stage 2: Relationship Modelling  → GNN / KNN (graph or spatial relationships)
Stage 3: Learned Ranking         → LambdaMART (optimal weight discovery)
```

**Proximity is the dominant ranking factor** across all models (50% weight in ground-truth labels), ensuring recommendations stay within the same geographic region.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    preprocessing.py                      │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ BERT Sentiment   │  │ Kaggle Review│  │ BERT Review│ │
│  │ Analyser (92%)   │  │ Enrichment   │  │ Selector   │ │
│  └────────┬────────┘  └──────┬───────┘  └─────┬──────┘ │
│           └──────────────────┼────────────────┘         │
│                              ▼                           │
│           Preprocessed DataFrames (sub1, sub2, acc)      │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌─────────────┐ ┌──────────┐ ┌────────────────┐
   │  Model 1     │ │ Model 2  │ │   Model 3      │
   │  SBERT+GNN   │ │ BERT+    │ │ BERT+LambdaMART│
   │  +LambdaMART │ │ Kaggle+  │ │ (per type:     │
   │              │ │ LambdaMART│ │  Hotels/Dining/│
   │              │ │          │ │  Activities)   │
   └──────┬──────┘ └────┬─────┘ └───────┬────────┘
          │              │               │
          ▼              ▼               ▼
   "You May Also   "Popular Places  "Top 5 Hotels,
    Like" recs       Nearby" recs    Dining, Activities"
```

---

## Model 1: You May Also Like

> **Purpose**: Given a place a user is currently viewing, recommend 5 similar places within the same region.

### Components

#### 1. Sentence-BERT (SBERT) — Semantic Feature Extraction

**Model used**: `all-MiniLM-L6-v2` (22M parameters)

Converts each place's combined text into a 384-dimensional dense vector that captures semantic meaning. Understands synonyms, context, and semantic relationships unlike TF-IDF.

#### 2. Place-Place Graph Neural Network (GNN) — Relationship Modelling

**Architecture**: 2-layer Graph Convolutional Network (GCN) with dropout 0.3

Learns 64-dimensional place embeddings by propagating information through a graph of real relationships:
- Same district, same category, geographic proximity (≤ 10 km), semantic similarity (cosine > 0.5)

#### 3. LambdaMART — Learned Ranking

Directly optimises NDCG (ranking metric) using LightGBM in `lambdarank` mode. Ground-truth relevance:
```
relevance = 50% proximity + 30% rating + 20% same_district
```

---

## Model 2: Popular Places Nearby

> **Purpose**: Given a place, recommend the 5 most popular and well-reviewed places within 20 km.

### Components

#### 1. BERT Sentiment Analysis
**Model used**: `nlptown/bert-base-multilingual-uncased-sentiment` (110M parameters, ~92% accuracy)

#### 2. Kaggle Review Enrichment
Augments 5 reviews per place with 35,434 real tourist reviews via fuzzy matching (131 places matched).

#### 3. LambdaMART — Learned Ranking
Features: `[sentiment, popularity, proximity, kaggle_enriched, rating, same_district, coord_trust]`

---

## Model 3: Nearby Essentials

> **Purpose**: Given a place, recommend the **top 5** Hotels, Dining options, and Activities.

### Guaranteed Top 5 Results

Model 3 always returns exactly 5 recommendations per service type through a three-tier strategy:
1. **Primary search**: Finds services within `max_distance_km` (default 15 km)
2. **Radius expansion**: If fewer than 5 found, expands search beyond the initial radius
3. **Fallback**: If still fewer than 5, searches from the nearest neighbouring places with services

### Why Separate Models Per Service Type?

| Factor | Hotels | Dining | Activities |
|--------|--------|--------|------------|
| Budget importance | High | Medium | Low |
| Proximity importance | Medium | High | Medium |
| Rating importance | High | Medium | High |

### Ranking Feature Priorities

Features: `[sentiment, proximity, rating, budget, distance_raw]`

Each per-type LambdaMART model learns different feature importance weights automatically:
- **Hotels**: Budget and rating dominate
- **Dining**: Proximity and sentiment dominate
- **Activities**: Rating and proximity dominate

---

## Shared Components

### Preprocessing Pipeline (`preprocessing.py`)
1. BERT Sentiment Analysis for all reviews
2. Kaggle Review Enrichment (35,434 reviews)
3. BERT Review Selector (picks best positive review)
4. Feature Engineering (budget scores, outlier clipping, coordinate validation)

### LambdaMART (All Models)
Industry standard learning-to-rank using LightGBM in `lambdarank` mode. Directly optimises NDCG.

---

## Model Comparison

Each model is compared against two alternatives in `model_comparison.py`:

### Model 1 Alternatives

| Approach | Description | Why Not Chosen |
|----------|-------------|----------------|
| **TF-IDF + KNN** | Word-frequency matching with nearest neighbours | No semantic understanding |
| **Word2Vec + PCA** | Average word vectors with dimensionality reduction | Loses word order; no graph structure |

### Model 2 Alternatives

| Approach | Description | Why Not Chosen |
|----------|-------------|----------------|
| **TextBlob + Manual Weights** | Rule-based sentiment with hand-tuned weights | 65% accuracy; arbitrary weights |
| **VADER + Random Forest** | Lexicon-based sentiment with classification ranking | Classification ≠ ranking |

### Model 3 Alternatives

| Approach | Description | Why Not Chosen |
|----------|-------------|----------------|
| **TextBlob + Manual Weights** | Rule-based sentiment with suboptimal weight distribution | Over-weights sentiment, under-weights proximity |
| **TF-IDF + SVM** | Content matching with linear SVM classification | Linear kernel too weak for ranking; classification ≠ ranking |

---

## Evaluation Methodology

### Dual Evaluation

The system uses **two independent evaluation approaches**:

#### Evaluation 1: Rule-Based (Internal Ranking Data)
Uses the LambdaMART ranking data to evaluate binary classification (relevance ≥ 2 → Relevant). Tests whether the model correctly separates relevant from non-relevant candidates in its training data distribution.

#### Evaluation 2: Criteria-Based (Actual Recommendation Outputs)
Uses **independent criteria** that LambdaMART was **NOT directly trained to optimise**:

| Model | Relevant Criteria |
|-------|-------------------|
| Model 1 (Similar Places) | `avg_rating ≥ 3.5` AND `distance_km ≤ 15` |
| Model 2 (Popular Nearby) | `avg_rating ≥ 3.5` AND `distance_km ≤ 15` |
| Model 3 (Nearby Services) | `service_avg_rating ≥ 3.5` AND `distance_km ≤ 10` |

Includes train/test accuracy comparison with overfitting analysis.

### Outputs

| Output | Style | Description |
|--------|-------|-------------|
| Model Test table | Table 22 | Train/test accuracy with overfitting remarks |
| Classification reports | Figure 25 | Precision, recall, F1-score per class |
| Confusion matrix tables | Table 23 | TP/TN/FP/FN breakdown |
| Confusion matrix heatmaps | Figure 26 | Visual confusion matrices |
| Model comparison | Summary | Cross-model performance comparison |

### Metrics

| Metric | Description |
|--------|-------------|
| **NDCG@5** | Normalised Discounted Cumulative Gain at rank 5 |
| **Diversity** | Ratio of unique districts in recommendations |
| **Avg Distance** | Mean distance of recommendations (km) |
| **Coverage** | Fraction of catalogue appearing in recommendations |
| **Precision/Recall/F1** | Standard classification metrics |

---

## Integration

The system supports two input modes for integration with other modules:

### Mode 1: Search & Find
User types a place name and selects from the dataset via a searchable dropdown.

### Mode 2: External Input (Image Recognition Integration)
Another team member's image recognition model identifies a place from a photo. The resulting place name is passed to this system via:

```python
# URL query parameter
streamlit_app.py?place_name=Sigiriya

# Or programmatically
from streamlit_app import recommend_by_name
place_id, matched_name, score = recommend_by_name("Sigiriya", sub1_df)
```

The system performs fuzzy matching (threshold ≥ 60%) to find the closest place in the dataset.

---

## Project Structure

```
project_root/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── preprocessing.py                    # BERT sentiment + Kaggle enrichment
├── utils.py                            # Haversine distance utility
├── evaluation.py                       # Dual evaluation (rule-based + criteria-based)
├── model_comparison.py                 # Comparison against alternative approaches
├── test_all_models_outcomes.py         # Test: loads models, generates sample recommendations
├── streamlit_app.py                    # Streamlit UI with search + external input
│
├── models/
│   ├── model_1_you_may_also_like.py    # SBERT + GNN + LambdaMART
│   ├── model_2_popular_nearby.py       # BERT Sentiment + Kaggle + LambdaMART
│   ├── model_3_nearby_essentials.py    # BERT + Per-Type LambdaMART (top 5 guaranteed)
│   ├── sbert_cache/                    # Cached SBERT model
│   └── bert_cache/                     # Cached BERT sentiment model
│
├── output/
│   ├── preprocessed/                   # Preprocessed DataFrames (joblib)
│   │   ├── submodel_1.joblib
│   │   ├── submodel_2.joblib
│   │   └── accommodation.joblib
│   ├── trained_models/                 # Trained model states (joblib)
│   │   ├── model_1.joblib
│   │   ├── model_2.joblib
│   │   └── model_3.joblib
│   └── evaluation/                     # Evaluation output images (PNG)
│
├── data/
│   ├── submodel_1.csv                  # 608 places
│   ├── submodel_2.csv                  # 608 places with reviews
│   ├── nearby_accommodation.csv        # 3,870 services
│   ├── user_likes.csv                  # User interaction data
│   ├── user_reviews.csv
│   └── user_ratings.csv
│
└── Destination Reviews (final).csv     # Kaggle: 35,434 tourist reviews
```

---

## Setup and Installation

### Prerequisites

- Python 3.11
- ~2 GB disk space for model downloads (BERT + SBERT)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | ≥ 2.2.0 | Data manipulation |
| `numpy` | ≥ 1.26.4 | Numerical operations |
| `scikit-learn` | ≥ 1.4.0 | KNN, preprocessing, metrics |
| `torch` | ≥ 2.0.0 | Neural network backend |
| `transformers` | ≥ 4.30.0 | BERT sentiment model |
| `sentence-transformers` | ≥ 2.2.0 | SBERT embeddings |
| `torch-geometric` | ≥ 2.3.0 | Graph Neural Network |
| `lightgbm` | ≥ 4.0.0 | LambdaMART ranking |
| `scipy` | ≥ 1.10.0 | Spatial computations |
| `joblib` | ≥ 1.3.0 | Model serialisation |
| `streamlit` | ≥ 1.30.0 | Web UI |
| `textblob` | ≥ 0.17.1 | Baseline comparison |
| `matplotlib` | ≥ 3.7.0 | Evaluation visualisation |

---

## Usage

### Execution Order

```bash
# Step 1: Preprocess — BERT sentiment + Kaggle enrichment (~3 min)
python preprocessing.py

# Step 2: Train Model 1 — SBERT + GNN + LambdaMART (~30 sec)
python models/model_1_you_may_also_like.py

# Step 3: Train Model 2 — BERT Sentiment + LambdaMART (~5 sec)
python models/model_2_popular_nearby.py

# Step 4: Train Model 3 — Per-type LambdaMART (~5 sec)
python models/model_3_nearby_essentials.py

# Step 5: Test — loads saved models (~5 sec)
python test_all_models_outcomes.py

# Step 6: Launch Streamlit UI
streamlit run streamlit_app.py

# Optional: Run evaluation (dual evaluation)
python evaluation.py

# Optional: Run model comparison
python model_comparison.py
```

Steps 1–4 only need to be run **once**. After that, `test_all_models_outcomes.py` loads the saved files and generates recommendations instantly.

### External Input (Integration)

Pass a place name via URL query parameter:
```
http://localhost:8501/?place_name=Sigiriya
```

### Use Models in Code

```python
import pandas as pd
from models.model_1_you_may_also_like import Model1_YouMayAlsoLike
from models.model_2_popular_nearby import Model2_PopularNearby
from models.model_3_nearby_essentials import Model3_NearbyEssentials

sub1 = pd.read_pickle('output/preprocessed/submodel_1.joblib')

model1 = Model1_YouMayAlsoLike(sub1)
model1.load('output/trained_models/model_1.joblib')

model2 = Model2_PopularNearby(pd.DataFrame())
model2.load('output/trained_models/model_2.joblib')

model3 = Model3_NearbyEssentials(pd.DataFrame(), pd.DataFrame())
model3.load('output/trained_models/model_3.joblib')

place_id = 'ChIJ...'
similar_places = model1.recommend(place_id, top_n=5)
popular_nearby = model2.recommend(place_id, top_n=5)
nearby_hotels = model3.recommend(place_id, service_type='Hotels', top_n=5)
nearby_dining = model3.recommend(place_id, service_type='Dining', top_n=5)
nearby_activities = model3.recommend(place_id, service_type='Activities', top_n=5)
```

---

## Datasets

### Source Data

| Dataset | Records | Source |
|---------|---------|--------|
| `submodel_1.csv` | 608 places | Google Places API |
| `submodel_2.csv` | 608 places (5 reviews each) | Google Places API |
| `nearby_accommodation.csv` | 3,870 services | Google Places API |
| `Destination Reviews (final).csv` | 35,434 reviews | Kaggle |

### Data Quality Fixes Applied

1. **District coordinate correction**: 91 places with GPS > 80 km from district centroid corrected
2. **Duplicate coordinates**: 117 places with identical GPS from non-adjacent districts flagged with coordinate trust scoring
3. **Review count outliers**: Clipped at 95th percentile
4. **Missing reviews**: Filled with descriptive placeholder text
5. **Budget normalisation**: String budget ranges converted to 1–5 scale

---

## Results

### Sample Output (Query: Nuwara Eliya Post Office)

**Model 1 — You May Also Like**:
| Rank | Place | District | Distance | Rating |
|------|-------|----------|----------|--------|
| 1 | Post Office Museum | Nuwara Eliya | 0.0 km | 4.6 |
| 2 | Holy Trinity Church | Nuwara Eliya | 0.58 km | 4.6 |
| 3 | St. Andrew's Hotel | Nuwara Eliya | 0.94 km | 4.6 |
| 4 | Gayathri Pitam | Nuwara Eliya | 1.18 km | 4.5 |
| 5 | Lover's Leap Waterfall | Nuwara Eliya | 2.35 km | 4.5 |

**Model 3 — Nearby Hotels** (Top 5 guaranteed):
| Rank | Service | Distance | Rating | Budget |
|------|---------|----------|--------|--------|
| 1 | The Grand Hotel Nuwara Eliya | 0.39 km | 4.6 | Rs. 15,000+ |
| 2 | Yara Nuwara Eliya Hotel | 0.18 km | 4.6 | Rs. 7,500–15,000 |
| 3 | The Lynden Grove | 0.62 km | 4.5 | Rs. 2,500–7,500 |
| 4 | Dahlia Mount View Hotel | 1.08 km | 4.5 | Rs. 2,500–7,500 |
| 5 | Royal Mount Hotel | 1.16 km | 4.4 | Rs. 500–2,500 |

---

## References

1. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
2. Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017.
3. Burges, C. J. (2010). *From RankNet to LambdaRank to LambdaMART: An Overview*. Microsoft Research Technical Report.
4. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019.
5. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS 2017.
