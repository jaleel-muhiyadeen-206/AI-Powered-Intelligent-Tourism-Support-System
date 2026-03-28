"""
Model 2: Popular Places Nearby
================================
BERT Sentiment + Kaggle Enrichment + LambdaMART Ranking

Architecture:
    1. BERT Sentiment pre-computed in preprocessing (avg_review_sentiment column)
    2. Kaggle Enrichment: 35,434 real tourist reviews for 131 matched destinations
    3. LambdaMART ranking with proximity-dominant relevance labels

Ground-Truth Relevance (0-4):
    40% proximity + 25% rating + 15% same_district + 10% sentiment
    + 5% kaggle_enriched + 5% coordinate_trust
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import joblib
import os
import sys
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import haversine_distance
from preprocessing import are_districts_adjacent

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DEFAULT_REVIEW = "A hidden gem waiting to be explored."


class Model2_PopularNearby:
    def __init__(self, submodel_2_df):
        self.df = submodel_2_df.reset_index(drop=True)
        self.knn_model = None
        self.lambdamart_model = None
        self.best_params = None
        self.test_ndcg = None
        self.trained = False

    def _make_ranking_data(self, query_indices, max_dist=20):
        feats, labels, groups = [], [], []
        for qi in query_indices:
            q = self.df.iloc[qi]
            q_coords = np.radians([[q['latitude'], q['longitude']]])
            dists, idxs = self.knn_model.kneighbors(q_coords, n_neighbors=min(30, len(self.df)))
            gf, gl = [], []

            for d_rad, ci in zip(dists[0], idxs[0]):
                if ci == qi:
                    continue
                d_km = haversine_distance(
                    q['latitude'], q['longitude'],
                    self.df.iloc[ci]['latitude'], self.df.iloc[ci]['longitude']
                )
                if d_km > max_dist:
                    continue
                c = self.df.iloc[ci]
                same_d = 1.0 if c['district'] == q['district'] else 0.0
                has_bad = c.get('has_shared_coords', False)
                adjacent = are_districts_adjacent(q['district'], c['district'])

                if has_bad and not same_d and not adjacent:
                    coord_trust = 0.0
                elif has_bad and not same_d:
                    coord_trust = 0.5
                else:
                    coord_trust = 1.0

                sent = c.get('avg_review_sentiment', 0.0)
                pop = min(c.get('review_count_clipped', c.get('review_count', 0)) / 5000, 1.0)
                prox = 1.0 / (1.0 + d_km / 5.0)
                kaggle = 1.0 if c.get('kaggle_review_count', 0) > 0 else 0.0
                rating = c['avg_rating'] / 5.0 if c['avg_rating'] > 0 else 0.0

                gf.append([sent, pop, prox, kaggle, rating, same_d, coord_trust])
                prox_f = max(0.0, 1.0 - d_km / max_dist)
                rel = (prox_f * 0.40 + rating * 0.25 + max(0, sent) * 0.10 +
                       kaggle * 0.05 + same_d * 0.15 + coord_trust * 0.05)
                gl.append(min(4, int(rel * 4)))

            if len(gf) >= 3:
                feats.extend(gf)
                labels.extend(gl)
                groups.append(len(gf))

        return (np.array(feats) if feats else np.empty((0, 7)), np.array(labels), groups)

    def _tune_lambdamart(self, X, y, groups):
        param_grid = [
            {'learning_rate': 0.05, 'num_leaves': 15, 'num_boost_round': 80},
            {'learning_rate': 0.05, 'num_leaves': 31, 'num_boost_round': 100},
            {'learning_rate': 0.1, 'num_leaves': 31, 'num_boost_round': 80},
            {'learning_rate': 0.1, 'num_leaves': 15, 'num_boost_round': 120},
        ]
        best_score, best_cfg = -1, param_grid[0]
        for cfg in param_grid:
            ds = lgb.Dataset(X, label=y, group=groups)
            params = {
                'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5],
                'verbose': -1, 'min_data_in_leaf': 5,
                'learning_rate': cfg['learning_rate'], 'num_leaves': cfg['num_leaves'],
            }
            model = lgb.train(params, ds, num_boost_round=cfg['num_boost_round'],
                              valid_sets=[ds], callbacks=[lgb.log_evaluation(0)])
            score = model.best_score.get('training', {}).get('ndcg@5', 0)
            if score > best_score:
                best_score = score
                best_cfg = cfg
        print(f"    Best: lr={best_cfg['learning_rate']}, leaves={best_cfg['num_leaves']}, "
              f"rounds={best_cfg['num_boost_round']}")
        return best_cfg

    @staticmethod
    def _ndcg_at_k(scores, k=5):
        rel = np.array(scores[:k], dtype=float)
        if rel.sum() == 0:
            return 0.0
        dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
        ideal = np.sort(rel)[::-1]
        idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
        return dcg / idcg if idcg > 0 else 0.0

    def train(self):
        print("\n" + "=" * 70)
        print("TRAINING MODEL 2: POPULAR PLACES NEARBY")
        print("  BERT Sentiment + Kaggle Enrichment + LambdaMART")
        print("=" * 70)

        print("\n[1/4] KNN fitting...")
        coords = self.df[['latitude', 'longitude']].values
        self.knn_model = NearestNeighbors(
            n_neighbors=min(30, len(self.df)), metric='haversine', algorithm='ball_tree'
        )
        self.knn_model.fit(np.radians(coords))
        print(f"    {len(self.df)} locations indexed")

        print("\n[2/4] Train/test split (80/20)...")
        rng = np.random.RandomState(42)
        all_idx = rng.permutation(len(self.df))
        split = int(0.8 * len(all_idx))
        train_idx, test_idx = all_idx[:split], all_idx[split:]
        print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")

        X_train, y_train, g_train = self._make_ranking_data(train_idx)
        X_test, y_test, g_test = self._make_ranking_data(test_idx)

        print("\n[3/4] Hyperparameter tuning...")
        self.best_params = self._tune_lambdamart(X_train, y_train, g_train)

        print("\n[4/4] Training final LambdaMART...")
        ds = lgb.Dataset(X_train, label=y_train, group=g_train)
        params = {
            'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5],
            'verbose': -1, 'min_data_in_leaf': 5,
            'learning_rate': self.best_params['learning_rate'],
            'num_leaves': self.best_params['num_leaves'],
        }
        self.lambdamart_model = lgb.train(
            params, ds, num_boost_round=self.best_params['num_boost_round'],
            valid_sets=[ds], callbacks=[lgb.log_evaluation(0)]
        )

        imp = self.lambdamart_model.feature_importance(importance_type='gain')
        names = ['sentiment', 'popularity', 'proximity', 'kaggle_enriched',
                 'rating', 'same_district', 'coord_trust']
        total = max(sum(imp), 1)
        print("\n    Feature importance:")
        for n, v in zip(names, imp):
            print(f"      {n}: {v / total * 100:.1f}%")

        if len(X_test) > 0:
            preds = self.lambdamart_model.predict(X_test)
            ndcg_scores = []
            offset = 0
            for g in g_test:
                chunk = list(zip(preds[offset:offset + g], y_test[offset:offset + g]))
                chunk.sort(key=lambda x: x[0], reverse=True)
                ndcg_scores.append(self._ndcg_at_k([c[1] for c in chunk]))
                offset += g
            self.test_ndcg = np.mean(ndcg_scores)
            print(f"\n    TEST NDCG@5: {self.test_ndcg:.4f}")

        kaggle_count = (self.df.get('kaggle_review_count', pd.Series([0])) > 0).sum()
        print(f"    Avg BERT sentiment: {self.df['avg_review_sentiment'].mean():.3f}")
        print(f"    Kaggle-enriched places: {kaggle_count}")

        self.trained = True
        print("\n" + "=" * 70)
        print("MODEL 2 TRAINING COMPLETE")
        print("=" * 70)

    def recommend(self, place_id, top_n=5, max_distance_km=20):
        if not self.trained:
            raise ValueError("Model not trained.")
        cur = self.df[self.df['place_id'] == place_id]
        if cur.empty:
            return pd.DataFrame()
        cur = cur.iloc[0]

        q_coords = np.radians([[cur['latitude'], cur['longitude']]])
        dists, idxs = self.knn_model.kneighbors(q_coords, n_neighbors=min(30, len(self.df)))

        candidates = []
        for d_rad, idx in zip(dists[0], idxs[0]):
            if self.df.iloc[idx]['place_id'] == place_id:
                continue
            d_km = haversine_distance(
                cur['latitude'], cur['longitude'],
                self.df.iloc[idx]['latitude'], self.df.iloc[idx]['longitude']
            )
            if d_km > max_distance_km:
                continue

            c = self.df.iloc[idx]
            same_d_bool = c['district'] == cur['district']
            has_bad = c.get('has_shared_coords', False)
            adjacent = are_districts_adjacent(cur['district'], c['district'])

            if has_bad and not same_d_bool and not adjacent:
                continue
            coord_trust = 0.5 if (has_bad and not same_d_bool) else 1.0

            sent = c.get('avg_review_sentiment', 0.0)
            pop = min(c.get('review_count_clipped', c.get('review_count', 0)) / 5000, 1.0)
            prox = 1.0 / (1.0 + d_km / 5.0)
            kaggle = 1.0 if c.get('kaggle_review_count', 0) > 0 else 0.0
            rating = c['avg_rating'] / 5.0 if c['avg_rating'] > 0 else 0.0
            same_d = 1.0 if same_d_bool else 0.0

            feat = np.array([[sent, pop, prox, kaggle, rating, same_d, coord_trust]])
            score = self.lambdamart_model.predict(feat)[0]

            review = c.get('display_review', '')
            if pd.isna(review) or not str(review).strip():
                review = DEFAULT_REVIEW

            candidates.append({
                'name': c['place_name'], 'district': c['district'],
                'image': c.get('image_url', ''), 'rating': round(c['avg_rating'], 1),
                'distance_km': round(d_km, 2), 'review': str(review)[:200],
                'sentiment_score': round(sent, 3), 'popularity_score': round(pop, 3),
                'proximity_score': round(prox, 3), 'final_score': round(float(score), 3),
            })

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        rows = []
        for rank, c in enumerate(candidates[:top_n], 1):
            rows.append({'rank': rank, **c})
        return pd.DataFrame(rows)

    def save(self, path):
        state = {
            'df': self.df, 'knn_model': self.knn_model,
            'lambdamart_model': self.lambdamart_model, 'best_params': self.best_params,
            'test_ndcg': self.test_ndcg, 'trained': self.trained,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(state, path, compress=3)
        print(f"  Model 2 saved to {path}")

    def load(self, path):
        state = joblib.load(path)
        self.df = state['df']
        self.knn_model = state['knn_model']
        self.lambdamart_model = state['lambdamart_model']
        self.best_params = state['best_params']
        self.test_ndcg = state['test_ndcg']
        self.trained = state['trained']
        print(f"  Model 2 loaded from {path}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    sub2 = joblib.load('output/preprocessed/submodel_2.joblib')
    print(f"Loaded submodel_2: {sub2.shape}")
    model = Model2_PopularNearby(sub2)
    model.train()
    model.save('output/trained_models/model_2.joblib')