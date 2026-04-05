"""
Model 3: Nearby Essentials
============================
BERT Sentiment + Per-Service-Type LambdaMART Ranking

Architecture:
    Separate LambdaMART models for each service category:
        - Hotels: prioritises budget, rating, proximity
        - Dining: prioritises proximity, sentiment
        - Activities: prioritises rating, proximity

    Dual-path recommendation:
        1. Dedicated services path: For 259 places that have curated
           services in the dataset (5 per category = 15 total), rank
           among those specific services using LambdaMART.
        2. Geographic fallback path: For 349 places without dedicated
           services, use KNN geographic search across all services
           and rank with LambdaMART.
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

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SERVICE_TYPES = ['Hotels', 'Dining', 'Activities']
DEFAULT_SERVICE_REVIEW = "This service is awaiting its first detailed review."


class Model3_NearbyEssentials:
    def __init__(self, submodel_1_df, accommodation_df):
        self.places = submodel_1_df.reset_index(drop=True)
        self.services = accommodation_df.reset_index(drop=True)
        self.service_data = {}
        self.lambdamart = {}
        self.best_params = {}
        self.test_ndcg = {}
        self.place_knn = None
        self.trained = False
        # Maps place_id -> {service_type: DataFrame of dedicated services}
        self.place_service_map = {}

    def _build_place_service_map(self):
        """
        Build a lookup: place_id -> {Hotels: df, Dining: df, Activities: df}
        from the raw services data. Only populated for places that have
        dedicated services in nearby_accommodation.csv (259 places).
        """
        self.place_service_map = {}
        if self.services is None or self.services.empty:
            return

        # Use 'service_type' column directly (original categories from dataset)
        # Also check for 'classified_type' if it was added during training
        type_col = 'classified_type' if 'classified_type' in self.services.columns else 'service_type'

        for pid in self.services['place_id'].unique():
            place_svcs = self.services[self.services['place_id'] == pid]
            type_map = {}
            for stype in SERVICE_TYPES:
                typed = place_svcs[place_svcs[type_col] == stype]
                if not typed.empty:
                    type_map[stype] = typed.reset_index(drop=True)
            if type_map:
                self.place_service_map[pid] = type_map

        print(f"  Built place-service map: {len(self.place_service_map)} places with dedicated services")

    def _classify_type(self, raw_type):
        t = str(raw_type).lower()
        if any(w in t for w in ['hotel', 'resort', 'inn', 'lodge', 'guest', 'hostel', 'villa']):
            return 'Hotels'
        if any(w in t for w in ['restaurant', 'cafe', 'food', 'dining', 'bakery', 'bar']):
            return 'Dining'
        return 'Activities'

    def _make_ranking_data(self, stype, place_indices, max_dist=15):
        svc = self.service_data.get(stype)
        if svc is None or len(svc) < 5:
            return np.empty((0, 5)), np.array([]), []

        svc_coords = svc[['service_latitude', 'service_longitude']].values
        knn = NearestNeighbors(
            n_neighbors=min(20, len(svc)), metric='haversine', algorithm='ball_tree'
        )
        knn.fit(np.radians(svc_coords))
        feats, labels, groups = [], [], []

        for pi in place_indices:
            p = self.places.iloc[pi]
            q = np.radians([[p['latitude'], p['longitude']]])
            dists, idxs = knn.kneighbors(q, n_neighbors=min(20, len(svc)))
            gf, gl = [], []

            for d_rad, si in zip(dists[0], idxs[0]):
                s = svc.iloc[si]
                d_km = haversine_distance(
                    p['latitude'], p['longitude'],
                    s['service_latitude'], s['service_longitude']
                )
                if d_km > max_dist:
                    continue
                sent = max(0, s.get('service_sentiment', 0.0))
                prox = 1.0 / (1.0 + d_km / 3.0)
                rating = s.get('service_avg_rating', 3.0) / 5.0
                budget = s.get('budget_score', 0.5)

                gf.append([sent, prox, rating, budget, d_km])
                prox_f = max(0.0, 1.0 - d_km / max_dist)
                rel = prox_f * 0.50 + rating * 0.30 + sent * 0.10 + (1 - budget / 5.0) * 0.10
                gl.append(min(4, int(rel * 4)))

            if len(gf) >= 3:
                feats.extend(gf)
                labels.extend(gl)
                groups.append(len(gf))

        return (np.array(feats) if feats else np.empty((0, 5)), np.array(labels), groups)

    def _tune(self, X, y, groups):
        param_grid = [
            {'learning_rate': 0.05, 'num_leaves': 15, 'rounds': 80},
            {'learning_rate': 0.1, 'num_leaves': 31, 'rounds': 100},
            {'learning_rate': 0.1, 'num_leaves': 15, 'rounds': 120},
        ]
        best_score, best_cfg = -1, param_grid[0]
        for cfg in param_grid:
            ds = lgb.Dataset(X, label=y, group=groups)
            params = {
                'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5],
                'verbose': -1, 'min_data_in_leaf': 3,
                'learning_rate': cfg['learning_rate'], 'num_leaves': cfg['num_leaves'],
            }
            m = lgb.train(params, ds, num_boost_round=cfg['rounds'],
                          valid_sets=[ds], callbacks=[lgb.log_evaluation(0)])
            s = m.best_score.get('training', {}).get('ndcg@5', 0)
            if s > best_score:
                best_score = s
                best_cfg = cfg
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
        print("TRAINING MODEL 3: NEARBY ESSENTIALS")
        print("  BERT Sentiment + LambdaMART (per service type)")
        print("=" * 70)

        self.services['classified_type'] = self.services['service_type'].apply(self._classify_type)
        for st in SERVICE_TYPES:
            self.service_data[st] = self.services[
                self.services['classified_type'] == st
            ].reset_index(drop=True)
            print(f"\n  {st}: {len(self.service_data[st])} services")

        # Build the place-service map for dedicated services
        self._build_place_service_map()

        self.place_knn = NearestNeighbors(n_neighbors=5, metric='haversine', algorithm='ball_tree')
        self.place_knn.fit(np.radians(self.places[['latitude', 'longitude']].values))

        rng = np.random.RandomState(42)
        all_idx = rng.permutation(len(self.places))
        split = int(0.8 * len(all_idx))
        train_idx, test_idx = all_idx[:split], all_idx[split:]
        print(f"\n  80/20 split: {len(train_idx)} train, {len(test_idx)} test")

        for st in SERVICE_TYPES:
            print(f"\n  --- Training {st} ---")
            X_tr, y_tr, g_tr = self._make_ranking_data(st, train_idx)
            X_te, y_te, g_te = self._make_ranking_data(st, test_idx)

            if len(X_tr) < 10:
                print(f"    Insufficient data for {st}, skipping")
                continue

            bp = self._tune(X_tr, y_tr, g_tr)
            self.best_params[st] = bp
            print(f"    Tuned: lr={bp['learning_rate']}, leaves={bp['num_leaves']}, rounds={bp['rounds']}")

            ds = lgb.Dataset(X_tr, label=y_tr, group=g_tr)
            params = {
                'objective': 'lambdarank', 'metric': 'ndcg', 'ndcg_eval_at': [5],
                'verbose': -1, 'min_data_in_leaf': 3,
                'learning_rate': bp['learning_rate'], 'num_leaves': bp['num_leaves'],
            }
            self.lambdamart[st] = lgb.train(
                params, ds, num_boost_round=bp['rounds'],
                valid_sets=[ds], callbacks=[lgb.log_evaluation(0)]
            )

            imp = self.lambdamart[st].feature_importance(importance_type='gain')
            names = ['sentiment', 'proximity', 'rating', 'budget', 'distance_raw']
            total = max(sum(imp), 1)
            print("    Feature importance (ranking priorities):")
            for n, v in zip(names, imp):
                print(f"      {n}: {v / total * 100:.1f}%")

            if len(X_te) > 0:
                preds = self.lambdamart[st].predict(X_te)
                ndcg_scores = []
                offset = 0
                for g in g_te:
                    chunk = list(zip(preds[offset:offset + g], y_te[offset:offset + g]))
                    chunk.sort(key=lambda x: x[0], reverse=True)
                    ndcg_scores.append(self._ndcg_at_k([c[1] for c in chunk]))
                    offset += g
                self.test_ndcg[st] = np.mean(ndcg_scores)
                print(f"    TEST NDCG@5: {self.test_ndcg[st]:.4f}")

        self.trained = True
        print("\n" + "=" * 70)
        print("MODEL 3 TRAINING COMPLETE")
        print("=" * 70)

    def _score_service(self, s, cur, model):
        """Compute LambdaMART score for a single service relative to a place."""
        d_km = haversine_distance(
            cur['latitude'], cur['longitude'],
            s['service_latitude'], s['service_longitude']
        )
        sent = max(0, s.get('service_sentiment', 0.0))
        prox = 1.0 / (1.0 + d_km / 3.0)
        rating = s.get('service_avg_rating', 3.0) / 5.0
        budget = s.get('budget_score', 0.5)

        if model:
            feat = np.array([[sent, prox, rating, budget, d_km]])
            score = float(model.predict(feat)[0])
        else:
            score = prox * 0.5 + rating * 0.3 + sent * 0.1 + (1 - budget / 5.0) * 0.1

        return score, d_km, sent, prox, rating, budget

    def _build_candidate(self, s, score, d_km, sent, prox, rating, budget):
        """Build a candidate dict from a service row and its computed scores."""
        review = s.get('service_display_review', '')
        if pd.isna(review) or not str(review).strip():
            review = DEFAULT_SERVICE_REVIEW

        return {
            'name': s.get('service_name', 'Unknown'),
            'image': s.get('service_image_url', ''),
            'rating': round(s.get('service_avg_rating', 0), 1),
            'distance_km': round(d_km, 2), 'review': str(review)[:200],
            'budget': s.get('service_budget_lkr', 'N/A'),
            'sentiment_score': round(sent, 3), 'proximity_score': round(prox, 3),
            'rating_score': round(rating, 3), 'budget_score': round(budget, 3),
            'final_score': round(score, 3),
        }

    def _recommend_dedicated(self, place_id, cur, service_type, model, top_n=5):
        """
        Recommend from a place's own dedicated services.
        Used for the 259 places that have curated services in the dataset.
        Ranks all services of the given type using LambdaMART and returns top_n.
        """
        dedicated = self.place_service_map.get(place_id, {}).get(service_type)
        if dedicated is None or dedicated.empty:
            return []

        candidates = []
        for _, s in dedicated.iterrows():
            score, d_km, sent, prox, rating, budget = self._score_service(s, cur, model)
            candidates.append(self._build_candidate(s, score, d_km, sent, prox, rating, budget))

        return candidates

    def _recommend_geographic(self, place_id, cur, service_type, model, top_n=5,
                              max_distance_km=15):
        """
        Recommend using geographic KNN search across all services.
        Used for the 349 places without dedicated services in the dataset.
        Falls back to expanding radius and nearest-place lookup if needed.
        """
        svc = self.service_data.get(service_type)
        if svc is None or len(svc) == 0:
            return []

        svc_coords = np.radians(svc[['service_latitude', 'service_longitude']].values)
        n_neighbors = min(50, len(svc))
        knn = NearestNeighbors(
            n_neighbors=n_neighbors, metric='haversine', algorithm='ball_tree'
        )
        knn.fit(svc_coords)

        q = np.radians([[cur['latitude'], cur['longitude']]])
        dists, idxs = knn.kneighbors(q, n_neighbors=n_neighbors)

        candidates = []
        effective_max = max_distance_km

        for d_rad, si in zip(dists[0], idxs[0]):
            s = svc.iloc[si]
            score, d_km, sent, prox, rating, budget = self._score_service(s, cur, model)
            if d_km > effective_max:
                continue
            candidates.append(self._build_candidate(s, score, d_km, sent, prox, rating, budget))

        # If fewer than top_n candidates within max_distance, expand search radius
        if len(candidates) < top_n:
            for d_rad, si in zip(dists[0], idxs[0]):
                s = svc.iloc[si]
                score, d_km, sent, prox, rating, budget = self._score_service(s, cur, model)
                if d_km <= effective_max:
                    continue  # already added

                candidates.append(self._build_candidate(s, score, d_km, sent, prox, rating, budget))
                if len(candidates) >= top_n:
                    break

        # If still fewer than top_n, try fallback to nearest place with services
        if len(candidates) < top_n:
            q = np.radians([[cur['latitude'], cur['longitude']]])
            p_dists, p_idxs = self.place_knn.kneighbors(q, n_neighbors=5)
            for pi in p_idxs[0]:
                fb = self.places.iloc[pi]
                if fb['place_id'] == place_id:
                    continue
                fb_q = np.radians([[fb['latitude'], fb['longitude']]])
                fb_dists, fb_idxs = knn.kneighbors(fb_q, n_neighbors=n_neighbors)
                for fd_rad, fsi in zip(fb_dists[0], fb_idxs[0]):
                    s = svc.iloc[fsi]
                    sname = s.get('service_name', 'Unknown')
                    if any(c['name'] == sname for c in candidates):
                        continue

                    score, d_km, sent, prox, rating, budget = self._score_service(s, cur, model)
                    candidates.append(self._build_candidate(s, score, d_km, sent, prox, rating, budget))
                    if len(candidates) >= top_n:
                        break
                if len(candidates) >= top_n:
                    break

        return candidates

    def recommend(self, place_id, service_type='Hotels', top_n=5, max_distance_km=15):
        """
        Main recommendation entry point.

        Dual-path logic:
            1. If place_id has dedicated services in the dataset (259 places):
               -> rank among those specific services using LambdaMART
            2. If place_id does NOT have dedicated services (349 places):
               -> use geographic KNN + LambdaMART from the full service pool
        """
        if not self.trained:
            raise ValueError("Model not trained.")
        cur = self.places[self.places['place_id'] == place_id]
        if cur.empty:
            return pd.DataFrame()
        cur = cur.iloc[0]

        svc = self.service_data.get(service_type)
        if svc is None or len(svc) == 0:
            return pd.DataFrame()

        model = self.lambdamart.get(service_type)

        # Dual-path: check if this place has dedicated services
        has_dedicated = (place_id in self.place_service_map
                         and service_type in self.place_service_map[place_id])

        if has_dedicated:
            # Path 1: Rank among the place's own dedicated services
            candidates = self._recommend_dedicated(place_id, cur, service_type, model, top_n)
        else:
            # Path 2: Geographic fallback for places not in the dataset
            candidates = self._recommend_geographic(
                place_id, cur, service_type, model, top_n, max_distance_km
            )

        # Sort by LambdaMART score and assign ranks
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        seen = set()
        rows = []
        rank = 1
        for c in candidates:
            if c['name'] in seen:
                continue
            seen.add(c['name'])
            rows.append({'rank': rank, **c})
            rank += 1
            if rank > top_n:
                break
        return pd.DataFrame(rows)

    def save(self, path):
        state = {
            'places': self.places, 'services': self.services,
            'service_data': self.service_data, 'lambdamart': self.lambdamart,
            'best_params': self.best_params, 'test_ndcg': self.test_ndcg,
            'place_knn': self.place_knn, 'trained': self.trained,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(state, path, compress=3)
        print(f"  Model 3 saved to {path}")

    def load(self, path):
        state = joblib.load(path)
        self.places = state['places']
        self.services = state['services']
        self.service_data = state['service_data']
        self.lambdamart = state['lambdamart']
        self.best_params = state['best_params']
        self.test_ndcg = state['test_ndcg']
        self.place_knn = state['place_knn']
        self.trained = state['trained']
        # Build the place-service map from loaded data (no retrain needed)
        self._build_place_service_map()
        print(f"  Model 3 loaded from {path}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    sub1 = joblib.load('output/preprocessed/submodel_1.joblib')
    acc = joblib.load('output/preprocessed/accommodation.joblib')
    print(f"Loaded submodel_1: {sub1.shape}")
    print(f"Loaded accommodation: {acc.shape}")
    model = Model3_NearbyEssentials(sub1, acc)
    model.train()
    model.save('output/trained_models/model_3.joblib')