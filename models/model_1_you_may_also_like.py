"""
Model 1: You May Also Like
===========================
SBERT Semantic Similarity + Place-Place GNN + LambdaMART Ranking

Architecture:
    1. SBERT (all-MiniLM-L6-v2) encodes place descriptions for semantic similarity
    2. Place-Place GNN (2-layer GCN) learns embeddings from graph structure
    3. LambdaMART (LightGBM) ranks candidates using learned feature weights

Ground-Truth Relevance (0-4):
    40% proximity + 25% rating + 25% same_district + 10% coordinate_trust
"""

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
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
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2'
SBERT_LOCAL_DIR = os.path.join(PROJECT_ROOT, 'models', 'sbert_local')
SBERT_CACHE_DIR = os.path.join(PROJECT_ROOT, 'models', 'sbert_cache')


def _load_sbert():
    if os.path.isdir(SBERT_LOCAL_DIR) and os.listdir(SBERT_LOCAL_DIR):
        return SentenceTransformer(SBERT_LOCAL_DIR)
    os.makedirs(SBERT_CACHE_DIR, exist_ok=True)
    model = SentenceTransformer(SBERT_MODEL_NAME, cache_folder=SBERT_CACHE_DIR)
    os.makedirs(SBERT_LOCAL_DIR, exist_ok=True)
    model.save(SBERT_LOCAL_DIR)
    return model


class PlaceGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index)
        return x


class Model1_YouMayAlsoLike:
    def __init__(self, submodel_1_df):
        self.df = submodel_1_df.reset_index(drop=True)
        self.sbert_model = None
        self.sbert_embeddings = None
        self.gnn_model = None
        self.gnn_embeddings = None
        self.knn_model = None
        self.lambdamart_model = None
        self.best_params = None
        self.test_ndcg = None
        self.trained = False

    def _build_graph(self):
        n = len(self.df)
        src, dst = [], []
        districts = self.df['district'].values
        categories = self.df['place_category'].values
        lats, lons = self.df['latitude'].values, self.df['longitude'].values
        sbert_sim = cosine_similarity(self.sbert_embeddings)

        for i in range(n):
            for j in range(i + 1, n):
                edge = False
                if districts[i] == districts[j]:
                    edge = True
                if categories[i] == categories[j]:
                    edge = True
                d = haversine_distance(lats[i], lons[i], lats[j], lons[j])
                if d <= 10.0:
                    edge = True
                if sbert_sim[i, j] > 0.5:
                    edge = True
                if edge:
                    src.extend([i, j])
                    dst.extend([j, i])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        x = torch.tensor(self.sbert_embeddings, dtype=torch.float32)
        graph = Data(x=x, edge_index=edge_index)
        print(f"    Graph: {n} nodes, {len(src) // 2} edges")
        return graph

    def _train_gnn(self, graph, epochs=100, lr=0.01):
        model = PlaceGNN(graph.x.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        n = graph.x.shape[0]
        adj_true = torch.zeros(n, n)
        adj_true[graph.edge_index[0], graph.edge_index[1]] = 1.0

        model.train()
        for ep in range(epochs):
            opt.zero_grad()
            emb = model(graph)
            pred = torch.sigmoid(emb @ emb.t())
            loss = F.binary_cross_entropy(pred, adj_true)
            loss.backward()
            opt.step()
            if (ep + 1) % 50 == 0:
                print(f"      GNN epoch {ep + 1}/{epochs}  loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            out = model(graph).numpy()
        self.gnn_model = model
        return out

    def _make_ranking_data(self, query_indices, max_dist=20):
        sbert_sim = cosine_similarity(self.sbert_embeddings)
        gnn_sim = cosine_similarity(self.gnn_embeddings)
        feats, labels, groups = [], [], []

        for qi in query_indices:
            q = self.df.iloc[qi]
            q_coords = np.radians([[q['latitude'], q['longitude']]])
            dists, idxs = self.knn_model.kneighbors(q_coords, n_neighbors=min(30, len(self.df)))
            gf, gl = [], []

            for d_rad, ci in zip(dists[0], idxs[0]):
                if ci == qi:
                    continue
                d_km = d_rad * 6371
                if d_km > max_dist:
                    continue
                c = self.df.iloc[ci]
                same_district = 1.0 if c['district'] == q['district'] else 0.0
                has_bad_coords = c.get('has_shared_coords', False)
                adjacent = are_districts_adjacent(q['district'], c['district'])

                if has_bad_coords and not same_district and not adjacent:
                    coord_trust = 0.0
                elif has_bad_coords and not same_district:
                    coord_trust = 0.5
                else:
                    coord_trust = 1.0

                f = [
                    sbert_sim[qi, ci], gnn_sim[qi, ci],
                    1.0 / (1.0 + d_km / 5.0), same_district,
                    1.0 if c['place_category'] == q['place_category'] else 0.0,
                    c['avg_rating'] / 5.0 if c['avg_rating'] > 0 else 0.0,
                    coord_trust,
                ]
                prox_factor = max(0.0, 1.0 - d_km / max_dist)
                rating_factor = c['avg_rating'] / 5.0 if c['avg_rating'] > 0 else 0.0
                relevance = (prox_factor * 0.40 + rating_factor * 0.25 +
                             same_district * 0.25 + coord_trust * 0.10)
                gf.append(f)
                gl.append(min(4, int(relevance * 4)))

            if len(gf) >= 3:
                feats.extend(gf)
                labels.extend(gl)
                groups.append(len(gf))

        return (np.array(feats) if feats else np.empty((0, 7)), np.array(labels), groups)

    def _tune_lambdamart(self, X_train, y_train, groups_train):
        param_grid = [
            {'learning_rate': 0.05, 'num_leaves': 15, 'num_boost_round': 80},
            {'learning_rate': 0.05, 'num_leaves': 31, 'num_boost_round': 100},
            {'learning_rate': 0.1, 'num_leaves': 31, 'num_boost_round': 80},
            {'learning_rate': 0.1, 'num_leaves': 15, 'num_boost_round': 120},
        ]
        best_score, best_cfg = -1, param_grid[0]
        for cfg in param_grid:
            ds = lgb.Dataset(X_train, label=y_train, group=groups_train)
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
        print("TRAINING MODEL 1: YOU MAY ALSO LIKE")
        print("  SBERT + Place-Place GNN + LambdaMART")
        print("=" * 70)

        print("\n[1/6] SBERT encoding...")
        self.sbert_model = _load_sbert()
        self.sbert_embeddings = self.sbert_model.encode(
            self.df['combined_features'].tolist(), show_progress_bar=True, batch_size=32
        )
        print(f"    Shape: {self.sbert_embeddings.shape}")

        print("\n[2/6] Building graph + training GNN...")
        graph = self._build_graph()
        self.gnn_embeddings = self._train_gnn(graph, epochs=100)
        print(f"    GNN embeddings: {self.gnn_embeddings.shape}")

        print("\n[3/6] KNN fitting...")
        coords = self.df[['latitude', 'longitude']].values
        self.knn_model = NearestNeighbors(
            n_neighbors=min(30, len(self.df)), metric='haversine', algorithm='ball_tree'
        )
        self.knn_model.fit(np.radians(coords))

        print("\n[4/6] Train/test split (80/20)...")
        rng = np.random.RandomState(42)
        all_idx = np.arange(len(self.df))
        rng.shuffle(all_idx)
        split = int(0.8 * len(all_idx))
        train_idx, test_idx = all_idx[:split], all_idx[split:]
        print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")

        X_train, y_train, g_train = self._make_ranking_data(train_idx)
        X_test, y_test, g_test = self._make_ranking_data(test_idx)

        print("\n[5/6] Hyperparameter tuning...")
        self.best_params = self._tune_lambdamart(X_train, y_train, g_train)

        print("\n[6/6] Training final LambdaMART...")
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
        names = ['sbert_sim', 'gnn_sim', 'proximity', 'same_district',
                 'same_category', 'rating', 'coord_trust']
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

        self.trained = True
        print("\n" + "=" * 70)
        print("MODEL 1 TRAINING COMPLETE")
        print("=" * 70)

    def recommend(self, place_id, top_n=5, max_distance_km=20):
        if not self.trained:
            raise ValueError("Model not trained.")
        cur = self.df[self.df['place_id'] == place_id]
        if cur.empty:
            return pd.DataFrame()
        cur = cur.iloc[0]
        ci = self.df[self.df['place_id'] == place_id].index[0]

        q_coords = np.radians([[cur['latitude'], cur['longitude']]])
        dists, idxs = self.knn_model.kneighbors(q_coords, n_neighbors=min(30, len(self.df)))

        sbert_sim_row = cosine_similarity(
            self.sbert_embeddings[ci:ci + 1], self.sbert_embeddings
        ).flatten()
        gnn_sim_row = cosine_similarity(
            self.gnn_embeddings[ci:ci + 1], self.gnn_embeddings
        ).flatten()

        candidates = []
        for d_rad, idx in zip(dists[0], idxs[0]):
            if idx == ci:
                continue
            d_km = haversine_distance(
                cur['latitude'], cur['longitude'],
                self.df.iloc[idx]['latitude'], self.df.iloc[idx]['longitude']
            )
            if d_km > max_distance_km:
                continue

            p = self.df.iloc[idx]
            same_district = p['district'] == cur['district']
            has_bad_coords = p.get('has_shared_coords', False)
            adjacent = are_districts_adjacent(cur['district'], p['district'])

            if has_bad_coords and not same_district and not adjacent:
                continue
            coord_trust = 0.5 if (has_bad_coords and not same_district) else 1.0

            feat = np.array([[
                sbert_sim_row[idx], gnn_sim_row[idx],
                1.0 / (1.0 + d_km / 5.0),
                1.0 if same_district else 0.0,
                1.0 if p['place_category'] == cur['place_category'] else 0.0,
                p['avg_rating'] / 5.0 if p['avg_rating'] > 0 else 0.0,
                coord_trust,
            ]])
            score = self.lambdamart_model.predict(feat)[0]
            candidates.append({
                'place': p, 'distance_km': round(d_km, 2),
                'sbert_score': round(float(sbert_sim_row[idx]), 3),
                'gnn_score': round(float(gnn_sim_row[idx]), 3),
                'proximity_score': round(1.0 / (1.0 + d_km / 5.0), 3),
                'final_score': round(float(score), 3),
            })

        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        rows = []
        for rank, c in enumerate(candidates[:top_n], 1):
            p = c['place']
            rows.append({
                'rank': rank, 'name': p['place_name'],
                'image': p.get('image_url', ''), 'type': p['place_category'],
                'district': p['district'], 'rating': round(p['avg_rating'], 1),
                'review': p.get('display_review', ''), 'distance_km': c['distance_km'],
                'sbert_score': c['sbert_score'], 'gnn_score': c['gnn_score'],
                'proximity_score': c['proximity_score'], 'final_score': c['final_score'],
            })
        return pd.DataFrame(rows)

    def save(self, path):
        state = {
            'df': self.df, 'sbert_embeddings': self.sbert_embeddings,
            'gnn_embeddings': self.gnn_embeddings, 'knn_model': self.knn_model,
            'lambdamart_model': self.lambdamart_model, 'best_params': self.best_params,
            'test_ndcg': self.test_ndcg, 'trained': self.trained,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(state, path, compress=3)
        print(f"  Model 1 saved to {path}")

    def load(self, path):
        state = joblib.load(path)
        self.df = state['df']
        self.sbert_embeddings = state['sbert_embeddings']
        self.gnn_embeddings = state['gnn_embeddings']
        self.knn_model = state['knn_model']
        self.lambdamart_model = state['lambdamart_model']
        self.best_params = state['best_params']
        self.test_ndcg = state['test_ndcg']
        self.trained = state['trained']
        self.sbert_model = _load_sbert()
        print(f"  Model 1 loaded from {path}")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    sub1 = joblib.load('output/preprocessed/submodel_1.joblib')
    print(f"Loaded submodel_1: {sub1.shape}")
    model = Model1_YouMayAlsoLike(sub1)
    model.train()
    model.save('output/trained_models/model_1.joblib')