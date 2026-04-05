"""
Chapter 8 – Evaluation & Testing
==================================
Comprehensive evaluation of the Tourism Recommendation System.

Two evaluation approaches:

Evaluation 1 (Rule-Based):
    Uses internal ranking data to evaluate LambdaMART predictions.
    Binary classification: relevance >= 2 → Relevant, else Not Relevant.

Evaluation 2 (Criteria-Based):
    Uses actual recommendation outputs evaluated against independent criteria
    that LambdaMART was NOT trained to optimise:

    Model 1 (Similar Places):
        Relevant if avg_rating >= 3.5 AND distance_km <= 15

    Model 2 (Popular Nearby):
        Relevant if avg_rating >= 3.5 AND distance_km <= 15

    Model 3 (Nearby Services):
        Relevant if service_avg_rating >= 3.5 AND distance_km <= 10

Outputs:
    1. Model Test table (Table 22 style)
    2. Classification reports (Figure 25 style)
    3. Confusion matrix tables (Table 23 style)
    4. Confusion matrix heatmaps (Figure 26 style)
    5. Model comparison summary

All figures saved to output/evaluation/
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
import joblib
import warnings

warnings.filterwarnings('ignore')

from utils import haversine_distance
from preprocessing import are_districts_adjacent
from models.model_1_you_may_also_like import Model1_YouMayAlsoLike
from models.model_2_popular_nearby import Model2_PopularNearby
from models.model_3_nearby_essentials import Model3_NearbyEssentials

OUT_DIR = os.path.join('output', 'evaluation')
os.makedirs(OUT_DIR, exist_ok=True)


def _ndcg_at_k(scores, k=5):
    rel = np.array(scores[:k], dtype=float)
    if rel.sum() == 0:
        return 0.0
    dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))
    ideal = np.sort(rel)[::-1]
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg > 0 else 0.0


def _diversity(districts):
    if not districts:
        return 0.0
    return len(set(districts)) / len(districts)


def _binarise(labels, threshold=2):
    return (np.array(labels) >= threshold).astype(int)


def _predict_binary(model_lgb, X, y_true_bin):
    scores = model_lgb.predict(X)
    true_relevant_ratio = y_true_bin.mean()
    cutoff = np.percentile(scores, 100 * (1 - true_relevant_ratio))
    return (scores >= cutoff).astype(int), scores


# ═══════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════

def _save_classification_report_image(report_dict, title, path):
    rows = []
    ordered_keys = [k for k in report_dict if k not in ('accuracy', 'macro avg', 'weighted avg')]
    for cls_name in ordered_keys:
        m = report_dict[cls_name]
        rows.append([cls_name,
                     f"{m['precision']:.2f}",
                     f"{m['recall']:.2f}",
                     f"{m['f1-score']:.2f}",
                     f"{int(m['support'])}"])

    for avg_key in ('accuracy', 'macro avg', 'weighted avg'):
        if avg_key in report_dict:
            m = report_dict[avg_key]
            if avg_key == 'accuracy':
                rows.append([avg_key,
                             f"{m:.2f}" if isinstance(m, float) else f"{m['precision']:.2f}",
                             f"{m:.2f}" if isinstance(m, float) else f"{m['recall']:.2f}",
                             f"{m:.2f}" if isinstance(m, float) else f"{m['f1-score']:.2f}",
                             ""])
            else:
                rows.append([avg_key,
                             f"{m['precision']:.2f}",
                             f"{m['recall']:.2f}",
                             f"{m['f1-score']:.2f}",
                             f"{int(m['support'])}"])

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.8))
    ax.axis('off')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    col_labels = ['', 'Precision', 'Recall', 'F1-Score', 'Support']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            cell = table[i, j]
            if i >= len(rows) - 1:
                cell.set_facecolor('#ecf0f1')
                cell.set_text_props(fontweight='bold')
            elif i % 2 == 0:
                cell.set_facecolor('#f7f9fa')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def _save_confusion_matrix_image(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_ylabel('Actual', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            colour = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, f'{cm[i, j]}',
                    ha='center', va='center', color=colour,
                    fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def _save_confusion_table(cm, labels, title, path):
    rows = []
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        rows = [
            ['True Positive',
             'The number of relevant recommendations correctly classified as Relevant.',
             str(tp)],
            ['True Negative',
             'The number of not-relevant recommendations correctly classified as Not Relevant.',
             str(tn)],
            ['False Positive',
             'The number of not-relevant recommendations misclassified as Relevant.',
             str(fp)],
            ['False Negative',
             'The number of relevant recommendations misclassified as Not Relevant.',
             str(fn)],
        ]
    else:
        for i, lbl in enumerate(labels):
            tp_i = cm[i, i]
            rows.append(['True Positive',
                         f'{lbl} correctly classified as {lbl}.',
                         str(tp_i)])
        for i, lbl_true in enumerate(labels):
            for j, lbl_pred in enumerate(labels):
                if i != j and cm[i, j] > 0:
                    rows.append(['Misclassification',
                                 f'{lbl_true} misclassified as {lbl_pred}.',
                                 str(cm[i, j])])

    fig, ax = plt.subplots(figsize=(13, 0.55 * len(rows) + 1.8))
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    col_labels = ['Metric', 'Description', 'Value']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                     cellLoc='center', colWidths=[0.16, 0.66, 0.08])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)

    for j in range(3):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(3):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#f7f9fa')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def _save_model_test_table(rows_data, title, path):
    fig, ax = plt.subplots(figsize=(14, 0.6 * len(rows_data) + 2.0))
    ax.axis('off')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    col_labels = ['Model', 'Description', 'Train\nAccuracy', 'Test\nAccuracy', 'Remark']
    table = ax.table(cellText=rows_data, colLabels=col_labels, loc='center',
                     cellLoc='center',
                     colWidths=[0.16, 0.38, 0.12, 0.12, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#f7f9fa')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


def _save_model_comparison_image(comparison_data, path):
    rows = []
    for entry in comparison_data:
        rows.append([
            entry['model'],
            entry['approach'],
            f"{entry['precision']:.2f}",
            f"{entry['recall']:.2f}",
            f"{entry['f1']:.2f}",
            f"{entry['accuracy']:.2f}",
            f"{int(entry['samples'])}",
        ])

    fig, ax = plt.subplots(figsize=(15, 0.6 * len(rows) + 2.2))
    ax.axis('off')
    ax.set_title('Model Comparison — Tourism Recommendation System',
                 fontsize=15, fontweight='bold', pad=20)

    col_labels = ['Model', 'Approach', 'Precision', 'Recall',
                  'F1-Score', 'Accuracy', 'Samples']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center',
                     cellLoc='center',
                     colWidths=[0.18, 0.30, 0.09, 0.09, 0.09, 0.09, 0.09])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#f7f9fa')

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation 1: Rule-Based (internal ranking data)
# ═══════════════════════════════════════════════════════════════════════

def _make_ranking_data_m1(model, test_indices, max_dist=20):
    from sklearn.metrics.pairwise import cosine_similarity
    sbert_sim = cosine_similarity(model.sbert_embeddings)
    gnn_sim = cosine_similarity(model.gnn_embeddings)
    feats, labels, groups = [], [], []

    for qi in test_indices:
        q = model.df.iloc[qi]
        q_coords = np.radians([[q['latitude'], q['longitude']]])
        dists, idxs = model.knn_model.kneighbors(q_coords, n_neighbors=min(30, len(model.df)))
        gf, gl = [], []

        for d_rad, ci in zip(dists[0], idxs[0]):
            if ci == qi:
                continue
            d_km = d_rad * 6371
            if d_km > max_dist:
                continue
            c = model.df.iloc[ci]
            same_district = 1.0 if c['district'] == q['district'] else 0.0
            has_bad = c.get('has_shared_coords', False)
            adjacent = are_districts_adjacent(q['district'], c['district'])
            if has_bad and not same_district and not adjacent:
                coord_trust = 0.0
            elif has_bad and not same_district:
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
            prox = max(0.0, 1.0 - d_km / max_dist)
            rel = prox * 0.40 + (c['avg_rating'] / 5.0 if c['avg_rating'] > 0 else 0.0) * 0.25 \
                   + same_district * 0.25 + coord_trust * 0.10
            gf.append(f)
            gl.append(min(4, int(rel * 4)))

        if len(gf) >= 3:
            feats.extend(gf)
            labels.extend(gl)
            groups.append(len(gf))

    return (np.array(feats) if feats else np.empty((0, 7)),
            np.array(labels), groups)


def _make_ranking_data_m2(model, test_indices, max_dist=20):
    feats, labels, groups = [], [], []

    for qi in test_indices:
        q = model.df.iloc[qi]
        q_coords = np.radians([[q['latitude'], q['longitude']]])
        dists, idxs = model.knn_model.kneighbors(q_coords, n_neighbors=min(30, len(model.df)))
        gf, gl = [], []

        for d_rad, ci in zip(dists[0], idxs[0]):
            if ci == qi:
                continue
            d_km = haversine_distance(q['latitude'], q['longitude'],
                                      model.df.iloc[ci]['latitude'], model.df.iloc[ci]['longitude'])
            if d_km > max_dist:
                continue
            c = model.df.iloc[ci]
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
            rel = prox_f * 0.40 + rating * 0.25 + max(0, sent) * 0.10 \
                  + kaggle * 0.05 + same_d * 0.15 + coord_trust * 0.05
            gl.append(min(4, int(rel * 4)))

        if len(gf) >= 3:
            feats.extend(gf)
            labels.extend(gl)
            groups.append(len(gf))

    return (np.array(feats) if feats else np.empty((0, 7)),
            np.array(labels), groups)


def _make_ranking_data_m3(model, stype, test_indices, max_dist=15):
    from sklearn.neighbors import NearestNeighbors
    svc = model.service_data.get(stype)
    if svc is None or len(svc) < 5:
        return np.empty((0, 5)), np.array([]), []

    svc_coords = svc[['service_latitude', 'service_longitude']].values
    knn = NearestNeighbors(n_neighbors=min(20, len(svc)),
                           metric='haversine', algorithm='ball_tree')
    knn.fit(np.radians(svc_coords))
    feats, labels, groups = [], [], []

    for pi in test_indices:
        p = model.places.iloc[pi]
        q = np.radians([[p['latitude'], p['longitude']]])
        dists, idxs = knn.kneighbors(q, n_neighbors=min(20, len(svc)))
        gf, gl = [], []

        for d_rad, si in zip(dists[0], idxs[0]):
            s = svc.iloc[si]
            d_km = haversine_distance(p['latitude'], p['longitude'],
                                      s['service_latitude'], s['service_longitude'])
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

    return (np.array(feats) if feats else np.empty((0, 5)),
            np.array(labels), groups)


def evaluate_model_rulebased(model_name, lgb_model, X_test, y_test, g_test, suffix):
    """Evaluation 1: Rule-based binary classification from ranking data."""
    y_true_bin = _binarise(y_test)
    y_pred_bin, scores = _predict_binary(lgb_model, X_test, y_true_bin)

    ndcg_scores = []
    offset = 0
    for g in g_test:
        chunk = list(zip(scores[offset:offset + g], y_test[offset:offset + g]))
        chunk.sort(key=lambda x: x[0], reverse=True)
        ndcg_scores.append(_ndcg_at_k([c[1] for c in chunk]))
        offset += g
    avg_ndcg = np.mean(ndcg_scores)

    report = classification_report(y_true_bin, y_pred_bin,
                                   target_names=['Not Relevant', 'Relevant'],
                                   output_dict=True, zero_division=0)
    report_text = classification_report(y_true_bin, y_pred_bin,
                                        target_names=['Not Relevant', 'Relevant'],
                                        zero_division=0)
    print(f"\n    Classification Report:\n{report_text}")
    print(f"    NDCG@5: {avg_ndcg:.4f}")

    cm = confusion_matrix(y_true_bin, y_pred_bin)

    _save_classification_report_image(
        report, f'{model_name} — Classification Report (Rule-Based)',
        os.path.join(OUT_DIR, f'eval1_classification_report_{suffix}.png'))
    _save_confusion_matrix_image(
        cm, ['Not Relevant', 'Relevant'],
        f'{model_name} — Confusion Matrix (Rule-Based)',
        os.path.join(OUT_DIR, f'eval1_confusion_matrix_{suffix}.png'))
    _save_confusion_table(
        cm, ['Not Relevant', 'Relevant'],
        f'Table: {model_name} — Confusion Matrix (Rule-Based)',
        os.path.join(OUT_DIR, f'eval1_confusion_table_{suffix}.png'))

    return {
        'model': model_name, 'ndcg': avg_ndcg,
        'precision': precision_score(y_true_bin, y_pred_bin, zero_division=0),
        'recall': recall_score(y_true_bin, y_pred_bin, zero_division=0),
        'f1': f1_score(y_true_bin, y_pred_bin, zero_division=0),
        'accuracy': accuracy_score(y_true_bin, y_pred_bin),
        'report': report, 'cm': cm, 'samples': len(y_test),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Evaluation 2: Criteria-Based (actual recommendation outputs)
# ═══════════════════════════════════════════════════════════════════════

def _eval2_model1(model1, test_place_ids):
    print("\n  ── Model 1: Similar Places (Criteria-Based) ──")
    y_true, y_pred = [], []
    evaluated = 0
    for pid in test_place_ids:
        recs = model1.recommend(pid, top_n=5, max_distance_km=20)
        if recs.empty:
            continue
        evaluated += 1
        for _, row in recs.iterrows():
            is_relevant = (row['rating'] >= 3.5) and (row['distance_km'] <= 15)
            y_true.append(1 if is_relevant else 0)
            if row['rank'] <= 3 and row['final_score'] > 0:
                y_pred.append(1)
            elif row['final_score'] > np.median(recs['final_score'].values):
                y_pred.append(1)
            else:
                y_pred.append(0)
    print(f"    Evaluated {evaluated} query places, {len(y_true)} recommendations")
    return np.array(y_true), np.array(y_pred)


def _eval2_model2(model2, test_place_ids):
    print("\n  ── Model 2: Popular Places Nearby (Criteria-Based) ──")
    y_true, y_pred = [], []
    evaluated = 0
    for pid in test_place_ids:
        recs = model2.recommend(pid, top_n=5, max_distance_km=20)
        if recs.empty:
            continue
        evaluated += 1
        for _, row in recs.iterrows():
            is_relevant = (row['rating'] >= 3.5) and (row['distance_km'] <= 15)
            y_true.append(1 if is_relevant else 0)
            if row['rank'] <= 3 and row['final_score'] > 0:
                y_pred.append(1)
            elif row['final_score'] > np.median(recs['final_score'].values):
                y_pred.append(1)
            else:
                y_pred.append(0)
    print(f"    Evaluated {evaluated} query places, {len(y_true)} recommendations")
    return np.array(y_true), np.array(y_pred)


def _eval2_model3(model3, test_place_ids):
    print("\n  ── Model 3: Nearby Services (Criteria-Based) ──")
    y_true, y_pred = [], []
    evaluated = 0
    for pid in test_place_ids:
        for stype in ['Hotels', 'Dining', 'Activities']:
            recs = model3.recommend(pid, service_type=stype, top_n=5, max_distance_km=15)
            if recs.empty:
                continue
            evaluated += 1
            for _, row in recs.iterrows():
                is_relevant = (row['rating'] >= 3.5) and (row['distance_km'] <= 10)
                y_true.append(1 if is_relevant else 0)
                if row['rank'] <= 3 and row['final_score'] > 0:
                    y_pred.append(1)
                elif row['final_score'] > np.median(recs['final_score'].values):
                    y_pred.append(1)
                else:
                    y_pred.append(0)
    print(f"    Evaluated {evaluated} query-service pairs, {len(y_true)} recommendations")
    return np.array(y_true), np.array(y_pred)


def _compute_train_accuracy(model, train_place_ids, model_num):
    y_true, y_pred = [], []
    for pid in train_place_ids[:80]:
        if model_num == 1:
            recs = model.recommend(pid, top_n=5, max_distance_km=20)
        elif model_num == 2:
            recs = model.recommend(pid, top_n=5, max_distance_km=20)
        else:
            recs = model.recommend(pid, service_type='Hotels', top_n=5, max_distance_km=15)
        if recs.empty:
            continue
        for _, row in recs.iterrows():
            if model_num in (1, 2):
                is_relevant = (row['rating'] >= 3.5) and (row['distance_km'] <= 15)
            else:
                is_relevant = (row['rating'] >= 3.5) and (row['distance_km'] <= 10)
            y_true.append(1 if is_relevant else 0)
            if row['rank'] <= 3 and row['final_score'] > 0:
                y_pred.append(1)
            elif row['final_score'] > np.median(recs['final_score'].values):
                y_pred.append(1)
            else:
                y_pred.append(0)
    if y_true:
        return accuracy_score(y_true, y_pred)
    return 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Overall 3-class classification report
# ═══════════════════════════════════════════════════════════════════════

def overall_classification_report(results, prefix=''):
    print(f"\n  ── Overall Recommendation System — Classification Report {prefix} ──")

    class_names = ['Similar Places', 'Popular Sites', 'Nearby Services']
    precisions, recalls, f1s, supports = [], [], [], []

    for res in results:
        precisions.append(res['precision'])
        recalls.append(res['recall'])
        f1s.append(res['f1'])
        supports.append(res['samples'])

    total_support = sum(supports)
    weights = [s / total_support if total_support > 0 else 0 for s in supports]

    macro_p = np.mean(precisions)
    macro_r = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    weighted_p = sum(p * w for p, w in zip(precisions, weights))
    weighted_r = sum(r * w for r, w in zip(recalls, weights))
    weighted_f1 = sum(f * w for f, w in zip(f1s, weights))

    overall_correct = sum(r['cm'].diagonal().sum() for r in results)
    overall_total = sum(r['cm'].sum() for r in results)
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0

    report_dict = {}
    for name, p, r, f, s in zip(class_names, precisions, recalls, f1s, supports):
        report_dict[name] = {'precision': p, 'recall': r, 'f1-score': f, 'support': s}
    report_dict['accuracy'] = overall_acc
    report_dict['macro avg'] = {'precision': macro_p, 'recall': macro_r,
                                'f1-score': macro_f1, 'support': total_support}
    report_dict['weighted avg'] = {'precision': weighted_p, 'recall': weighted_r,
                                   'f1-score': weighted_f1, 'support': total_support}

    print(f"\n    {'':>18s} {'precision':>10s} {'recall':>10s} {'f1-score':>10s} {'support':>10s}")
    print("    " + "-" * 58)
    for name in class_names:
        m = report_dict[name]
        print(f"    {name:>18s} {m['precision']:>10.2f} {m['recall']:>10.2f} "
              f"{m['f1-score']:>10.2f} {m['support']:>10d}")
    print("    " + "-" * 58)
    print(f"    {'accuracy':>18s} {'':>10s} {'':>10s} "
          f"{overall_acc:>10.2f} {overall_total:>10d}")
    print(f"    {'macro avg':>18s} {macro_p:>10.2f} {macro_r:>10.2f} "
          f"{macro_f1:>10.2f} {total_support:>10d}")
    print(f"    {'weighted avg':>18s} {weighted_p:>10.2f} {weighted_r:>10.2f} "
          f"{weighted_f1:>10.2f} {total_support:>10d}")

    tag = prefix.strip('() ').replace(' ', '_').lower() if prefix else ''
    fname = f'overall_classification_report_{tag}.png' if tag else 'overall_classification_report.png'
    _save_classification_report_image(
        report_dict,
        f'Recommendation System — Classification Report {prefix}',
        os.path.join(OUT_DIR, fname))

    return report_dict


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("CHAPTER 8 — EVALUATION & TESTING")
    print("Tourism Recommendation System")
    print("=" * 80)

    # ── Load data & models ─────────────────────────────────────────────
    print("\n[1/8] Loading preprocessed data & trained models...")

    sub1 = joblib.load('output/preprocessed/submodel_1.joblib')
    sub2 = joblib.load('output/preprocessed/submodel_2.joblib')
    acc = joblib.load('output/preprocessed/accommodation.joblib')
    print(f"    Submodel 1: {sub1.shape}")
    print(f"    Submodel 2: {sub2.shape}")
    print(f"    Accommodation: {acc.shape}")

    model1 = Model1_YouMayAlsoLike(sub1)
    model1.load('output/trained_models/model_1.joblib')

    model2 = Model2_PopularNearby(sub2)
    model2.load('output/trained_models/model_2.joblib')

    model3 = Model3_NearbyEssentials(sub1, acc)
    model3.load('output/trained_models/model_3.joblib')

    # ── Train/Test split ───────────────────────────────────────────────
    rng = np.random.RandomState(42)
    all_idx_m1 = np.arange(len(model1.df))
    rng.shuffle(all_idx_m1)
    split_m1 = int(0.8 * len(all_idx_m1))
    test_idx_m1 = all_idx_m1[split_m1:]

    rng2 = np.random.RandomState(42)
    all_idx_m2 = rng2.permutation(len(model2.df))
    split_m2 = int(0.8 * len(all_idx_m2))
    test_idx_m2 = all_idx_m2[split_m2:]

    rng3 = np.random.RandomState(42)
    all_idx_m3 = rng3.permutation(len(model3.places))
    split_m3 = int(0.8 * len(all_idx_m3))
    test_idx_m3 = all_idx_m3[split_m3:]

    # ══════════════════════════════════════════════════════════════════
    #  EVALUATION 1: Rule-Based (internal ranking data)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("EVALUATION 1: RULE-BASED (Internal Ranking Data)")
    print("=" * 80)

    print("\n[2/8] Evaluating models using rule-based ground truth...")

    print("\n  ── Model 1: Similar Places ──")
    X_test_m1, y_test_m1, g_test_m1 = _make_ranking_data_m1(model1, test_idx_m1)
    eval1_m1 = None
    if len(X_test_m1) > 0:
        eval1_m1 = evaluate_model_rulebased(
            'Model 1 (Similar Places)', model1.lambdamart_model,
            X_test_m1, y_test_m1, g_test_m1, 'model1')

    print("\n  ── Model 2: Popular Places ──")
    X_test_m2, y_test_m2, g_test_m2 = _make_ranking_data_m2(model2, test_idx_m2)
    eval1_m2 = None
    if len(X_test_m2) > 0:
        eval1_m2 = evaluate_model_rulebased(
            'Model 2 (Popular Nearby)', model2.lambdamart_model,
            X_test_m2, y_test_m2, g_test_m2, 'model2')

    print("\n  ── Model 3: Nearby Services ──")
    all_y_true_m3, all_y_pred_m3 = [], []
    ndcg_all_m3 = []
    for stype in ['Hotels', 'Dining', 'Activities']:
        lm = model3.lambdamart.get(stype)
        if lm is None:
            continue
        X_test, y_test, g_test = _make_ranking_data_m3(model3, stype, test_idx_m3)
        if len(X_test) == 0:
            continue
        scores = lm.predict(X_test)
        y_true_bin = _binarise(y_test)
        true_relevant_ratio = y_true_bin.mean()
        cutoff = np.percentile(scores, 100 * (1 - true_relevant_ratio))
        y_pred_bin = (scores >= cutoff).astype(int)
        all_y_true_m3.extend(y_true_bin)
        all_y_pred_m3.extend(y_pred_bin)
        offset = 0
        for g in g_test:
            chunk = list(zip(scores[offset:offset + g], y_test[offset:offset + g]))
            chunk.sort(key=lambda x: x[0], reverse=True)
            ndcg_all_m3.append(_ndcg_at_k([c[1] for c in chunk]))
            offset += g
        print(f"    {stype}: {len(X_test)} candidate pairs evaluated")

    eval1_m3 = None
    if all_y_true_m3:
        y_true_m3 = np.array(all_y_true_m3)
        y_pred_m3 = np.array(all_y_pred_m3)
        avg_ndcg_m3 = np.mean(ndcg_all_m3)
        report = classification_report(y_true_m3, y_pred_m3,
                                       target_names=['Not Relevant', 'Relevant'],
                                       output_dict=True, zero_division=0)
        report_text = classification_report(y_true_m3, y_pred_m3,
                                            target_names=['Not Relevant', 'Relevant'],
                                            zero_division=0)
        print(f"\n    Classification Report:\n{report_text}")
        print(f"    NDCG@5: {avg_ndcg_m3:.4f}")
        cm = confusion_matrix(y_true_m3, y_pred_m3)

        _save_classification_report_image(
            report, 'Model 3 (Nearby Services) — Classification Report (Rule-Based)',
            os.path.join(OUT_DIR, 'eval1_classification_report_model3.png'))
        _save_confusion_matrix_image(
            cm, ['Not Relevant', 'Relevant'],
            'Model 3 — Confusion Matrix (Rule-Based)',
            os.path.join(OUT_DIR, 'eval1_confusion_matrix_model3.png'))
        _save_confusion_table(
            cm, ['Not Relevant', 'Relevant'],
            'Table: Model 3 — Confusion Matrix (Rule-Based)',
            os.path.join(OUT_DIR, 'eval1_confusion_table_model3.png'))

        eval1_m3 = {
            'model': 'Model 3 (Nearby Services)',
            'approach': 'BERT Sentiment + LambdaMART (per type)',
            'ndcg': avg_ndcg_m3,
            'precision': precision_score(y_true_m3, y_pred_m3, zero_division=0),
            'recall': recall_score(y_true_m3, y_pred_m3, zero_division=0),
            'f1': f1_score(y_true_m3, y_pred_m3, zero_division=0),
            'accuracy': accuracy_score(y_true_m3, y_pred_m3),
            'report': report, 'cm': cm, 'samples': len(y_true_m3),
        }

    print("\n[3/8] Generating Evaluation 1 overall report...")
    eval1_results = [r for r in [eval1_m1, eval1_m2, eval1_m3] if r is not None]
    if eval1_results:
        overall_classification_report(eval1_results, '(Rule-Based)')

    # ══════════════════════════════════════════════════════════════════
    #  EVALUATION 2: Criteria-Based (actual recommendation outputs)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("EVALUATION 2: CRITERIA-BASED (Actual Recommendation Outputs)")
    print("=" * 80)

    print("\n[4/8] Creating train/test split for criteria evaluation...")
    rng_c = np.random.RandomState(42)
    all_idx_c1 = np.arange(len(sub1))
    rng_c.shuffle(all_idx_c1)
    split_c1 = int(0.8 * len(all_idx_c1))
    train_ids_c1 = sub1.iloc[all_idx_c1[:split_c1]]['place_id'].tolist()
    test_ids_c1 = sub1.iloc[all_idx_c1[split_c1:]]['place_id'].tolist()

    rng_c2 = np.random.RandomState(42)
    all_idx_c2 = rng_c2.permutation(len(sub2))
    split_c2 = int(0.8 * len(all_idx_c2))
    train_ids_c2 = sub2.iloc[all_idx_c2[:split_c2]]['place_id'].tolist()
    test_ids_c2 = sub2.iloc[all_idx_c2[split_c2:]]['place_id'].tolist()

    rng_c3 = np.random.RandomState(42)
    all_idx_c3 = rng_c3.permutation(len(sub1))
    split_c3 = int(0.8 * len(all_idx_c3))
    train_ids_c3 = sub1.iloc[all_idx_c3[:split_c3]]['place_id'].tolist()
    test_ids_c3 = sub1.iloc[all_idx_c3[split_c3:]]['place_id'].tolist()

    print(f"    Model 1: {len(train_ids_c1)} train, {len(test_ids_c1)} test")
    print(f"    Model 2: {len(train_ids_c2)} train, {len(test_ids_c2)} test")
    print(f"    Model 3: {len(train_ids_c3)} train, {len(test_ids_c3)} test")

    print("\n[5/8] Evaluating models on actual recommendation outputs...")
    y_true_c1, y_pred_c1 = _eval2_model1(model1, test_ids_c1)
    y_true_c2, y_pred_c2 = _eval2_model2(model2, test_ids_c2)
    y_true_c3, y_pred_c3 = _eval2_model3(model3, test_ids_c3)

    print("\n[6/8] Computing train accuracy for overfitting analysis...")
    train_acc_c1 = _compute_train_accuracy(model1, train_ids_c1, 1)
    train_acc_c2 = _compute_train_accuracy(model2, train_ids_c2, 2)
    train_acc_c3 = _compute_train_accuracy(model3, train_ids_c3, 3)

    test_acc_c1 = accuracy_score(y_true_c1, y_pred_c1) if len(y_true_c1) else 0
    test_acc_c2 = accuracy_score(y_true_c2, y_pred_c2) if len(y_true_c2) else 0
    test_acc_c3 = accuracy_score(y_true_c3, y_pred_c3) if len(y_true_c3) else 0

    def _remark(train, test):
        diff = train - test
        if diff > 0.10:
            return "Overfitting"
        elif diff < -0.05:
            return "Underfitting"
        else:
            return "Good fit"

    print("\n    Model Test Summary (Criteria-Based):")
    print(f"    {'Model':<35s} {'Train Acc':>10s} {'Test Acc':>10s} {'Remark':<15s}")
    print("    " + "-" * 70)
    for name, tr, te in [
        ("SBERT + GNN + LambdaMART", train_acc_c1, test_acc_c1),
        ("BERT + Kaggle + LambdaMART", train_acc_c2, test_acc_c2),
        ("BERT + per-type LambdaMART", train_acc_c3, test_acc_c3),
    ]:
        print(f"    {name:<35s} {tr:>10.0%} {te:>10.0%} {_remark(tr, te):<15s}")

    model_test_rows = [
        ['SBERT + GNN +\nLambdaMART',
         'SBERT embeddings + GNN graph\nencoding + LambdaMART ranking.',
         f"{train_acc_c1:.0%}", f"{test_acc_c1:.0%}", _remark(train_acc_c1, test_acc_c1)],
        ['BERT Sentiment +\nKaggle + LambdaMART',
         'BERT sentiment + 35,434 Kaggle\nreviews + LambdaMART ranking.',
         f"{train_acc_c2:.0%}", f"{test_acc_c2:.0%}", _remark(train_acc_c2, test_acc_c2)],
        ['BERT + per-type\nLambdaMART',
         'Separate LambdaMART per service\ntype (Hotels/Dining/Activities).',
         f"{train_acc_c3:.0%}", f"{test_acc_c3:.0%}", _remark(train_acc_c3, test_acc_c3)],
    ]

    _save_model_test_table(
        model_test_rows,
        'Table: Recommendation System — Model Test (Criteria-Based)',
        os.path.join(OUT_DIR, 'eval2_model_test_table.png'))

    print("\n[7/8] Generating Evaluation 2 classification reports and confusion matrices...")

    model_configs = [
        ('Model 1 (Similar Places)', 'SBERT + GNN + LambdaMART',
         y_true_c1, y_pred_c1, 'model1'),
        ('Model 2 (Popular Nearby)', 'BERT Sentiment + Kaggle + LambdaMART',
         y_true_c2, y_pred_c2, 'model2'),
        ('Model 3 (Nearby Services)', 'BERT Sentiment + LambdaMART (per type)',
         y_true_c3, y_pred_c3, 'model3'),
    ]

    eval2_results = []
    for model_name, approach, y_true, y_pred, suffix in model_configs:
        print(f"\n    {model_name}:")
        report = classification_report(y_true, y_pred,
                                       target_names=['Not Relevant', 'Relevant'],
                                       output_dict=True, zero_division=0)
        report_text = classification_report(y_true, y_pred,
                                            target_names=['Not Relevant', 'Relevant'],
                                            zero_division=0)
        print(f"{report_text}")
        cm = confusion_matrix(y_true, y_pred)

        _save_classification_report_image(
            report, f'{model_name} — Classification Report (Criteria-Based)',
            os.path.join(OUT_DIR, f'eval2_classification_report_{suffix}.png'))
        _save_confusion_matrix_image(
            cm, ['Not Relevant', 'Relevant'],
            f'{model_name} — Confusion Matrix (Criteria-Based)',
            os.path.join(OUT_DIR, f'eval2_confusion_matrix_{suffix}.png'))
        _save_confusion_table(
            cm, ['Not Relevant', 'Relevant'],
            f'Table: {model_name} — Confusion Matrix (Criteria-Based)',
            os.path.join(OUT_DIR, f'eval2_confusion_table_{suffix}.png'))

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1_val = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)

        eval2_results.append({
            'model': model_name, 'approach': approach,
            'precision': prec, 'recall': rec, 'f1': f1_val, 'accuracy': acc,
            'report': report, 'cm': cm, 'samples': len(y_true),
        })

    overall_classification_report(eval2_results, '(Criteria-Based)')
    _save_model_comparison_image(eval2_results,
                                 os.path.join(OUT_DIR, 'eval2_model_comparison.png'))

    # ── Final Summary ──────────────────────────────────────────────────
    print("\n[8/8] Final Summary")
    print("=" * 80)
    print("EVALUATION 1 (Rule-Based):")
    print(f"    {'Model':<28s} {'NDCG@5':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Acc':>8s}")
    print("    " + "-" * 68)
    for r in eval1_results:
        print(f"    {r['model']:<28s} {r.get('ndcg', 0):>8.4f} {r['precision']:>8.2f} "
              f"{r['recall']:>8.2f} {r['f1']:>8.2f} {r['accuracy']:>8.2f}")

    print("\nEVALUATION 2 (Criteria-Based):")
    print(f"    {'Model':<28s} {'Train':>8s} {'Test':>8s} {'Prec':>8s} {'Recall':>8s} "
          f"{'F1':>8s} {'Remark':<15s}")
    print("    " + "-" * 85)
    for r, (tr, te) in zip(eval2_results, [(train_acc_c1, test_acc_c1),
                                            (train_acc_c2, test_acc_c2),
                                            (train_acc_c3, test_acc_c3)]):
        print(f"    {r['model']:<28s} {tr:>8.0%} {te:>8.0%} {r['precision']:>8.2f} "
              f"{r['recall']:>8.2f} {r['f1']:>8.2f} {_remark(tr, te):<15s}")

    print(f"\n  All evaluation outputs saved to: {os.path.abspath(OUT_DIR)}/")
    print("    Evaluation 1 (Rule-Based):")
    print("      eval1_classification_report_model[1-3].png")
    print("      eval1_confusion_matrix_model[1-3].png")
    print("      eval1_confusion_table_model[1-3].png")
    print("    Evaluation 2 (Criteria-Based):")
    print("      eval2_model_test_table.png")
    print("      eval2_classification_report_model[1-3].png")
    print("      eval2_confusion_matrix_model[1-3].png")
    print("      eval2_confusion_table_model[1-3].png")
    print("      eval2_model_comparison.png")
    print("    Overall:")
    print("      overall_classification_report_rule-based.png")
    print("      overall_classification_report_criteria-based.png")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
