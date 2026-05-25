#!/usr/bin/env python3
"""
train_model.py — Train a CW classifier from user-labeled data.

Reads training_data.csv (produced by cwiden --validate),
trains a gradient boosted tree ensemble, and exports the model
as a JSON file that the C++ runtime can load for inference.

Usage:
    python3 scripts/train_model.py training_data.csv --output model.json
"""

import argparse
import csv
import json
import sys
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix


FEATURE_NAMES = [
    'snr_db', 'effective_bw', 'shape_factor', 'headroom_db',
    'bimodality_coeff', 'on_off_ratio', 'rhythm_score', 'wpm_estimate',
    'spectral_entropy', 'peak_stability',
]


def load_csv(path: str):
    """Load labeled rows from training_data.csv."""
    X, y, freqs = [], [], []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row.get('user_label', -1))
            if label < 0:
                continue  # skip unlabeled rows

            features = []
            for name in FEATURE_NAMES:
                val = row.get(name, '0')
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    features.append(0.0)

            X.append(features)
            y.append(label)
            freqs.append(float(row.get('freq_hz', 0)))

    return np.array(X), np.array(y), np.array(freqs)


def tree_to_dict(tree, feature_names):
    """Convert a sklearn DecisionTreeRegressor to a JSON-serializable dict."""
    t = tree.tree_
    
    def recurse(node_id):
        if t.children_left[node_id] == -1:  # leaf
            return {'leaf': float(t.value[node_id][0, 0])}
        
        feature_idx = int(t.feature[node_id])
        threshold = float(t.threshold[node_id])
        
        return {
            'feature': feature_idx,
            'feature_name': feature_names[feature_idx] if feature_idx < len(feature_names) else f'f{feature_idx}',
            'threshold': threshold,
            'left': recurse(int(t.children_left[node_id])),
            'right': recurse(int(t.children_right[node_id])),
        }
    
    return recurse(0)


def export_model(clf, feature_names, output_path, stats):
    """Export GBT model as JSON for C++ inference."""
    model = {
        'version': 1,
        'type': 'gradient_boosting',
        'n_estimators': len(clf.estimators_),
        'learning_rate': clf.learning_rate,
        'init_value': float(clf.init_.class_prior_[1]) if hasattr(clf.init_, 'class_prior_') else 0.0,
        'feature_names': feature_names,
        'feature_importances': [float(x) for x in clf.feature_importances_],
        'trees': [],
        'stats': stats,
    }

    # Compute init raw prediction (log-odds)
    p = float(clf.init_.class_prior_[1]) if hasattr(clf.init_, 'class_prior_') else 0.5
    p = max(1e-6, min(1 - 1e-6, p))
    model['init_raw'] = float(np.log(p / (1 - p)))

    for i, stage in enumerate(clf.estimators_):
        tree = stage[0]
        model['trees'].append(tree_to_dict(tree, feature_names))

    with open(output_path, 'w') as f:
        json.dump(model, f, indent=2)

    print(f"Model saved to {output_path}")
    print(f"  Trees: {len(model['trees'])}")
    print(f"  Features: {len(feature_names)}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train CW classifier from labeled data')
    parser.add_argument('input', help='Path to training_data.csv')
    parser.add_argument('--output', '-o', default='model.json', help='Output model path')
    parser.add_argument('--estimators', '-n', type=int, default=100, help='Number of trees')
    parser.add_argument('--depth', '-d', type=int, default=4, help='Max tree depth')
    parser.add_argument('--rate', '-r', type=float, default=0.1, help='Learning rate')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading data from {args.input}...")
    X, y, freqs = load_csv(args.input)

    if len(y) == 0:
        print("ERROR: No labeled rows found. Label signals with Y/N in --validate mode first.", file=sys.stderr)
        sys.exit(1)

    n_cw = int(np.sum(y == 1))
    n_noise = int(np.sum(y == 0))
    print(f"  Labeled samples: {len(y)} ({n_cw} CW, {n_noise} noise)")

    if n_cw < 5 or n_noise < 5:
        print("WARNING: Need at least 5 of each class for reliable training.", file=sys.stderr)
        if n_cw == 0 or n_noise == 0:
            print("ERROR: Both CW and noise labels required.", file=sys.stderr)
            sys.exit(1)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)

    # Train
    print(f"\nTraining GBT ({args.estimators} trees, depth={args.depth}, lr={args.rate})...")
    clf = GradientBoostingClassifier(
        n_estimators=args.estimators,
        max_depth=args.depth,
        learning_rate=args.rate,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validation
    if len(y) >= 10:
        n_splits = min(5, min(n_cw, n_noise))
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            print(f"  Cross-val accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    clf.fit(X, y)

    # Evaluation
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    print(f"\nTraining accuracy: {(y_pred == y).mean():.1%}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Noise', 'CW']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    print("\nFeature Importance:")
    importances = clf.feature_importances_
    for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        bar = '#' * int(imp * 40)
        print(f"  {name:20s} {imp:.3f} {bar}")

    # Stats
    stats = {
        'n_samples': int(len(y)),
        'n_cw': n_cw,
        'n_noise': n_noise,
        'train_accuracy': float((y_pred == y).mean()),
    }

    # Export
    export_model(clf, FEATURE_NAMES, args.output, stats)

    # Show uncertain samples (for active learning reference)
    uncertain = np.abs(y_prob - 0.5) < 0.15
    if uncertain.any():
        print(f"\nMost uncertain samples ({uncertain.sum()}):")
        for idx in np.where(uncertain)[0][:10]:
            print(f"  {freqs[idx]/1000:.3f} kHz  prob={y_prob[idx]:.2f}  label={'CW' if y[idx] else 'noise'}")


if __name__ == '__main__':
    main()
