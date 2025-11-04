from __future__ import annotations
import argparse
import warnings
from typing import List
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

# pgmpy import changed in newer versions; this keeps the script working across versions.
try:
    from pgmpy.models import BayesianModel  # pgmpy <= 0.1.24
except Exception:  # pragma: no cover
    from pgmpy.models.BayesianNetwork import BayesianNetwork as BayesianModel  # pgmpy >= 0.1.25

from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# Data helpers (It can be modified when I get the real dataset for now I have used a synthetic data for testing.)

def make_synthetic(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    # I have created a tiny synthetic dataset that mimics a purchase funnel. All features are discrete/categorical so BN learning is straightforward.
    rng = np.random.default_rng(seed)

    traffic_source = rng.choice(['search', 'ads', 'direct', 'social'], size=n, p=[0.45, 0.25, 0.20, 0.10])
    device = rng.choice(['mobile', 'desktop'], size=n, p=[0.65, 0.35])
    prior_purchaser = rng.choice([0, 1], size=n, p=[0.80, 0.20])

    # Funnel steps,added a bit of noise to make the data feel more like what you'd see in the real world.
    used_search = (traffic_source == 'search').astype(int)
    applied_filters = (used_search * (rng.random(n) < 0.6)).astype(int)
    added_to_cart = ((applied_filters | (rng.random(n) < 0.15)) * (rng.random(n) < 0.5)).astype(int)
    reached_checkout = (added_to_cart * (rng.random(n) < 0.55)).astype(int)
    viewed_shipping = ((reached_checkout | (rng.random(n) < 0.1)) * (rng.random(n) < 0.7)).astype(int)

    # Intent depends on checkout step, prior purchaser, and a small device effect
    base = 0.05 + 0.40 * reached_checkout + 0.15 * prior_purchaser + 0.05 * (device == 'desktop')
    intent = (rng.random(n) < np.clip(base, 0, 0.95)).astype(int)

    df = pd.DataFrame({
        'traffic_source': traffic_source,
        'device': device,
        'prior_purchaser': prior_purchaser,
        'used_search': used_search,
        'applied_filters': applied_filters,
        'added_to_cart': added_to_cart,
        'reached_checkout': reached_checkout,
        'viewed_shipping': viewed_shipping,
        'intent': intent,
    })
    return df


def load_or_make(data_path: str | None) -> pd.DataFrame:
    if data_path is None:
        print("[info] No --data provided; generating a small synthetic dataset.")
        return make_synthetic(n=2500)
    df = pd.read_csv(data_path)
    if 'intent' not in df.columns:
        raise ValueError("CSV must contain a binary column named 'intent'.")
    return df


# BN model: build, fit, predict

def build_simple_dag(features: List[str]) -> BayesianModel:
    # A small, readable funnel-shaped DAG ending at 'intent'. Only adds edges for nodes present in the dataset.

    nodes = set(features) | {'intent'}

    def have(*cols):
        return all(c in nodes for c in cols)

    edges = []
    if have('traffic_source', 'used_search'):
        edges.append(('traffic_source', 'used_search'))
    if have('used_search', 'applied_filters'):
        edges.append(('used_search', 'applied_filters'))
    if have('applied_filters', 'added_to_cart'):
        edges.append(('applied_filters', 'added_to_cart'))
    if have('added_to_cart', 'reached_checkout'):
        edges.append(('added_to_cart', 'reached_checkout'))
    if have('reached_checkout', 'intent'):
        edges.append(('reached_checkout', 'intent'))
    if have('viewed_shipping', 'intent'):
        edges.append(('viewed_shipping', 'intent'))
    if have('prior_purchaser', 'intent'):
        edges.append(('prior_purchaser', 'intent'))
    if have('device', 'intent'):
        edges.append(('device', 'intent'))

    # Safety: connect any remaining feature directly to intent so it contributes
    for f in features:
        if f != 'intent' and not any(child == 'intent' and parent == f for parent, child in edges):
            edges.append((f, 'intent'))

    model = BayesianModel(edges)
    return model


def ensure_discrete(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    # Convert all non-target columns to string or int categories. BN learning in pgmpy expects discrete states; this keeps the tutorial simple.

    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        if pd.api.types.is_float_dtype(out[c]):
            # Bin continuous columns into 4 quantiles for a quick demo
            try:
                out[c] = pd.qcut(out[c], q=4, duplicates='drop').astype(str)
            except Exception:
                out[c] = out[c].astype(str)
        elif pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_bool_dtype(out[c]):
            # Keep ints/bools as small categories (strings)
            out[c] = out[c].astype(int).astype(str)
        else:
            out[c] = out[c].astype(str)
    return out


def fit_bn(train: pd.DataFrame, features: List[str]) -> tuple[BayesianModel, VariableElimination]:
    model = build_simple_dag(features)
    # Using a tiny equivalent sample size to smooth rare states
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.fit(train[features + ['intent']],
                  estimator=BayesianEstimator,
                  prior_type='BDeu',
                  equivalent_sample_size=1.0)
    infer = VariableElimination(model)
    return model, infer


def predict_proba_intent(infer, X: pd.DataFrame) -> np.ndarray:
    # Returns P(intent=1) for each row in X. Handles both int [0,1] and str ['0','1'] state names safely. Also avoids index-mismatch by resetting index.

    X = X.reset_index(drop=True)  # avoid out-of-bounds writes
    p = np.zeros(len(X), dtype=float)

    def _p1_from_query(q) -> float:
        # flatten values
        vals = np.asarray(q.values).ravel()
        # try to read state names
        names = q.state_names.get('intent', None)
        if names is None:
            # no names? assume last entry is intent=1
            return float(vals[-1])

        # match either '1' or 1
        if '1' in names:
            idx = names.index('1')
        elif 1 in names:
            idx = names.index(1)
        else:
            # fallback: pick the "larger" state as positive
            try:
                nums = [float(str(s)) for s in names]
                idx = int(np.argmax(nums))
            except Exception:
                idx = len(names) - 1
        return float(vals[idx])

    for pos, row in enumerate(X.itertuples(index=False, name=None)):
        evidence = {col: val for col, val in zip(X.columns, row) if pd.notna(val)}
        q = infer.query(variables=['intent'], evidence=evidence, show_progress=False)
        p[pos] = _p1_from_query(q)

    return p


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default=None, help='Path to CSV with intent labels (0/1).')
    ap.add_argument('--outdir', type=str, default='./bn_out')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    df = load_or_make(args.data)

    # Keeping a small, clear set of columns if present
    candidate_cols = [
        'traffic_source', 'device', 'prior_purchaser', 'used_search', 'applied_filters',
        'added_to_cart', 'reached_checkout', 'viewed_shipping'
    ]
    features = [c for c in candidate_cols if c in df.columns]
    if 'intent' not in df.columns:
        raise ValueError("Expect a binary 'intent' column (0/1)")

    # Train/test splitting
    X = df[features].copy()
    y = df['intent'].astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    # Ensure discrete categories for BN learning
    train_disc = ensure_discrete(pd.concat([X_train, y_train], axis=1), exclude=['intent'])
    val_disc = ensure_discrete(pd.concat([X_val, y_val], axis=1), exclude=['intent'])

    # Fit BN
    model, infer = fit_bn(train_disc, features)

    # Predict P(intent=1) on validation
    p_val = predict_proba_intent(infer, val_disc[features])

    # Simple metrics to check performance
    auroc = roc_auc_score(y_val, p_val)
    aupr = average_precision_score(y_val, p_val)
    brier = brier_score_loss(y_val, p_val)

    print("\nValidation metrics (simple):")
    print({"AUROC": round(auroc, 4), "AUPR": round(aupr, 4), "Brier": round(brier, 4)})


    # Exporting intent CPTs
    outdir = Path(args.outdir) # define output directory if missing
    outdir.mkdir(parents=True, exist_ok=True)

    cpd_intent = model.get_cpds('intent')

    # handle both int and str state names
    intent_states = list(cpd_intent.state_names.get('intent', []))
    intent_states = [str(s) for s in intent_states] if intent_states else [str(i) for i in
                                                                           range(cpd_intent.cardinality[0])]

    # geting parent variables and their states
    parents = cpd_intent.variables[:-1]  # all except the 'intent'
    parent_states = [cpd_intent.state_names[p] for p in parents] if parents else []

    # building MultiIndex columns for parent state combos
    if parents:
        col_tuples = list(product(*parent_states))
        col_index = pd.MultiIndex.from_tuples(col_tuples, names=[str(p) for p in parents])
    else:
        col_index = pd.Index(["<no_parents>"])

    # building DataFrame (rows = intent states, columns = parent combos)
    vals = cpd_intent.get_values()
    cpt_wide = pd.DataFrame(vals, index=intent_states, columns=col_index)
    cpt_wide.index.name = 'intent_state'

    # saving both wide and tidy versions
    cpt_wide.to_csv(outdir / 'intent_cpt_wide.csv')

    if parents:
        cpt_long = (
            cpt_wide.stack(list(range(len(parents))))
            .rename('prob')
            .reset_index()
        )
        cpt_long.columns = ['intent_state'] + [str(p) for p in parents] + ['prob']
    else:
        cpt_long = pd.DataFrame({'intent_state': intent_states, 'prob': vals.ravel()})

    cpt_long.to_csv(outdir / 'intent_cpt_long.csv', index=False)

    print(f"Saved CPTs:\n- {outdir / 'intent_cpt_wide.csv'} (matrix)\n- {outdir / 'intent_cpt_long.csv'} (tidy)")

    # Also saving the validation predictions
    pd.DataFrame({"p_intent": p_val, "y": y_val.to_numpy()}).to_csv(outdir / 'val_predictions.csv', index=False)
    print(f"Saved validation predictions to: {outdir / 'val_predictions.csv'}\n")


if __name__ == '__main__':
    main()