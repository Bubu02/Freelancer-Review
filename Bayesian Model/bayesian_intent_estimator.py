#!/usr/bin/env python3
"""
Bayesian Intent Estimator (pgmpy)

- Trains a small Bayesian Network for purchase intent.
- Scores a validation split.
- Exports:
    - val_predictions.csv
    - intent_cpt_wide.csv  (matrix view)
    - intent_cpt_long.csv  (tidy view for analysis)

"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# pgmpy shuffled names around in newer releases; keep this import robust.
# (Yes, this looks a bit defensive, but future-you will thank past-you.)
try:
    from pgmpy.models import BayesianNetwork  # <= 0.1.24
except Exception:
    from pgmpy.models.BayesianNetwork import BayesianNetwork  # >= 0.1.25

from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# ---------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    seed: int = 42
    # Small equivalent sample size adds a bit of smoothing to CPTs
    cpd_equiv_n: float = 1.0
    outdir: Path = Path("./bn_out")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def make_synthetic(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    # Tiny synthetic dataset that roughly mimics an e-commerce funnel. Everything is discrete so pgmpy has an easy time learning CPTs.
    rng = np.random.default_rng(seed)

    traffic_source = rng.choice(["search", "ads", "direct", "social"], size=n, p=[0.45, 0.25, 0.20, 0.10])
    device = rng.choice(["mobile", "desktop"], size=n, p=[0.65, 0.35])
    prior_purchaser = rng.choice([0, 1], size=n, p=[0.80, 0.20])

    # Funnel-ish signals
    used_search = (traffic_source == "search").astype(int)
    applied_filters = (used_search * (rng.random(n) < 0.6)).astype(int)
    added_to_cart = ((applied_filters | (rng.random(n) < 0.15)) * (rng.random(n) < 0.5)).astype(int)
    reached_checkout = (added_to_cart * (rng.random(n) < 0.55)).astype(int)
    viewed_shipping = ((reached_checkout | (rng.random(n) < 0.1)) * (rng.random(n) < 0.7)).astype(int)

    # Intent: mostly driven by checkout + some prior + tiny device bump
    base = 0.05 + 0.40 * reached_checkout + 0.15 * prior_purchaser + 0.05 * (device == "desktop")
    intent = (rng.random(n) < np.clip(base, 0, 0.95)).astype(int)

    return pd.DataFrame(
        {
            "traffic_source": traffic_source,
            "device": device,
            "prior_purchaser": prior_purchaser,
            "used_search": used_search,
            "applied_filters": applied_filters,
            "added_to_cart": added_to_cart,
            "reached_checkout": reached_checkout,
            "viewed_shipping": viewed_shipping,
            "intent": intent,
        }
    )


def load_or_make(path: str | None) -> pd.DataFrame:

    # Creating a small synthetic dataset. This keeps the script runnable. Just for the absence of dataset.
    if path is None:
        logging.info("No --data provided; generating a small synthetic dataset.")
        return make_synthetic()
    df = pd.read_csv(path)
    if "intent" not in df.columns:
        raise ValueError("CSV must contain a binary column named 'intent'.")
    return df


def ensure_discrete(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    pgmpy (and BNs in general) want discrete states.
    - Integers/bools: keep as small categories (cast to str).
    - Floats: quick & dirty 4-quantile binning (again to str).
    This is a didactic choice, not a law—swap with your real preprocessing if needed.
    """
    out = df.copy()
    for c in out.columns:
        if c == target:
            continue
        if pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].astype(int).astype(str)
        elif pd.api.types.is_float_dtype(out[c]):
            try:
                out[c] = pd.qcut(out[c], q=4, duplicates="drop").astype(str)
            except Exception:
                # If qcut chokes (all constants, etc.), just coerce to str and move on.
                out[c] = out[c].astype(str)
        else:
            out[c] = out[c].astype(str)
    return out


# ---------------------------------------------------------------------
# Model: build a simple, readable DAG and fit it
# ---------------------------------------------------------------------

def build_dag(features: List[str], target: str = "intent") -> BayesianNetwork:
    """A modest funnel-shaped DAG:
      traffic_source -> used_search -> applied_filters -> added_to_cart -> reached_checkout -> intent
      And a few direct arrows into intent where that’s reasonable (device, viewed_shipping, prior_purchaser).
    Anything else we didn’t explicitly wire gets a direct edge into intent so it still contributes.
    """
    nodes = set(features) | {target}

    def have(*cols: str) -> bool:
        return all(c in nodes for c in cols)

    edges: List[Tuple[str, str]] = []
    if have("traffic_source", "used_search"):
        edges.append(("traffic_source", "used_search"))
    if have("used_search", "applied_filters"):
        edges.append(("used_search", "applied_filters"))
    if have("applied_filters", "added_to_cart"):
        edges.append(("applied_filters", "added_to_cart"))
    if have("added_to_cart", "reached_checkout"):
        edges.append(("added_to_cart", "reached_checkout"))

    for parent in ["reached_checkout", "viewed_shipping", "prior_purchaser", "device"]:
        if have(parent, target):
            edges.append((parent, target))

    # Anything not covered above still gets a say.
    for f in features:
        if f != target and not any(p == f and c == target for p, c in edges):
            edges.append((f, target))

    return BayesianNetwork(edges)


def fit_bn(
    train: pd.DataFrame,
    features: List[str],
    target: str,
    equiv_n: float,
) -> tuple[BayesianNetwork, VariableElimination]:
    # Learn CPTs with a small BDeu prior (equiv_n). Nothing fancy; just enough smoothing that rare cells don’t explode.

    model = build_dag(features, target)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            train[features + [target]],
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=equiv_n,
        )
    return model, VariableElimination(model)


# =====================================================================
# Inference utilities
# =====================================================================

def _p1_from_query(q) -> float:
    """
    Extract P(target=1) from pgmpy’s query result.
    State labels can be '0'/'1' or 0/1 depending on data—handle both.
    When in doubt, pick the numerically larger state as “positive”.
    """
    vals = np.asarray(q.values).ravel()
    names = q.state_names.get(list(q.variables)[0], None)
    if names is None:
        return float(vals[-1])
    if "1" in names:
        return float(vals[names.index("1")])
    if 1 in names:
        return float(vals[names.index(1)])
    # Fallback: try to coerce to numbers; if that fails, take the last slot.
    try:
        idx = int(np.argmax([float(str(s)) for s in names]))
    except Exception:
        idx = len(names) - 1
    return float(vals[idx])


def predict_proba(infer: VariableElimination, X: pd.DataFrame, target: str) -> np.ndarray:
    """
    P(target=1) row by row. We reset the index to avoid the classic “assign by original index”
    bug when you subselect with iloc earlier in the pipeline.
    """
    X = X.reset_index(drop=True)
    out = np.zeros(len(X), dtype=float)
    for i, row in enumerate(X.itertuples(index=False, name=None)):
        evidence = {c: v for c, v in zip(X.columns, row) if pd.notna(v)}
        q = infer.query([target], evidence=evidence, show_progress=False)
        out[i] = _p1_from_query(q)
    return out


# ---------------------------------------------------------------------
# CPT export (wide + long)
# ---------------------------------------------------------------------

def export_cpt(model: BayesianNetwork, node: str, outdir: Path) -> None:
    """
    Save the node’s CPT in two flavors:
    - wide: rows = child states, columns = every parent-state combo (truth-table style)
    - long: tidy DataFrame (one row per combo) for analysis and plotting
    """
    cpd = model.get_cpds(node)

    # Child state labels (accept ints or strings; write as strings for CSV stability)
    child_states = cpd.state_names.get(node, [])
    child_states = [str(s) for s in child_states] if child_states else [str(i) for i in range(cpd.cardinality[0])]

    # Parent axes and labels
    parents = cpd.variables[:-1]
    parent_states = [cpd.state_names[p] for p in parents] if parents else []

    # Build column labels for every parent-state combo
    if parents:
        cols = pd.MultiIndex.from_tuples(list(product(*parent_states)), names=[str(p) for p in parents])
    else:
        cols = pd.Index(["<no_parents>"])

    # Matrix is [child_states, parent_configurations]
    vals = cpd.get_values()
    cpt_wide = pd.DataFrame(vals, index=child_states, columns=cols)
    cpt_wide.index.name = f"{node}_state"
    cpt_wide.to_csv(outdir / f"{node}_cpt_wide.csv")

    # Tidy version is much easier to groupby/plot
    if parents:
        cpt_long = (
            cpt_wide
            .stack(list(range(len(parents))))
            .rename("prob")
            .reset_index()
        )
        cpt_long.columns = [f"{node}_state"] + [str(p) for p in parents] + ["prob"]
    else:
        cpt_long = pd.DataFrame({f"{node}_state": child_states, "prob": vals.ravel()})

    cpt_long.to_csv(outdir / f"{node}_cpt_long.csv", index=False)


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def eval_probs(y_true: Iterable[int], p: np.ndarray) -> dict:
    # Quick sanity metrics. If you need CIs or calibration curves, do that in the notebook.
    y = np.asarray(list(y_true)).astype(int)
    return {
        "AUROC": round(roc_auc_score(y, p), 4),
        "AUPR": round(average_precision_score(y, p), 4),
        "Brier": round(brier_score_loss(y, p), 4),
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bayesian intent estimation (pgmpy).")
    ap.add_argument("--data", type=str, default=None, help="Path to CSV with 'intent' column.")
    ap.add_argument("--outdir", type=Path, default=Path("./bn_out"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=str, default="INFO", help="DEBUG, INFO, WARNING, ...")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    cfg = TrainConfig(seed=args.seed, outdir=args.outdir)

    df = load_or_make(args.data)

    # Features we hope to find; use what's actually present so this doesn't blow up on a new dataset.
    candidates = [
        "traffic_source", "device", "prior_purchaser", "used_search",
        "applied_filters", "added_to_cart", "reached_checkout", "viewed_shipping",
    ]
    features = [c for c in candidates if c in df.columns]
    target = "intent"

    if target not in df.columns:
        raise ValueError("Expect a binary 'intent' column (0/1).")

    # Standard split
    X = df[features].copy()
    y = df[target].astype(int)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    # Discretize for BN learning. This is deliberately simple for explanation/demo.
    train_disc = ensure_discrete(pd.concat([X_tr, y_tr], axis=1), target=target)
    val_disc = ensure_discrete(pd.concat([X_va, y_va], axis=1), target=target)

    logging.info("Fitting Bayesian network...")
    model, infer = fit_bn(train_disc, features, target, equiv_n=cfg.cpd_equiv_n)

    logging.info("Scoring validation set...")
    p_val = predict_proba(infer, val_disc[features], target=target)
    metrics = eval_probs(y_va, p_val)
    logging.info("Validation metrics: %s", metrics)

    # Persist outputs for the analysis notebook
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"p_intent": p_val, "y": y_va.to_numpy()}).to_csv(cfg.outdir / "val_predictions.csv", index=False)
    export_cpt(model, node=target, outdir=cfg.outdir)
    logging.info("Saved outputs to %s", cfg.outdir.resolve())


if __name__ == "__main__":
    main()