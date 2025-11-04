# tests/test_bayesian_intent_estimator.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------
# Avoid OpenMP duplicate runtime crashes on some Windows setups.
# If the env is already clean this is a no-op; if not, it prevents aborts.
# (We prefer stability over flakiness in CI.)
# ---------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Skip the whole module if pgmpy isn't available (or torch issues)
pgmpy = pytest.importorskip("pgmpy", reason="pgmpy not installed; skip BN tests")

import bayesian_intent_estimator as bn  # make sure PYTHONPATH includes project root


def test_make_synthetic_shape_and_columns():
    df = bn.make_synthetic(n=200, seed=123)
    # Must have the 'intent' target and at least a few expected predictors
    expected = {
        "traffic_source", "device", "prior_purchaser", "used_search",
        "applied_filters", "added_to_cart", "reached_checkout", "viewed_shipping",
        "intent",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) == 200
    assert set(df["intent"].unique()) <= {0, 1}


def test_full_pipeline_produces_outputs(tmp_path: Path):
    # Small synthetic dataset to keep tests quick
    df = bn.make_synthetic(n=300, seed=7)

    # Mirror script behavior
    candidates = [
        "traffic_source", "device", "prior_purchaser", "used_search",
        "applied_filters", "added_to_cart", "reached_checkout", "viewed_shipping",
    ]
    features = [c for c in candidates if c in df.columns]
    target = "intent"

    X = df[features].copy()
    y = df[target].astype(int)

    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Discretize (BN wants discrete states)
    train_disc = bn.ensure_discrete(pd.concat([X_tr, y_tr], axis=1), target=target)
    val_disc = bn.ensure_discrete(pd.concat([X_va, y_va], axis=1), target=target)

    # Fit BN
    model, infer = bn.fit_bn(train_disc, features, target, equiv_n=1.0)

    # Model should have a CPD for the target
    assert model.get_cpds(target) is not None

    # Predict probabilities
    p_val = bn.predict_proba(infer, val_disc[features], target=target)
    assert isinstance(p_val, np.ndarray)
    assert len(p_val) == len(y_va)
    assert np.all(np.isfinite(p_val))
    assert np.all((p_val >= 0.0) & (p_val <= 1.0))

    # Metrics should compute without error
    m = bn.eval_probs(y_va, p_val)
    for k in ("AUROC", "AUPR", "Brier"):
        assert k in m
        assert isinstance(m[k], float)

    # Export CPTs to a temp outdir
    outdir = tmp_path / "bn_out"
    outdir.mkdir(exist_ok=True, parents=True)
    bn.export_cpt(model, node=target, outdir=outdir)

    # Files must exist and be non-empty
    wide = outdir / "intent_cpt_wide.csv"
    long = outdir / "intent_cpt_long.csv"
    assert wide.exists() and wide.stat().st_size > 0
    assert long.exists() and long.stat().st_size > 0

    # Basic sanity on CPT content
    cpt_long = pd.read_csv(long, header=0)
    assert "prob" in cpt_long.columns
    assert cpt_long["prob"].between(0, 1).all()

    # Each parent configuration should normalize over child states (~1.0)
    # Build a list of parent columns (everything except the child-state and prob)
    parent_cols = [c for c in cpt_long.columns if c not in {f"{target}_state", "prob"}]
    sums = (
        cpt_long
        .groupby(parent_cols, dropna=False)["prob"]
        .sum()
        .to_numpy()
    )
    assert np.allclose(sums, 1.0, atol=1e-6)


def test_predict_proba_index_safety(tmp_path: Path):
    # Regression: ensure no index mismatch when X has a non-contiguous index
    df = bn.make_synthetic(n=120, seed=99)
    features = [
        "traffic_source", "device", "prior_purchaser", "used_search",
        "applied_filters", "added_to_cart", "reached_checkout", "viewed_shipping",
    ]
    target = "intent"
    X = df[features].iloc[::2].copy()   # make a gappy index
    y = df[target].iloc[::2].astype(int)

    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, random_state=123, stratify=y
    )
    train_disc = bn.ensure_discrete(pd.concat([X_tr, y_tr], axis=1), target=target)
    val_disc = bn.ensure_discrete(pd.concat([X_va, y_va], axis=1), target=target)

    model, infer = bn.fit_bn(train_disc, features, target, equiv_n=1.0)
    p = bn.predict_proba(infer, val_disc[features], target=target)

    # length should match; no IndexError
    assert len(p) == len(y_va)
    # and probabilities are valid
    assert np.all((p >= 0) & (p <= 1))