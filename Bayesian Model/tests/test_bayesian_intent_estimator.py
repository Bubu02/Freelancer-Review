# tests/test_bayesian_intent_estimator.py
import os
from pathlib import Path

import warnings
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------
# Stabilize Windows CI/runs where multiple OpenMP runtimes get loaded.
# Harmless if already clean; prevents libiomp5md.dll duplicate aborts.
# ---------------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Skip the whole test module if pgmpy isn't installed (or import fails)
pgmpy = pytest.importorskip("pgmpy", reason="pgmpy not installed; skip BN tests")

# Import the module under test (project root must be on PYTHONPATH)
import bayesian_intent_estimator as bn


# 1) Basic synthetic data sanity: shape, columns, labels
def test_make_synthetic_shape_and_columns():
    df = bn.make_synthetic(n=200, seed=123)
    expected = {
        "traffic_source",
        "device",
        "prior_purchaser",
        "used_search",
        "applied_filters",
        "added_to_cart",
        "reached_checkout",
        "viewed_shipping",
        "intent",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) == 200
    assert set(df["intent"].unique()) <= {0, 1}


# 2) Full pipeline: fit → predict → metrics → export CPTs (+ normalization check)
def test_full_pipeline_produces_outputs(tmp_path: Path):
    df = bn.make_synthetic(n=300, seed=7)

    candidates = [
        "traffic_source",
        "device",
        "prior_purchaser",
        "used_search",
        "applied_filters",
        "added_to_cart",
        "reached_checkout",
        "viewed_shipping",
    ]
    features = [c for c in candidates if c in df.columns]
    target = "intent"

    X = df[features].copy()
    y = df[target].astype(int)

    from sklearn.model_selection import train_test_split

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # BN wants discrete states for predictors
    train_disc = bn.ensure_discrete(pd.concat([X_tr, y_tr], axis=1), target=target)
    val_disc = bn.ensure_discrete(pd.concat([X_va, y_va], axis=1), target=target)

    model, infer = bn.fit_bn(train_disc, features, target, equiv_n=1.0)

    # Target CPD must exist
    assert model.get_cpds(target) is not None

    # Predict probabilities on validation
    p_val = bn.predict_proba(infer, val_disc[features], target=target)
    assert isinstance(p_val, np.ndarray)
    assert len(p_val) == len(y_va)
    assert np.all(np.isfinite(p_val))
    assert np.all((p_val >= 0.0) & (p_val <= 1.0))

    # Metrics should compute fine
    m = bn.eval_probs(y_va, p_val)
    for k in ("AUROC", "AUPR", "Brier"):
        assert k in m and isinstance(m[k], float)

    # Export CPTs and check files exist
    outdir = tmp_path / "bn_out"
    outdir.mkdir(exist_ok=True, parents=True)
    bn.export_cpt(model, node=target, outdir=outdir)

    wide = outdir / "intent_cpt_wide.csv"
    long = outdir / "intent_cpt_long.csv"
    assert wide.exists() and wide.stat().st_size > 0
    assert long.exists() and long.stat().st_size > 0

    # CPT long sanity: probs in [0,1] and normalize per parent configuration
    cpt_long = pd.read_csv(long)
    assert "prob" in cpt_long.columns
    assert cpt_long["prob"].between(0, 1).all()

    parent_cols = [c for c in cpt_long.columns if c not in {f"{target}_state", "prob"}]
    sums = cpt_long.groupby(parent_cols, dropna=False)["prob"].sum().to_numpy()
    assert np.allclose(sums, 1.0, atol=1e-6)


# 3) Index safety: non-contiguous X index must not break predict_proba
def test_predict_proba_index_safety():
    df = bn.make_synthetic(n=120, seed=99)
    features = [
        "traffic_source",
        "device",
        "prior_purchaser",
        "used_search",
        "applied_filters",
        "added_to_cart",
        "reached_checkout",
        "viewed_shipping",
    ]
    target = "intent"

    # Make a gappy index to catch position vs label mistakes
    X = df[features].iloc[::2].copy()
    y = df[target].iloc[::2].astype(int)

    from sklearn.model_selection import train_test_split

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.25, random_state=123, stratify=y
    )
    train_disc = bn.ensure_discrete(pd.concat([X_tr, y_tr], axis=1), target=target)
    val_disc = bn.ensure_discrete(pd.concat([X_va, y_va], axis=1), target=target)

    model, infer = bn.fit_bn(train_disc, features, target, equiv_n=1.0)
    p = bn.predict_proba(infer, val_disc[features], target=target)

    assert len(p) == len(y_va)
    assert np.all((p >= 0) & (p <= 1))


# 4) Reproducibility: same seed → identical synthetic dataset
def test_make_synthetic_reproducible():
    df1 = bn.make_synthetic(n=200, seed=123)
    df2 = bn.make_synthetic(n=200, seed=123)
    assert df1.equals(df2)


# 5) Discretization behavior: ints/bools→categories; floats→binned
def test_ensure_discrete_casts_and_bins():
    raw = pd.DataFrame(
        {
            "int_col": [0, 1, 1, 0, 1],
            "bool_col": [True, False, True, False, True],
            "float_col": [0.1, 0.2, 0.9, 0.4, 0.6],
            "intent": [0, 1, 0, 1, 0],
        }
    )
    disc = bn.ensure_discrete(raw, target="intent")
    # Non-target columns become object (string labels)
    assert disc["int_col"].dtype == object
    assert disc["bool_col"].dtype == object
    assert disc["float_col"].dtype == object
    # Target remains numeric
    assert pd.api.types.is_integer_dtype(disc["intent"]) or pd.api.types.is_bool_dtype(
        disc["intent"]
    )


# 6) Loader guardrail: CSV without 'intent' should raise
def test_load_or_make_raises_without_intent(tmp_path: Path):
    bad = tmp_path / "no_intent.csv"
    pd.DataFrame({"foo": [1, 2, 3]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="must contain a binary column named 'intent'"):
        _ = bn.load_or_make(str(bad))


# 7) CPT export when target has **no parents** (only intent column present)
def test_export_cpt_no_parents(tmp_path: Path):
    import warnings
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import BayesianEstimator

    df = pd.DataFrame({"intent": [0, 1, 0, 1, 1, 0, 1, 0]})
    target = "intent"

    # Discretize (no-op here except coercions)
    train_disc = bn.ensure_discrete(df, target=target)

    # Build a one-node BN (no parents)
    model = BayesianNetwork()
    model.add_node(target)

    # Fit prior CPD for 'intent'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train_disc[[target]], estimator=BayesianEstimator)

    # Export CPTs
    outdir = tmp_path / "bn_out"
    outdir.mkdir(parents=True, exist_ok=True)
    bn.export_cpt(model, node=target, outdir=outdir)

    # Check wide CSV has "<no_parents>" and sums to ~1 across child states
    wide = pd.read_csv(outdir / "intent_cpt_wide.csv")
    assert "<no_parents>" in wide.columns
    assert np.isclose(wide["<no_parents>"].sum(), 1.0, atol=1e-6)


