#!/usr/bin/env python3
"""
CatBoost Intent Estimator

- Trains a CatBoostClassifier for purchase intent.
- Handles mixed tabular data and categorical columns automatically.
- Evaluates on a validation split and (optionally) calibrates probabilities.
- Exports:
    - val_predictions.csv
    - feature_importances.csv
    - catboost_model.cbm  (serialized model)

"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.2
    seed: int = 42
    # Reasonable tutorial defaults; tune later
    iterations: int = 1200
    depth: int = 6
    learning_rate: float = 0.08
    l2_leaf_reg: float = 3.0
    # Calibration is helpful if you feed probs into downstream pricing logic
    calibrate: bool = True
    outdir: Path = Path("./cb_out")


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------

def load_data(path: Optional[str]) -> pd.DataFrame:
    """
    Load a CSV with a binary 'intent' column.
    We keep it strict here—no synthetic fallback like the BN script.
    """
    if not path:
        raise ValueError("Please provide --data /path/to/sessions.csv")
    df = pd.read_csv(path)
    if "intent" not in df.columns:
        raise ValueError("CSV must contain a binary column named 'intent'.")
    return df


def detect_categorical_columns(X: pd.DataFrame) -> List[int]:
    """
    CatBoost needs column indices for categoricals.
    We treat object/category/bool as categorical; the rest are numeric.
    """
    cat_idx: List[int] = []
    for i, c in enumerate(X.columns):
        dtype = X[c].dtype
        if str(dtype).startswith("category") or dtype == object or str(dtype) == "bool":
            cat_idx.append(i)
    return cat_idx


# ---------------------------------------------------------------------
# Training / Inference
# ---------------------------------------------------------------------

def train_catboost(
    X_tr: pd.DataFrame,
    y_tr: Iterable[int],
    X_va: Optional[pd.DataFrame],
    y_va: Optional[Iterable[int]],
    cfg: TrainConfig,
) -> tuple[CatBoostClassifier, Optional[IsotonicRegression], List[int]]:
    """
    Fit CatBoost with basic params and optional isotonic calibration.
    We return the fitted model, a calibrator (or None), and the categorical column indices.
    """
    cat_idx = detect_categorical_columns(X_tr)
    y_tr = np.asarray(list(y_tr)).astype(int)

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)

    model = CatBoostClassifier(
        iterations=cfg.iterations,
        depth=cfg.depth,
        learning_rate=cfg.learning_rate,
        l2_leaf_reg=cfg.l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="AUC",
        verbose=100,
        random_seed=cfg.seed,
        allow_writing_files=False,  # keep local dir clean
    )

    eval_set = None
    if X_va is not None and y_va is not None:
        eval_set = Pool(X_va, np.asarray(list(y_va)).astype(int), cat_features=cat_idx)

    model.fit(train_pool, eval_set=eval_set)

    calibrator = None
    if cfg.calibrate and X_va is not None and y_va is not None:
        p_raw = model.predict_proba(X_va)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(p_raw, np.asarray(list(y_va)).astype(int))

    return model, calibrator, cat_idx


def predict_proba(model: CatBoostClassifier, X: pd.DataFrame, calibrator: Optional[IsotonicRegression], cat_idx: List[int]) -> np.ndarray:
    """
    Predict probabilities; if a calibrator is present, map raw probs -> calibrated probs.
    """
    pool = Pool(X, cat_features=cat_idx)
    p = model.predict_proba(pool)[:, 1]
    if calibrator is not None:
        p = calibrator.transform(p)
    return p


def eval_probs(y_true: Iterable[int], p: np.ndarray) -> dict:
    """Quick sanity metrics for classification."""
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
    ap = argparse.ArgumentParser(description="CatBoost intent estimator")
    ap.add_argument("--data", type=str, required=True, help="Path to CSV with 'intent' column.")
    ap.add_argument("--outdir", type=Path, default=Path("./cb_out"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=str, default="INFO", help="DEBUG, INFO, WARNING, ...")
    # Model hyperparams (keep overridable from CLI)
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--l2", type=float, default=3.0)
    ap.add_argument("--no_calibration", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    cfg = TrainConfig(
        seed=args.seed,
        iterations=args.iters,
        depth=args.depth,
        learning_rate=args.lr,
        l2_leaf_reg=args.l2,
        calibrate=not args.no_calibration,
        outdir=args.outdir,
    )

    df = load_data(args.data)

    # Grab features present in your dataset (be permissive; don't explode on missing ones)
    candidates = [
        "traffic_source", "device", "prior_purchaser", "used_search",
        "applied_filters", "added_to_cart", "reached_checkout", "viewed_shipping",
    ]
    features = [c for c in candidates if c in df.columns]
    target = "intent"

    if target not in df.columns:
        raise ValueError("Expect a binary 'intent' column (0/1).")

    X = df[features].copy()
    y = df[target].astype(int)

    # Standard stratified split; reproducible by seed
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    logging.info("Training CatBoost...")
    model, calibrator, cat_idx = train_catboost(X_tr, y_tr, X_va, y_va, cfg)

    logging.info("Scoring validation set...")
    p_val = predict_proba(model, X_va, calibrator, cat_idx)
    metrics = eval_probs(y_va, p_val)
    logging.info("Validation metrics: %s", metrics)

    # Outputs
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    # 1) predictions
    pd.DataFrame({"p_intent": p_val, "y": y_va.to_numpy()}).to_csv(cfg.outdir / "val_predictions.csv", index=False)
    # 2) feature importances (PredictionValuesChange is stable/ interpretable for CatBoost)
    fi = model.get_feature_importance(Pool(X_tr, label=y_tr, cat_features=cat_idx), type="PredictionValuesChange")
    pd.DataFrame({"feature": X.columns, "importance": fi}).sort_values("importance", ascending=False)\
        .to_csv(cfg.outdir / "feature_importances.csv", index=False)
    # 3) model file
    model.save_model(str(cfg.outdir / "catboost_model.cbm"))

    logging.info("Saved outputs to %s", cfg.outdir.resolve())


if __name__ == "__main__":
    main()
