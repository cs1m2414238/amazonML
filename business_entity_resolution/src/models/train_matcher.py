"""Train LightGBM binary classifier for pairwise business entity matching."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def train_matching_model(
    train_pairs_path: str | Path,
    output_dir: str | Path = "output/models",
    val_split_fraction: float = 0.15,
    random_seed: int = 42,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    num_leaves: int = 31,
    n_estimators: int = 300,
    n_jobs: int = 4,
) -> dict[str, Any]:
    """Train LightGBM model with early stopping on query-partitioned validation split."""
    train_pairs_path = Path(train_pairs_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading training pairs from {train_pairs_path}...")
    df = pd.read_csv(train_pairs_path)
    logging.info(f"Loaded {len(df)} pairs across {df['s1_id'].nunique()} unique S1 queries.")

    # Identify feature columns
    excluded_cols = {"s1_id", "target_id", "label"}
    feature_cols = [c for c in df.columns if c not in excluded_cols]
    logging.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Query-level train/validation partition for early stopping
    unique_s1 = df["s1_id"].unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(unique_s1)

    n_val = max(1, int(len(unique_s1) * val_split_fraction))
    val_s1_set = set(unique_s1[:n_val])
    train_s1_set = set(unique_s1[n_val:])

    train_mask = df["s1_id"].isin(train_s1_set)
    val_mask = df["s1_id"].isin(val_s1_set)

    X_train = df.loc[train_mask, feature_cols].astype(np.float32)
    y_train = df.loc[train_mask, "label"].astype(np.int32)
    X_val = df.loc[val_mask, feature_cols].astype(np.float32)
    y_val = df.loc[val_mask, "label"].astype(np.int32)

    logging.info(f"Train split: {len(X_train)} pairs ({y_train.sum()} pos, {(y_train == 0).sum()} neg)")
    logging.info(f"Val split: {len(X_val)} pairs ({y_val.sum()} pos, {(y_val == 0).sum()} neg)")

    # Model definition
    model = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=n_estimators,
        random_state=random_seed,
        n_jobs=n_jobs,
        verbose=-1,
    )

    logging.info("Training LightGBM classifier with early stopping...")
    t0 = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["binary_logloss", "auc"],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )
    train_time = time.perf_counter() - t0
    logging.info(f"Training completed in {train_time:.2f} seconds. Best iteration: {model.best_iteration_}")

    # Evaluate on validation split
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_probs))
    val_pr_auc = float(average_precision_score(y_val, val_probs))
    val_loss = float(log_loss(y_val, val_probs))

    logging.info(f"Validation Split Metrics - AUC: {val_auc:.4f} | PR-AUC: {val_pr_auc:.4f} | LogLoss: {val_loss:.4f}")

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances.tolist()), key=lambda x: x[1], reverse=True)
    logging.info("Top 10 Feature Importances:")
    for f_name, imp in feat_imp[:10]:
        logging.info(f"  {f_name}: {imp}")

    # Save artifacts
    model_txt_path = output_dir / "lightgbm_matcher.txt"
    model.booster_.save_model(str(model_txt_path))
    logging.info(f"Saved Booster text model to {model_txt_path}")

    model_joblib_path = output_dir / "lightgbm_matcher.joblib"
    joblib.dump(model, model_joblib_path)
    logging.info(f"Saved joblib model to {model_joblib_path}")

    schema_path = output_dir / "feature_schema.json"
    schema_data = {
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "normalization_version": "v1.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_data, f, indent=2)
    logging.info(f"Saved feature schema to {schema_path}")

    manifest_data = {
        "metadata": {
            "model_type": "LightGBM",
            "license": "MIT",
            "lightgbm_version": lgb.__version__,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "random_seed": random_seed,
            "training_time_seconds": round(train_time, 2),
            "best_iteration": model.best_iteration_,
        },
        "hyperparameters": {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "n_estimators": n_estimators,
            "n_jobs": n_jobs,
        },
        "data_summary": {
            "total_pairs": len(df),
            "train_pairs": len(X_train),
            "val_pairs": len(X_val),
            "total_s1_queries": df["s1_id"].nunique(),
        },
        "validation_split_metrics": {
            "roc_auc": round(val_auc, 4),
            "pr_auc": round(val_pr_auc, 4),
            "log_loss": round(val_loss, 4),
        },
        "top_features": feat_imp[:15],
    }

    manifest_path = output_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logging.info(f"Saved training manifest to {manifest_path}")

    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM business entity matcher")
    parser.add_argument("--train-pairs", type=str, default="output/models/train_pairs.csv")
    parser.add_argument("--output-dir", type=str, default="output/models")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_matching_model(
        train_pairs_path=args.train_pairs,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
