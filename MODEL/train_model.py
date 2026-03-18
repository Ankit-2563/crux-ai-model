import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
DEFAULT_DATA_PATH  = Path("./ev_battery_10M.csv")
DEFAULT_MODEL_PATH = Path("ev_battery_model.pkl")
REPORT_PATH        = Path("training_report.txt")

FEATURES    = ["temperature", "voltage", "current", "power"]
TARGET      = "battery_state"
CLASSES     = ["NORMAL", "ALERT", "CRITICAL"]

TEST_SIZE    = 0.20   # 80 / 20 split
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# Feature bounds — must mirror generate_dataset.py exactly
#
#   temperature : [0.0, 80.0)  °C  — strictly below 80
#   voltage     : [8.0, 13.0)  V   — strictly below 13
#   current     : [0.0, 18.0]  A   — inclusive 18
#   power       : [0.0, 234.0) W   — derived: voltage × current
#                                     max ≈ nextafter(13.0) × 18 < 234
# ──────────────────────────────────────────────
FEATURE_BOUNDS = {
    #           (  min,    max,   max_inclusive )
    "temperature": ( 0.0,   80.0,  False),
    "voltage":     ( 8.0,   13.0,  False),
    "current":     ( 0.0,   18.0,  True ),
    "power":       ( 0.0,  234.0,  False),
}

# ──────────────────────────────────────────────
# Label thresholds — mirrors generate_dataset.py exactly
#
# CRITICAL (priority 1 — overrides all):
#   temperature >= 75  OR  voltage < 8.5  OR  current >= 17.5
#
# ALERT (priority 2 — overrides NORMAL):
#   temperature >= 70  OR  voltage < 9.5  OR  current > 15
#
# NORMAL (priority 3):
#   temperature < 40  AND  voltage >= 11.5  AND  current <= 10
# ──────────────────────────────────────────────
CRITICAL_TEMP_THRESH = 75.0
CRITICAL_VOLT_THRESH =  8.5
CRITICAL_CURR_THRESH = 17.5   # NOTE: dataset uses 17.5, not 18.0

ALERT_TEMP_THRESH    = 70.0
ALERT_VOLT_THRESH    =  9.5
ALERT_CURR_THRESH    = 15.0

NORMAL_TEMP_THRESH   = 40.0
NORMAL_VOLT_THRESH   = 11.5
NORMAL_CURR_THRESH   = 10.0

# Decision Tree hyper-parameters  (tuned for this label-space geometry)
DT_PARAMS = dict(
    criterion         = "gini",
    max_depth         = 10,          # deep enough to capture all 3 class boundaries
    min_samples_split = 50,          # avoid tiny, noisy splits on 10M rows
    min_samples_leaf  = 20,          # floor on leaf size
    class_weight      = "balanced",  # handles any class imbalance gracefully
    random_state      = RANDOM_STATE,
)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def load_data(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load CSV; optionally subsample for faster iteration."""
    log.info("Loading data from %s …", path)
    t0 = time.perf_counter()

    if max_rows:
        # Efficient stratified subsample without reading the full file
        df = pd.read_csv(path, nrows=max_rows, dtype={
            "temperature"  : "float32",
            "voltage"      : "float32",
            "current"      : "float32",
            "power"        : "float32",
            "battery_state": "category",
        })
        log.info("Loaded %s rows (subsampled from %s)", f"{len(df):,}", path.name)
    else:
        # Full 10M read — uses category dtype to minimise RAM
        df = pd.read_csv(path, dtype={
            "temperature"  : "float32",
            "voltage"      : "float32",
            "current"      : "float32",
            "power"        : "float32",
            "battery_state": "category",
        })
        log.info("Loaded %s rows in %.1f s", f"{len(df):,}", time.perf_counter() - t0)

    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    """Fail fast on missing columns, unexpected nulls, or out-of-bound values."""
    # ── Column presence ──
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # ── Null check ──
    null_counts = df[FEATURES + [TARGET]].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values detected:\n{null_counts[null_counts > 0]}")

    # ── Class label check ──
    unexpected = set(df[TARGET].unique()) - set(CLASSES)
    if unexpected:
        raise ValueError(f"Unexpected class labels in target: {unexpected}")

    # ── Feature bounds check (mirrors FEATURE_BOUNDS) ──
    violations = []
    for feat, (lo, hi, hi_inclusive) in FEATURE_BOUNDS.items():
        if feat not in df.columns:
            continue
        col = df[feat]
        if col.min() < lo:
            violations.append(
                f"  {feat}: min {col.min():.4f} is below lower bound {lo}"
            )
        if hi_inclusive:
            if col.max() > hi:
                violations.append(
                    f"  {feat}: max {col.max():.4f} exceeds upper bound {hi} (inclusive)"
                )
        else:
            if col.max() >= hi:
                violations.append(
                    f"  {feat}: max {col.max():.4f} meets or exceeds upper bound {hi} (exclusive)"
                )
    if violations:
        raise ValueError("Feature bound violations:\n" + "\n".join(violations))

    log.info("✅  Data validation passed.")


def validate_inference_bounds(X: np.ndarray) -> None:
    """
    Raise ValueError when live / production data falls outside the training
    distribution defined by FEATURE_BOUNDS.

    Call this before clf.predict() in any inference pipeline.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, len(FEATURES))
        Feature matrix in the same column order as FEATURES.
    """
    violations = []
    for i, feat in enumerate(FEATURES):
        lo, hi, hi_inclusive = FEATURE_BOUNDS[feat]
        col = X[:, i]
        col_min, col_max = float(col.min()), float(col.max())

        if col_min < lo:
            violations.append(
                f"  {feat}: min {col_min:.4f} < lower bound {lo}"
            )
        if hi_inclusive:
            if col_max > hi:
                violations.append(
                    f"  {feat}: max {col_max:.4f} > upper bound {hi} (inclusive)"
                )
        else:
            if col_max >= hi:
                violations.append(
                    f"  {feat}: max {col_max:.4f} >= upper bound {hi} (exclusive)"
                )

    if violations:
        raise ValueError(
            "Inference input out of training distribution:\n" + "\n".join(violations)
        )


def build_report(
    model         : DecisionTreeClassifier,
    le            : LabelEncoder,
    X_test        : np.ndarray,
    y_test        : np.ndarray,
    y_pred        : np.ndarray,
    elapsed_train : float,
    elapsed_infer : float,
    n_train       : int,
    n_test        : int,
) -> str:
    """Return a detailed text report string."""
    class_names = le.classes_.tolist()

    # Build a human-readable bounds summary for the report
    bounds_lines = []
    for feat, (lo, hi, hi_inclusive) in FEATURE_BOUNDS.items():
        hi_bracket = "]" if hi_inclusive else ")"
        bounds_lines.append(f"  {feat:<15}  [{lo}, {hi}{hi_bracket}")

    threshold_lines = [
        f"  CRITICAL  temp >= {CRITICAL_TEMP_THRESH}  |  volt < {CRITICAL_VOLT_THRESH}"
        f"  |  curr >= {CRITICAL_CURR_THRESH}",
        f"  ALERT     temp >= {ALERT_TEMP_THRESH}  |  volt < {ALERT_VOLT_THRESH}"
        f"  |  curr > {ALERT_CURR_THRESH}",
        f"  NORMAL    temp <  {NORMAL_TEMP_THRESH}  AND volt >= {NORMAL_VOLT_THRESH}"
        f"  AND curr <= {NORMAL_CURR_THRESH}",
    ]

    lines = [
        "=" * 70,
        "EV Battery Decision Tree — Training Report",
        "=" * 70,
        f"Training rows : {n_train:,}",
        f"Test rows     : {n_test:,}",
        f"Train time    : {elapsed_train:.2f} s",
        f"Inference time: {elapsed_infer:.4f} s  ({n_test / elapsed_infer:,.0f} rows/s)",
        "",
        "── Feature bounds (training distribution) ──",
    ] + bounds_lines + [
        "",
        "── Label thresholds ──",
    ] + threshold_lines + [
        "",
        "── Hyper-parameters ──",
        json.dumps(DT_PARAMS, indent=2),
        "",
        "── Overall Accuracy ──",
        f"  {accuracy_score(y_test, y_pred) * 100:.4f} %",
        "",
        "── Macro F1 ──",
        f"  {f1_score(y_test, y_pred, average='macro') * 100:.4f} %",
        "",
        "── Per-class Report ──",
        classification_report(y_test, y_pred, target_names=class_names, digits=4),
        "── Confusion Matrix ──",
        "  Rows = Actual, Columns = Predicted",
        f"  Classes: {class_names}",
        str(confusion_matrix(y_test, y_pred)),
        "",
        "── Feature Importances ──",
    ]
    for feat, imp in sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: -x[1]
    ):
        lines.append(f"  {feat:<15}  {imp:.6f}")

    lines += [
        "",
        "── Tree Structure (depth ≤ 4 preview) ──",
        export_text(model, feature_names=FEATURES, max_depth=4),
        "=" * 70,
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def train(data_path: Path, model_path: Path, max_rows: int | None) -> None:
    # 1. Load & validate
    df = load_data(data_path, max_rows)
    validate_dataframe(df)

    # 2. Encode target labels → integers (required by sklearn metrics)
    le = LabelEncoder()
    le.fit(CLASSES)          # deterministic ordering: ALERT=0, CRITICAL=1, NORMAL=2
    y = le.transform(df[TARGET].astype(str))
    X = df[FEATURES].values.astype(np.float32)

    log.info("Class distribution (full dataset):")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        log.info("  %-10s  %9d  (%.2f%%)", le.classes_[u], c, 100 * c / len(y))

    # 3. Stratified train / test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    log.info("Split → train: %s  |  test: %s", f"{len(X_train):,}", f"{len(X_test):,}")

    # 4. Train
    log.info("Training Decision Tree  (params: %s) …", DT_PARAMS)
    clf = DecisionTreeClassifier(**DT_PARAMS)
    t_train = time.perf_counter()
    clf.fit(X_train, y_train)
    elapsed_train = time.perf_counter() - t_train
    log.info("Training complete in %.2f s  |  tree depth: %d  |  leaves: %d",
             elapsed_train, clf.get_depth(), clf.get_n_leaves())

    # 5. Evaluate
    #    validate_inference_bounds is called here to confirm test data stays
    #    within the training distribution (sanity check on the split).
    validate_inference_bounds(X_test)
    t_infer = time.perf_counter()
    y_pred  = clf.predict(X_test)
    elapsed_infer = time.perf_counter() - t_infer

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    log.info("Accuracy: %.4f %%   Macro-F1: %.4f %%", acc * 100, f1 * 100)

    # 6. Build & save report
    report = build_report(
        clf, le, X_test, y_test, y_pred,
        elapsed_train, elapsed_infer,
        len(X_train), len(X_test),
    )
    print("\n" + report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    log.info("Report saved → %s", REPORT_PATH.resolve())

    # 7. Serialise model bundle as .pkl
    model_bundle = {
        "model"           : clf,
        "label_encoder"   : le,
        "features"        : FEATURES,
        "classes"         : CLASSES,
        "feature_bounds"  : FEATURE_BOUNDS,       # stored for inference-time validation
        "label_thresholds": {                      # mirrors generate_dataset.py exactly
            "critical": {
                "temp_ge" : CRITICAL_TEMP_THRESH,
                "volt_lt" : CRITICAL_VOLT_THRESH,
                "curr_ge" : CRITICAL_CURR_THRESH,
            },
            "alert": {
                "temp_ge" : ALERT_TEMP_THRESH,
                "volt_lt" : ALERT_VOLT_THRESH,
                "curr_gt" : ALERT_CURR_THRESH,
            },
            "normal": {
                "temp_lt" : NORMAL_TEMP_THRESH,
                "volt_ge" : NORMAL_VOLT_THRESH,
                "curr_le" : NORMAL_CURR_THRESH,
            },
        },
        "dt_params"       : DT_PARAMS,
        "accuracy"        : float(acc),
        "macro_f1"        : float(f1),
        "tree_depth"      : clf.get_depth(),
        "n_leaves"        : clf.get_n_leaves(),
        "trained_on_rows" : len(X_train),
        "sklearn_version" : __import__("sklearn").__version__,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = model_path.stat().st_size / 1024
    log.info("✅  Model saved → %s  (%.1f KB)", model_path.resolve(), size_kb)


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EV Battery Decision Tree")
    parser.add_argument("--data",  type=Path, default=DEFAULT_DATA_PATH,
                        help="Path to the CSV dataset")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="Output path for the .pkl model bundle")
    parser.add_argument("--rows",  type=int,  default=None,
                        help="Subsample N rows for quick iteration (omit for full 10M)")
    args = parser.parse_args()

    if not args.data.exists():
        log.error("Dataset not found: %s", args.data.resolve())
        sys.exit(1)

    train(args.data, args.model, args.rows)
