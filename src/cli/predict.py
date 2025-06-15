#!/usr/bin/env python
"""src/cli/predict.py — Generate a Kaggle submission CSV for the *Steel Plate Defect Prediction* TFG project.

▸ **Auto‑detects folds** by scanning ``models/``.
▸ **Preserves feature names** when predicting with LightGBM to avoid the common
  *"X does not have valid feature names"* warning.
▸ Default output now points to ``reports/submissions/`` so submissions are
  grouped in your repo.

Typical usage (no need to pass --folds or --out):

```bash
python -m src.cli.predict --model-name lgbm
```
"""
from __future__ import annotations

import argparse
import glob
import logging
import re
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLS: list[str] = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

LOGGER = logging.getLogger("predict")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_folds(model_dir: Path, pattern: str = "*_preproc_fold*.pkl") -> list[int]:
    """Return sorted list of fold indices detected in *model_dir*."""
    files = glob.glob(str(model_dir / pattern))
    if not files:
        raise FileNotFoundError(
            f"No preprocessing artefacts found in {model_dir} matching {pattern}"
        )
    fold_ids: list[int] = []
    for path in files:
        m = re.search(r"fold(\d+)", Path(path).stem)
        if m:
            fold_ids.append(int(m.group(1)))
    if not fold_ids:
        raise FileNotFoundError(
            f"Could not parse fold indices from artefacts in {model_dir}."
        )
    fold_ids = sorted(set(fold_ids))
    LOGGER.info("Detected %d folds: %s", len(fold_ids), fold_ids)
    return fold_ids


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:  # noqa: D401
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Generate Kaggle submission CSV by averaging per‑fold predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="lgbm",
        help="Prefix used when saving classifier artefacts (e.g. 'lgbm').",
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing the saved models and preprocessors (one per fold).",
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("data/raw/playground-series-s4e3/test.csv"),
        help="Path to Kaggle test.csv file.",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=None,
        help="Number of CV folds trained. If omitted, autodetect by scanning --model-dir.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("reports/submissions/")
        / f"{date.today().isoformat()}_submission.csv",
        help="Output CSV path for the submission file.",
    )
    return p.parse_args(args=argv)


def load_artifact(path: Path):
    """Load a *joblib* artefact, with a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected model artefact not found: {path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:  # noqa: C901
    args = parse_args(argv)

    # Normalise paths
    model_dir: Path = args.model_dir.expanduser().resolve()
    test_csv: Path = args.test_csv.expanduser().resolve()
    args.out = args.out.expanduser().resolve()

    # Determine folds automatically if not provided via CLI
    if args.folds is None:
        fold_ids = detect_folds(model_dir, pattern=f"{args.model_name}_preproc_fold*.pkl")
    else:
        fold_ids = list(range(args.folds))
        LOGGER.info("Using user‑specified fold count: %s", fold_ids)

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    if not test_csv.exists():
        LOGGER.error("Test CSV not found at %s", test_csv)
        sys.exit(1)

    df_test = pd.read_csv(test_csv)
    X_test = df_test.drop(columns=["id"])

    # Prepare container for predictions.
    preds = np.zeros((len(df_test), len(TARGET_COLS)), dtype=np.float32)

    # ------------------------------------------------------------------
    # Loop over folds.
    # ------------------------------------------------------------------
    for fold in fold_ids:
        LOGGER.info("Processing fold %d", fold)

        # Load preprocessing pipeline
        preproc_path = model_dir / f"{args.model_name}_preproc_fold{fold}.pkl"
        preproc = load_artifact(preproc_path)
        X_trans = preproc.transform(X_test)

        # ------------------------------------------------------------------
        # Preserve feature names to avoid LightGBM warning
        # ------------------------------------------------------------------
        if not isinstance(X_trans, pd.DataFrame):
            try:
                feat_names = preproc.get_feature_names_out()
                X_trans = pd.DataFrame(X_trans, columns=feat_names)
            except Exception:  # noqa: BLE001 — best‑effort, fall back to ndarray
                pass

        # ------------------------------------------------------------------
        # Predict per label
        # ------------------------------------------------------------------
        for idx, label in enumerate(TARGET_COLS):
            clf_path = model_dir / f"{args.model_name}_label{idx}_fold{fold}.pkl"
            clf = load_artifact(clf_path)
            preds[:, idx] += clf.predict_proba(X_trans)[:, 1]

    # Average probabilities across folds and clip.
    preds /= len(fold_ids)
    preds = np.clip(preds, 1e-6, 1 - 1e-6)

    # Build submission DataFrame.
    df_submit = pd.DataFrame(preds, columns=TARGET_COLS)
    df_submit.insert(0, "id", df_test["id"].values)

    # Ensure output directory exists and save.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_submit.to_csv(args.out, index=False)

    LOGGER.info("Saved submission shape %s → %s", df_submit.shape, args.out)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
