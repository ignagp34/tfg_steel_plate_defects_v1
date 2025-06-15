"""cli.train – Command-line interface to launch cross‑validated training
for the *Steel‑Plate Defect Prediction* TFG project.

This script is a **thin wrapper** around :class:`src.models.trainer.Trainer`.
It parses CLI arguments, configures logging and delegates the heavy lifting
(preprocessing, CV loop, persistence) to the training engine.

python -m src.cli.train --n-splits 5 --model-name lgbm --no-oversample
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import MODEL_DIR, REPORT_DIR, RANDOM_STATE
from src.models.trainer import Trainer, TrainerConfig

# ---------------------------------------------------------------------------
# Logging --------------------------------------------------------------------
# ---------------------------------------------------------------------------
_LOG_FMT = "%(__asctime__)s [%(levelname)s] %(name)s – %(message)s"  # noqa: WPS323
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments."""

    parser = argparse.ArgumentParser(
        prog="cli.train",
        description="Launch K‑fold cross‑validated training and persist artefacts.",
    )

    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="lgbm",
        help="Key understood by build_estimator() (e.g. 'lgbm', 'rf').",
    )
    parser.add_argument(
        "--no-oversample",
        dest="oversample",
        action="store_false",
        help="Disable RandomOverSampler for minority classes.",
    )
    parser.set_defaults(oversample=True)

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Global RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=MODEL_DIR,
        help="Directory where trained models will be saved.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Directory where CV metrics CSV will be saved.",
    )

    return parser.parse_args()


def _configure_logging() -> None:
    """Configure root logger with uniform format and datefmt."""

    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, datefmt=_DATE_FMT)


def main() -> None:  # pragma: no cover
    """Entry‑point executed by ``python -m src.cli.train``."""

    _configure_logging()
    args = _parse_args()

    cfg = TrainerConfig(
        n_splits=args.n_splits,
        oversample=args.oversample,
        random_state=args.random_state,
        model_name=args.model_name,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
    )

    trainer = Trainer(cfg)
    df_cv = trainer.run_cv()

    print(f"✓ Training finished · mean CV AUC = {df_cv['mean_auc'].mean():.5f}")


if __name__ == "__main__":
    main()
