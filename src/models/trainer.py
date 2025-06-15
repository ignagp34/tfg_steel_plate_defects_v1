"""Training engine for the *Steel Plate Defect Prediction* TFG project.

This module provides a reusable :class:`Trainer` that is consumed by the
CLI wrapper ``src/cli/train.py``.

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
import warnings
import inspect

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Local imports --------------------------------------------------------------
from src.config import MODEL_DIR, REPORT_DIR, RANDOM_STATE, TARGET_COLS
from src.data.loading import load_data
from src.data.split import make_holdout_split, make_folds
from src.pipeline.preprocessing import build_preprocessing_pipeline
from src.models.builders import build_estimator

# Silence noisy sklearn future warnings (keeps logs clean in CLI) -----------
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\..*",
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass ----------------------------------------------------
# ---------------------------------------------------------------------------
@dataclass
class TrainerConfig:
    """Container for all run-time hyper-parameters handled by the CLI."""

    n_splits: int = 5
    oversample: bool = True
    random_state: int = RANDOM_STATE
    model_name: str = "lgbm"
    model_dir: Path = MODEL_DIR
    report_dir: Path = REPORT_DIR

    def asdict(self) -> Dict[str, Any]:  # handy for logging
        return {
            "n_splits": self.n_splits,
            "oversample": self.oversample,
            "random_state": self.random_state,
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "report_dir": self.report_dir,
        }

# ---------------------------------------------------------------------------
# Helper functions (lifted from the validated ``training.py``) --------------
# ---------------------------------------------------------------------------
_DEF_EPS = float(np.finfo("float32").eps)


def _compute_pos_weights(y: np.ndarray) -> List[float]:
    """Balanced positive-class weights for each defect label.

    Returns a list *w_pos/w_neg* so that the minority class has higher weight.
    We cap the minimum at *eps* to avoid zero weights (LightGBM requirement).
    """

    classes = np.array([0, 1])
    weights: List[float] = []
    for k in range(y.shape[1]):
        w_neg, w_pos = compute_class_weight(
            class_weight="balanced", classes=classes, y=y[:, k]
        )
        ratio = w_neg / w_pos if w_pos else _DEF_EPS
        weights.append(max(ratio, _DEF_EPS))
    return weights


def _roc_auc_by_class(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return per-label AUC plus the macro mean."""

    aucs = {
        col: roc_auc_score(y_true[:, i], y_pred[:, i])
        for i, col in enumerate(TARGET_COLS)
    }
    aucs["mean_auc"] = float(np.mean(list(aucs.values())))
    return aucs

def _custom_oversample_per_label(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    label_name: str,
    random_state: int,
    fixed_pos_counts: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Si `label_name` está en fixed_pos_counts, hace oversample de la clase positiva
    hasta el número fijo indicado. En caso contrario, devuelve (X_arr, y_arr) sin cambios.

    - X_arr, y_arr: datos de entrenamiento transformados (arrays) para una única etiqueta.
    - label_name: el nombre de la etiqueta actual p.ej. "Pastry" o "Dirtiness".
    - random_state: semilla para reproducibilidad.
    - fixed_pos_counts: diccionario {"Pastry": 1800, "Dirtiness": 1600} con el
      número EXACTO de positivos deseados tras el oversample.

    Retorna:
        X_res, y_res: arrays tras aplicar (o no) RandomOverSampler.
    """
    # Si esta etiqueta no está en fixed_pos_counts, no hacemos nada
    if label_name not in fixed_pos_counts:
        return X_arr, y_arr

    # Recuperamos cuántos positivos queremos para esta etiqueta
    target_pos = fixed_pos_counts[label_name]

    # Construir sampling_strategy: dejamos los ceros igual, aumentamos la clase 1
    from collections import Counter
    counts = Counter(y_arr)
    n_neg = counts[0]  # tamaño de la clase 0 en este fold
    # n_pos_antes = counts[1]  # (opcional) número de positivos actuales

    sampling_dict = {0: n_neg, 1: target_pos}

    ros = RandomOverSampler(sampling_strategy=sampling_dict, random_state=random_state)
    X_res, y_res = ros.fit_resample(X_arr, y_arr)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Trainer class -------------------------------------------------------------
# ---------------------------------------------------------------------------
class Trainer:
    """Cross-validated trainer wrapper.

    Parameters
    ----------
    cfg
        Dataclass with all runtime options; typically built in the CLI.
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    def __init__(self, cfg: TrainerConfig) -> None:
        self.cfg = cfg
        self.model_dir = cfg.model_dir
        self.report_dir = cfg.report_dir
        self.random_state = cfg.random_state

        self.logger = _LOGGER
        self.logger.info("Trainer initialised with config: %s", cfg.asdict())

        # Lazily initialised attributes ----------------------------------
        self.preproc = None  # fitted ColumnTransformer or FunctionTransformer
        self.classifiers = None  # list[Estimator]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_cv(self) -> pd.DataFrame:
        """Run K-fold CV, save artefacts and return metrics DataFrame."""

        # 1. Data handling ------------------------------------------------
        X, y, train_idx, hold_idx, folds = self._prepare_data()

        # 2. Pipelines & estimators --------------------------------------
        self._prepare_transformers_and_models(y[train_idx])

        # 3. Fold loop ----------------------------------------------------
        cv_metrics: List[Dict[str, float]] = []
        for fold_id, val_idx in enumerate(folds):
            tr_idx = np.setdiff1d(train_idx, val_idx)
            self.logger.info(
                "Fold %s – train %s · val %s", fold_id, len(tr_idx), len(val_idx)
            )
            fold_metrics = self._train_fold(fold_id, tr_idx, val_idx, X, y)
            cv_metrics.append(fold_metrics)

        # 4. Aggregate & persist metrics ---------------------------------
        df_cv = pd.DataFrame(cv_metrics)
        df_cv.to_csv(self.report_dir / f"{self.cfg.model_name}_cv_metrics.csv", index=False)

        self.logger.info("Mean AUC per fold:\n%s", df_cv[["fold", "mean_auc"]])
        self.logger.info("Global mean AUC = %.5f", df_cv["mean_auc"].mean())
        return df_cv

    # ------------------------------------------------------------------
    # Private helpers (single-responsibility, logic untouched) ----------
    # ------------------------------------------------------------------
    def _prepare_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
        """Load data, build indices and CV folds."""
        X, y_raw = load_data()
        self.logger.info("Loaded %s rows × %s features", len(X), X.shape[1])

        # Ensure ndarray targets ---------------------------------------
        y = y_raw[TARGET_COLS].values if isinstance(y_raw, pd.DataFrame) else y_raw

        train_idx, hold_idx = make_holdout_split(X, y)
        folds = make_folds(train_idx, y, self.cfg.n_splits)
        return X, y, train_idx, hold_idx, folds

    def _prepare_transformers_and_models(self, y_train: np.ndarray) -> None:
        """Instantiate preprocessing pipeline and one-vs-rest estimators."""
        pos_weights = _compute_pos_weights(y_train)
        self.preproc = build_preprocessing_pipeline()
        self.classifiers = build_estimator(self.cfg.model_name, pos_weights)

    def _train_fold(
        self,
        fold_id: int,
        tr_idx: np.ndarray,
        val_idx: np.ndarray,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Fit models on a single fold and return its metrics dict."""

        # Split raw data -------------------------------------------------
        X_tr_raw, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va_raw, y_va = X.iloc[val_idx], y[val_idx]

        # Fit transformers only on training portion ---------------------
        X_tr_arr = self.preproc.fit_transform(X_tr_raw)
        X_va_arr = self.preproc.transform(X_va_raw)

        # Optional feature names (if supported by the transformer) -------
        feat_names_arr = getattr(self.preproc, "get_feature_names_out", lambda: None)()
        feat_names: List[str] | None = list(map(str, feat_names_arr)) if feat_names_arr is not None else None

        # DataFrame helps LightGBM understand feature names --------------
        X_va_df = (
            pd.DataFrame(X_va_arr, columns=feat_names)
            if feat_names
            else pd.DataFrame(X_va_arr)
        )

        y_pred_cols: List[np.ndarray] = []

        # Iterate classifiers (one-vs-rest) -----------------------------
        for k, clf in enumerate(self.classifiers):
            X_k_arr = X_tr_arr
            y_k = y_tr[:, k]

        # Optional oversampling on training fold -------------------
        # ---------------------------------------------------------------------
        for k, clf in enumerate(self.classifiers):
            X_k_arr = X_tr_arr
            y_k     = y_tr[:, k]
            label   = TARGET_COLS[k]  # p.ej. "Pastry", "Z_Scratch", "Dirtiness", etc.
        
            # -------------------------------------------------------------
            # Nuevo bloque: oversample “Pastry” y “Dirtiness” con valores fijos
            # -------------------------------------------------------------
            if self.cfg.oversample:
                fixed_counts = {
                    "Pastry":   2000,
                    "Dirtiness": 1800,
                }
        
                X_k_arr, y_k = _custom_oversample_per_label(
                    X_arr=X_k_arr,
                    y_arr=y_k,
                    label_name=label,
                    random_state=self.random_state,
                    fixed_pos_counts=fixed_counts,
                )
            # -------------------------------------------------------------

            # Convert arrays to DataFrame if we have feature names ------
            X_k_df = (
                pd.DataFrame(X_k_arr, columns=feat_names)
                if feat_names
                else pd.DataFrame(X_k_arr)
            )

            # -----------------------------------------------------------------
            # NEW: Only pass ``feature_name`` if `.fit` supports that kwarg
            # -----------------------------------------------------------------
            fit_kwargs: Dict[str, Any] = {}
            if feat_names:
                # Some third‑party estimators (LightGBM, XGBoost) accept it …
                if "feature_name" in inspect.signature(clf.fit).parameters:
                    fit_kwargs["feature_name"] = feat_names

            clf.fit(X_k_df, y_k, **fit_kwargs)

            proba = clf.predict_proba(X_va_df)[:, 1]
            y_pred_cols.append(proba)

        # Metrics & persistence per fold ---------------------------------
        y_pred = np.vstack(y_pred_cols).T  # shape (n_val, n_labels)
        fold_metrics = _roc_auc_by_class(y_va, y_pred)
        fold_metrics["fold"] = fold_id

        self._save_fold_artifacts(fold_id)
        return fold_metrics

    def _save_fold_artifacts(self, fold_id: int) -> None:
        """Persist fitted estimators and preprocessor for the current fold."""
        for k, clf in enumerate(self.classifiers):
            joblib.dump(
                clf,
                self.model_dir / f"{self.cfg.model_name}_label{k}_fold{fold_id}.pkl",
            )
        joblib.dump(self.preproc, self.model_dir / f"{self.cfg.model_name}_preproc_fold{fold_id}.pkl")


# ----------------------------------------------------------------------------
# Module test hook -----------------------------------------------------------
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal smoke test when executed directly
    cfg = TrainerConfig(n_splits=3, model_name="rf")
    # Trainer(cfg).run_cv()
    trainer = Trainer(cfg)
    df_cv = trainer.run_cv()
    print(f"✓ Training finished · mean CV AUC = {df_cv['mean_auc'].mean():.5f}")
