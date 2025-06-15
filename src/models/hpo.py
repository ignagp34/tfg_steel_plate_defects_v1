from __future__ import annotations

from pathlib import Path
import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
import optuna

from sklearn.metrics import roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    from lightgbm import LGBMClassifier
except ImportError:  # noqa: WPS440 (allow bare except for optional dep)
    LGBMClassifier = None

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from src.config import RANDOM_STATE, HPO_DIR, ROOT_DIR
from src.data.loading import load_data
from src.data.split import make_holdout_split, make_folds
from src.pipeline.preprocessing import build_preprocessing_pipeline
from src.models.trainer import _compute_pos_weights

# python -m src.models.hpo rf --trials 100 --n-splits 10

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_prefixes(params: dict[str, Any]) -> dict[str, Any]:
    """Quita prefijos (``rf_``, ``lgb_``, ``lr_``) de los hiperparámetros.

    Optuna usa prefijos para evitar colisiones entre modelos (p.ej. ``lgb_num_leaves``),
    pero el estimador final —``LGBMClassifier``, ``RandomForestClassifier``, etc.— sólo
    entiende los nombres "limpios".  Esta función convierte, por ejemplo::

        {"lgb_num_leaves": 185, "rf_max_depth": 10} -> {"num_leaves": 185, "max_depth": 10}
    """
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if key.startswith(("rf_", "lgb_", "lr_")):
            cleaned[key.split("_", 1)[1]] = value  # quita el prefijo y conserva el resto
        else:
            cleaned[key] = value
    return cleaned

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_model_from_trial(model_name: str, trial: optuna.Trial, pos_weights: list[float]):
    """Devuelve un estimador multilabel para el *trial* actual."""

    if model_name == "rf":
        base = RandomForestClassifier(
            n_estimators      = trial.suggest_int("rf_n_estimators", 200, 1000, step=100),
            max_depth         = trial.suggest_int("rf_max_depth", 4, 30) if
                                trial.suggest_categorical("rf_use_depth", [True, False]) else None,
            min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 10),
            min_samples_leaf  = trial.suggest_int("rf_min_samples_leaf", 1, 8),
            class_weight      = "balanced_subsample",
            n_jobs            = -1,
            random_state      = RANDOM_STATE,
        )
        return MultiOutputClassifier(base, n_jobs=-1)

    if model_name == "logreg":
        penalty = trial.suggest_categorical("lr_penalty", ["l1", "l2", "elasticnet"])
        l1_ratio = None
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("lr_l1_ratio", 0.0, 1.0)

        base = LogisticRegression(
            solver       = "saga",
            max_iter     = 3000,
            penalty      = penalty,
            l1_ratio     = l1_ratio,
            C            = trial.suggest_float("lr_C", 0.01, 10.0, log=True),
            class_weight = "balanced",
            n_jobs       = -1,
            random_state = RANDOM_STATE,
        )
        return MultiOutputClassifier(base, n_jobs=-1)

    if model_name == "lgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM no está instalado en el entorno.")

        estimators: list[LGBMClassifier] = []
        for w in pos_weights:
            estimators.append(
                LGBMClassifier(
                    n_estimators      = trial.suggest_int("lgb_n_estimators", 300, 1500),
                    learning_rate     = trial.suggest_float("lgb_learning_rate", 1e-3, 0.3, log=True),
                    num_leaves        = trial.suggest_int("lgb_num_leaves", 31, 255),
                    max_depth         = trial.suggest_int("lgb_max_depth", -1, 16),
                    min_child_weight  = trial.suggest_float("lgb_min_child_weight", 1e-3, 10.0, log=True),
                    subsample         = trial.suggest_float("lgb_subsample", 0.6, 1.0),
                    colsample_bytree  = trial.suggest_float("lgb_colsample_bytree", 0.6, 1.0),
                    reg_alpha         = trial.suggest_float("lgb_reg_alpha", 1e-8, 10.0, log=True),
                    reg_lambda        = trial.suggest_float("lgb_reg_lambda", 1e-8, 10.0, log=True),
                    objective         = "binary",
                    random_state      = RANDOM_STATE,
                    n_jobs            = -1,
                    scale_pos_weight  = w,
                    verbosity         = -1,
                    min_gain_to_split = 0.0,
                )
            )
        return estimators  # list[LGBMClassifier]

    raise ValueError(f"Modelo '{model_name}' no soportado.")

# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _objective(trial: optuna.Trial, model_name: str, X, y, train_idx, folds):  # noqa: WPS211
    """AUC macro media para un conjunto de hiperparámetros sugeridos."""

    pos_weights = _compute_pos_weights(y[train_idx])
    model = build_model_from_trial(model_name, trial, pos_weights)
    preproc = build_preprocessing_pipeline()

    fold_aucs: list[float] = []
    for fold_no, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(train_idx, val_idx)

        # Pre-procesamiento (misma canalización para todos los folds)
        X_tr_arr = preproc.fit_transform(X.iloc[tr_idx])
        X_va_arr = preproc.transform(X.iloc[val_idx])
        y_tr, y_va = y[tr_idx], y[val_idx]

        feature_names = preproc.get_feature_names_out()
        X_tr = pd.DataFrame(X_tr_arr, columns=feature_names)
        X_va = pd.DataFrame(X_va_arr, columns=feature_names)

        if isinstance(model, list):  # LightGBM «uno-por-clase»
            for k, clf in enumerate(model):  # noqa: WPS518 (explicit index OK)
                clf.fit(X_tr, y_tr[:, k])
            y_hat = np.stack([clf.predict_proba(X_va)[:, 1] for clf in model], axis=1)
        else:  # MultiOutput wrapper (RF / LogReg)
            model.fit(X_tr, y_tr)
            y_hat = np.stack([p[:, 1] for p in model.predict_proba(X_va)], axis=1)

        fold_auc = roc_auc_score(y_va, y_hat, average="macro")
        fold_aucs.append(fold_auc)

        trial.report(fold_auc, step=fold_no)
        if trial.should_prune():
            raise optuna.TrialPruned

    return float(np.mean(fold_aucs))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_hpo(model_name: str, n_trials: int = 50, n_splits: int = 5):  # noqa: WPS231
    """Ejecuta Optuna y guarda los mejores hiperparámetros sin prefijos."""

    X, y = load_data()
    train_idx, _ = make_holdout_split(X, y, test_size=0.15)
    folds = make_folds(train_idx, y, n_splits=n_splits)

    study_name = f"hpo_{model_name}"
    storage = f"sqlite:///{HPO_DIR}/{study_name}.db"

    study = optuna.create_study(
        study_name     = study_name,
        storage        = storage,
        load_if_exists = True,
        direction      = "maximize",
        sampler        = optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner         = optuna.pruners.MedianPruner(n_warmup_steps=1),
    )

    study.optimize(
        lambda tr: _objective(tr, model_name, X, y, train_idx, folds),
        n_trials          = n_trials,
        show_progress_bar = True,
    )

    best_params_path = HPO_DIR / f"best_params_{model_name}.json"
    cleaned = _strip_prefixes(study.best_trial.params)
    with best_params_path.open("w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, indent=2, ensure_ascii=False)

    print("\nMejores hiperparámetros (limpios):")
    print(json.dumps(cleaned, indent=2, ensure_ascii=False))
    print(f"\nAUC media validación: {study.best_value:.5f}")
    print(f"Guardado en: {best_params_path.relative_to(ROOT_DIR)}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameter optimisation (Optuna)")
    parser.add_argument("model", choices=["rf", "logreg", "lgbm"], help="Modelo a optimizar")
    parser.add_argument("--trials", type=int, default=50, help="Número de ensayos de Optuna")
    parser.add_argument("--n-splits", type=int, default=5, help="Nº de folds CV en cada ensayo")
    args = parser.parse_args()

    run_hpo(args.model, args.trials, args.n_splits)
