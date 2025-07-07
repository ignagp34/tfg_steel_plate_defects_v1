"""src/models/builders.py
-------------------------------------------------
Factory que construye una lista de clasificadores
(one‑vs‑rest) para cada una de las siete etiquetas
usadas en el TFG.  Añade soporte para MLP («mlp»)
y garantiza la compatibilidad de los hiperparámetros
importados desde Optuna (se normalizan los nombres
quitando prefijos) y la coherencia entre «penalty» y
«solver» en la regresión logística.
"""
from __future__ import annotations

import json
import logging
from inspect import signature
from pathlib import Path
from typing import List

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Config (import seguro)
# ---------------------------------------------------------------------------
try:
    from src.config import HPO_DIR, RANDOM_STATE, N_JOBS  # type: ignore
except ImportError:  # pragma: no cover
    from src.config import HPO_DIR, RANDOM_STATE  # type: ignore
    N_JOBS = -1

# Silencia la verbosidad de LightGBM en consola
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Utilidades auxiliares
# ---------------------------------------------------------------------------

def _normalize_optuna_names(params: dict) -> dict:
    """Quita prefijos añadidos por Optuna (rf_, lgb_, lr_, mlp_)"""
    cleaned = {}
    for k, v in params.items():
        if k.startswith(("rf_", "lgb_", "lr_", "mlp_")):
            cleaned[k.split("_", 1)[1]] = v
        else:
            cleaned[k] = v
    return cleaned


def _load_best_params(model_name: str) -> dict:
    """Carga los mejores hiperparámetros desde JSON (si existe)."""
    fp = Path(HPO_DIR) / f"best_params_{model_name}.json"
    if fp.is_file():
        try:
            return _normalize_optuna_names(json.loads(fp.read_text()))
        except Exception:  # fichero corrupto → ignora
            return {}
    return {}


def _filter_params(estimator_cls, params: dict) -> dict:
    """Devuelve solo los kwargs aceptados por `estimator_cls.__init__`."""
    valid = signature(estimator_cls.__init__).parameters
    return {k: v for k, v in params.items() if k in valid}

# ---------------------------------------------------------------------------
# Fábrica principal
# ---------------------------------------------------------------------------

def build_estimator(name: str, pos_weights: List[float]):
    """Devuelve una lista de estimadores (uno por etiqueta).

    Parameters
    ----------
    name : {"lgbm", "rf", "logreg", "mlp"}
        Algoritmo base.
    pos_weights : list[float]
        Peso positivo por etiqueta.
    """
    n_labels = len(pos_weights)

    # -----------------------------------------------------------
    # LightGBM --------------------------------------------------
    # -----------------------------------------------------------
    if name == "lgbm":
        base_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbosity=-1,
            verbose=-1,
            min_gain_to_split=0.0,
        )
        base_params.update(_load_best_params("lgbm"))
        base_params = _filter_params(LGBMClassifier, base_params)
        return [
            LGBMClassifier(**base_params, scale_pos_weight=pos_weights[k])
            for k in range(n_labels)
        ]

    # -----------------------------------------------------------
    # Random Forest --------------------------------------------
    # -----------------------------------------------------------
    if name == "rf":
        base_params = dict(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
        base_params.update(_load_best_params("rf"))
        base_params = _filter_params(RandomForestClassifier, base_params)
        return [
            RandomForestClassifier(**base_params, class_weight={0: 1, 1: pos_weights[k]})
            for k in range(n_labels)
        ]

    # -----------------------------------------------------------
    # Regresión logística --------------------------------------
    # -----------------------------------------------------------
    if name in {"logreg", "lr", "logistic"}:
        base_params = dict(
            penalty="l2",
            solver="lbfgs",
            C=1.0,
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )
        base_params.update(_load_best_params("logreg"))

        # Asegura coherencia penalty/solver
        if base_params.get("penalty") in {"l1", "elasticnet"} and base_params.get("solver") not in {"saga", "liblinear"}:
            base_params["solver"] = "saga"  # compatible con l1 y elasticnet

        base_params = _filter_params(LogisticRegression, base_params)
        return [
            LogisticRegression(**base_params, class_weight={0: 1, 1: pos_weights[k]})
            for k in range(n_labels)
        ]

    # -----------------------------------------------------------
    # Multi‑Layer Perceptron -----------------------------------
    # -----------------------------------------------------------
    if name == "mlp":
        base_params = dict(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            batch_size=256,
            max_iter=200,
            alpha=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        base_params.update(_load_best_params("mlp"))
        base_params = _filter_params(MLPClassifier, base_params)
        return [MLPClassifier(**base_params) for _ in range(n_labels)]

    # -----------------------------------------------------------
    # Modelo desconocido ---------------------------------------
    # -----------------------------------------------------------
    raise ValueError(
        f"Modelo '{name}' no soportado. Elige entre 'lgbm', 'rf', 'logreg' o 'mlp'."
    )
