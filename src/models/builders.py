# src/models/builders.py

import json
from typing import List
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import HPO_DIR, RANDOM_STATE


def _load_best_params(name: str) -> dict:
    """
    Lee parámetros óptimos guardados en JSON bajo HPO_DIR.
    Si no existe el archivo, devuelve un dict vacío.
    """
    fp = HPO_DIR / f"best_params_{name}.json"
    if fp.exists():
        return json.loads(fp.read_text())
    return {}


def build_estimator(name: str, pos_weights: List[float]):
    """
    Crea una lista de clasificadores (uno por cada etiqueta) con los pesos 
    de las clases proporcionados en `pos_weights`. 
    - name: 'lgbm', 'rf' o 'logreg' (case‐insensitive).
    - pos_weights: lista de floats de longitud = número de etiquetas.
    """
    name = name.lower()
    n_labels = len(pos_weights)

    if name == "lgbm":
        base_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
            min_gain_to_split=0.0,
        )
        base_params.update(_load_best_params("lgbm"))
        print("hiperparámetros:", base_params)

        classifiers = [
            LGBMClassifier(scale_pos_weight=pos_weights[k], **base_params)
            for k in range(n_labels)
        ]

    elif name == "rf":
        base_params = dict(
            n_estimators=400,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        base_params.update(_load_best_params("rf"))

        classifiers = [
            RandomForestClassifier(
                class_weight={0: 1, 1: pos_weights[k]}, **base_params
            )
            for k in range(n_labels)
        ]

    elif name in {"logreg", "lr"}:
        base_params = dict(
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        base_params.update(_load_best_params("logreg"))

        classifiers = [
            LogisticRegression(
                class_weight={0: 1, 1: pos_weights[k]}, **base_params
            )
            for k in range(n_labels)
        ]

    else:
        raise ValueError(
            f"Modelo '{name}' no soportado. Usa 'lgbm', 'rf' o 'logreg'."
        )

    return classifiers
