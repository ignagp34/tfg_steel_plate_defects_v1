import joblib
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold
from src.config import DATA_RAW, DATA_SPLIT, RANDOM_STATE,TARGET_COLS  

def make_holdout_split(X, y, test_size=0.15):
    """
    Genera (o recarga) índices de train/holdout con MultilabelStratifiedShuffleSplit,
    asegurando al menos 2 positivos por etiqueta en cada part.
    """
    holdout_path = DATA_SPLIT / "holdout_idx.pkl"
    if holdout_path.exists():
        split = joblib.load(holdout_path)
        return split["train_idx"], split["hold_idx"]

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=RANDOM_STATE
    )
    for train_idx, hold_idx in msss.split(X, y):
        y_tr, y_hd = y[train_idx], y[hold_idx]
        pos_tr = y_tr.sum(axis=0)
        pos_hd = y_hd.sum(axis=0)
        if np.all(pos_tr >= 2) and np.all(pos_hd >= 2):
            joblib.dump(
                {"train_idx": train_idx, "hold_idx": hold_idx},
                holdout_path
            )
            return train_idx, hold_idx

    raise RuntimeError("No se pudo generar hold-out con ≥2 positivos en cada etiqueta.")


def make_folds(train_idx, y, n_splits=5):
    """
    Genera (o recarga) listas de índices de validación para CV estratificado multilabel.
    """
    fold_paths = [DATA_SPLIT / f"fold_{i}.pkl" for i in range(n_splits)]
    if all(p.exists() for p in fold_paths):
        return [joblib.load(p) for p in fold_paths]

    # Para reclasificar, necesitamos el CSV original
    df = pd.read_csv(DATA_RAW / "train.csv")
    X_rem = df.iloc[train_idx, :].drop(columns=TARGET_COLS)
    y_rem = y[train_idx]

    strat = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    folds = []
    for _, val_local_ix in strat.split(X_rem, y_rem):
        val_idx_global = train_idx[val_local_ix]
        folds.append(val_idx_global)

    for i, val_idx in enumerate(folds):
        joblib.dump(val_idx, fold_paths[i])

    return folds
