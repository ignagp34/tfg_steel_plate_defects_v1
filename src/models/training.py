
# standard
import joblib
import numpy as np
import pandas as pd

# third-party
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

# local
from src import ROOT_DIR
from src.data.loading import load_data
from src.data.split import make_holdout_split, make_folds
from src.pipeline.preprocessing import build_preprocessing_pipeline
from src.models.builders import build_estimator
from src.config import MODEL_DIR, RANDOM_STATE, REPORT_DIR, TARGET_COLS 
import warnings

# Oculta TODOS los FutureWarning cuyo módulo empiece por "sklearn."
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\..*"
)


# %%

def compute_pos_weights(y):
    weights = []
    classes = np.array([0, 1])
    for k in range(y.shape[1]):
        w_neg, w_pos = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y[:, k]
        )
        weights.append(w_neg / w_pos)
    return weights

def roc_auc_by_class(y_true, y_pred):
    aucs = {}
    for i, col in enumerate(TARGET_COLS):
        aucs[col] = roc_auc_score(y_true[:, i], y_pred[:, i])
    aucs["mean_auc"] = np.mean(list(aucs.values()))
    return aucs

def run_cv(model_name: str = "lgbm",
           n_splits: int = 5,
           do_oversample: bool = True):

    X, y = load_data()
    train_idx, hold_idx = make_holdout_split(X, y)
    folds = make_folds(train_idx, y, n_splits)
    pos_weights  = compute_pos_weights(y[train_idx])
    preproc      = build_preprocessing_pipeline()
    classifiers  = build_estimator(model_name, pos_weights)   
    cv_metrics = []

    for fold_id, val_idx in enumerate(folds):
        tr_idx = np.setdiff1d(train_idx, val_idx)

        X_tr_raw, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va_raw, y_va = X.iloc[val_idx], y[val_idx]


        X_tr_arr = preproc.fit_transform(X_tr_raw)   
        X_va_arr = preproc.transform(X_va_raw)       
        feature_names = preproc.get_feature_names_out()


        X_va_df = pd.DataFrame(X_va_arr, columns=feature_names)


        y_pred_cols = []

        for k, clf in enumerate(classifiers):
            X_k_arr = X_tr_arr          
            y_k     = y_tr[:, k]


            if do_oversample:
                ros = RandomOverSampler(
                    sampling_strategy="not majority",
                    random_state=RANDOM_STATE
                )
                X_k_arr, y_k = ros.fit_resample(X_k_arr, y_k)


            X_k_df = pd.DataFrame(X_k_arr, columns=feature_names)


            clf.fit(X_k_df, y_k)
            proba = clf.predict_proba(X_va_df)[:, 1]
            y_pred_cols.append(proba)


        y_pred = np.vstack(y_pred_cols).T
        fold_metrics = roc_auc_by_class(y_va, y_pred)
        fold_metrics["fold"] = fold_id
        cv_metrics.append(fold_metrics)


        for k, clf in enumerate(classifiers):
            joblib.dump(
                clf,
                MODEL_DIR / f"{model_name}_label{k}_fold{fold_id}.pkl"
            )
        joblib.dump(preproc, MODEL_DIR / f"preproc_fold{fold_id}.pkl")


    df_cv = pd.DataFrame(cv_metrics)
    df_cv.to_csv(REPORT_DIR / f"{model_name}_cv_metrics.csv", index=False)

    print("AUC media por fold:\n", df_cv[["fold", "mean_auc"]])
    print("AUC global:", df_cv["mean_auc"].mean())
    return df_cv


if __name__ == "__main__":
    import sys
    alg = sys.argv[1] if len(sys.argv) > 1 else "lgbm"
    run_cv(model_name=alg)

