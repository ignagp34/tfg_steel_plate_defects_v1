import pandas as pd
from src.config import DATA_RAW, TARGET_COLS  


FEATURE_COLS = None

def clean_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas físicamente imposibles. No hay valores nulos en este dataset.
    """
    global FEATURE_COLS
    if FEATURE_COLS is None:
        FEATURE_COLS = [c for c in df.columns if c not in TARGET_COLS + ["id"]]

    mask = (
        (df["Y_Maximum"] >= df["Y_Minimum"]) &
        (df["X_Maximum"] >= df["X_Minimum"])
    )
    return df.loc[mask].reset_index(drop=True)


def load_data():
    """
    Lee train.csv desde la carpeta RAW y devuelve X (DataFrame) y y (ndarray de 7 etiquetas).
    """
    df = pd.read_csv(DATA_RAW / "train.csv")
    df = clean_raw_df(df)  # limpia antes de devolver
    X = df.drop(columns=TARGET_COLS + ["id"])
    y = df[TARGET_COLS].values
    return X, y
