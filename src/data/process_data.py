
import pandas as pd
from pathlib import Path
from joblib import dump

from src.pipeline.preprocessing import build_preprocessing_pipeline

ROOT      = Path(__file__).resolve().parents[2]
RAW_DIR   = ROOT / "data/raw/playground-series-s4e3"
INTERIM   = ROOT / "data/interim"
INTERIM.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(RAW_DIR / "train.csv")
    X = df.drop(columns=[
        "Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"
    ])
    y = df[[
        "Pastry","Z_Scratch","K_Scatch","Stains","Dirtiness","Bumps","Other_Faults"
    ]]
    
    pipeline = build_preprocessing_pipeline()
    X_prep = pipeline.fit_transform(X)
    
    df_prep = pd.DataFrame(
        X_prep,
        columns=pipeline.get_feature_names_out(),  # sklearn≥1.1
        index=df.index
    )
    df_prep[y.columns] = y.values
    
    df_prep.to_csv(INTERIM / "processed_train.csv", index=True)
    
    dump(pipeline, INTERIM / "preprocessor.joblib")

if __name__ == "__main__":
    main()

