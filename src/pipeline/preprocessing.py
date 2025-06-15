"""
Archivo: src/pipeline/preprocessing.py
Construye un Pipeline de preprocesamiento reproducible.

Estrategia:
1. Generar/combinar features personalizados (FeatureGenerator).
2. Winsorizar y aplicar log-transform solo a LOG_COLS.
3. Escalar:
   • QUANT_COLS  → QuantileTransformer (output_distribution='normal').
   • ROBUST_COLS → RobustScaler   (resistente a outliers).
   • STD_COLS    → StandardScaler (escala ligera).
   • Todo lo demás se deja passthrough.
"""

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from src.features.transformers import (
    LogTransformer,
    Winsorizer,
    FeatureGenerator,
    ColumnDropper,
)

# --- Definición de grupos de columnas ----------------------------
# 1) Columnas que primero se log-transforman (ya muy asimétricas)
LOG_COLS: List[str] = [
    "Pixels_Areas",
    "X_Perimeter",
    "Y_Perimeter",
    "Maximum_of_Luminosity",
    # 'Max_to_Mean_Luminosity', No aparece
    'Mean_Luminosity',
    'Outside_X_Index',
    'Y_Maximum',

]

# 2) Columnas que necesitan QuantileTransformer (unimodales muy sesgadas)
QUANT_COLS: List[str] = [
    'Edges_Index', 
    'Edges_Y_Index', 
    'Height', 
    'Length_of_Conveyer', 
    'Width',
    'Aspect_Ratio',
]

# 3) Columnas moderadamente sesgadas o con outliers → RobustScaler
ROBUST_COLS: List[str] = [
    'LogOfAreas', 
    'Log_X_Index', 
    'Y_Center_norm',
    'X_Center_norm',

]

# 4) Columnas que simplemente escalamos a media 0 / var 1
RAW_COORDS: List[str] = [
    'Empty_Index', 
    'Log_Y_Index', 
    'Minimum_of_Luminosity', 
    'Orientation_Index', 
    'Outside_Global_Index', 
    'Square_Index',  
    'X_Maximum', 
    'X_Minimum',
    'Steel_Plate_Thickness',
    ]
STD_COLS: List[str] = RAW_COORDS  # puedes añadir más aquí


# 5) Columnas redundantes
REDUNDANT_COLS = [
    'Y_Minimum', 
    'Sum_of_Luminosity', 
    'Edges_X_Index', # ej. correlación ~0.97 con Edges_Index
    'SigmoidOfAreas', 
    'Luminosity_Index', 
    'TypeOfSteel_A300'
]


# 5) El resto de columnas pasan sin modificar (remainder='passthrough')
# -----------------------------------------------------------------

def build_preprocessing_pipeline() -> Pipeline:
    """Devuelve un Pipeline de preprocesamiento para usar antes del modelado."""

    # Transformaciones específicas
    winsorizer = Winsorizer(cols=LOG_COLS)
    log_transform = LogTransformer(cols=LOG_COLS)

    # Escaladores
    qt_scaler = QuantileTransformer(
        output_distribution="normal", random_state=42
    )
    robust_scaler = RobustScaler()
    std_scaler = StandardScaler()

    # ColumnTransformer: reparte cada escalador a su grupo de columnas
    scaler = ColumnTransformer(
        transformers=[
            ("qt", qt_scaler, QUANT_COLS),
            ("robust", robust_scaler, ROBUST_COLS),
            ("std", std_scaler, STD_COLS),
            # columnas ya log-transformadas → passthrough
            ("log_pass", "passthrough", LOG_COLS),
        ],
        remainder="passthrough",  # lo no listado se copia tal cual
        verbose_feature_names_out=False,  # mantiene nombres originales
        n_jobs=-1,
    )

    # Pipeline global
    preprocessor = Pipeline(
        steps=[
            ("feature_gen", FeatureGenerator()),
            ("drop_redundant", ColumnDropper(REDUNDANT_COLS)),
            ("winsorize", winsorizer),
            ("log", log_transform),
            ("scale", scaler),
        ]
    )

    return preprocessor
