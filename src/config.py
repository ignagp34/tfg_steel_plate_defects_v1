# from src import ROOT_DIR
from pathlib import Path
"""Configuración centralizada del proyecto *TFG-Steel-Plate-Defects*.

Este módulo reúne todas las rutas de carpetas, constantes y parámetros por
defecto. Importa las variables que necesites en lugar de hard-codearlas en
múltiples archivos – así facilitas el mantenimiento y la reproducibilidad.

"""

# ────────────────────────────────
#  Rutas base del proyecto
# ────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR      = ROOT_DIR / "data"
DATA_RAW      = DATA_DIR / "raw" / "playground-series-s4e3"
DATA_INTERIM  = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

# Directorio donde se guardan los índices de splits (hold-out, CV, etc.)
DATA_SPLIT = DATA_INTERIM / "splits"

# Archivo con estadísticas de columnas (generado en el EDA)
STATS_PATH = ROOT_DIR / "reports/tables/summary_statistics.csv"
REPORT_DIR = ROOT_DIR / "reports/tables"

# Carpeta para modelos serializados y artefactos
MODEL_DIR = ROOT_DIR / "models_artifacts"

# Hiperparámetros
HPO_DIR    = ROOT_DIR / "reports/hpo"

# ────────────────────────────────
#  Constantes de ML / preprocesado
# ────────────────────────────────

# Etiquetas binarias (mismo orden que en Kaggle)
TARGET_COLS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

# Semilla global para reproducibilidad
RANDOM_STATE = 42

# Factor por defecto para la regla IQR en detección de outliers
IQR_FACTOR = 3.0  # 1.5 es clásico; 3.0 más laxo

# Tamaño de test (proporción) para el hold-out inicial
TEST_SIZE = 0.15

# Número mínimo de positivos por etiqueta que se exige en cada split
MIN_POSITIVE_PER_CLASS = 2

# ────────────────────────────────
#  Helper: crea carpetas si no existen (opcional)
# ────────────────────────────────
for _dir in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, DATA_EXTERNAL, DATA_SPLIT, MODEL_DIR):
    _dir.mkdir(parents=True, exist_ok=True)