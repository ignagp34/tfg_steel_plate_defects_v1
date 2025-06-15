import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """Aplica log1p a las columnas indicadas."""
    def __init__(self, cols=None):
        self.cols = cols or []

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X_ = X.copy()
        for c in self.cols:
            X_[c] = np.log1p(X_[c])
        return X_
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)

class Winsorizer(BaseEstimator, TransformerMixin):
    """Capea los valores fuera de [p1, p99] para cada columna."""
    def __init__(self, cols=None, lower_pct=0.1, upper_pct=99.9):
        self.cols = cols or []
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.bounds_ = {}

    def fit(self, X, y=None):
        for c in self.cols:
            p1, p99 = np.percentile(X[c], [self.lower_pct, self.upper_pct])
            self.bounds_[c] = (p1, p99)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X_ = X.copy()
        for c, (p1, p99) in self.bounds_.items():
            X_[c] = np.clip(X_[c], p1, p99)
        return X_

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        return np.asarray(input_features, dtype=object)

class FeatureGenerator(BaseEstimator, TransformerMixin):
    """Genera features derivadas sin filtrar datos de validación‐test."""

    def __init__(self, img_width: int = 1700, img_height: int = 12_800_000):
        self.img_width = img_width
        self.img_height = img_height

    # ---------------------------------------------------------------------
    # AJUSTE (calculamos y guardamos el percentil 95 ≈ ‘muy grande’)
    # ---------------------------------------------------------------------
    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.p95_ = np.percentile(X["Pixels_Areas"], 95)  # ← ¡una sola vez!
        return self

    # ---------------------------------------------------------------------
    # TRANSFORMACIÓN (usa el percentil ya aprendido)
    # ---------------------------------------------------------------------
    def transform(self, X):
        X_ = X.copy()

        # dimensiones y ratios
        X_["Width"] = X_["X_Maximum"] - X_["X_Minimum"]
        X_["Height"] = X_["Y_Maximum"] - X_["Y_Minimum"]
        X_["Aspect_Ratio"] = X_["Width"] / (X_["Height"] + 1e-6)

        # posición normalizada
        X_["X_Center_norm"] = (X_["X_Minimum"] + X_["X_Maximum"]) / 2 / self.img_width
        X_["Y_Center_norm"] = (X_["Y_Minimum"] + X_["Y_Maximum"]) / 2 / self.img_height

        # luminosidad media y relación pico-media
        X_["Mean_Luminosity"] = X_["Sum_of_Luminosity"] / (X_["Pixels_Areas"] + 1e-6)
        X_["Max_to_Mean_Luminosity"] = (
            X_["Maximum_of_Luminosity"] / (X_["Mean_Luminosity"] + 1e-6)
        )

        # flag de área muy grande (sin recalcular el p95)
        X_["is_very_large"] = (X_["Pixels_Areas"] > self.p95_).astype(int)

        return X_

    # ---------------------------------------------------------------------
    # Salida de nombres de columnas
    # ---------------------------------------------------------------------
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        extra = np.array(
            [
                "Width",
                "Height",
                "Aspect_Ratio",
                "X_Center_norm",
                "Y_Center_norm",
                "Mean_Luminosity",
                "Max_to_Mean_Luminosity",
                "is_very_large",
            ],
            dtype=object,
        )
        return np.concatenate([np.asarray(input_features, dtype=object), extra])


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Elimina columnas fijas conocidas de antemano."""
    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return X.drop(columns=self.cols)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
        mask = ~np.isin(input_features, self.cols)
        return np.asarray(input_features, dtype=object)[mask]
    