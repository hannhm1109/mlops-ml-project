# src/features.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer

def _clip(X):
    """Clip values to [-3, 3] range to handle outliers"""
    return X.clip(-3, 3)

def build_numeric_preprocess():
    """
    Prétraitement amélioré :
    - imputation médiane
    - standardisation
    - clipping pour gérer les outliers
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clip", FunctionTransformer(_clip)),
    ])