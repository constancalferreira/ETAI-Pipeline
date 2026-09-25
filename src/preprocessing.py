import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_diagnostics import flag_invalid_values
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler
from category_encoders import CountEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


TARGET = "two_year_recid"
SENSITIVE_ATTR = "race"
COMPAS_OWN_SCORE = ["decile_score", "score_text"]
DROP_ALWAYS = ["id", TARGET, SENSITIVE_ATTR] + COMPAS_OWN_SCORE

NUMERIC_FEATURES = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count"]
CATEGORICAL_FEATURES = ["sex", "age_cat", "c_charge_degree"]
MNAR_INDICATOR_SOURCES = ["priors_count", "c_charge_degree"]


def canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


# category cleanup, domain-rule/placeholder -> Step 1
def clean_dataset(df: pd.DataFrame, diagnosis: dict) -> pd.DataFrame:
    out = df.copy()
    placeholder_tokens = set(diagnosis["placeholder_tokens"])

    for col in ["priors_count", "prior_offenses"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    out.loc[out["age"].notna() & ~out["age"].between(18, 100), "age"] = np.nan
    out.loc[~out["decile_score"].between(1, 10), "decile_score"] = np.nan
    out.loc[out["juv_fel_count"].notna() & (out["juv_fel_count"] < 0), "juv_fel_count"] = np.nan
    out.loc[out["priors_count"].notna() & ((out["priors_count"] < 0) | (out["priors_count"] > 60)), "priors_count"] = np.nan

    out = canonicalize_categories(out, diagnosis["canonical_categories"], placeholder_tokens)

    out = out.drop_duplicates()
    if "id" in out.columns:
        out = out.drop_duplicates(subset="id", keep="first")

    cols_to_drop = [c for c in diagnosis["redundant_columns"] if c in out.columns and c != "id"]
    out = out.drop(columns=cols_to_drop)

    return out

#  adds <col>_was_missing flags for the MNAR columns -> Step 3
def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in MNAR_INDICATOR_SOURCES:
        out[f"{col}_was_missing"] = out[col].isna().astype(int)

    return out


# returns (X, y, extras); y is None on label-free data -> Step 3
def split_features_target(df: pd.DataFrame):
    df = add_missingness_indicators(df)
    y = df[TARGET] if TARGET in df.columns else None

    extras_cols = [c for c in [SENSITIVE_ATTR, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]
    X = df[feature_cols]

    return X, y, extras

_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
_ENCODERS = {
    "onehot": lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda: TargetEncoder(handle_unknown="value", handle_missing="value"),
}

# the leak-safe ColumnTransformer factory -> Step 3
def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    encoder_name = preprocessing_config["encoder"]
    scaler_name = preprocessing_config["scaler"]
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_indicator_sources = preprocessing_config.get("mnar_indicator_sources", [])
    imputation = preprocessing_config.get("imputation", {})

    scaler_factory = _SCALERS[scaler_name]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[encoder_name]()

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])

    indicator_cols = [f"{c}_was_missing" for c in mnar_indicator_sources]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_cols),
    ])


# the stratified train/test split -> Week 2 
def split_train_test(X, y, extras, test_size: float, random_state: int):
    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test, extras_train, extras_test