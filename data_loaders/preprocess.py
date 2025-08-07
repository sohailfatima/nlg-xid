import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def build_preprocess_pipeline(categorical_cols, numeric_cols):
    cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    num = StandardScaler()
    pre = ColumnTransformer([
        ('cat', cat, categorical_cols),
        ('num', num, numeric_cols),
    ], remainder='drop')
    return pre

def split_xy(df, label_col, drop_cols=None):
    drop_cols = drop_cols or []
    X = df.drop(columns=[label_col] + drop_cols, errors='ignore')
    y = df[label_col]
    cats = y.unique()
    cats.sort()
    cats = [str(c) for c in cats if isinstance(c, str)]  # Ensure all categories are strings
    y = y.map(lambda x: x if isinstance(x, int) else cats.index(x))

    return X, y

def train_val_test_split(X, y, test_size=0.2, val_size=0.1, seed=42, stratify=True):
    strat = y if stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=strat)
    strat_temp = y_temp if stratify else None
    val_ratio = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1.0 - val_ratio, random_state=seed, stratify=strat_temp)
    return X_train, y_train, X_val, y_val, X_test, y_test
