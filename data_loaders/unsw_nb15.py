import pandas as pd

# Common NB15 columns: 'proto','service','state','label' (0 benign, 1 attack), 'attack_cat' for multi-class
DEFAULT_CATEGORICAL = ['proto', 'service', 'state']

def load_nb15(csv_path):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        for cand in ['Label', 'labels', 'class']:
            if cand in df.columns:
                df = df.rename(columns={cand: 'label'})
                break
    return df

def infer_columns(df, provided_cats=None):
    categorical = provided_cats or [c for c in DEFAULT_CATEGORICAL if c in df.columns]
    # prefer attack_cat if you want multi-class; default to 'label' (binary) else
    label_col = 'attack_cat' if 'attack_cat' in df.columns else 'label'
    drop_cols = []
    numeric = [c for c in df.columns if c not in categorical + drop_cols + [label_col] and pd.api.types.is_numeric_dtype(df[c])]
    return categorical, numeric, label_col, drop_cols
