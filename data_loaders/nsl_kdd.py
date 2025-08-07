import pandas as pd

# Expected columns: includes 'label' (multi-class) or 'attack_cat' mapping provided externally if needed.
# Categorical: protocol_type, service, flag
# Numeric: everything else excluding label-like columns

DEFAULT_CATEGORICAL = ['protocol_type', 'service', 'flag']

NSL_KDD_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
    'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label',
    'difficulty_level'
]

def load_nsl_kdd(csv_path):
    df = pd.read_csv(csv_path, names=NSL_KDD_COLUMNS, header=None)
    return df

def infer_columns(df, provided_cats=None):
    categorical = provided_cats or [c for c in DEFAULT_CATEGORICAL if c in df.columns]
    label_col = 'label' if 'label' in df.columns else None
    drop_cols = [c for c in ['difficulty_level'] if c in df.columns]
    numeric = [c for c in df.columns if c not in categorical + drop_cols + [label_col] and pd.api.types.is_numeric_dtype(df[c])]
    return categorical, numeric, label_col, drop_cols
