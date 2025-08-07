import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

def build_xgb_model(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8):
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method='hist'
    )
    return clf

def train_xgb(pipeline, X_train, y_train, X_val=None, y_val=None):
    pipeline.fit(X_train, y_train)
    return pipeline
