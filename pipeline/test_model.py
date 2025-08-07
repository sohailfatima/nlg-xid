import argparse
import yaml
import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn.functional as F

from data_loaders.nsl_kdd import load_nsl_kdd, infer_columns as infer_nsl
from data_loaders.unsw_nb15 import load_nb15, infer_columns as infer_nb15
from data_loaders.preprocess import split_xy, build_preprocess_pipeline
from models.autoencoder import SklearnAutoencoder

ATTACK_TO_CAT = {
    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "mailbomb": "DoS",
    "apache2": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",
    "worm": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    # normal
    "normal": "normal"
}

def main():
    parser = argparse.ArgumentParser(description='Test trained models for intrusion detection')
    parser.add_argument('--dataset', choices=['nsl-kdd', 'nb15'], required=True,
                        help='Dataset to use for testing')
    parser.add_argument('--train_path', required=True,
                        help='Path to train data file')
    parser.add_argument('--test_path', required=True,
                        help='Path to test data file')
    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (.pkl)')
    parser.add_argument('--config', default='configs/defaults.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model_type', choices=['xgb', 'ae'], default='xgb',
                        help='Model type to test')
    parser.add_argument('--ae_weights', default=None,
                        help='Path to AE weights (.pth), required for AE')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Load test data
    print(f"Loading test data from {args.test_path}")
    if args.dataset == 'nsl-kdd':
        train_df = load_nsl_kdd(args.train_path)
        test_df = load_nsl_kdd(args.test_path)
        test_df['label'] = test_df['label'].apply(lambda lbl: ATTACK_TO_CAT.get(lbl, 'unknown'))
        categorical, numeric, label_col, drop_cols = infer_nsl(test_df)
    else:
        train_df = load_nb15(args.train_path)
        test_df = load_nb15(args.test_path)
        categorical, numeric, label_col, drop_cols = infer_nb15(test_df)

    print(f"Loaded {len(test_df)} test samples")

    # Split features and labels
    X_train, y_train = split_xy(train_df, label_col, drop_cols)
    X_test, y_test = split_xy(test_df, label_col, drop_cols)
    
    pre = build_preprocess_pipeline(categorical, numeric)
    X_train_copy = pre.fit(X_train)
    X_test_copy = pre.transform(X_test)

    # Load model
    with open(args.model_path, 'rb') as f:
        pipe = pickle.load(f)

    print(f"Testing {args.model_type.upper()} model...")
    print(y_test.unique())

    if args.model_type == 'xgb':
        # XGBoost: predict and report
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        # Error can be 1-acc
        print(f"Test Error: {1-acc:.4f}")

    else:  # Autoencoder
        # AE: get AE from pipeline, load weights if provided
        ae = pipe.named_steps['clf']
        if args.ae_weights:
            ae.load_model(args.ae_weights)
        # Predict (reconstruction), compute CELoss
        # X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
        # Use the scaler from AE model
        X_test_scaled = ae.scaler.transform(X_test_copy)
        # Convert to tensor
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
        y_test_tensor = torch.tensor(y_test_np, dtype=torch.long).to(device)
        # Forward pass
        outputs = ae.model(X_test_tensor)
        ce_loss = F.cross_entropy(outputs, y_test_tensor)
        print(f"Test CrossEntropyLoss: {ce_loss.item():.4f}")
        # For AE, you may want to get predicted labels
        y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        acc = accuracy_score(y_test_np, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y_test_np, y_pred))
        print(f"Test Error: {1-acc:.4f}")

if __name__ == '__main__':
    main()