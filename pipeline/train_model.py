#!/usr/bin/env python3
"""
Train models and save them to disk for later evaluation.
This script handles both XGBoost and PyTorch Autoencoder models.
"""

import argparse
import yaml
import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline

from data_loaders.nsl_kdd import load_nsl_kdd, infer_columns as infer_nsl
from data_loaders.unsw_nb15 import load_nb15, infer_columns as infer_nb15
from data_loaders.preprocess import build_preprocess_pipeline, split_xy
from models.xgb import build_xgb_model, train_xgb
from models.autoencoder import SklearnAutoencoder
from utils.seed import set_seed

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
    parser = argparse.ArgumentParser(description='Train and save models for intrusion detection')
    parser.add_argument('--dataset', choices=['nsl-kdd', 'nb15'], required=True,
                        help='Dataset to use for training')
    parser.add_argument('--train_path', required=True,
                        help='Path to training data file')
    parser.add_argument('--model', choices=['xgb', 'ae'], default='xgb',
                        help='Model type to train')
    parser.add_argument('--config', default='configs/defaults.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', default='trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--model_name', default=None,
                        help='Custom name for saved model (default: auto-generated)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg.get('seed', 42))

    # Load training data
    print(f"Loading training data from {args.train_path}")
    if args.dataset == 'nsl-kdd':
        train_df = load_nsl_kdd(args.train_path)
        train_df['label'] = train_df['label'].apply(lambda lbl: ATTACK_TO_CAT.get(lbl, 'unknown'))
        categorical, numeric, label_col, drop_cols = infer_nsl(train_df)
    else:  # nb15
        train_df = load_nb15(args.train_path)
        categorical, numeric, label_col, drop_cols = infer_nb15(train_df)

    print(f"Loaded {len(train_df)} training samples")
    
    # Split features and labels
    X_train, y_train = split_xy(train_df, label_col, drop_cols)
    
    # Build preprocessing pipeline
    pre = build_preprocess_pipeline(categorical, numeric)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate model name if not provided
    if args.model_name is None:
        args.model_name = f"{args.dataset}_{args.model}_model"
    
    model_path = os.path.join(args.output_dir, f"{args.model_name}.pkl")
    metadata_path = os.path.join(args.output_dir, f"{args.model_name}_metadata.pkl")
    
    print(f"Training {args.model.upper()} model...")
    
    if args.model == 'xgb':
        # Build and train XGBoost model
        print(y_train.unique())
        clf = build_xgb_model(**cfg['model']['xgb'])
        pipe = Pipeline([('pre', pre), ('clf', clf)])
        pipe = train_xgb(pipe, X_train, y_train)
        
        # Save the pipeline
        with open(model_path, 'wb') as f:
            pickle.dump(pipe, f)
        
        print(f"XGBoost model trained and saved to {model_path}")
        
    else:  # autoencoder
        # Build and train Autoencoder
        print(y_train.unique())
        ae = SklearnAutoencoder(**cfg['model']['ae'])
        pipe = Pipeline([('pre', pre), ('clf', ae)])
        pipe.fit(X_train, y_train)
        
        # Save the preprocessing pipeline separately
        with open(model_path, 'wb') as f:
            pickle.dump(pipe, f)
        
        # Save the PyTorch autoencoder model separately for better loading
        ae_model_path = os.path.join(args.output_dir, f"{args.model_name}_autoencoder.pth")
        ae.save_model(ae_model_path)
        
        print(f"Autoencoder model trained and saved to {model_path}")
        print(f"PyTorch autoencoder weights saved to {ae_model_path}")
    
    # Save metadata about the training
    metadata = {
        'dataset': args.dataset,
        'model_type': args.model,
        'train_path': args.train_path,
        'config_used': cfg,
        'feature_info': {
            'categorical': categorical,
            'numeric': numeric,
            'label_col': label_col,
            'drop_cols': drop_cols
        },
        'training_samples': len(train_df),
        'model_path': model_path
    }
    
    if args.model == 'ae':
        metadata['autoencoder_weights_path'] = ae_model_path
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Training metadata saved to {metadata_path}")
    print("Training completed successfully!")
    
    


if __name__ == '__main__':
    main()
