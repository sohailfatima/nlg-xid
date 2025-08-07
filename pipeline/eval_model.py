#!/usr/bin/env python3
"""
Load trained models and run experiments/evaluations without retraining.
This script supports running explanations, metrics, and LLM-as-judge evaluations.
"""

import argparse
import json
import yaml
import os
import pickle
import numpy as np
from sklearn.pipeline import Pipeline
import sys
import os

# Add parent directory of data_loaders to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loaders.nsl_kdd import load_nsl_kdd, infer_columns as infer_nsl
from data_loaders.unsw_nb15 import load_nb15, infer_columns as infer_nb15
from data_loaders.preprocess import split_xy
from explain.shap_utils import shap_for_xgb, shap_for_scorer
from explain.glossary import load_glossary_for_dataset
from utils.io import save_json, load_json
from utils.seed import set_seed
from pipeline.generate_explanations import build_explanations


def load_trained_model(model_path, metadata_path):
    """Load a trained model and its metadata"""
    print(f"Loading trained model from {model_path}")
    
    # Load metadata
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load the pipeline
    with open(model_path, 'rb') as f:
        pipe = pickle.load(f)
    
    # For autoencoders, also load the PyTorch weights if available
    if metadata['model_type'] == 'ae' and 'autoencoder_weights_path' in metadata:
        ae_weights_path = metadata['autoencoder_weights_path']
        if os.path.exists(ae_weights_path):
            print(f"Loading PyTorch autoencoder weights from {ae_weights_path}")
            pipe.named_steps['clf'].load_model(ae_weights_path)
        else:
            print(f"Warning: PyTorch weights file not found at {ae_weights_path}")
    
    return pipe, metadata


def main():
    parser = argparse.ArgumentParser(description='Run experiments on trained models')
    parser.add_argument('--model_path', required=True,
                        help='Path to trained model (.pkl file)')
    parser.add_argument('--test_path', required=True,
                        help='Path to test data file')
    parser.add_argument('--glossary_path', default=None,
                        help='Path to custom glossary file (optional)')
    parser.add_argument('--explain', choices=['rules', 'llm', 'hybrid', 'shap_only'], 
                        default='rules', help='Explanation method to use')
    parser.add_argument('--config', default='configs/defaults.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', default='outputs',
                        help='Directory to save results')
    parser.add_argument('--ablation_llm_inputs', 
                        choices=['label', 'label+features', 'label+shap', 'full'], 
                        default='full', help='LLM input ablation setting')
    parser.add_argument('--llm_provider', choices=['stub', 'ollama', 'huggingface','openrouter'], 
                        default='stub', help='LLM provider to use')
    parser.add_argument('--llm_model', default=None,
                        help='Specific LLM model name')
    parser.add_argument('--judge', action='store_true',
                        help='Enable LLM-as-judge evaluation')
    parser.add_argument('--experiment_name', default=None,
                        help='Custom name for this experiment (for output files)')
    parser.add_argument('--api_key', default=None,
                        help='API key for LLM provider (if required)')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg.get('seed', 42))

    # Load trained model and metadata
    metadata_path = args.model_path.replace('.pkl', '_metadata.pkl')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    pipe, metadata = load_trained_model(args.model_path, metadata_path)
    
    # Extract information from metadata
    dataset = metadata['dataset']
    model_type = metadata['model_type']
    feature_info = metadata['feature_info']
    categorical = feature_info['categorical']
    numeric = feature_info['numeric']
    label_col = feature_info['label_col']
    drop_cols = feature_info['drop_cols']
    
    print(f"Loaded {model_type.upper()} model trained on {dataset} dataset")
    print(f"Original training samples: {metadata['training_samples']}")

    # Load test data
    print(f"Loading test data from {args.test_path}")
    if dataset == 'nsl-kdd':
        test_df = load_nsl_kdd(args.test_path)
        # Verify column structure matches training
        test_categorical, test_numeric, test_label_col, test_drop_cols = infer_nsl(test_df)
    else:  # nb15
        test_df = load_nb15(args.test_path)
        test_categorical, test_numeric, test_label_col, test_drop_cols = infer_nb15(test_df)
    
    print(f"Loaded {len(test_df)} test samples")
    
    # Split test features and labels
    X_test, y_test = split_xy(test_df, label_col, drop_cols)
    
    # Load glossary
    if args.glossary_path:
        glossary = load_json(args.glossary_path)
    else:
        # Load dataset-specific glossary from YAML files
        glossary = load_glossary_for_dataset(dataset)

    print(f"Running inference and generating explanations...")
    
    # Generate predictions and SHAP values
    if model_type == 'xgb':
        # XGBoost model
        shap_df, top_list = shap_for_xgb(pipe, X_test, 
                                        cfg['shap']['background_samples'], 
                                        cfg['top_k'])
        # Predictions
        preds_proba = pipe.predict_proba(X_test)
        preds = pipe.classes_[np.argmax(preds_proba, axis=1)]
        
        # Prepare per-instance SHAP dict
        shap_dicts = [dict(zip(shap_df.columns, shap_df.iloc[i].values)) 
                     for i in range(shap_df.shape[0])]
        top_lists = [top_list for _ in range(len(shap_dicts))]
        
    else:  # autoencoder
        # Autoencoder model (now classification-based)
        pre_fitted = pipe.named_steps['pre']
        ae_model = pipe.named_steps['clf']
        
        # Use predict_proba for scoring function
        scorer = lambda X_trans: ae_model.predict_proba(X_trans)[:, 1]  # Use positive class probability
        
        shap_df, top_list = shap_for_scorer(pipe, scorer, X_test, 
                                          cfg['shap']['background_samples'], 
                                          cfg['top_k'], 
                                          cfg['shap']['nsamples_kernel'])
        
        # Get predictions directly from the model
        # Get numeric predictions from the model
        numeric_preds = ae_model.predict(pre_fitted.transform(X_test))
        
        # Map numeric predictions to class labels based on dataset
        if dataset == 'nb15':
            # NB15 class mapping used during training
            class_names = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode', 
                          'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']
        else:  # nsl-kdd
            # For NSL-KDD, we'd need to define the class mapping here
            # This would depend on how the autoencoder was trained for NSL-KDD
            class_names = ['normal', 'dos', 'probe', 'r2l', 'u2r']  # Common NSL-KDD classes
        
        # Map numeric predictions to class names
        preds = [class_names[int(pred)] if int(pred) < len(class_names) else f'unknown_{pred}' 
                for pred in numeric_preds]
        
        shap_dicts = [dict(zip(shap_df.columns, shap_df.iloc[i].values)) 
                     for i in range(shap_df.shape[0])]
        top_lists = [top_list for _ in range(len(shap_dicts))]

    # Build instances as dict (original feature space for readability)
    instances = [X_test.iloc[i].to_dict() for i in range(len(shap_dicts))]
    
    # Generate explanations
    print(f"Generating {args.explain} explanations...")
    exps = build_explanations(
        dataset=dataset, 
        preds=[str(p) for p in preds], 
        shap_frames=shap_dicts, 
        instances=instances,
        glossary=glossary, 
        top_lists=top_lists, 
        mode=args.explain, 
        llm_provider=args.llm_provider,
        llm_inputs=args.ablation_llm_inputs, 
        llm_variant=cfg['llm']['variant'], 
        llm_model=args.llm_model,
        api_key=args.api_key
    )

    # Evaluate explanations
    print("Computing evaluation metrics...")
    metrics = {}
    
    # Import metrics functions when needed
    from eval.metrics import shap_fidelity_check
    
    # Check for gold references for BLEU evaluation
    exp_name = args.experiment_name or f"{dataset}_{model_type}_{args.explain}"
    gold_path = os.path.join(args.output_dir, f"{dataset}_gold_refs.json")
    
    if os.path.exists(gold_path):
        gold = json.load(open(gold_path))
        # Expect a dict with "refs": List[List[str]] aligned to exps
        bleu = 0.0
        try:
            from eval.metrics import bleu_score
            bleu = bleu_score(exps, gold['refs'], cfg['eval']['bleu_ngram'])
        except Exception as e:
            print(f"Warning: BLEU computation failed: {e}")
            bleu = -1.0
        metrics['bleu'] = bleu
    else:
        print(f"No gold references found at {gold_path}, skipping BLEU evaluation")

    # SHAP fidelity proxy
    metrics['shap_fidelity'] = shap_fidelity_check(exps, top_lists)

    # LLM-as-judge evaluation (optional)
    if args.judge or cfg['eval'].get('judge', False):
        print("Running LLM-as-judge evaluation...")
        from eval.judge import LLMJudge
        
        judge = LLMJudge(
            provider=cfg['eval'].get('judge_provider', 'stub'), 
            model=cfg['eval'].get('judge_model', 'gpt-4o-mini'), 
            weights=cfg['eval'].get('judge_weights', None),
            api_key=args.api_key
        )
        
        labels_for_cases = [str(p) for p in preds]
        def snip(d: dict): 
            return ", ".join(f"{k}={v}" for k, v in list(d.items())[:20])
        instance_snips = [snip(inst) for inst in instances]
        
        judge_scores = judge.score_cases(labels_for_cases, exps, top_lists, 
                                       shap_dicts, instance_snips)
        
        # Save judge scores separately
        os.makedirs(args.output_dir, exist_ok=True)
        save_json(judge_scores, 
                 os.path.join(args.output_dir, f"{exp_name}_judge.json"))
        
        metrics['judge_overall'] = judge_scores.get('summary', {}).get('overall', None)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save explanations
    explanations_output = {
        'explanations': exps,
        'metadata': {
            'model_type': model_type,
            'dataset': dataset,
            'test_samples': len(X_test),
            'explanation_method': args.explain,
            'llm_provider': args.llm_provider,
            'llm_model': args.llm_model,
            'experiment_name': exp_name
        }
    }
    
    save_json(explanations_output, 
             os.path.join(args.output_dir, f"{exp_name}_explanations.json"))
    
    # Save metrics
    metrics_output = {
        'metrics': metrics,
        'metadata': {
            'model_type': model_type,
            'dataset': dataset,
            'test_samples': len(X_test),
            'explanation_method': args.explain,
            'experiment_name': exp_name
        }
    }
    
    save_json(metrics_output, 
             os.path.join(args.output_dir, f"{exp_name}_metrics.json"))
    
    print(f"Results saved to {args.output_dir}:")
    print(f"  - Explanations: {exp_name}_explanations.json")
    print(f"  - Metrics: {exp_name}_metrics.json")
    if args.judge or cfg['eval'].get('judge', False):
        print(f"  - Judge scores: {exp_name}_judge.json")
    
    # Print summary metrics
    print("\nEvaluation Summary:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")
    
    print("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
