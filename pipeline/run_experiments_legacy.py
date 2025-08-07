import argparse, json, yaml, os
import numpy as np
from sklearn.pipeline import Pipeline
from data_loaders.nsl_kdd import load_nsl_kdd, infer_columns as infer_nsl
from data_loaders.unsw_nb15 import load_nb15, infer_columns as infer_nb15
from data_loaders.preprocess import build_preprocess_pipeline, split_xy
from models.xgb import build_xgb_model, train_xgb
from models.autoencoder import SklearnAutoencoder
from explain.shap_utils import shap_for_xgb, shap_for_scorer
from explain.glossary import load_glossary_for_dataset
from utils.io import save_json, load_json
from utils.seed import set_seed
from pipeline.generate_explanations import build_explanations

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['nsl-kdd','nb15'], required=True)
    parser.add_argument('--train_path', required=True, help='Path to training data file')
    parser.add_argument('--test_path', required=True, help='Path to test data file')
    parser.add_argument('--glossary_path', default=None)
    parser.add_argument('--model', choices=['xgb','ae'], default='xgb')
    parser.add_argument('--explain', choices=['rules','llm','hybrid','shap_only'], default='rules')
    parser.add_argument('--config', default='configs/defaults.yaml')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--ablation_llm_inputs', choices=['label','label+features','label+shap','full'], default='full')
    parser.add_argument('--llm_provider', choices=['stub','ollama','huggingface'], default='stub', help='LLM provider to use')
    parser.add_argument('--llm_model', default=None, help='Specific LLM model name (e.g., llama2, meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument('--judge', action='store_true', help='Enable LLM-as-judge evaluation (overrides config).')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))

    # Load data
    if args.dataset == 'nsl-kdd':
        train_df = load_nsl_kdd(args.train_path)
        test_df = load_nsl_kdd(args.test_path)
        categorical, numeric, label_col, drop_cols = infer_nsl(train_df)
    else:
        train_df = load_nb15(args.train_path)
        test_df = load_nb15(args.test_path)
        categorical, numeric, label_col, drop_cols = infer_nb15(train_df)

    if args.glossary_path:
        glossary = load_json(args.glossary_path)
    else:
        # Load dataset-specific glossary from YAML files
        glossary = load_glossary_for_dataset(args.dataset)

    # Split X and y for train and test sets
    X_train, y_train = split_xy(train_df, label_col, drop_cols)
    X_test, y_test = split_xy(test_df, label_col, drop_cols)
    
    pre = build_preprocess_pipeline(categorical, numeric)

    # Build model pipeline
    if args.model == 'xgb':
        clf = build_xgb_model(**cfg['model']['xgb'])
        pipe = Pipeline([('pre', pre), ('clf', clf)])
        pipe = train_xgb(pipe, X_train, y_train)
        # SHAP for XGB
        shap_df, top_list = shap_for_xgb(pipe, X_test, cfg['shap']['background_samples'], cfg['top_k'])
        # Predictions
        preds_proba = pipe.predict_proba(X_test)
        preds = pipe.classes_[np.argmax(preds_proba, axis=1)]
        # Prepare per-instance SHAP dict
        shap_dicts = [dict(zip(shap_df.columns, shap_df.iloc[i].values)) for i in range(shap_df.shape[0])]
        top_lists = [top_list for _ in range(len(shap_dicts))]
    else:
        # Autoencoder: fit on benign/normal only (y==0 or 'normal')
        ae = SklearnAutoencoder(**cfg['model']['ae'])
        pipe = Pipeline([('pre', pre), ('clf', ae)])
        pipe.fit(X_train, y_train)
        # scorer in transformed space
        pre_fitted = pipe.named_steps['pre']
        ae_model = pipe.named_steps['clf']
        scorer = lambda X_trans: ae_model.decision_function(X_trans)
        shap_df, top_list = shap_for_scorer(pipe, scorer, X_test, cfg['shap']['background_samples'], cfg['top_k'], cfg['shap']['nsamples_kernel'])
        scores = ae_model.decision_function(pre_fitted.transform(X_test))
        preds = (scores > ae_model.threshold).astype(int)
        shap_dicts = [dict(zip(shap_df.columns, shap_df.iloc[i].values)) for i in range(shap_df.shape[0])]
        top_lists = [top_list for _ in range(len(shap_dicts))]

    # Build instances as dict (original feature space for readability)
    instances = [X_test.iloc[i].to_dict() for i in range(len(shap_dicts))]
    # Explanations
    exps = build_explanations(
        dataset=args.dataset, preds=[str(p) for p in preds], shap_frames=shap_dicts, instances=instances,
        glossary=glossary, top_lists=top_lists, mode=args.explain, llm_provider=args.llm_provider,
        llm_inputs=args.ablation_llm_inputs, llm_variant=cfg['llm']['variant'], llm_model=args.llm_model
    )

    # Optional evaluation: BLEU requires gold references (provide at outputs/gold_refs.json if available)
    gold_path = os.path.join(args.output_dir, f"{args.dataset}_gold_refs.json")
    metrics = {}
    
    # Import metrics functions when needed
    from eval.metrics import shap_fidelity_check
    
    if os.path.exists(gold_path):
        gold = json.load(open(gold_path))
        # Expect a dict with "refs": List[List[str]] aligned to exps
        bleu = 0.0
        try:
            from eval.metrics import bleu_score
            bleu = bleu_score(exps, gold['refs'], cfg['eval']['bleu_ngram'])
        except Exception:
            bleu = -1.0
        metrics['bleu'] = bleu

    # SHAP fidelity proxy
    metrics['shap_fidelity'] = shap_fidelity_check(exps, top_lists)

    # LLM-as-judge (optional)
    if args.judge or cfg['eval'].get('judge', False):
        from eval.judge import LLMJudge
        judge = LLMJudge(provider=cfg['eval'].get('judge_provider','stub'), model=cfg['eval'].get('judge_model','gpt-4o-mini'), weights=cfg['eval'].get('judge_weights', None))
        labels_for_cases = [str(p) for p in preds]
        def snip(d: dict): return ", ".join(f"{k}={v}" for k,v in list(d.items())[:20])
        instance_snips = [snip(inst) for inst in instances]
        judge_scores = judge.score_cases(labels_for_cases, exps, top_lists, shap_dicts, instance_snips)
        from utils.io import save_json
        save_json(judge_scores, os.path.join(args.output_dir, f"{args.dataset}_{args.model}_{args.explain}_judge.json"))
        metrics['judge_overall'] = judge_scores.get('summary',{}).get('overall', None)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    save_json({'explanations': exps}, os.path.join(args.output_dir, f"{args.dataset}_{args.model}_{args.explain}_explanations.json"))
    save_json({'metrics': metrics}, os.path.join(args.output_dir, f"{args.dataset}_{args.model}_{args.explain}_metrics.json"))
    print("Saved explanations and metrics to", args.output_dir)

if __name__ == '__main__':
    main()
