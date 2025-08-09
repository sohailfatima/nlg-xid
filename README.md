# NLG + XAI for Intrusion Detection (NSL-KDD & UNSW-NB15)

End-to-end pipeline to:
1) preprocess NSL-KDD and UNSW-NB15,
2) train Autoencoder (PyTorch-based unsupervised) and XGBoost (supervised),
3) compute SHAP explanations,
4) generate rule-based and LLM-augmented natural-language explanations,
5) run ablations and evaluate with BLEU and LLM-as-judge.

## Two-Stage Workflow

This codebase separates model training from evaluation to allow efficient multiple experiments:

1. **Training Stage**: Train models once and save them to disk
2. **Evaluation Stage**: Load trained models and run multiple experiments/ablations

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
```

### Training and Evaluation

#### Step 1: Train Models

```bash
# Train XGBoost on NSL-KDD
python3 eval.py --dataset nsl-kdd --train_path data/nsl_kdd/KDDTrain+.txt \
--test_path data/nsl_kdd/KDDTest+.txt --model_type xgb \
--model_path trained_models/nsl-kdd_xgb_model.pkl

# Train Autoencoder on NB15  
python3 eval.py --dataset nb15 --train_path data/unsw_nb15/nb15_train.csv \
--test_path data/unsw_nb15/nb15_test.csv --model_type ae \
--model_path trained_models/nb15_ae_model.pkl 
--ae_weights trained_models/nb15_ae_model_autoencoder.pth
```

#### Step 2: Run Evaluations (Multiple Times)

```bash
# Evaluate with rule-based explanations
python eval_model.py --model_path trained_models/nsl_kdd_xgb.pkl \
  --test_path data/nsl_kdd/KDDTest+.txt \
  --explain rules

# Evaluate with LLM explanations
python eval_model.py --model_path trained_models/nsl_kdd_xgb.pkl \
  --test_path data/nsl_kdd/KDDTest+.txt \
  --explain llm --llm_provider ollama --llm_model llama2

```

### Automated Ablation Studies

```bash
# Run comprehensive ablation study
!python3 pipeline/run_ablations.py \
    --model_path trained_models/nsl-kdd_ae_model.pkl \
    --test_path data/nsl_kdd/KDDTrain+200.txt \
    --api_key [api_key] \
    --llm_provider openrouter \
    --llm_model meta-llama/llama-3.1-8b-instruct \
    --explain_methods rules llm shap_only\
    --output_dir results/ablation_kdd_ae_train
```

## Features

- **Dataset Support**: NSL-KDD and UNSW-NB15 with dataset-aware feature glossaries
- **Models**: XGBoost (supervised) and PyTorch Autoencoder (unsupervised anomaly detection)
- **Explanations**: Rule-based, LLM-generated, SHAP-only
- **LLM Support**: Stub, Ollama, HuggingFace and OpenRouter providers with customizable models
- **Evaluation**: SHAP fidelity, BLEU scores, Rule Coverage, and LLM-as-judge evaluation
- **Efficient Workflow**: Train once, evaluate multiple times without retraining

## Directory Structure

- `train_model.py` - Train and save models
- `eval_model.py` - Load models and run evaluations (test model performance)  
- `run_ablations.py` - Automated ablation studies
- `trained_models/` - Saved model files and metadata
- `results/` - Evaluation results and explanations
- `explain/` - Explanation generation modules
- `eval/` - Evaluation metrics and LLM-as-judge
- `models/` - Model implementations
- `data_loaders/` - Data loading and preprocessing
- `data/` - Full test and train datasets + 200 sample balanced evalaution sets for ablations
- `Experiments.ipynb` - Output of all experiments ran

## Configuration

Configuration is managed via `configs/defaults.yaml`. 
